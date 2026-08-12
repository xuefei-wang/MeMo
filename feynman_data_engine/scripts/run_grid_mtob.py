"""MTOB grid: engine x budget x seed -> data-gen -> Qwen3-8B LoRA-SFT -> closed-book
chrF eval -> curve. Resumable. Data-gen hits the shared Qwen3 vLLM endpoints; SFT
and eval are pinned round-robin to GPUs."""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")
TRAIN_GPUS = ["4", "5", "6"]
BASE = "Qwen/Qwen3-8B"


def cell_dir(root, mode, budget, seed):
    return root / f"{mode}__b{budget}__s{seed}"


def sh(cmd, gpu, log):
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as f:
        return subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "runs" / "grid_mtob"))
    ap.add_argument("--budgets", type=int, nargs="+", default=[6000, 20000])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--modes", nargs="+", default=["extraction_only", "feynman_core"])
    ap.add_argument("--gen_conc", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--cheatsheet_eval", action="store_true",
                    help="also eval feynman cells with base model + cheatsheet as "
                         "amortized context (no SFT) -- the feynman-cheatsheet method")
    args = ap.parse_args()

    root = Path(args.out); root.mkdir(parents=True, exist_ok=True)
    cells = list(itertools.product(args.modes, args.budgets, args.seeds))
    print(f"[grid-mtob] {len(cells)} cells -> {root}")

    # ---- Phase A: data-gen (concurrent, shared endpoints) ----
    def gen_one(cell):
        mode, budget, seed = cell
        d = cell_dir(root, mode, budget, seed)
        if (d / "dataset.jsonl").exists() and (d / "meta.json").exists():
            return cell, "skip"
        rc = sh([PY, str(ROOT / "scripts" / "run_engine_mtob.py"), "--mode", mode,
                 "--budget", str(budget), "--seed", str(seed), "--out", str(d)],
                None, d / "gen.log")
        return cell, "ok" if rc == 0 else f"FAIL:{rc}"

    print("[grid-mtob] Phase A: data-gen ...")
    with ThreadPoolExecutor(max_workers=args.gen_conc) as ex:
        for c, s in ex.map(gen_one, cells):
            print(f"  gen {c} -> {s}", flush=True)

    # ---- Phase B: SFT Qwen3-8B (round-robin GPUs) ----
    def train_one(idx_cell):
        idx, cell = idx_cell
        mode, budget, seed = cell
        d = cell_dir(root, mode, budget, seed)
        adir = d / "adapter"
        if (adir / "adapter_config.json").exists():
            return cell, "skip"
        if not (d / "dataset.jsonl").exists():
            return cell, "no-data"
        gpu = TRAIN_GPUS[idx % len(TRAIN_GPUS)]
        rc = sh([PY, str(ROOT / "learner" / "sft.py"), "--base", BASE,
                 "--dataset", str(d / "dataset.jsonl"), "--out", str(adir),
                 "--epochs", str(args.epochs), "--bs", "2", "--grad_accum", "8",
                 "--max_len", "1024"], gpu, d / "sft.log")
        return cell, "ok" if rc == 0 else f"FAIL:{rc}"

    print("[grid-mtob] Phase B: SFT ...")
    with ThreadPoolExecutor(max_workers=len(TRAIN_GPUS)) as ex:
        for c, s in ex.map(train_one, list(enumerate(cells))):
            print(f"  sft {c} -> {s}", flush=True)

    # ---- Phase C: closed-book chrF eval (round-robin GPUs) ----
    def eval_one(idx_cell):
        idx, cell = idx_cell
        mode, budget, seed = cell
        d = cell_dir(root, mode, budget, seed)
        adir = d / "adapter"; outp = d / "eval.json"
        if outp.exists():
            return cell, "skip"
        if not (adir / "adapter_config.json").exists():
            return cell, "no-adapter"
        gpu = TRAIN_GPUS[idx % len(TRAIN_GPUS)]
        rc = sh([PY, str(ROOT / "learner" / "eval_mtob.py"), "--base", BASE,
                 "--adapter", str(adir), "--direction", "ke", "--context", "none",
                 "--n", "50", "--no_think", "--bs", "16", "--max_new", "128",
                 "--out", str(outp)], gpu, d / "eval.log")
        return cell, "ok" if rc == 0 else f"FAIL:{rc}"

    print("[grid-mtob] Phase C: eval ...")
    with ThreadPoolExecutor(max_workers=len(TRAIN_GPUS)) as ex:
        for c, s in ex.map(eval_one, list(enumerate(cells))):
            print(f"  eval {c} -> {s}", flush=True)

    # ---- Phase C2: feynman-cheatsheet eval (base model + cheatsheet, no SFT) ----
    # The blog's amortized-context method: prepend the engine's failure-targeted
    # cheat-sheet at inference instead of (or on top of) weight updates. Only feynman
    # cells produce a cheatsheet.txt; extraction cells are skipped.
    if args.cheatsheet_eval:
        def cheat_eval_one(idx_cell):
            idx, cell = idx_cell
            mode, budget, seed = cell
            d = cell_dir(root, mode, budget, seed)
            cheat = d / "cheatsheet.txt"; outp = d / "eval_cheatsheet.json"
            if outp.exists():
                return cell, "skip"
            if not cheat.exists():
                return cell, "no-cheatsheet"
            gpu = TRAIN_GPUS[idx % len(TRAIN_GPUS)]
            rc = sh([PY, str(ROOT / "learner" / "eval_mtob.py"), "--base", BASE,
                     "--direction", "ke", "--context", "cheatsheet",
                     "--cheatsheet_file", str(cheat), "--n", "50", "--no_think",
                     "--bs", "16", "--max_new", "128", "--out", str(outp)],
                    gpu, d / "eval_cheatsheet.log")
            return cell, "ok" if rc == 0 else f"FAIL:{rc}"

        print("[grid-mtob] Phase C2: feynman-cheatsheet eval (base + cheatsheet) ...")
        with ThreadPoolExecutor(max_workers=len(TRAIN_GPUS)) as ex:
            for c, s in ex.map(cheat_eval_one, list(enumerate(cells))):
                print(f"  cheat-eval {c} -> {s}", flush=True)

    # ---- Phase D: aggregate ----
    rows = []
    for cell in cells:
        mode, budget, seed = cell
        d = cell_dir(root, mode, budget, seed)
        try:
            meta = json.loads((d / "meta.json").read_text())
            ev = json.loads((d / "eval.json").read_text())["summary"]
            row = {"mode": mode, "budget": budget, "seed": seed,
                   "synth_tokens": meta["emitted_synthetic_tokens"],
                   "gen_tokens": meta["total_gen_completion_tokens"],
                   "n_notes": meta["n_examples"] - 375,
                   "chrf": ev["corpus_chrf"]}
            cheat_p = d / "eval_cheatsheet.json"
            if cheat_p.exists():  # feynman-cheatsheet method (base model + cheatsheet)
                row["chrf_cheatsheet"] = \
                    json.loads(cheat_p.read_text())["summary"]["corpus_chrf"]
            rows.append(row)
        except Exception as e:  # noqa: BLE001
            rows.append({"mode": mode, "budget": budget, "seed": seed, "error": str(e)})
    (root / "results.json").write_text(json.dumps(rows, indent=2))
    print(f"[grid-mtob] wrote {root/'results.json'}")
    for r in rows:
        if "error" in r:
            print(f"  {r['mode']:16s} b{r['budget']:>6} s{r['seed']}  ERROR {r['error']}")
        else:
            cheat = f" cheatsheet={r['chrf_cheatsheet']:.2f}" if "chrf_cheatsheet" in r else ""
            print(f"  {r['mode']:16s} b{r['budget']:>6} s{r['seed']}  "
                  f"synth={r['synth_tokens']:>6} gen={r['gen_tokens']:>7} "
                  f"chrF={r['chrf']:.2f}{cheat}")


if __name__ == "__main__":
    main()
