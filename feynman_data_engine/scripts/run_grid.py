"""Grid orchestrator: engine x budget x seed -> data-gen -> SFT -> eval -> curve.

Resumable (skips cells whose outputs already exist). Data-gen runs concurrently
against the shared vLLM servers; SFT and eval are pinned round-robin to GPUs.
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")
TRAIN_GPUS = ["3", "4", "5", "6"]


def cell_dir(root: Path, mode: str, budget: int, seed: int) -> Path:
    return root / f"{mode}__b{budget}__s{seed}"


def sh(cmd: list[str], gpu: str | None, log: Path) -> int:
    import os
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as f:
        p = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "runs" / "grid"))
    ap.add_argument("--budgets", type=int, nargs="+", default=[30000, 80000, 160000])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--modes", nargs="+", default=["extraction_only", "feynman_core"])
    ap.add_argument("--gen_conc", type=int, default=6)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--eval_n", type=int, default=100)
    args = ap.parse_args()

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    cells = list(itertools.product(args.modes, args.budgets, args.seeds))
    print(f"[grid] {len(cells)} cells -> {root}")

    # ---- Phase A: data generation (concurrent, shared servers) ----
    def gen_one(cell):
        mode, budget, seed = cell
        d = cell_dir(root, mode, budget, seed)
        if (d / "dataset.jsonl").exists() and (d / "meta.json").exists():
            return (cell, "skip")
        rc = sh([PY, str(ROOT / "scripts" / "run_engine.py"),
                 "--mode", mode, "--budget", str(budget), "--seed", str(seed),
                 "--out", str(d)], gpu=None, log=d / "gen.log")
        return (cell, "ok" if rc == 0 else f"FAIL:{rc}")

    print("[grid] Phase A: data-gen ...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.gen_conc) as ex:
        for cell, status in ex.map(gen_one, cells):
            print(f"  gen {cell} -> {status}", flush=True)
    print(f"[grid] Phase A done in {time.time()-t0:.0f}s")

    # ---- Phase B: SFT (round-robin GPUs) ----
    def train_one(idx_cell):
        idx, cell = idx_cell
        mode, budget, seed = cell
        d = cell_dir(root, mode, budget, seed)
        adir = d / "adapter"
        if (adir / "adapter_config.json").exists():
            return (cell, "skip")
        if not (d / "dataset.jsonl").exists():
            return (cell, "no-data")
        gpu = TRAIN_GPUS[idx % len(TRAIN_GPUS)]
        rc = sh([PY, str(ROOT / "learner" / "sft.py"),
                 "--dataset", str(d / "dataset.jsonl"), "--out", str(adir),
                 "--epochs", str(args.epochs)], gpu=gpu, log=d / "sft.log")
        return (cell, "ok" if rc == 0 else f"FAIL:{rc}")

    print("[grid] Phase B: SFT ...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(TRAIN_GPUS)) as ex:
        for cell, status in ex.map(train_one, list(enumerate(cells))):
            print(f"  sft {cell} -> {status}", flush=True)
    print(f"[grid] Phase B done in {time.time()-t0:.0f}s")

    # ---- Phase C: eval (round-robin GPUs) ----
    def eval_one(idx_cell):
        idx, cell = idx_cell
        mode, budget, seed = cell
        d = cell_dir(root, mode, budget, seed)
        adir = d / "adapter"
        outp = d / "eval.json"
        if outp.exists():
            return (cell, "skip")
        if not (adir / "adapter_config.json").exists():
            return (cell, "no-adapter")
        gpu = TRAIN_GPUS[idx % len(TRAIN_GPUS)]
        rc = sh([PY, str(ROOT / "learner" / "eval_learner.py"),
                 "--adapter", str(adir), "--n", str(args.eval_n),
                 "--out", str(outp)], gpu=gpu, log=d / "eval.log")
        return (cell, "ok" if rc == 0 else f"FAIL:{rc}")

    print("[grid] Phase C: eval ...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(TRAIN_GPUS)) as ex:
        for cell, status in ex.map(eval_one, list(enumerate(cells))):
            print(f"  eval {cell} -> {status}", flush=True)
    print(f"[grid] Phase C done in {time.time()-t0:.0f}s")

    # ---- Phase D: aggregate ----
    rows = []
    for cell in cells:
        mode, budget, seed = cell
        d = cell_dir(root, mode, budget, seed)
        try:
            meta = json.loads((d / "meta.json").read_text())
            ev = json.loads((d / "eval.json").read_text())["summary"]
            rows.append({"mode": mode, "budget": budget, "seed": seed,
                         "emitted_tokens": meta["emitted_tokens"],
                         "gen_completion_tokens": meta["total_gen_completion_tokens"],
                         "n_examples": meta["n_examples"],
                         "within20": ev["within20"], "within10": ev["within10"],
                         "median_rel_err": ev["median_rel_err"],
                         "mean_rel_err": ev["mean_rel_err"],
                         "exact_acc": ev["exact_acc"]})
        except Exception as e:  # noqa: BLE001
            rows.append({"mode": mode, "budget": budget, "seed": seed, "error": str(e)})
    (root / "results.json").write_text(json.dumps(rows, indent=2))
    print(f"[grid] wrote {root/'results.json'}")
    for r in rows:
        if "error" in r:
            print(f"  {r['mode']:16s} b{r['budget']:>7} s{r['seed']}  ERROR {r['error']}")
        else:
            print(f"  {r['mode']:16s} b{r['budget']:>7} s{r['seed']}  "
                  f"emit={r['emitted_tokens']:>7} gen={r['gen_completion_tokens']:>8} "
                  f"within20={r['within20']:.3f} medRelErr={r['median_rel_err']:.3f}")


if __name__ == "__main__":
    main()
