"""Head-to-head grid: for each seed build ONE shared pool -> sieve_gen + feynman_gen
datasets -> matched LoRA-SFT (SIEVE's MTOB config) -> closed-book chrF eval -> compare.
Only the selection policy differs, so the chrF gap is the pure data-engine effect."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")
ARMS = ["sieve_gen", "feynman_gen", "gold_only"]


def sh(cmd, gpu, log):
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as f:
        return subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "runs" / "grid_h2h"))
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--pool_factor", type=float, default=3.0)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--epochs", type=float, default=5.0)
    ap.add_argument("--train_gpus", nargs="+", default=["6"])
    ap.add_argument("--base", default="Qwen/Qwen3-8B")
    args = ap.parse_args()
    root = Path(args.out); root.mkdir(parents=True, exist_ok=True)
    tg = args.train_gpus

    # ---- Phase A: build pools (sequential; each pool build is already concurrent) ----
    print(f"[h2h-grid] Phase A: pools (n={args.n} x{args.pool_factor}) ...", flush=True)
    for seed in args.seeds:
        d = root / f"s{seed}"
        if (d / "sieve_gen" / "dataset.jsonl").exists():
            print(f"  pool s{seed} -> skip"); continue
        rc = sh([PY, str(ROOT / "scripts" / "run_headtohead.py"), "--n", str(args.n),
                 "--seed", str(seed), "--pool_factor", str(args.pool_factor),
                 "--concurrency", str(args.concurrency), "--out", str(d)],
                None, d / "gen.log")
        print(f"  pool s{seed} -> {'ok' if rc == 0 else f'FAIL:{rc}'}", flush=True)

    cells = [(seed, arm) for seed in args.seeds for arm in ARMS]

    # ---- Phase B: SFT (matched config, round-robin GPUs) ----
    def train(idx_cell):
        idx, (seed, arm) = idx_cell
        d = root / f"s{seed}" / arm
        adir = d / "adapter"
        if (adir / "adapter_config.json").exists():
            return (seed, arm), "skip"
        if not (d / "dataset.jsonl").exists():
            return (seed, arm), "no-data"
        gpu = tg[idx % len(tg)]
        # LoRA (not SIEVE's full-FT), so lr is LoRA-appropriate 2e-4, NOT their 1e-5;
        # SIEVE's 5 epochs + effective-batch 64 kept. Identical across arms => unbiased.
        rc = sh([PY, str(ROOT / "learner" / "sft.py"), "--base", args.base,
                 "--dataset", str(d / "dataset.jsonl"), "--out", str(adir),
                 "--epochs", str(args.epochs), "--lr", "2e-4", "--bs", "4",
                 "--grad_accum", "16", "--max_len", "512"], gpu, d / "sft.log")
        return (seed, arm), "ok" if rc == 0 else f"FAIL:{rc}"

    print("[h2h-grid] Phase B: SFT (5ep, LoRA lr2e-4, eff-batch 64) ...", flush=True)
    with ThreadPoolExecutor(max_workers=len(tg)) as ex:
        for c, s in ex.map(train, list(enumerate(cells))):
            print(f"  sft {c} -> {s}", flush=True)

    # ---- Phase C: closed-book chrF eval ----
    def evl(idx_cell):
        idx, (seed, arm) = idx_cell
        d = root / f"s{seed}" / arm
        adir, outp = d / "adapter", d / "eval.json"
        if outp.exists():
            return (seed, arm), "skip"
        if not (adir / "adapter_config.json").exists():
            return (seed, arm), "no-adapter"
        gpu = tg[idx % len(tg)]
        rc = sh([PY, str(ROOT / "learner" / "eval_mtob.py"), "--base", args.base,
                 "--adapter", str(adir), "--direction", "ke", "--context", "none",
                 "--n", "50", "--no_think", "--bs", "16", "--max_new", "128",
                 "--out", str(outp)], gpu, d / "eval.log")
        return (seed, arm), "ok" if rc == 0 else f"FAIL:{rc}"

    print("[h2h-grid] Phase C: eval ...", flush=True)
    with ThreadPoolExecutor(max_workers=len(tg)) as ex:
        for c, s in ex.map(evl, list(enumerate(cells))):
            print(f"  eval {c} -> {s}", flush=True)

    # ---- Phase D: aggregate ----
    rows = []
    for seed, arm in cells:
        d = root / f"s{seed}" / arm
        try:
            ev = json.loads((d / "eval.json").read_text())["summary"]
            meta = json.loads((d / "meta.json").read_text())
            rows.append({"seed": seed, "arm": arm, "chrf": ev["corpus_chrf"],
                         "sel_median_student_chrf": meta.get("selected_median_student_chrf"),
                         "n": ev["n"]})
        except Exception as e:  # noqa: BLE001
            rows.append({"seed": seed, "arm": arm, "error": str(e)})
    (root / "results.json").write_text(json.dumps(rows, indent=2))
    print(f"\n[h2h-grid] results -> {root/'results.json'}")
    for arm in ARMS:
        cs = [r["chrf"] for r in rows if r.get("arm") == arm and "chrf" in r]
        if cs:
            print(f"  {arm:12s} chrF: " + ", ".join(f"{c:.2f}" for c in cs)
                  + f"  mean {statistics.mean(cs):.2f}")


if __name__ == "__main__":
    main()
