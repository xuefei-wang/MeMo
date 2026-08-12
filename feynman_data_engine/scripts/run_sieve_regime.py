"""SIEVE-regime chase at scale: 16K synthetic + section-level ("full-book") teacher,
four data-engine arms on ONE closed-book chrF axis.

  sieve_gen         : SIEVE-GEN recipe, random 16K of the shared pool  (diverse coverage)
  feynman_gen       : same pool, the hardest-for-student 16K tail       (frontier selection)
  feynman_cheatsheet: the Feynman loop's failure-targeted study notes   (amortized context)
  gold_only         : the 375 gold pairs alone                          (anchor baseline)

Pool arms train hard-SFT (default) or soft context-distillation (--train soft); the
cheatsheet + gold arms train in their native hard-SFT mode. The cheatsheet arm is also
evaluated weight-free (base + cheatsheet) and combined (SFT + cheatsheet) per the blog.
Only the DATA ENGINE differs across arms, so the chrF spread is the data-engine effect."""
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
POOL_ARMS = ["sieve_gen", "feynman_gen"]
ALL_ARMS = ["sieve_gen", "feynman_gen", "feynman_cheatsheet", "gold_only"]


def sh(cmd, gpu, log):
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as f:
        return subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode


def synth_token_budget(dataset_path: Path, n_gold: int, base: str) -> int:
    """Emitted synthetic training tokens of the pool arm = fair budget for the
    token-metered cheatsheet engine (match the synthetic signal volume, not #examples)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base)
    rows = [json.loads(l) for l in dataset_path.read_text().splitlines() if l.strip()]
    synth = rows[n_gold:]  # gold base is prepended first
    total = 0
    for ex in synth:
        total += len(tok.encode("\n".join(m["content"] for m in ex["messages"])))
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "runs" / "sieve_regime"))
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--pool_factor", type=float, default=1.5)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--n_sections", type=int, default=4)
    ap.add_argument("--train", choices=["hard", "soft"], default="hard",
                    help="pool arms: hard-label SFT or soft context-distillation")
    ap.add_argument("--gpus", nargs="+", default=["4", "6"])
    ap.add_argument("--base", default="Qwen/Qwen3-8B")
    ap.add_argument("--gen_url", default="http://localhost:8002/v1")
    ap.add_argument("--gen_model", default="student")
    ap.add_argument("--cheatsheet_budget", type=int, default=60000,
                    help="cap on the cheatsheet arm's synthetic budget. run_feynman is a "
                         "SEQUENTIAL loop and the sheet saturates at 4000 chars, so token-"
                         "matching to 16K synth (~1.1M tok) is both ~6h slow and category-"
                         "wrong -- a cheatsheet is amortized/compressed by design.")
    args = ap.parse_args()
    root = Path(args.out); root.mkdir(parents=True, exist_ok=True)
    tg = args.gpus

    # ---- Phase A: generation (pool + cheatsheet), sequential per seed ----
    print(f"[sieve] Phase A: generate (n={args.n} x{args.pool_factor}, "
          f"section teacher, {len(args.seeds)} seed) ...", flush=True)
    for seed in args.seeds:
        d = root / f"s{seed}"
        if not (d / "sieve_gen" / "records.jsonl").exists():
            rc = sh([PY, str(ROOT / "scripts" / "run_headtohead.py"), "--n", str(args.n),
                     "--seed", str(seed), "--pool_factor", str(args.pool_factor),
                     "--concurrency", str(args.concurrency), "--teacher_ctx", "section",
                     "--n_sections", str(args.n_sections), "--out", str(d)],
                    None, d / "gen_pool.log")
            print(f"  pool s{seed} -> {'ok' if rc == 0 else f'FAIL:{rc}'}", flush=True)
        else:
            print(f"  pool s{seed} -> skip")
        # cheatsheet arm: budget-matched to the pool arm's synthetic token volume
        cdir = d / "feynman_cheatsheet"
        if not (cdir / "dataset.jsonl").exists():
            meta = json.loads((d / "sieve_gen" / "meta.json").read_text())
            synth_b = synth_token_budget(d / "sieve_gen" / "dataset.jsonl",
                                         meta.get("n_gold", 375), args.base)
            budget = min(synth_b, args.cheatsheet_budget)  # cap: sheet saturates + serial
            print(f"  cheatsheet s{seed}: budget={budget} tok "
                  f"(min of sieve-synth {synth_b}, cap {args.cheatsheet_budget})", flush=True)
            rc = sh([PY, str(ROOT / "scripts" / "run_engine_mtob.py"),
                     "--mode", "feynman_core", "--budget", str(budget), "--seed", str(seed),
                     "--gen_url", args.gen_url, "--gen_model", args.gen_model,
                     "--student_url", args.gen_url, "--student_model", args.gen_model,
                     # diagnoser book context must fit the 16K endpoint (full 70K-char
                     # book = ~22.8K tok overflows); ~42K chars ~= 13.7K tok leaves room.
                     "--book_chars", "42000",
                     "--out", str(cdir)], None, d / "gen_cheatsheet.log")
            print(f"  cheatsheet s{seed} -> {'ok' if rc == 0 else f'FAIL:{rc}'}", flush=True)
        else:
            print(f"  cheatsheet s{seed} -> skip")

    cells = [(seed, arm) for seed in args.seeds for arm in ALL_ARMS]

    # ---- Phase B: train (round-robin GPUs) ----
    def train(idx_cell):
        idx, (seed, arm) = idx_cell
        d = root / f"s{seed}" / arm
        adir = d / "adapter"
        if (adir / "adapter_config.json").exists():
            return (seed, arm), "skip"
        gpu = tg[idx % len(tg)]
        soft = args.train == "soft" and arm in POOL_ARMS
        if soft:
            if not (d / "records.jsonl").exists():
                return (seed, arm), "no-records"
            cmd = [PY, str(ROOT / "learner" / "soft_distill.py"),
                   "--records", str(d / "records.jsonl"),
                   "--sections", str(root / f"s{seed}" / "sections.json"),
                   "--base", args.base, "--out", str(adir), "--epochs", str(args.epochs)]
        else:
            if not (d / "dataset.jsonl").exists():
                return (seed, arm), "no-data"
            cmd = [PY, str(ROOT / "learner" / "sft.py"), "--base", args.base,
                   "--dataset", str(d / "dataset.jsonl"), "--out", str(adir),
                   "--epochs", str(args.epochs), "--lr", "2e-4", "--bs", "4",
                   "--grad_accum", "16", "--max_len", "512"]
        rc = sh(cmd, gpu, d / "train.log")
        return (seed, arm), "ok" if rc == 0 else f"FAIL:{rc}"

    print(f"[sieve] Phase B: train ({args.train} for pool arms, hard for gold/cheatsheet) ...",
          flush=True)
    with ThreadPoolExecutor(max_workers=len(tg)) as ex:
        for c, s in ex.map(train, list(enumerate(cells))):
            print(f"  train {c} -> {s}", flush=True)

    # ---- Phase C: closed-book chrF eval (+ cheatsheet-context readings) ----
    def evl(idx_cell):
        idx, (seed, arm) = idx_cell
        d = root / f"s{seed}" / arm
        adir, outp = d / "adapter", d / "eval.json"
        gpu = tg[idx % len(tg)]
        base_eval = [PY, str(ROOT / "learner" / "eval_mtob.py"), "--base", args.base,
                     "--direction", "ke", "--n", "50", "--no_think", "--bs", "16",
                     "--max_new", "128"]
        done = []
        if not outp.exists() and (adir / "adapter_config.json").exists():
            rc = sh(base_eval + ["--adapter", str(adir), "--context", "none",
                                 "--out", str(outp)], gpu, d / "eval.log")
            done.append(f"closed-book:{'ok' if rc == 0 else rc}")
        if arm == "feynman_cheatsheet" and (d / "cheatsheet.txt").exists():
            cf = str(d / "cheatsheet.txt")
            bc = d / "eval_base_cheat.json"      # weight-free amortized context
            if not bc.exists():
                rc = sh(base_eval + ["--context", "cheatsheet", "--cheatsheet_file", cf,
                                     "--out", str(bc)], gpu, d / "eval_base_cheat.log")
                done.append(f"base+cheat:{'ok' if rc == 0 else rc}")
            sc = d / "eval_sft_cheat.json"       # weights + context combined
            if not sc.exists() and (adir / "adapter_config.json").exists():
                rc = sh(base_eval + ["--adapter", str(adir), "--context", "cheatsheet",
                                     "--cheatsheet_file", cf, "--out", str(sc)],
                        gpu, d / "eval_sft_cheat.log")
                done.append(f"sft+cheat:{'ok' if rc == 0 else rc}")
        return (seed, arm), (", ".join(done) or "skip")

    print("[sieve] Phase C: eval ...", flush=True)
    with ThreadPoolExecutor(max_workers=len(tg)) as ex:
        for c, s in ex.map(evl, list(enumerate(cells))):
            print(f"  eval {c} -> {s}", flush=True)

    # ---- Phase D: aggregate ----
    def chrf_of(p: Path):
        try:
            s = json.loads(p.read_text())["summary"]
            return s.get("corpus_chrf")
        except Exception:
            return None

    rows = []
    for seed, arm in cells:
        d = root / f"s{seed}" / arm
        rows.append({"seed": seed, "arm": arm,
                     "chrf_closedbook": chrf_of(d / "eval.json"),
                     "chrf_base_cheat": chrf_of(d / "eval_base_cheat.json"),
                     "chrf_sft_cheat": chrf_of(d / "eval_sft_cheat.json")})
    (root / "results.json").write_text(json.dumps(rows, indent=2))
    print(f"\n[sieve] results -> {root/'results.json'}  (floor 15.9, SIEVE 24.48)")
    for arm in ALL_ARMS:
        cs = [r["chrf_closedbook"] for r in rows
              if r["arm"] == arm and r["chrf_closedbook"] is not None]
        if cs:
            print(f"  {arm:18s} closed-book chrF: "
                  + ", ".join(f"{c:.2f}" for c in cs) + f"  mean {statistics.mean(cs):.2f}")
    for r in rows:
        if r["arm"] == "feynman_cheatsheet":
            print(f"  [cheatsheet s{r['seed']}] base+cheat={r['chrf_base_cheat']} "
                  f"sft+cheat={r['chrf_sft_cheat']}")


if __name__ == "__main__":
    main()
