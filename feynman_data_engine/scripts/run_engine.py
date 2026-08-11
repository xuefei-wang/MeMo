"""Run a data engine to produce an SFT dataset at a given emitted-token budget."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transformers import AutoTokenizer  # noqa: E402
from common.ledger import Ledger  # noqa: E402
from common.llm import LLM, Endpoint  # noqa: E402
from data import rulearena as ra  # noqa: E402
from engine.engine import (run_extraction, run_feynman, split_concepts)  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["extraction_only", "feynman_core"], required=True)
    ap.add_argument("--budget", type=int, required=True, help="emitted training tokens")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gen_url", default="http://localhost:8001/v1")
    ap.add_argument("--gen_model", default="gen")
    ap.add_argument("--student_url", default="http://localhost:8002/v1")
    ap.add_argument("--student_model", default="student")
    ap.add_argument("--learner_base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.learner_base)
    ledger = Ledger(f"{args.mode}_b{args.budget}_s{args.seed}")
    gen = LLM(Endpoint(args.gen_url, args.gen_model), ledger)
    rules_ctx = ra.load_corpus(include_tables=True)
    concepts = split_concepts(rules_ctx)
    print(f"[engine] mode={args.mode} budget={args.budget} seed={args.seed} "
          f"concepts={len(concepts)}")

    t0 = time.time()
    if args.mode == "extraction_only":
        data = run_extraction(gen, tok, ledger, args.budget, args.seed, rules_ctx)
    else:
        student = LLM(Endpoint(args.student_url, args.student_model), ledger)
        data = run_feynman(gen, student, tok, ledger, args.budget, args.seed,
                           rules_ctx, concepts)
    dt = time.time() - t0

    ds_path = outdir / "dataset.jsonl"
    with ds_path.open("w") as f:
        for ex in data:
            f.write(json.dumps(ex) + "\n")
    ledger.save(outdir / "ledger.json")
    meta = {"mode": args.mode, "budget": args.budget, "seed": args.seed,
            "n_examples": len(data), "wall_s": round(dt, 1),
            "emitted_tokens": ledger.emitted_training_tokens,
            "total_gen_completion_tokens": ledger.total_generator_completion_tokens}
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[engine] done: {len(data)} examples, {ledger.emitted_training_tokens} "
          f"emitted toks, {ledger.total_generator_completion_tokens} gen-completion "
          f"toks, {dt:.0f}s -> {ds_path}")


if __name__ == "__main__":
    main()
