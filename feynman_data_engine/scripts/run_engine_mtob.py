"""Run the MTOB data engine to produce an SFT translation dataset at a budget."""
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
from data import mtob  # noqa: E402
from engine.engine_mtob import run_extraction, run_feynman, chunk_book  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["extraction_only", "feynman_core"], required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--direction", default="ke")
    ap.add_argument("--gen_url", default="http://localhost:8001/v1")
    ap.add_argument("--gen_model", default="gen")
    ap.add_argument("--student_url", default="http://localhost:8002/v1")
    ap.add_argument("--student_model", default="student")
    ap.add_argument("--learner_base", default="Qwen/Qwen3-8B")
    ap.add_argument("--book_chars", type=int, default=70000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.learner_base)
    ledger = Ledger(f"mtob_{args.mode}_b{args.budget}_s{args.seed}")
    gen = LLM(Endpoint(args.gen_url, args.gen_model), ledger)

    book_ctx = mtob.load_grammar_book("medium")[:args.book_chars]
    wordlist = mtob.load_wordlist()
    gold_pairs = [(p.source, p.target) for p in mtob.train_as_pairs(args.direction)]
    concepts = chunk_book(book_ctx, chunk_chars=2000)
    print(f"[mtob-engine] mode={args.mode} budget={args.budget} seed={args.seed} "
          f"gold_pairs={len(gold_pairs)} concepts={len(concepts)} vocab={len(wordlist)}")

    t0 = time.time()
    if args.mode == "extraction_only":
        data = run_extraction(gen, tok, ledger, args.budget, args.seed, book_ctx,
                              wordlist, gold_pairs, concepts)
    else:
        student = LLM(Endpoint(args.student_url, args.student_model), ledger)
        data = run_feynman(gen, student, tok, ledger, args.budget, args.seed,
                           book_ctx, wordlist, gold_pairs, concepts)
    dt = time.time() - t0

    with (outdir / "dataset.jsonl").open("w") as f:
        for ex in data:
            f.write(json.dumps(ex) + "\n")
    ledger.save(outdir / "ledger.json")
    gold_base = getattr(ledger, "gold_base_tokens", 0)
    (outdir / "meta.json").write_text(json.dumps({
        "mode": args.mode, "budget": args.budget, "seed": args.seed,
        "direction": args.direction, "n_examples": len(data), "wall_s": round(dt, 1),
        "emitted_synthetic_tokens": ledger.emitted_training_tokens,  # budget axis
        "gold_base_tokens": gold_base,  # shared, un-budgeted foundation
        "total_train_tokens": ledger.emitted_training_tokens + gold_base,
        "total_gen_completion_tokens": ledger.total_generator_completion_tokens}, indent=2))
    print(f"[mtob-engine] done: {len(data)} ex, synth={ledger.emitted_training_tokens} "
          f"gold_base={gold_base} gen={ledger.total_generator_completion_tokens} {dt:.0f}s")


if __name__ == "__main__":
    main()
