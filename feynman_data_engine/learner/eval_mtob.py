"""HF-based MTOB translation eval (for models vLLM 0.6.6 can't serve, e.g. Qwen3).

Reuses the translation prompt (eval/translate.py) and chrF (eval/chrf.py); swaps
generation to HF transformers (like eval_learner.py). Used for the Qwen3-8B
capability/contamination gate and later for the trained-learner eval.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import mtob  # noqa: E402
from eval.translate import build_prompt, _clean  # noqa: E402
from eval.chrf import corpus_chrf, sentence_chrf  # noqa: E402
from learner.eval_learner import load_model, generate_batch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-8B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--direction", default="ke", choices=["ke", "ek"])
    ap.add_argument("--context", choices=["none", "gold"], default="none")
    ap.add_argument("--book_size", default="medium")
    ap.add_argument("--max_book_chars", type=int, default=80000)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--max_new", type=int, default=128)
    ap.add_argument("--max_input_len", type=int, default=31000)
    ap.add_argument("--no_think", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    test = mtob.load_test(args.direction)[:args.n]
    ctx = None
    if args.context == "gold":
        ctx = mtob.load_grammar_book(args.book_size)[:args.max_book_chars]
    prompts = [build_prompt(p.source, args.direction, ctx) for p in test]

    tok, model = load_model(args.base, args.adapter)
    enable_thinking = False if args.no_think else None
    print(f"[mtob-eval] base={args.base} dir={args.direction} ctx={args.context} "
          f"n={len(test)} think={enable_thinking}")
    outs = generate_batch(tok, model, prompts, max_new=args.max_new, bs=args.bs,
                          max_input_len=args.max_input_len, enable_thinking=enable_thinking)

    hyps = [_clean(o) for o in outs]
    refs = [p.target for p in test]
    score = corpus_chrf(hyps, refs)
    samples = [{"src": p.source, "hyp": h, "ref": r,
                "chrf": round(sentence_chrf(h, r), 2)}
               for p, h, r in zip(test, hyps, refs)]
    summary = {"base": args.base, "adapter": args.adapter, "direction": args.direction,
               "context": args.context, "n": len(test), "corpus_chrf": round(score, 3)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "samples": samples}, indent=2))
    print(f"[mtob-eval] corpus_chrf={score:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
