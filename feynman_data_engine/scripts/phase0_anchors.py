"""Phase-0 headroom gate (G1).

Measures the two anchors on the LEARNER BASE model:
  * closed-book  -- base model, no corpus access  -> floor
  * ICL-gold     -- base model, grammar book in context -> ceiling
The band between them is the headroom any data engine competes inside.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.ledger import Ledger  # noqa: E402
from common.llm import LLM, Endpoint  # noqa: E402
from data import mtob  # noqa: E402
from eval.translate import run_eval  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_url", default="http://localhost:8001/v1")
    ap.add_argument("--model", default="gen")  # Qwen2.5-3B served as 'gen'
    ap.add_argument("--direction", default="ke")
    ap.add_argument("--book_size", default="medium")
    ap.add_argument("--max_book_chars", type=int, default=90000)
    ap.add_argument("--out", default="runs/phase0")
    args = ap.parse_args()

    outdir = Path(__file__).resolve().parent.parent / args.out
    outdir.mkdir(parents=True, exist_ok=True)

    ledger = Ledger("phase0_anchors")
    llm = LLM(Endpoint(args.base_url, args.model), ledger)
    test = mtob.load_test(args.direction)
    print(f"[phase0] direction={args.direction} n_test={len(test)}")

    # floor: closed-book
    print("[phase0] closed-book ...")
    cb = run_eval(llm, test, args.direction, context=None, purpose="anchor_closedbook")
    print(f"  closed-book chrF = {cb['corpus_chrf']}")

    # ceiling: ICL-gold (grammar book in context)
    book = mtob.load_grammar_book(args.book_size)
    if len(book) > args.max_book_chars:  # fit within the base model's context window
        book = book[:args.max_book_chars]
    print(f"[phase0] ICL-gold (book={args.book_size}, {len(book)} chars) ...")
    icl = run_eval(llm, test, args.direction, context=book, purpose="anchor_iclgold",
                   max_tokens=256)
    print(f"  ICL-gold chrF   = {icl['corpus_chrf']}")

    band = round(icl["corpus_chrf"] - cb["corpus_chrf"], 3)
    verdict = "GO (wide band)" if band >= 15 else "NARROW -- consider swapping base"
    result = {
        "direction": args.direction, "model": args.model, "book_size": args.book_size,
        "closed_book_chrf": cb["corpus_chrf"], "icl_gold_chrf": icl["corpus_chrf"],
        "headroom_band": band, "verdict": verdict,
    }
    print(f"\n[phase0] HEADROOM BAND = {band} chrF  -> {verdict}")
    (outdir / f"anchors_{args.direction}.json").write_text(json.dumps({
        "result": result, "closed_book": cb, "icl_gold": icl,
        "ledger": ledger.summary(),
    }, indent=2))
    print(f"[phase0] wrote {outdir / f'anchors_{args.direction}.json'}")


if __name__ == "__main__":
    main()
