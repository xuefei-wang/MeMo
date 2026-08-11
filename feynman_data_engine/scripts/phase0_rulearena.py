"""Phase-0 headroom gate (G1) for RuleArena airline.

closed-book (no rules) -> floor ; ICL-gold (rules+tables in context) -> ceiling.
Exact fee totals are unknowable without the corpus, so closed-book should be ~0.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.ledger import Ledger  # noqa: E402
from common.llm import LLM, Endpoint  # noqa: E402
from data import rulearena  # noqa: E402
from eval.rulearena_eval import run_eval  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_url", default="http://localhost:8001/v1")
    ap.add_argument("--model", default="gen")
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="runs/phase0")
    args = ap.parse_args()

    outdir = Path(__file__).resolve().parent.parent / args.out
    outdir.mkdir(parents=True, exist_ok=True)

    ledger = Ledger("phase0_rulearena")
    llm = LLM(Endpoint(args.base_url, args.model), ledger)
    probs = rulearena.load_problems(args.level)[:args.n]
    print(f"[phase0-ra] level={args.level} n={len(probs)}")

    print("[phase0-ra] closed-book ...")
    cb = run_eval(llm, probs, context=None, purpose="anchor_closedbook")
    print(f"  closed-book acc = {cb['accuracy']}  (parse {cb['parse_rate']})")

    corpus = rulearena.load_corpus(include_tables=True)
    print(f"[phase0-ra] ICL-gold (corpus {len(corpus)} chars) ...")
    icl = run_eval(llm, probs, context=corpus, purpose="anchor_iclgold")
    print(f"  ICL-gold acc    = {icl['accuracy']}  (parse {icl['parse_rate']})")

    band = round(icl["accuracy"] - cb["accuracy"], 4)
    verdict = "GO (headroom exists)" if band >= 0.10 else "NARROW"
    print(f"\n[phase0-ra] HEADROOM BAND = {band}  -> {verdict}")
    (outdir / f"rulearena_anchors_L{args.level}.json").write_text(json.dumps({
        "level": args.level, "n": len(probs),
        "closed_book_acc": cb["accuracy"], "icl_gold_acc": icl["accuracy"],
        "closed_book_parse": cb["parse_rate"], "icl_gold_parse": icl["parse_rate"],
        "headroom_band": band, "verdict": verdict,
        "ledger": ledger.summary(),
    }, indent=2))
    print(f"[phase0-ra] wrote {outdir / f'rulearena_anchors_L{args.level}.json'}")


if __name__ == "__main__":
    main()
