"""Build ONE shared pool of synthetic MTOB queries, then emit BOTH engine datasets
from it (sieve_gen = representative sample, feynman_gen = hardest-student tail). Same
pool + same teacher answers => the datasets differ ONLY in selection policy.

Hits Qwen3-8B-Base (selection) + Qwen3-8B (instruct/teacher/student)."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transformers import AutoTokenizer  # noqa: E402
from common.ledger import Ledger  # noqa: E402
from common.llm import LLM, Endpoint  # noqa: E402
from data import mtob  # noqa: E402
from engine.headtohead_mtob import (build_pool, select_sieve, select_feynman,  # noqa: E402
                                    to_sft, book_sections)


def _gold_examples(gold_pairs):
    from eval.translate import build_prompt
    return [{"messages": build_prompt(k, "ke", None)
             + [{"role": "assistant", "content": e}]} for k, e in gold_pairs]


def _write(outdir: Path, name: str, data, ledger, extra, gold_rows=None,
           records=None):
    d = outdir / name
    d.mkdir(parents=True, exist_ok=True)
    rows = (gold_rows or []) + data  # shared gold base anchors correctness under SFT
    with (d / "dataset.jsonl").open("w") as f:
        for ex in rows:
            f.write(json.dumps(ex) + "\n")
    if records is not None:  # (kal, eng, pieces) for soft distillation
        with (d / "records.jsonl").open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
    meta = {"engine": name, "n_synth": len(data), "n_gold": len(gold_rows or []),
            "n_examples": len(rows), **extra}
    (d / "meta.json").write_text(json.dumps(meta, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="matched budget = #examples/arm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool_factor", type=float, default=3.0)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--base_url", default="http://localhost:8003/v1")
    ap.add_argument("--base_model", default="base")
    ap.add_argument("--qwen_url", default="http://localhost:8002/v1")
    ap.add_argument("--qwen_model", default="student")
    ap.add_argument("--learner_base", default="Qwen/Qwen3-8B")
    ap.add_argument("--book_chars", type=int, default=115000)
    ap.add_argument("--teacher_ctx", choices=["pieces", "section"], default="section",
                    help="teacher conditions on the whole book section (strong) or just "
                         "the 6-8 selected pieces (SIEVE-faithful per-example)")
    ap.add_argument("--n_sections", type=int, default=4,
                    help="fixed, prefix-cached book sections for the section teacher")
    ap.add_argument("--no_gold_base", action="store_true",
                    help="omit the shared gold base (de-risk showed it's needed)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.learner_base)
    ledger = Ledger(f"h2h_pool_n{args.n}_s{args.seed}")
    base = LLM(Endpoint(args.base_url, args.base_model), ledger)
    qwen = LLM(Endpoint(args.qwen_url, args.qwen_model), ledger)

    book = mtob.load_grammar_book("medium")[:args.book_chars]
    sections = book_sections(book, args.n_sections)
    # persist the fixed sections so soft_distill can rebuild the teacher context by idx
    (outdir / "sections.json").write_text(json.dumps(sections))
    gold = [(p.source, p.target) for p in mtob.train_as_pairs("ke")]
    m = int(args.n * args.pool_factor)
    print(f"[h2h] pool m={m} (n={args.n} x{args.pool_factor}) seed={args.seed} "
          f"teacher={args.teacher_ctx} sections={len(sections)}", flush=True)

    t0 = time.time()
    pool = build_pool(base, qwen, qwen, qwen, ledger, m, args.seed, sections, gold,
                      concurrency=args.concurrency, teacher_mode=args.teacher_ctx)
    dt = time.time() - t0
    scores = sorted(c["student_chrf"] for c in pool)
    dist = {"min": round(scores[0], 1), "median": round(statistics.median(scores), 1),
            "max": round(scores[-1], 1)} if scores else {}
    print(f"[h2h] pool built: {len(pool)} usable, student-chrF {dist} in {dt:.0f}s",
          flush=True)

    sieve = select_sieve(pool, args.n, args.seed)
    feyn = select_feynman(pool, args.n)
    gold_rows = None if args.no_gold_base else _gold_examples(gold)
    common = {"n_target": args.n, "seed": args.seed, "pool_size": len(pool),
              "pool_wall_s": round(dt, 1), "pool_student_chrf": dist,
              "gen_completion_tokens": ledger.total_generator_completion_tokens}
    # per-arm difficulty of the SELECTED set (the whole point of the contrast)
    s_sel = statistics.median(c["student_chrf"] for c in sieve)
    f_sel = statistics.median(c["student_chrf"] for c in feyn)
    _write(outdir, "sieve_gen", to_sft(sieve, tok, ledger), ledger,
           {**common, "selected_median_student_chrf": round(s_sel, 1)}, gold_rows,
           records=sieve)
    _write(outdir, "feynman_gen", to_sft(feyn, tok, ledger), ledger,
           {**common, "selected_median_student_chrf": round(f_sel, 1)}, gold_rows,
           records=feyn)
    # gold-only baseline: how much does ANY synthetic add over the anchor alone?
    if gold_rows is not None:
        _write(outdir, "gold_only", [], ledger, {**common, "note": "gold base only"},
               gold_rows)
    ledger.save(outdir / "ledger.json")
    print(f"[h2h] wrote sieve_gen (sel-median chrF {s_sel:.1f}) + feynman_gen "
          f"(sel-median chrF {f_sel:.1f}) + gold_only, {args.n} synth ex each "
          f"(+{0 if gold_rows is None else len(gold_rows)} gold)", flush=True)


if __name__ == "__main__":
    main()
