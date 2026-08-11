"""Analyze the MTOB grid: extraction_only vs feynman_core at matched budget.

Reports, per budget:
  - corpus chrF per seed + seed mean (the headline metric)
  - a per-sentence PAIRED bootstrap on mean sentence-chrF (feynman - extraction),
    pooling seeds; the 50 ke test sentences are shared+ordered across all cells,
    so sample i pairs across modes. This is a proxy for significance with more
    power than n=2 seeds -- clearly labelled as per-sentence, NOT corpus chrF.
  - both budget axes: emitted training tokens and total generator compute
    (completion tokens; plus prefill-inclusive with a prefix-cache caveat).

Usage: python scripts/analyze_mtob.py [--root runs/grid_mtob] [--boot 5000]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _prng(seed):
    s = seed & 0xFFFFFFFF

    def nxt(n):  # xorshift -> index in [0,n)
        nonlocal s
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= s >> 17
        s ^= (s << 5) & 0xFFFFFFFF
        return s % n
    return nxt


def load_cell(root, mode, budget, seed):
    d = root / f"{mode}__b{budget}__s{seed}"
    ev, meta, ledg = d / "eval.json", d / "meta.json", d / "ledger.json"
    if not ev.exists():
        return None
    e = json.loads(ev.read_text())
    m = json.loads(meta.read_text()) if meta.exists() else {}
    lg = json.loads(ledg.read_text()) if ledg.exists() else {}
    return {"corpus": e["summary"]["corpus_chrf"],
            "sent": [s["chrf"] for s in e["samples"]],
            "synth": m.get("emitted_synthetic_tokens"),
            "gen_compl": lg.get("axis_total_generator_completion_tokens"),
            "gen_total": lg.get("axis_total_generator_tokens"),
            "n_notes": (m.get("n_examples", 375) - 375)}


def paired_boot(fey_rows, ext_rows, nboot, seed=12345):
    """Paired bootstrap on mean sentence-chrF diff, pooling seeds within a budget.
    Pairs by (seed-index, sentence-index). Returns (mean_diff, lo, hi, p_two)."""
    pairs = []
    for f, e in zip(fey_rows, ext_rows):
        pairs += list(zip(f["sent"], e["sent"]))
    n = len(pairs)
    base = sum(a - b for a, b in pairs) / n
    nxt = _prng(seed)
    diffs = []
    for _ in range(nboot):
        acc = 0.0
        for _ in range(n):
            a, b = pairs[nxt(n)]
            acc += a - b
        diffs.append(acc / n)
    diffs.sort()
    lo = diffs[int(0.025 * nboot)]
    hi = diffs[int(0.975 * nboot)]
    ge = sum(1 for d in diffs if d <= 0)
    p = 2 * min(ge, nboot - ge) / nboot
    return base, lo, hi, p, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "runs" / "grid_mtob"))
    ap.add_argument("--budgets", type=int, nargs="+", default=[6000, 20000])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--boot", type=int, default=5000)
    args = ap.parse_args()
    root = Path(args.root)

    floor = None
    fj = root / "base_closedbook.json"
    if fj.exists():
        floor = json.loads(fj.read_text())["summary"]["corpus_chrf"]

    print(f"\n{'='*72}\nMTOB grid: extraction_only vs feynman_core  (closed-book ke chrF)")
    if floor is not None:
        print(f"floor (untrained Qwen3-8B, no context): {floor:.2f} chrF")
    print("="*72)

    for b in args.budgets:
        ext = [load_cell(root, "extraction_only", b, s) for s in args.seeds]
        fey = [load_cell(root, "feynman_core", b, s) for s in args.seeds]
        ext = [r for r in ext if r]
        fey = [r for r in fey if r]
        if not ext or not fey:
            print(f"\n[budget {b}] incomplete (ext={len(ext)} fey={len(fey)})")
            continue

        em = sum(r["corpus"] for r in ext) / len(ext)
        fm = sum(r["corpus"] for r in fey) / len(fey)
        print(f"\n[budget {b}]  emitted training tokens")
        print(f"  extraction corpus chrF: "
              + ", ".join(f"{r['corpus']:.2f}" for r in ext) + f"  mean {em:.2f}")
        print(f"  feynman    corpus chrF: "
              + ", ".join(f"{r['corpus']:.2f}" for r in fey) + f"  mean {fm:.2f}")
        print(f"  Δ corpus chrF (fey - ext): {fm - em:+.2f}")
        if floor is not None:
            print(f"  lift over floor: ext {em - floor:+.2f}, fey {fm - floor:+.2f}")

        # only pair seeds present in BOTH
        n = min(len(ext), len(fey))
        base, lo, hi, p, npair = paired_boot(fey[:n], ext[:n], args.boot)
        star = "  *sig*" if (lo > 0 or hi < 0) else "  (n.s.)"
        print(f"  per-sentence paired bootstrap (n={npair} sent-pairs, {n} seed(s)):")
        print(f"    mean sent-chrF Δ {base:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  "
              f"p={p:.3f}{star}")

        # budget axes
        ec = sum(r["gen_compl"] or 0 for r in ext) / len(ext)
        fc = sum(r["gen_compl"] or 0 for r in fey) / len(fey)
        et = sum(r["gen_total"] or 0 for r in ext) / len(ext)
        ft = sum(r["gen_total"] or 0 for r in fey) / len(fey)
        en = sum(r["n_notes"] for r in ext) / len(ext)
        fn = sum(r["n_notes"] for r in fey) / len(fey)
        print(f"  generator compute (completion tok): ext {ec:.0f}  fey {fc:.0f}"
              f"  ({(fc-ec)/ec*100:+.0f}%)")
        print(f"  generator compute (incl. prefill):  ext {et:.0f}  fey {ft:.0f}"
              f"  (prefill dominated by re-sent book; ~free under vLLM prefix cache)")
        print(f"  #notes emitted: ext {en:.0f}  fey {fn:.0f}  "
              f"(feynman = many short failure-diagnoses; ext = few long chunk-notes)")

    print(f"\n{'='*72}\n")


if __name__ == "__main__":
    main()
