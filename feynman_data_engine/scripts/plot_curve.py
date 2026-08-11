"""Aggregate grid results.json into efficiency curves on BOTH budget axes."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODE_STYLE = {"extraction_only": ("tab:gray", "o", "extraction (SIEVE-shaped)"),
              "feynman_core": ("tab:red", "s", "feynman (failure loop)")}


def agg(rows, xaxis, yaxis):
    """mode -> sorted list of (mean_x, mean_y, y_lo, y_hi) grouped by budget."""
    by_mode_budget = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if "error" in r:
            continue
        by_mode_budget[r["mode"]][r["budget"]].append(r)
    out = {}
    for mode, budgets in by_mode_budget.items():
        pts = []
        for budget, rs in sorted(budgets.items()):
            xs = [r[xaxis] for r in rs]
            ys = [r[yaxis] for r in rs]
            mx = sum(xs) / len(xs)
            my = sum(ys) / len(ys)
            lo, hi = (min(ys), max(ys)) if len(ys) > 1 else (my, my)
            pts.append((mx, my, lo, hi))
        out[mode] = sorted(pts)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--base", default=None, help="base_closedbook.json for floor line")
    ap.add_argument("--out", default=None)
    ap.add_argument("--yaxis", default="within20",
                    choices=["within20", "within10", "median_rel_err", "exact_acc"])
    args = ap.parse_args()

    rows = json.loads(Path(args.results).read_text())
    floor = None
    if args.base and Path(args.base).exists():
        floor = json.loads(Path(args.base).read_text())["summary"].get(args.yaxis)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, xaxis, xlabel in [
        (axes[0], "emitted_tokens", "emitted training tokens"),
        (axes[1], "gen_completion_tokens", "total generator-completion tokens"),
    ]:
        curves = agg(rows, xaxis, args.yaxis)
        for mode, pts in curves.items():
            color, marker, label = MODE_STYLE.get(mode, ("k", "x", mode))
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            los = [p[2] for p in pts]; his = [p[3] for p in pts]
            ax.plot(xs, ys, color=color, marker=marker, label=label, lw=2)
            ax.fill_between(xs, los, his, color=color, alpha=0.15)
        if floor is not None:
            ax.axhline(floor, color="black", ls="--", lw=1, label=f"base closed-book ({floor})")
        ax.set_xlabel(xlabel); ax.set_ylabel(args.yaxis)
        ax.set_xscale("log"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(f"Feynman vs extraction data engine — RuleArena airline ({args.yaxis})")
    fig.tight_layout()
    out = args.out or str(Path(args.results).parent / f"curve_{args.yaxis}.png")
    fig.savefig(out, dpi=130)
    print(f"[plot] wrote {out}")

    # text table
    print(f"\n{'mode':18s}{'budget':>9}{'emit':>9}{'gen':>10}{'  '+args.yaxis:>12}")
    for r in sorted(rows, key=lambda r: (r.get('mode',''), r.get('budget',0), r.get('seed',0))):
        if "error" in r:
            continue
        print(f"{r['mode']:18s}{r['budget']:>9}{r['emitted_tokens']:>9}"
              f"{r['gen_completion_tokens']:>10}{r[args.yaxis]:>12.4f}")


if __name__ == "__main__":
    main()
