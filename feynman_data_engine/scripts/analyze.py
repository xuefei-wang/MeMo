"""Rigorous readout of the grid: pool per-problem results across seeds, bootstrap
a CI on within20, and report the feynman-minus-extraction gap with its CI so we
can tell signal from seed noise."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def within_flags(results: list[dict], frac: float = 0.20) -> list[int]:
    flags = []
    for r in results:
        p, g = r.get("pred"), r["gold"]
        flags.append(1 if (p is not None and abs(p - g) <= max(1.0, frac * g)) else 0)
    return flags


def boot_ci(flags: list[int], n_boot: int = 5000, seed: int = 0):
    rng = random.Random(seed)
    n = len(flags)
    if n == 0:
        return (0.0, 0.0, 0.0)
    means = []
    for _ in range(n_boot):
        s = sum(flags[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    return (sum(flags) / n, means[int(0.025 * n_boot)], means[int(0.975 * n_boot)])


def diff_ci(fey: list[int], ext: list[int], n_boot: int = 5000, seed: int = 0):
    """Bootstrap CI on (mean(fey) - mean(ext)) with independent resampling."""
    rng = random.Random(seed)
    diffs = []
    nf, ne = len(fey), len(ext)
    for _ in range(n_boot):
        mf = sum(fey[rng.randrange(nf)] for _ in range(nf)) / nf
        me = sum(ext[rng.randrange(ne)] for _ in range(ne)) / ne
        diffs.append(mf - me)
    diffs.sort()
    return (sum(fey)/nf - sum(ext)/ne, diffs[int(0.025*n_boot)], diffs[int(0.975*n_boot)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="runs/grid")
    ap.add_argument("--frac", type=float, default=0.20)
    args = ap.parse_args()
    grid = Path(args.grid)

    # pool per-problem flags across seeds, keyed by (mode, budget)
    pooled = defaultdict(list)
    gen_tokens = defaultdict(list)
    per_seed = defaultdict(dict)  # (mode,budget) -> {seed: within20}
    for ev in grid.glob("*/eval.json"):
        cell = ev.parent.name  # mode__bBUDGET__sSEED
        try:
            mode, b, s = cell.split("__")
            budget = int(b[1:]); seed = int(s[1:])
        except Exception:
            continue
        results = json.loads(ev.read_text())["results"]
        flags = within_flags(results, args.frac)
        pooled[(mode, budget)] += flags
        per_seed[(mode, budget)][seed] = sum(flags) / len(flags) if flags else 0.0
        meta = json.loads((ev.parent / "meta.json").read_text())
        gen_tokens[(mode, budget)].append(meta["total_gen_completion_tokens"])

    budgets = sorted({b for (_, b) in pooled})
    print(f"within{int(args.frac*100)} pooled across seeds, bootstrap 95% CI\n")
    print(f"{'budget':>8}  {'extraction (CI)':>26}  {'feynman (CI)':>26}  "
          f"{'fey-ext diff (CI)':>26}  signif")
    for budget in budgets:
        ext = pooled.get(("extraction_only", budget), [])
        fey = pooled.get(("feynman_core", budget), [])
        if not ext or not fey:
            continue
        em, elo, ehi = boot_ci(ext)
        fm, flo, fhi = boot_ci(fey)
        dm, dlo, dhi = diff_ci(fey, ext)
        sig = "YES" if (dlo > 0 or dhi < 0) else "no (CI spans 0)"
        print(f"{budget:>8}  {em:.3f} [{elo:.3f},{ehi:.3f}]  "
              f"{fm:.3f} [{flo:.3f},{fhi:.3f}]  "
              f"{dm:+.3f} [{dlo:+.3f},{dhi:+.3f}]  {sig}")
    print("\ngenerator-compute premium (feynman/extraction, mean gen tokens):")
    for budget in budgets:
        e = gen_tokens.get(("extraction_only", budget))
        f = gen_tokens.get(("feynman_core", budget))
        if e and f:
            print(f"  b{budget}: {sum(f)/len(f)/(sum(e)/len(e)):.2f}x")

    # seed-level PAIRED diffs (fey - ext at the same seed): the right unit of
    # replication for training-randomness noise.
    print("\nseed-level paired diffs (fey - ext within20, per seed):")
    for budget in budgets:
        fs = per_seed.get(("feynman_core", budget), {})
        es = per_seed.get(("extraction_only", budget), {})
        seeds = sorted(set(fs) & set(es))
        diffs = [fs[s] - es[s] for s in seeds]
        if not diffs:
            continue
        mean = sum(diffs) / len(diffs)
        allpos = all(d > 0 for d in diffs)
        print(f"  b{budget}: n_seeds={len(diffs)} diffs={[round(d,3) for d in diffs]} "
              f"mean={mean:+.3f} all_positive={allpos}")


if __name__ == "__main__":
    main()
