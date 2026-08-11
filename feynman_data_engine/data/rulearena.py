"""RuleArena (airline) loader.

Task: given passenger + bags, compute total cost = ticket + baggage fees, by
applying the airline's fee rules. The answer (a dollar total) is NOT stated in
the corpus -- it must be derived by applying the rules -> the right shape for a
data engine whose value is turning rules into applied procedure.

corpus  = reference_rules.txt + serialized fee tables (the specific fee amounts).
gold    = compute_answer(**info) (programmatic ground truth).
metric  = exact match on the total (closed-book floor ~ 0, since exact fees are
          unknowable without the corpus).
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent
AIRLINE = _HERE / "rulearena_airline"


@dataclass
class Problem:
    prompt: str
    info: dict
    gold: float


_TABLES_CACHE = None


def _tables():
    # compute_answer.py reads fee_tables/... relative to cwd; import with cwd set.
    global _TABLES_CACHE
    if _TABLES_CACHE is not None:
        return _TABLES_CACHE
    cwd = os.getcwd()
    sys.path.insert(0, str(AIRLINE))
    os.chdir(AIRLINE)
    try:
        from compute_answer import load_checking_fee  # noqa: E402
        _TABLES_CACHE = load_checking_fee()
        return _TABLES_CACHE
    finally:
        os.chdir(cwd)


def load_problems(level: int = 0) -> list[Problem]:
    tables = _tables()
    cwd = os.getcwd()
    sys.path.insert(0, str(AIRLINE))
    os.chdir(AIRLINE)
    try:
        from compute_answer import compute_answer  # noqa: E402
        rows = [json.loads(l) for l in
                (AIRLINE / "synthesized_problems" / f"comp_{level}.jsonl").read_text().splitlines()
                if l.strip()]
        out = []
        for r in rows:
            fee, _ = compute_answer(**r["info"], check_base_tables=tables)
            out.append(Problem(prompt=r["prompt"], info=r["info"], gold=float(fee)))
        return out
    finally:
        os.chdir(cwd)


_BAG_NAMES = ["backpack", "luggage box", "suitcase", "duffel bag", "garment bag"]
_US_CITIES = ["Seattle", "Denver", "Dallas", "Chicago", "Boston", "Atlanta",
              "Orlando", "Miami", "New York", "Los Angeles", "Phoenix", "Austin"]
_CLASSES = ["Main Cabin", "Main Cabin", "Basic Economy", "First", "Business"]
_NAMES = ["Sarah", "David", "Maria", "John", "Aisha", "Chen", "Omar", "Lena",
          "Raj", "Nina", "Tom", "Yuki"]


def sample_problem(rng, n_bags: int | None = None) -> dict:
    """Sample a structured airline info dict (U.S. domestic) for TRAINING data.
    Kept domestic to match eval level-0 distribution; gold via compute_answer."""
    import random as _r
    r: _r.Random = rng
    nb = n_bags or r.randint(2, 6)
    bags = []
    for i in range(nb):
        # sizes/weights spanning the fee thresholds (62in, 50/70lb) for coverage
        l = r.randint(20, 46); w = r.randint(12, 24); h = r.randint(6, 22)
        weight = r.choice([r.randint(8, 49), r.randint(50, 70), r.randint(71, 95)])
        bags.append({"id": i + 1, "name": r.choice(_BAG_NAMES),
                     "size": [l, w, h], "weight": weight})
    return {
        "base_price": r.randint(80, 300),
        "customer_class": r.choice(_CLASSES),
        "routine": "U.S.", "direction": r.choice([0, 1]),
        "bag_list": bags,
    }


def render_prompt(info: dict, name: str = "The passenger") -> str:
    src, dst = None, None
    lines = [f"{name} is a {info['customer_class']} passenger flying "
             f"within the U.S. with the following items:"]
    for i, b in enumerate(info["bag_list"], 1):
        s = b["size"]
        lines.append(f"{i}. A {b['name']}: {s[0]} x {s[1]} x {s[2]} inches, "
                     f"{b['weight']} lbs;")
    lines.append(f"The base ticket price is ${info['base_price']}.")
    lines.append("Compute the total cost (ticket + all baggage fees).")
    return "\n".join(lines)


def gold_for(info: dict) -> float:
    tables = _tables()
    cwd = os.getcwd()
    sys.path.insert(0, str(AIRLINE))
    os.chdir(AIRLINE)
    try:
        from compute_answer import compute_answer  # noqa: E402
        fee, _ = compute_answer(**info, check_base_tables=tables)
        return float(fee)
    finally:
        os.chdir(cwd)


def load_corpus(include_tables: bool = True) -> str:
    """The rules the engine must internalize: prose rules + fee tables."""
    parts = [(AIRLINE / "reference_rules.txt").read_text()]
    if include_tables:
        parts.append("\n\n# Checked bag fee tables\n")
        ft = AIRLINE / "fee_tables"
        for bag_num in range(1, 5):
            for direction, label in ((0, "U.S. departure"), (1, "U.S. arrival")):
                csv = (ft / f"bag_{bag_num}" / f"{direction}.csv").read_text()
                parts.append(f"\n## Fee for checked bag #{bag_num} ({label})\n{csv}")
    return "".join(parts)


if __name__ == "__main__":
    for lv in (0, 1, 2):
        ps = load_problems(lv)
        golds = [p.gold for p in ps]
        print(f"level {lv}: n={len(ps)} gold[min/mean/max]="
              f"{min(golds):.0f}/{sum(golds)/len(golds):.0f}/{max(golds):.0f}"
              f" avg_bags={sum(len(p.info['bag_list']) for p in ps)/len(ps):.1f}")
    print("corpus chars:", len(load_corpus()))
