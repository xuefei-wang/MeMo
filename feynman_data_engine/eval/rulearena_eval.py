"""RuleArena airline eval: run a model over problems, extract the final total,
score exact-match (and within-$1) against the computed gold."""
from __future__ import annotations

import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.llm import LLM  # noqa: E402
from data import rulearena  # noqa: E402

SYS = ("You are an expert airline fare assistant. Compute the passenger's TOTAL "
       "cost = ticket base price + all applicable checked/overweight/oversize "
       "baggage fees, by applying the airline's rules. Reason step by step. "
       "End your answer with a line exactly of the form:\nFINAL: <dollars>\n"
       "where <dollars> is a single integer with no $ sign or commas.")


def build_prompt(p: rulearena.Problem, context: str | None) -> list[dict]:
    user = ""
    if context:
        user += ("=== AIRLINE FEE RULES ===\n" + context +
                 "\n=== END RULES ===\n\n")
    user += p.prompt.rstrip()
    if not user.rstrip().endswith((".", ":", ";")):
        user += "\n\nCompute the total cost."
    return [{"role": "system", "content": SYS}, {"role": "user", "content": user}]


_NUM = re.compile(r"FINAL:\s*\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)")
_ANYNUM = re.compile(r"\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)")


_MAX_PLAUSIBLE = 100000.0  # realistic totals are < ~10k; guard the fallback


def extract_total(out: str) -> float | None:
    m = _NUM.search(out)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    # fallback: last PLAUSIBLE number in the text (ignore absurd IDs/years/dims)
    for tok in reversed(_ANYNUM.findall(out)):
        try:
            v = float(tok.replace(",", ""))
        except ValueError:
            continue
        if 0 <= v <= _MAX_PLAUSIBLE:
            return v
    return None


def run_eval(llm: LLM, problems: list[rulearena.Problem], context: str | None = None,
             purpose: str = "eval", max_workers: int = 32, max_tokens: int = 1200,
             temperature: float = 0.0, tol: float = 1.0) -> dict:
    def one(p: rulearena.Problem):
        msgs = build_prompt(p, context)
        out = llm.chat(msgs, purpose=purpose, temperature=temperature,
                       max_tokens=max_tokens)
        pred = extract_total(out)
        exact = pred is not None and abs(pred - p.gold) <= tol
        return {"pred": pred, "gold": p.gold, "exact": exact}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(one, problems))
    n = len(results)
    acc = sum(r["exact"] for r in results) / n if n else 0.0
    parsed = sum(r["pred"] is not None for r in results) / n if n else 0.0
    # graded metrics -- the exact-match ceiling collapses on a weak learner, so
    # these carry the G2 relative signal (feynman-data vs extraction-data).
    rel = [abs(r["pred"] - r["gold"]) / max(r["gold"], 1.0)
           for r in results if r["pred"] is not None]
    rel_sorted = sorted(rel)
    med_rel = rel_sorted[len(rel_sorted) // 2] if rel_sorted else 1.0
    within10 = sum(e <= 0.10 for e in rel) / n if n else 0.0
    within20 = sum(e <= 0.20 for e in rel) / n if n else 0.0
    return {"accuracy": round(acc, 4), "parse_rate": round(parsed, 4),
            "within10": round(within10, 4), "within20": round(within20, 4),
            "median_rel_err": round(med_rel, 4), "n": n, "results": results}


if __name__ == "__main__":
    print("extract test:", extract_total("... so FINAL: $1,245"))  # 1245.0
    print("extract fallback:", extract_total("the total is 949 dollars"))  # 949.0
