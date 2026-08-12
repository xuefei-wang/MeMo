"""Build a budget-B SIEVE-GEN SFT dataset by subsampling an existing pool dataset:
gold base (kept whole) + synthetic pairs until ~B synthetic training tokens. Lets the
notes-vs-pairs frontier reuse ONE 16K pool instead of regenerating per budget."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transformers import AutoTokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool_dataset", required=True, help="s*/sieve_gen/dataset.jsonl")
    ap.add_argument("--n_gold", type=int, default=375, help="gold base rows prepended")
    ap.add_argument("--budget", type=int, required=True, help="synthetic training tokens")
    ap.add_argument("--base", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    rows = [json.loads(l) for l in Path(args.pool_dataset).read_text().splitlines() if l.strip()]
    gold, synth = rows[:args.n_gold], rows[args.n_gold:]
    kept, tot = [], 0
    for ex in synth:
        if tot >= args.budget:
            break
        kept.append(ex)
        tot += len(tok.encode("\n".join(m["content"] for m in ex["messages"])))
    out_rows = gold + kept
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for ex in out_rows:
            f.write(json.dumps(ex) + "\n")
    print(f"[subsample] budget={args.budget} -> {len(kept)} synth ({tot} tok) "
          f"+ {len(gold)} gold = {len(out_rows)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
