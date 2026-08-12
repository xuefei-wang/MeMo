"""Inference-budget scaling eval for MTOB translation -- the blog's "expertise" axis.

Point accuracy (eval_mtob.py) cannot separate *cramming* (high accuracy at low inference
compute, flat as you spend more) from genuine *studying* (accuracy that scales with
inference compute). This eval sweeps an inference-compute knob and reports the whole curve
plus a single weighted-AUC "expertise" score, following Jacob Li's "Machine Studying":

    "An expert in a domain is an agent that can efficiently turn inference compute into
     accurate work" -- expertise = weighted area under performance-vs-inference-budget,
     weighting cheaper budgets more.

Inference-compute knob: self-consistency budget K = samples per sentence, selected by
reference-free MBR (pick the hypothesis with the highest mean pairwise chrF to the other
samples -- a consensus estimate that needs no gold). K is a clean, model-agnostic proxy
for inference compute; total generated tokens (~K x mean new tokens) is also recorded.

Efficiency trick: generate max(K) samples ONCE per sentence, then read every K off the
prefix samples[:K] -- the whole curve costs one generation pass at the largest budget.

Same model knobs as eval_mtob.py (--adapter, --context cheatsheet/gold/none), so the
extraction-SFT, feynman-SFT and feynman-cheatsheet methods each get a scaling curve on the
same axis: cramming shows as a flat/declining curve, studying as a rising one.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import mtob  # noqa: E402
from eval.chrf import corpus_chrf, sentence_chrf  # noqa: E402
from eval.translate import build_prompt, _clean  # noqa: E402
from learner.eval_learner import load_model  # noqa: E402
from learner.eval_mtob import strip_think  # noqa: E402


def _template(tok, msgs, enable_thinking):
    kw = {}
    if enable_thinking is not None:
        kw["enable_thinking"] = enable_thinking
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def sample_batch(tok, model, prompts, n_samples, max_new=128, bs=8,
                 max_input_len=31000, enable_thinking=None,
                 temperature=0.7, top_p=0.95):
    """Return (samples, gen_tokens): samples[i] is a list of n_samples decoded strings for
    prompt i; gen_tokens[i] is the list of their new-token counts. Sampling on (temp>0)."""
    samples: list[list[str]] = [[] for _ in prompts]
    gen_tokens: list[list[int]] = [[] for _ in prompts]
    for i in range(0, len(prompts), bs):
        chunk = prompts[i:i + bs]
        texts = [_template(tok, m, enable_thinking) for m in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_input_len).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=True,
                             temperature=temperature, top_p=top_p,
                             num_return_sequences=n_samples, pad_token_id=tok.pad_token_id)
        in_len = enc["input_ids"].shape[1]
        # generate() returns len(chunk)*n_samples rows, grouped per prompt
        for j in range(len(chunk)):
            for s in range(n_samples):
                new = gen[j * n_samples + s][in_len:]
                tok_ids = [t for t in new.tolist() if t != tok.pad_token_id]
                samples[i + j].append(tok.decode(new, skip_special_tokens=True))
                gen_tokens[i + j].append(len(tok_ids))
    return samples, gen_tokens


def mbr_select(hyps: list[str]) -> str:
    """Reference-free minimum-Bayes-risk pick: the hypothesis with the highest mean pairwise
    chrF to the others (consensus). Ties/degenerate -> first. Single hyp -> itself.

    Exclude by INDEX, not object identity: identical agreeing samples (interned to one
    object) must still reinforce each other's consensus -- that is exactly the agreement
    MBR rewards -- so `is not` would wrongly drop those votes."""
    if len(hyps) == 1:
        return hyps[0]
    best, best_score = hyps[0], -1.0
    for i, a in enumerate(hyps):
        mean = sum(sentence_chrf(a, b) for k, b in enumerate(hyps) if k != i) / (len(hyps) - 1)
        if mean > best_score:
            best, best_score = a, mean
    return best


def expertise_auc(curve: list[dict], decay: float) -> dict:
    """Weighted area under the chrF-vs-budget curve, cheaper budgets weighted more.

    Budget axis x = log2(K) (so K=1,2,4,8 -> 0,1,2,3, equally spaced). Trapezoidal integral
    of chrF over x, each trapezoid weighted by exp(-decay * x_mid) then normalized by the
    summed weights -> a weight-averaged chrF in chrF units. decay=0 is the plain (uniform)
    average height; larger decay emphasizes the cheap end. Reported alongside the raw curve
    so a downstream analysis can re-weight without re-running generation."""
    import math
    xs = [math.log2(p["K"]) for p in curve]
    ys = [p["chrf"] for p in curve]
    if len(curve) == 1:
        return {"decay": decay, "expertise": ys[0], "auc_uniform": ys[0]}
    num_w = den_w = num_u = den_u = 0.0
    for k in range(len(xs) - 1):
        dx = xs[k + 1] - xs[k]
        avg_h = 0.5 * (ys[k] + ys[k + 1])
        x_mid = 0.5 * (xs[k] + xs[k + 1])
        w = math.exp(-decay * x_mid)
        num_w += w * avg_h * dx; den_w += w * dx
        num_u += avg_h * dx; den_u += dx
    return {"decay": decay,
            "expertise": round(num_w / den_w, 3) if den_w else ys[0],
            "auc_uniform": round(num_u / den_u, 3) if den_u else ys[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-8B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--direction", default="ke", choices=["ke", "ek"])
    ap.add_argument("--context", choices=["none", "gold", "cheatsheet"], default="none")
    ap.add_argument("--cheatsheet_file", default=None)
    ap.add_argument("--book_size", default="medium")
    ap.add_argument("--max_book_chars", type=int, default=80000)
    ap.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 4, 8],
                    help="self-consistency K values (samples/answer) to sweep")
    ap.add_argument("--decay", type=float, default=0.5,
                    help="expertise weight decay over log2(K); larger favors cheap budgets")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max_new", type=int, default=128)
    ap.add_argument("--max_input_len", type=int, default=31000)
    ap.add_argument("--no_think", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    test = mtob.load_test(args.direction)[:args.n]
    ctx = None
    if args.context == "gold":
        ctx = mtob.load_grammar_book(args.book_size)[:args.max_book_chars]
    elif args.context == "cheatsheet":
        if not args.cheatsheet_file:
            ap.error("--context cheatsheet requires --cheatsheet_file")
        ctx = Path(args.cheatsheet_file).read_text()
    prompts = [build_prompt(p.source, args.direction, ctx) for p in test]
    refs = [p.target for p in test]

    max_k = max(args.budgets)
    tok, model = load_model(args.base, args.adapter)
    enable_thinking = False if args.no_think else None
    print(f"[scaling] base={args.base} adapter={args.adapter} ctx={args.context} "
          f"n={len(test)} Ks={args.budgets} temp={args.temperature} think={enable_thinking}")
    samples, gen_tokens = sample_batch(
        tok, model, prompts, n_samples=max_k, max_new=args.max_new, bs=args.bs,
        max_input_len=args.max_input_len, enable_thinking=enable_thinking,
        temperature=args.temperature, top_p=args.top_p)

    curve = []
    for K in sorted(args.budgets):
        hyps, mean_tok = [], 0.0
        for i in range(len(test)):
            picks = [_clean(strip_think(s)) for s in samples[i][:K]]
            hyps.append(mbr_select(picks))
            mean_tok += sum(gen_tokens[i][:K])
        score = corpus_chrf(hyps, refs)
        curve.append({"K": K, "chrf": round(score, 3),
                      "gen_tokens_per_answer": round(mean_tok / len(test), 1)})
        print(f"  K={K:>2}  chrF={score:.3f}  gen_tok/ans={mean_tok/len(test):.0f}")

    expertise = expertise_auc(curve, args.decay)
    summary = {"base": args.base, "adapter": args.adapter, "direction": args.direction,
               "context": args.context, "cheatsheet_file": args.cheatsheet_file,
               "n": len(test), "temperature": args.temperature,
               "context_chars": len(ctx) if ctx else 0,
               "curve": curve, **expertise}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary}, indent=2))
    print(f"[scaling] expertise(decay={args.decay})={expertise['expertise']} "
          f"auc_uniform={expertise['auc_uniform']} -> {args.out}")


if __name__ == "__main__":
    main()
