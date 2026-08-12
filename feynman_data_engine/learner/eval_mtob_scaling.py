"""Inference-budget scaling eval for MTOB translation -- the blog's "expertise" axis.

Point accuracy (eval_mtob.py) cannot separate *cramming* (high accuracy at low inference
compute, flat as you spend more) from genuine *studying* (accuracy that scales with
inference compute). This eval sweeps an inference-compute knob and reports the whole curve
plus a single weighted-AUC "expertise" score, following Jacob Li's "Machine Studying":

    "An expert in a domain is an agent that can efficiently turn inference compute into
     accurate work" -- expertise = weighted area under performance-vs-inference-budget,
     weighting cheaper budgets more.

The expertise score uses StudyBench's exact form: E = integral p(x) w(x) dx with a
log10-token budget axis and w(x) = ln(10) * 10^(-x) (see expertise_auc). Budget is real
tokens/answer (context prefill once + K x generation), so a small cheat-sheet context is
rewarded over the full grammar book -- the whole point of the efficiency framing.

Inference-compute knob: self-consistency budget K = samples per sentence, selected by
reference-free MBR (pick the hypothesis with the highest mean pairwise chrF to the other
samples -- a consensus estimate that needs no gold).

Efficiency trick: generate max(K) samples ONCE per sentence, then read every K off the
prefix samples[:K] -- the whole curve costs one generation pass at the largest budget.

Same model knobs as eval_mtob.py (--adapter, --context cheatsheet/gold/none), so the
extraction-SFT, feynman-SFT and feynman-cheatsheet methods each get a scaling curve on the
same axis: cramming shows as a flat/declining curve, studying as a rising one.
"""
from __future__ import annotations

import argparse
import json
import math
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


def expertise_auc(curve: list[dict], anchor_tokens: float = 3000.0) -> dict:
    """StudyBench expertise metric ("Machine Studying"): the weighted area under the
    performance-vs-inference-compute curve.

        E = integral_0^inf  p(x) w(x) dx ,   x = log10(tokens / anchor_tokens),
            w(x) = ln(10) * 10^(-x)      (integrates to 1 over [0, inf))

    p(x) is a STEP function of the measured budgets: score_i holds on [x_i, x_{i+1}),
    the region below the cheapest budget scores 0, and the last score is CARRIED FORWARD
    to infinity. Weight mass on [x_i, x_{i+1}) = 10^(-x_i) - 10^(-x_{i+1}); the tail
    [x_last, inf) has mass 10^(-x_last). So E is a weight-averaged score in the metric's
    own units (here chrF), with cheaper budgets weighted exponentially more -- doubling
    the compute halves the weight. This step/carry-forward form reproduces the blog's
    worked example (budgets 5k/10k/20k/100k @ 10/20/30/40% -> 10.8%).

    anchor_tokens is the x=0 floor: budgets below it score 0. The blog uses 3000 for its
    agentic tasks. MTOB single-sentence translation spends far fewer tokens/answer, so a
    3000 anchor can zero the whole curve; pass a task-scaled anchor (e.g. the cheapest
    budget's tokens) for within-MTOB resolution. The raw curve is always returned so a
    downstream analysis can re-anchor/re-weight without re-running generation."""
    pts = sorted((p for p in curve if p["total_tokens_per_answer"] >= anchor_tokens),
                 key=lambda p: p["total_tokens_per_answer"])
    if not pts:  # every budget is below the anchor floor -> no credited compute
        return {"anchor_tokens": anchor_tokens, "expertise": 0.0, "n_credited_budgets": 0}
    xs = [math.log10(p["total_tokens_per_answer"] / anchor_tokens) for p in pts]
    ys = [p["chrf"] for p in pts]
    e = 0.0
    for i in range(len(pts)):
        lo = 10.0 ** (-xs[i])
        hi = 10.0 ** (-xs[i + 1]) if i + 1 < len(pts) else 0.0  # tail carries to inf
        e += ys[i] * (lo - hi)
    return {"anchor_tokens": anchor_tokens, "expertise": round(e, 3),
            "n_credited_budgets": len(pts)}


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
    ap.add_argument("--expertise_anchor_tokens", type=float, default=3000.0,
                    help="StudyBench x=0 token floor; budgets below it score 0. Blog uses "
                         "3000 (agentic scale). Pass 0 to auto-anchor at the cheapest budget "
                         "for within-MTOB resolution.")
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
    # mean prompt tokens = the (context-inclusive) prefill each answer pays ONCE; the K
    # samples share prefill (num_return_sequences), so tokens/answer = prompt + K*gen.
    # This is why a small cheat-sheet context beats the full book on the expertise axis.
    prompt_lens = [len(tok(_template(tok, m, enable_thinking))["input_ids"]) for m in prompts]
    mean_prompt_tokens = sum(prompt_lens) / len(prompt_lens)
    print(f"[scaling] base={args.base} adapter={args.adapter} ctx={args.context} "
          f"n={len(test)} Ks={args.budgets} temp={args.temperature} think={enable_thinking} "
          f"mean_prompt_tok={mean_prompt_tokens:.0f}")
    samples, gen_tokens = sample_batch(
        tok, model, prompts, n_samples=max_k, max_new=args.max_new, bs=args.bs,
        max_input_len=args.max_input_len, enable_thinking=enable_thinking,
        temperature=args.temperature, top_p=args.top_p)

    curve = []
    for K in sorted(args.budgets):
        hyps, gen_sum = [], 0.0
        for i in range(len(test)):
            picks = [_clean(strip_think(s)) for s in samples[i][:K]]
            hyps.append(mbr_select(picks))
            gen_sum += sum(gen_tokens[i][:K])
        score = corpus_chrf(hyps, refs)
        gen_per_ans = gen_sum / len(test)
        total_per_ans = mean_prompt_tokens + gen_per_ans  # prefill once + K*gen
        curve.append({"K": K, "chrf": round(score, 3),
                      "gen_tokens_per_answer": round(gen_per_ans, 1),
                      "total_tokens_per_answer": round(total_per_ans, 1)})
        print(f"  K={K:>2}  chrF={score:.3f}  tok/ans={total_per_ans:.0f} "
              f"(prompt {mean_prompt_tokens:.0f} + gen {gen_per_ans:.0f})")

    # auto-anchor (--expertise_anchor_tokens 0) = the cheapest budget's tokens/answer, for
    # within-MTOB resolution when absolute tokens sit below the blog's 3k floor.
    anchor = args.expertise_anchor_tokens or min(p["total_tokens_per_answer"] for p in curve)
    expertise = expertise_auc(curve, anchor)
    summary = {"base": args.base, "adapter": args.adapter, "direction": args.direction,
               "context": args.context, "cheatsheet_file": args.cheatsheet_file,
               "n": len(test), "temperature": args.temperature,
               "context_chars": len(ctx) if ctx else 0,
               "mean_prompt_tokens": round(mean_prompt_tokens, 1),
               "curve": curve, **expertise}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary}, indent=2))
    print(f"[scaling] expertise(anchor={anchor:.0f}tok)={expertise['expertise']} "
          f"over {expertise['n_credited_budgets']}/{len(curve)} credited budgets -> {args.out}")


if __name__ == "__main__":
    main()
