"""Evaluate a trained learner (base + LoRA adapter, or bare base) on RuleArena
airline test problems, using HF transformers batched generation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import rulearena as ra  # noqa: E402
from eval.rulearena_eval import build_prompt, extract_total  # noqa: E402

TOL = 1.0


def load_model(base: str, adapter: str | None):
    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map={"": 0})
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    model.eval()
    return tok, model


@torch.no_grad()
def generate_batch(tok, model, prompts: list[list[dict]], max_new=700, bs=16):
    outs = []
    for i in range(0, len(prompts), bs):
        chunk = prompts[i:i + bs]
        texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                 for m in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=4096).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            new = gen[j][enc["input_ids"].shape[1]:]
            outs.append(tok.decode(new, skip_special_tokens=True))
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--context", choices=["none", "gold"], default="none")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    probs = ra.load_problems(args.level)[:args.n]
    ctx = ra.load_corpus(True) if args.context == "gold" else None
    prompts = [build_prompt(p, ctx) for p in probs]
    tok, model = load_model(args.base, args.adapter)
    print(f"[eval] {len(probs)} problems, adapter={args.adapter}")
    outs = generate_batch(tok, model, prompts)

    results = []
    for p, o in zip(probs, outs):
        pred = extract_total(o)
        results.append({"pred": pred, "gold": p.gold,
                        "exact": pred is not None and abs(pred - p.gold) <= TOL})
    n = len(results)
    rel = [abs(r["pred"] - r["gold"]) / max(r["gold"], 1.0)
           for r in results if r["pred"] is not None]
    rel_sorted = sorted(rel)
    summary = {
        "adapter": args.adapter, "level": args.level, "n": n,
        "exact_acc": round(sum(r["exact"] for r in results) / n, 4),
        "parse_rate": round(sum(r["pred"] is not None for r in results) / n, 4),
        "within10": round(sum(e <= 0.10 for e in rel) / n, 4),
        "within20": round(sum(e <= 0.20 for e in rel) / n, 4),
        "median_rel_err": round(rel_sorted[len(rel_sorted) // 2], 4) if rel_sorted else 1.0,
        "mean_rel_err": round(sum(rel) / len(rel), 4) if rel else 1.0,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    print(f"[eval] {json.dumps(summary)}")


if __name__ == "__main__":
    main()
