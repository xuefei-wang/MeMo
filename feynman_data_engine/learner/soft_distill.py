"""Soft (context) distillation with LoRA on a single GPU.

Teacher and student are the SAME Qwen3-8B; the ONLY difference is context. For each
synthetic example (kal sentence, teacher answer eng, applicable context):

  teacher forward = frozen base, prompt WITH the grammar context  -> distribution over eng
  student forward = base + LoRA,  prompt WITHOUT any context       -> distribution over eng
  loss = KL(teacher || student) over the answer tokens, temperature T (x T^2).

Two teacher-context modes:
  * pieces  : the 6-8 selected grammar pieces (short, SIEVE-faithful per example).
  * section : one of a FEW fixed book sections (~10-12K tok, "full-book teacher").
              A per-example 10K-token teacher forward would be ~20h/arm at 16K, so we
              cache each fixed section's KV ONCE (GQA -> ~1.5GB each, all resident) and
              forward only the short `kal`+answer suffix, cropping the cache back after.
              This is exact context distillation at book-section strength, cheaply.

The adapter is toggled off for the teacher pass (model.disable_adapter), so both passes
share the same frozen base weights -- no logit storage, no tokenizer drift, one GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.translate import build_prompt  # noqa: E402


def _render_prompt_ids(tok, kal, context):
    msgs = build_prompt(kal, "ke", context)
    try:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)


def _answer_logits(model, prompt_ids, eng_ids, device):
    """Plain forward: logits predicting the eng tokens, shape [len_eng, vocab]."""
    full = torch.tensor([prompt_ids + eng_ids], device=device)
    logits = model(full).logits[0]
    start = len(prompt_ids) - 1
    return logits[start:start + len(eng_ids)]


def _section_prefix_len(tok, section):
    """Longest shared token prefix over two different kal sentences in this section =
    the fixed [system + reference-preamble + section + 'Kalamang: '] block, i.e. the
    index where the variable kal begins. That prefix is identical for every example in
    the section, so its KV is cacheable."""
    a = _render_prompt_ids(tok, "an me qai barlong", section)
    b = _render_prompt_ids(tok, "ZZ maat balim yang", section)
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class SectionTeacher:
    """Caches each fixed section's teacher KV once; answers per example from the short
    suffix only. `model` must have LoRA -- we disable it so the KV is pure base."""

    def __init__(self, model, tok, sections, device):
        self.model, self.tok, self.dev = model, tok, device
        self.prefix_len, self.cache = {}, {}
        for i, sec in enumerate(sections):
            pl = _section_prefix_len(tok, sec)
            ids = _render_prompt_ids(tok, "x", sec)[:pl]
            cache = DynamicCache()
            with torch.no_grad(), model.disable_adapter():
                # logits_to_keep=1: don't run lm_head over all ~10K prefix positions
                model(torch.tensor([ids], device=device),
                      past_key_values=cache, use_cache=True, logits_to_keep=1)
            self.prefix_len[i], self.cache[i] = pl, cache
            print(f"[soft] section {i}: prefix {pl} tok cached", flush=True)

    def answer_logits(self, sec_idx, full_pids, eng_ids):
        pl = self.prefix_len[sec_idx]
        cache = self.cache[sec_idx]
        suffix = full_pids[pl:] + eng_ids
        n = len(suffix)
        inp = torch.tensor([suffix], device=self.dev)
        pos = torch.arange(pl, pl + n, device=self.dev).unsqueeze(0)
        attn = torch.ones((1, pl + n), device=self.dev, dtype=torch.long)
        cpos = torch.arange(pl, pl + n, device=self.dev)
        with torch.no_grad(), self.model.disable_adapter():
            out = self.model(inp, past_key_values=cache, use_cache=True,
                             position_ids=pos, attention_mask=attn,
                             cache_position=cpos).logits[0]
        cache.crop(pl)  # restore for the next example
        start = (len(full_pids) - 1) - pl  # first eng token is predicted at pos P-1
        return out[start:start + len(eng_ids)].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="jsonl with kal,eng,pieces,sec_idx")
    ap.add_argument("--sections", default=None,
                    help="sections.json -> section-teacher (full-book strength) mode")
    ap.add_argument("--base", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=5.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_answer_tokens", type=int, default=64)
    args = ap.parse_args()

    recs = [json.loads(l) for l in Path(args.records).read_text().splitlines() if l.strip()]
    recs = [r for r in recs if r.get("kal") and r.get("eng")]
    sections = json.loads(Path(args.sections).read_text()) if args.sections else None
    mode = "section" if sections else "pieces"
    print(f"[soft] {len(recs)} records -> {args.out}  mode={mode}", flush=True)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map={"": 0})
    model.config.use_cache = False
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.0,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    T = args.temperature

    teacher = SectionTeacher(model, tok, sections, dev) if sections else None

    # pre-tokenize: student prompt (closed-book, short) + teacher context, per record
    data = []
    for r in recs:
        eng_ids = tok.encode(r["eng"], add_special_tokens=False)[:args.max_answer_tokens]
        if not eng_ids:
            continue
        student_pids = _render_prompt_ids(tok, r["kal"], None)[-args.max_len:]
        if teacher is not None:
            si = int(r.get("sec_idx", 0))
            full_pids = _render_prompt_ids(tok, r["kal"], sections[si])
            data.append(("section", si, full_pids, student_pids, eng_ids))
        else:
            t_pids = _render_prompt_ids(tok, r["kal"], r.get("pieces"))[-args.max_len:]
            data.append(("pieces", t_pids, None, student_pids, eng_ids))
    print(f"[soft] {len(data)} usable examples", flush=True)

    n_steps = int(len(data) * args.epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps // args.grad_accum + 1)

    import random
    rng = random.Random(0)
    step = 0
    opt.zero_grad()
    running = 0.0
    for ep in range(int(args.epochs)):
        order = list(range(len(data)))
        rng.shuffle(order)
        for i in order:
            rec = data[i]
            with torch.no_grad():
                if rec[0] == "section":
                    _, si, full_pids, student_pids, eng_ids = rec
                    t_logits = teacher.answer_logits(si, full_pids, eng_ids)
                else:
                    _, t_pids, _, student_pids, eng_ids = rec
                    with model.disable_adapter():
                        t_logits = _answer_logits(model, t_pids, eng_ids, dev).float()
            s_logits = _answer_logits(model, student_pids, eng_ids, dev).float()
            t_logp = F.log_softmax(t_logits / T, dim=-1)
            s_logp = F.log_softmax(s_logits / T, dim=-1)
            loss = F.kl_div(s_logp, t_logp, reduction="batchmean", log_target=True) * (T * T)
            (loss / args.grad_accum).backward()
            running += loss.item()
            step += 1
            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                if (step // args.grad_accum) % 10 == 0:
                    print(f"[soft] ep{ep} step{step} kl={running/(10*args.grad_accum):.4f} "
                          f"lr={sched.get_last_lr()[0]:.2e}", flush=True)
                    running = 0.0

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"[soft] saved adapter -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
