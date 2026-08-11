"""LoRA-SFT the fixed learner (Qwen2.5-3B-Instruct) on an engine's dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def load_dataset(path: str) -> Dataset:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    return Dataset.from_list(rows)  # each row: {"messages": [...]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=2048)
    args = ap.parse_args()

    ds = load_dataset(args.dataset)
    print(f"[sft] {len(ds)} examples -> {args.out}")
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map={"": 0})
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")
    cfg = SFTConfig(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.05,
        logging_steps=5, save_strategy="no", bf16=True,
        max_length=args.max_len, packing=False, report_to=[],
        dataset_kwargs={"skip_prepare_dataset": False},
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tok, peft_config=lora)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"[sft] saved adapter -> {args.out}")


if __name__ == "__main__":
    main()
