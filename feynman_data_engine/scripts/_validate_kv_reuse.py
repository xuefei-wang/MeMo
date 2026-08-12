"""Validate SectionTeacher KV-reuse == a full teacher forward (answer logits)."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from data import mtob
from engine.headtohead_mtob import book_sections
from learner.soft_distill import SectionTeacher, _render_prompt_ids, _answer_logits

dev = "cuda"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype=torch.bfloat16,
                                             device_map={"": 0})
model.config.use_cache = False
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM"))
model.eval()

book = mtob.load_grammar_book("medium")[:115000]
sections = book_sections(book, 4)
teacher = SectionTeacher(model, tok, sections, dev)

cases = [(0, "an me qai barlong", "I am going to the garden"),
         (2, "maat balim yang", "the sea is calm"),
         (3, "kantop ma he", "that is a knife")]
worst_kl, worst_top1 = 0.0, True
for si, kal, eng in cases:
    eng_ids = tok.encode(eng, add_special_tokens=False)[:64]
    full_pids = _render_prompt_ids(tok, kal, sections[si])
    got = teacher.answer_logits(si, full_pids, eng_ids)
    with torch.no_grad(), model.disable_adapter():
        ref = _answer_logits(model, full_pids, eng_ids, dev).float()
    kl = torch.nn.functional.kl_div(
        got.log_softmax(-1), ref.log_softmax(-1), reduction="batchmean", log_target=True).item()
    top1 = bool((got.argmax(-1) == ref.argmax(-1)).all())
    maxdiff = (got - ref).abs().max().item()
    worst_kl = max(worst_kl, abs(kl)); worst_top1 = worst_top1 and top1
    print(f"sec{si} '{kal[:20]}': KL(got||ref)={kl:.2e} top1_match={top1} maxlogitdiff={maxdiff:.3f}")

print(f"\nWORST kl={worst_kl:.2e} all_top1_match={worst_top1}")
print("PASS" if worst_kl < 1e-2 and worst_top1 else "FAIL")
