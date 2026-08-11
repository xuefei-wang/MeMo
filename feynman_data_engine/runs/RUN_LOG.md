# Feynman Data Engine — Run Log

Autonomous build, 2026-08-11. Branch `feat/feynman-data-engine`.

## Environment
- 7× RTX 6000 Ada (48GB), 6 idle. 128 cores, 755GB RAM. Disk /data 100% (230G free).
- No real API keys → everything local. Cached models: Qwen2.5-3B/1.5B-Instruct,
  Qwen3-14B/32B (vLLM 0.6.6 can't serve Qwen3 arch), Llama-3.3-70B-Instruct.
- vLLM 0.6.6.post1 (works with Qwen2/Llama arch, NOT Qwen3). transformers 4.57, trl 1.9.2.
- Servers: gen=Qwen2.5-3B @:8001 (GPU1), student=Qwen2.5-1.5B @:8002 (GPU2).

## Built & working
- `common/ledger.py` — two-axis token accounting (emitted-training vs total-generator). Tested.
- `common/llm.py` — OpenAI-compatible client, per-call purpose logging, JSON extraction. 
- `data/mtob.py` — MTOB loader (375 train, 50 test/dir, 2531 wordlist, grammar book). Tested.
- `eval/chrf.py` — sacrebleu chrF. Tested.
- `eval/translate.py` — concurrent translation eval (shared by anchors + learner eval).
- `scripts/phase0_anchors.py` — G1 headroom gate.

## Findings
### G1 on MTOB with Qwen2.5-3B: NO HEADROOM → pivot
- KE: closed-book 14.7, ICL-gold 15.7 → band **1.0 chrF** (NARROW).
- EK: closed-book 13.7, ICL-gold 7.9 → band **-5.7** (ICL hurts).
- Diagnosis: 3B model can't do in-context language learning on MTOB. ICL-gold
  samples are fluent-but-hallucinated English; chrF floor inflated by shared
  English function-word characters. Known result — MTOB paper used frontier models.
- Implication: MTOB needs a strong learner (70B+), unfit for a CHEAP-learner
  efficiency-curve study. MTOB scaffolding retained; revisit with a larger learner
  as a later-phase question.

## Decision
Pivot thin slice to **RuleArena** (spec-designated fallback): rules-in-context is
tractable for a 3B model (honest headroom expected), and aligns with the
cheap-fixed-learner efficiency study.

## RuleArena airline (thin slice substrate)
- Corpus = reference_rules.txt + fee tables (~6k tok). gold via compute_answer oracle.
- G1 with exact-match: closed 0.0 / ICL 0.0 (3B can't nail multi-step arithmetic).
  BUT graded metric shows real headroom: median rel-err ICL 0.44 < closed 0.58;
  within-20% ICL 17% > closed 7%.
- **Reframe:** G2 is RELATIVE (feynman-data vs extraction-data, same 3B learner) on
  a GRADED metric -> does not need a high absolute ceiling. Proceed with 3B.

## Pipeline built (all working E2E)
- data/rulearena.py: loader + synthetic problem sampler + gold oracle.
- engine/engine.py: extraction_only (uniform) vs feynman_core (source-blind student
  probe -> skip passes, diagnose+teach+critic on failures = failure-driven curriculum).
- learner/sft.py (LoRA Qwen2.5-3B via trl) + eval_learner.py (HF batched, graded metrics).
- scripts/run_grid.py (resumable: gen -> SFT -> eval -> aggregate), plot_curve.py (2-axis).
- Smoke: feynman burns MORE gen-compute per emitted token than extraction (two-axis story visible).

## Reference floor
- Base Qwen2.5-3B closed-book (no training), 60 test probs: within20=0.0, median_rel_err=0.65.

## In flight
- Validation grid: {extraction, feynman} x b=30k x seed0, eval_n=60. Then full grid
  3 budgets x 2 seeds if signal separates the engines.
