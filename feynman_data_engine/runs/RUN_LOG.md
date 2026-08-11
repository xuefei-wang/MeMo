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

## First signal (validation, pre-fix, 30k/seed0, eval_n=60)
- extraction: within20=0.050, medRelErr=0.641, gen=23456
- feynman:    within20=0.033, medRelErr=0.757, gen=37740  (WORSE + 60% more compute)
- Diagnosis: NOT just noise. At low budget both train on ~identical data (same seed
  -> same problems; 1.5B student failed ~everything at near-exact threshold -> feynman
  never skipped -> == extraction + diagnose/recap overhead). Failure loop couldn't bite.

## Root-cause fix
- Loosened feynman "pass" to a graded 25% band. Student now discriminates: skipped
  3/16 easy problems in a 12k smoke -> budget concentrates on the hard frontier.
- Gen-compute overhead ~1.37x emitted (two-axis cost confirmed).

## Base references (n=100, learner Qwen2.5-3B, extractor-fixed)
- closed-book floor: within20=0.02, medRelErr=0.66.
- ICL-gold ceiling UNRELIABLE on 3B: rules (6k tok) exceed eval_learner max_length
  4096 -> problem truncated -> garbage. 3B can't use ICL anyway (phase0). Use floor only.

## Grid wave 1 results (post-fix, mean over 2 seeds; floor within20=0.02, medRelErr=0.66)
| cell | within20 | medRelErr | gen tokens |
|---|---|---|---|
| extraction 30k | 0.045 | 0.744 | 23.6k |
| feynman 30k    | 0.050 | 0.729 | 38.5k |
| extraction 90k | 0.060 | 0.776 | 69k |
| feynman 90k    | 0.070 | 0.840 | 113k |

Read:
- within20 (primary): feynman >= extraction at both budgets; feynman scales steeper
  (0.050->0.070 vs 0.045->0.060). BUT seed variance (extraction 30k: 0.02 vs 0.07)
  SWAMPS the gap -> not significant at 2 seeds.
- feynman costs ~1.6x generator compute -> on the gen-token axis, extraction wins.
- SFT RAISES medRelErr above untrained base (0.66): training adds variance
  (more near-hits AND more big misses), esp. at 90k.
- Net: weak/null, slight feynman edge on within20 that GROWS with budget -> motivates
  a higher-budget wave (the "needs budget to pay off" hypothesis).

## Grid wave 2 (200k added) — full 3-point curve
- within20 (mean/2 seeds): 30k ext .045/fey .050 | 90k .060/.070 | 200k .100/.125.
- feynman-ext diff GROWS: +0.005 -> +0.010 -> +0.025 (monotonic, both seeds same sign).
- Bootstrap CIs still span 0 (under-powered at 2 seeds); compute premium 1.5-1.6x.
- At 200k both engines lift off floor; extraction medRelErr 0.64 ~ base floor 0.66.
- Verdict: PROMISING but UNCONFIRMED. Feynman shows the predicted budget-scaling
  (advantage grows where failure-driven curricula should help). Next: more seeds +
  task-capable learner (>=14B), not a new mechanism.
- Deliverables: FINDINGS.md, runs/grid/curve_{within20,medrelerr}.png, results.json,
  scripts/{run_grid,plot_curve,analyze}.py. All committed.
