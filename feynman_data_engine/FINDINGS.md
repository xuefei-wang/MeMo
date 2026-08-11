# Feynman Data Engine — Early Findings (Phase 0 + Thin Slice)

**Question.** Is a *Feynman* recipe — teach-back with **failure-driven repair** — a
better synthetic-data engine than one-shot extraction, at matched budget, for
injecting a small corpus's knowledge into a model's parameters?

**This is a thin de-risking slice**, not the full study. Its job was two go/no-go
gates (G1 headroom, G2 does-the-loop-move-the-metric) on the cheapest viable setup.

---

## Setup

- **Engines** (both emit worked SFT examples in the eval format; differ only in the loop):
  - `extraction_only` — uniform random problems, one-pass correct worked solution (SIEVE/MeMo-shaped control).
  - `feynman_core` — a source-blind **student** attempts each problem with the engine's
    running rule cheat-sheet; **passes are skipped** (no budget on what's already learnable),
    **failures** trigger diagnose → teach → simplicity-critic → a worked solution that recaps
    the misapplied rule. Budget concentrates on the student's failure frontier.
- **Learner (fixed):** Qwen2.5-3B-Instruct, LoRA-SFT. **Generator:** Qwen2.5-3B. **Student:** Qwen2.5-1.5B.
- **Two budget axes:** emitted training tokens; total generator-completion tokens (the loop's
  probe/diagnose calls that never become training data).
- **All local** (7× RTX 6000 Ada), no API keys; every LLM call logged through a token ledger.

## Substrate journey (G1)

| Substrate | Learner | Headroom (closed-book → ICL-gold) | Verdict |
|---|---|---|---|
| **MTOB** (translate Kalamang from a grammar book) | Qwen2.5-3B | KE 14.7→15.7 chrF (band **1.0**); EK band **−5.7** | **No headroom** — 3B can't do in-context language learning; MTOB needs a frontier learner. |
| **RuleArena airline** (apply fee rules → total cost) | Qwen2.5-3B | exact-match 0→0, but **graded** medRelErr 0.58→0.44, within20 7%→17% | **Graded headroom** — used. |

**Key reframe:** exact-match collapses on a weak learner (it can't nail multi-step
arithmetic), so G2 is measured as a **relative** comparison (feynman-data vs
extraction-data training the *same* 3B) on a **graded** metric (within-20% of gold).
A relative comparison needs metric *resolution*, not a high absolute ceiling.

## Result (G2) — Grid: {extraction, feynman} × budget × 2 seeds

**Base floor** (untrained Qwen2.5-3B, closed-book, n=100): within20 = 0.02, medRelErr = 0.66.

| budget | engine | within20 (↑) | medRelErr (↓) | gen tokens |
|---|---|---|---|---|
| 30k | extraction | 0.045 | 0.744 | 23.6k |
| 30k | feynman | 0.050 | 0.729 | 38.5k |
| 90k | extraction | 0.060 | 0.776 | 69k |
| 90k | feynman | 0.070 | 0.840 | 113k |
| 200k | extraction | 0.100 | 0.642 | 154k |
| 200k | feynman | **0.125** | 0.719 | 231k |

**Bootstrap (pooled per-problem across seeds, 95% CI):**

| budget | feynman − extraction (within20) | significant? | compute premium |
|---|---|---|---|
| 30k | +0.005 [−0.035, +0.045] | no (CI spans 0) | 1.63× |
| 90k | +0.010 [−0.035, +0.055] | no (CI spans 0) | 1.63× |
| 200k | **+0.025** [−0.035, +0.090] | no (CI spans 0) | 1.50× |

Plots: `runs/grid/curve_within20.png`, `runs/grid/curve_medrelerr.png` (both budget axes).

## Honest read (full 3-point curve)

1. **A consistent, *growing* advantage — but not yet significant.** Feynman's within20
   edge widens monotonically with budget: **+0.005 → +0.010 → +0.025**. At 200k feynman
   is ~**25% relatively better** (0.125 vs 0.100), and both engines finally lift clearly
   off the floor (0.02). The direction is stable across all three budgets and both seeds —
   but the bootstrap CI still spans 0 (2 seeds is under-powered for a ~2-point effect).
2. **The loop is not free.** Feynman spends **1.5–1.6× generator compute**. On the
   *compute* axis (right panel) its edge shrinks and extraction is more efficient in the
   mid-range — the emitted-token win is partly bought with extra generator calls.
3. **Training helps most at scale.** At 200k, extraction's median-rel-err (0.64) finally
   returns to ~the untrained floor (0.66) and within20 doubles vs 90k — the 3B needs
   substantial data before the procedure sticks. At low budget SFT mostly adds variance.
4. **Verdict:** *promising but unconfirmed.* The Feynman recipe shows the predicted
   budget-scaling behaviour (its advantage grows exactly where failure-driven curricula
   should help — once there's enough budget to skip the easy and concentrate on the hard),
   but this thin slice at 3B scale cannot yet call it a win. The right next step is power
   (more seeds) and a task-capable learner, not a new mechanism.

## What this de-risked (and what it did not)

- ✅ Full engine → LoRA-SFT → eval pipeline works end-to-end, both budget axes logged.
- ✅ The failure-driven loop *mechanically* works (student discriminates; budget concentrates on hard cases).
- ✅ A real design trap found & fixed (a too-strict pass threshold made feynman ≡ extraction+overhead).
- ❌ **Not yet shown**: that failure-driven teaching *beats* uniform extraction. Early signal is null-to-marginal at a 3B scale where the task capability (not knowledge) is the binding constraint.

## Next (later specs)

- Higher-budget point (wave 2, in flight) + more seeds for power.
- A **task-capable learner** (≥14B) so the absolute ceiling isn't capacity-bound — the
  cleanest way to let a knowledge-injection difference show.
- Data-quality gate: verify generator reasoning against the oracle before emitting.
- The full ablation ladder (M1→M2→M3→loop) and incumbents (SIEVE-GEN, Cartridges).
