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

## Result (G2) — Grid: {extraction, feynman} × 3 budgets × **4 seeds**

**Base floor** (untrained Qwen2.5-3B, closed-book, n=100): within20 = 0.02, medRelErr = 0.66.

within20, mean over 4 seeds (per-seed spread in brackets):

| budget | extraction | feynman | fey − ext | compute premium |
|---|---|---|---|---|
| 30k  | 0.040 [0.02–0.07] | 0.048 [0.04–0.06] | +0.008 | 1.67× |
| 90k  | 0.068 [0.06–0.08] | 0.085 [0.07–0.12] | **+0.017** | 1.60× |
| 200k | 0.090 [0.05–0.12] | 0.098 [0.06–0.13] | +0.008 | 1.52× |

**Bootstrap (pooled per-problem, 95% CI) — none significant:**
30k +0.007 [−0.022,+0.035] · 90k +0.018 [−0.020,+0.055] · 200k +0.008 [−0.033,+0.048].

**Seed-level paired diffs (fey−ext, per seed):**
30k [−.01,.02,.01,.01] · 90k [.01,.01,.05,.00] · 200k [.01,.04,−.03,.01] — mostly
positive, but not unanimously; one 200k seed favours extraction.

Plots: `runs/grid/curve_within20.png`, `runs/grid/curve_medrelerr.png` (both budget axes).

## Honest read (full 3-budget × 4-seed grid)

1. **A small, consistent positive lean — within noise.** Feynman's mean within20 is
   ≥ extraction at every budget (+0.008, +0.017, +0.008), but seed variance is large
   (extraction at 200k spans 0.05–0.12) and every CI spans 0. **No confirmed advantage.**
2. **Methodological catch — power matters.** The 2-seed run suggested a *monotonically
   growing* edge (+0.005→+0.010→+0.025, both seeds positive at 200k). Two more seeds
   flattened it: the 200k diff fell to +0.008 and one new seed favoured extraction. The
   "growing trend" was partly a small-sample artifact — a caution logged for the full study.
3. **The loop is not free.** Feynman costs **1.5–1.7× generator compute**. On the *compute*
   axis its already-small edge all but vanishes; extraction is more compute-efficient
   mid-range. Any real Feynman value must clear this ~1.6× tax.
4. **Both engines clearly beat the floor and scale with budget** (0.02 → ~0.09–0.10 at
   200k). Synthetic worked-example data *works*; which recipe generates it barely moves
   the needle at 3B scale, where task *capability* — not knowledge coverage — is binding.
5. **Verdict:** *no confirmed win for Feynman on this slice.* A persistent but
   non-significant positive lean, bought with extra compute. The mechanism is sound and
   implemented; discriminating it needs (a) more seeds for power and (b) a **task-capable
   learner (≥14B)** so the ceiling isn't capacity-bound — not a new mechanism.

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

---

# MTOB @ Qwen3-8B — the knowledge-bound substrate (first grid)

The RuleArena verdict said the null was **capability-bound**: at 3B the binding
constraint was task capability, not knowledge coverage, so a knowledge-injection
recipe had no room to show. MTOB is the recommended fix — a translation task where
knowledge coverage (Kalamang grammar+lexicon) IS the binding constraint, on a
capable 8B learner. First grid: 2 modes x {6k,20k} emitted tokens x 2 seeds.

Setup: shared un-budgeted gold base (375 pairs) for BOTH engines; budget counts only
ADDED synthetic notes. Learner Qwen3-8B LoRA. Metric: closed-book ke chrF, n=50, no-think.
Reference frame: floor (untrained, no context) 15.9 chrF; ICL-gold ceiling ~27.

## Result (closed-book ke chrF)

| budget | extraction (2 seeds) | feynman (2 seeds) | Δ | per-sentence paired bootstrap |
|--------|----------------------|-------------------|-----|-------------------------------|
| 6k  | 21.14, 20.57 → 20.86 | 21.60, 21.22 → 21.41 | +0.56 | CI [-0.72,+1.57] p=0.55 n.s. |
| 20k | 21.82, 20.85 → 21.33 | 22.35, 22.65 → 22.50 | +1.17 | CI [+0.04,+2.28] **p=0.042 sig** |

1. **Feynman > extraction at matched emitted-token budget, and the gap WIDENS with
   budget** (+0.56 → +1.17). This is the efficiency-curve signature the hypothesis
   predicts: failure-targeted teaching compounds; uniform coverage saturates. Contrast
   the RuleArena 3B run where the "growing edge" was small-sample noise — here the
   *larger* budget has the *cleaner* signal, and the 20k per-sentence bootstrap crosses
   significance.
2. **Compute is now in Feynman's favour on the axis that matters.** Completion tokens
   (what gets generated): feynman uses FEWER (-16% @6k, -9% @20k) for higher chrF —
   the opposite of the RuleArena 1.6x tax. Prefill-inclusive it is 4.8x higher (many
   diagnose calls re-send the book), ~free under vLLM prefix-caching of the shared book
   prefix — flagged, not hidden.
3. **Mechanism visible in the data.** At 20k feynman emits 175 short failure-diagnoses
   vs extraction's 58 long chunk-notes — same budget, far more items, each aimed at the
   student's actual translation-failure frontier.
4. **Why the flip vs RuleArena.** MTOB is knowledge-bound on a capable learner, so the
   engine's job (inject the right knowledge) is the binding constraint — exactly where a
   targeting advantage can show. This supports the RuleArena diagnosis: the earlier null
   was substrate/scale, not a broken mechanism.

## Caveats (why this is a signal, not a verdict)

- **n=2 seeds** → seed-level significance is out of reach; the per-sentence bootstrap
  pools seeds as a proxy. The widening gap is the real signal; confirmation needs seeds.
- One benchmark, one direction (ke), one model, two budgets. No mid-budget point yet to
  confirm curve shape.

## Next

- More seeds at both budgets (in flight) for a seed-level paired test.
- A mid budget (~12k) to trace curve shape; the ek direction as a transfer check.
