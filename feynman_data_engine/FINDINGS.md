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
capable 8B learner. Grid: 2 modes x {6k,20k} emitted tokens x 4 seeds.

Setup: shared un-budgeted gold base (375 pairs) for BOTH engines; budget counts only
ADDED synthetic notes. Learner Qwen3-8B LoRA. Metric: closed-book ke chrF, n=50, no-think.
Reference frame: floor (untrained, no context) 15.9 chrF; ICL-gold ceiling ~27.

## Result (closed-book ke chrF, 4 seeds)

| budget | extraction (mean of 4) | feynman (mean of 4) | Δ | per-sentence paired bootstrap |
|--------|------------------------|---------------------|-----|-------------------------------|
| 6k  | 21.14, 20.57, 21.50, 21.21 → 21.10 | 21.60, 21.22, 21.23, 21.72 → 21.44 | +0.34 | CI [-0.47,+1.07] p=0.45 n.s. |
| 20k | 21.82, 20.85, 22.22, 21.88 → 21.69 | 22.35, 22.65, 22.39, 21.72 → 22.28 | +0.59 | CI [-0.36,+1.38] p=0.24 n.s. |

**The confirmatory seeds corrected an over-read.** A 2-seed cut showed +0.56/+1.17 with
the 20k per-sentence bootstrap crossing significance (p=0.042). Adding seeds 2,3 pulled
extraction's 20k mean up (the two new extraction seeds landed at 21.50/22.22) and the gap
fell to +0.34/+0.59 — **neither budget significant at n=4.** Same lesson as the RuleArena
run: two seeds over-read. The check did its job.

1. **A consistent, non-significant positive lean that still widens with budget** (+0.34 →
   +0.59). feynman ≥ extraction at both budgets and its mean is higher in every cell; the
   ordering is stable, the magnitude is inside seed noise. Directionally the
   efficiency-curve signature is intact; it is not yet a confirmed win.
2. **No compute penalty — a real advance over RuleArena.** On completion tokens (what gets
   generated) feynman uses FEWER (-15% @6k, -9% @20k) for its (slightly higher) chrF — the
   opposite of the RuleArena 1.6x tax. Prefill-inclusive it is ~4.8x higher (many diagnose
   calls re-send the book), ~free under vLLM prefix-caching of the shared book prefix —
   flagged, not hidden. So the lean is bought at no marginal generation cost.
3. **Mechanism visible in the data.** At 20k feynman emits ~176 short failure-diagnoses
   vs extraction's ~58 long chunk-notes — same budget, far more items, each aimed at the
   student's actual translation-failure frontier.
4. **Why it beats the RuleArena null even without significance.** MTOB is knowledge-bound
   on a capable learner, so the engine's job (inject the right knowledge) is the binding
   constraint. The lean is now positive-and-free rather than null-and-taxed — consistent
   with the RuleArena diagnosis that the earlier null was substrate/scale, not a broken
   mechanism.

## Verdict

*No confirmed win at n=4, but a materially better picture than RuleArena:* a stable
positive lean that widens with budget and costs no extra generation. The mechanism is
sound and the substrate is right; discriminating it needs **more power** (seeds/budgets),
not a new mechanism.

## Next

- More seeds + a mid budget (~12k) to trace curve shape and get seed-level power.
- ek direction as a transfer check.
- Data-quality gate (verify generator reasoning vs oracle before emitting).

---

# Head-to-head: Feynman-GEN vs SIEVE-GEN (matched-SFT, MTOB @ Qwen3-8B)

Goal: compare the Feynman data engine against SIEVE's on equal footing -- "match SIEVE,
differ only in the data engine." SIEVE = decompose the book -> diverse synthetic
translation queries -> distill. Fully faithful reproduction needs SIEVE's 8-GPU soft
distillation (top-k logits, ZeRO-3), infeasible on the available 2-3 GPUs, so we ran the
feasible **matched-SFT** variant (hard labels, applied equally to both arms).

Design: ONE shared candidate factory (SIEVE-GEN's exact 3-step recipe, their prompts;
gold sentences added as style anchors after a de-risk showed naive gen makes word-salad).
Each synthetic Kalamang sentence is translated by a teacher WITH the applicable
grammar/vocab (pseudo-gold answer + SFT target) and scored by a partially-competent
student (30 gold examples). The two engines differ ONLY in which N they keep from the
SAME scored pool: **sieve_gen** = a representative sample (diverse coverage); **feynman_gen**
= the lowest-student-chrF tail (the frontier). Identical LoRA-SFT + closed-book chrF eval.

## De-risk found two design walls (before any large run)

1. **Synthetic-only hard-label SFT trains BELOW floor** (sieve 14.6, feynman 11.1 vs floor
   15.9). Teacher translations of novel synthetic sentences are noisy labels; SFT overfits
   the noise. SIEVE's real 24.48 depends on SOFT distillation (noise-robust) -- the
   matched-SFT concession cannot reach that regime.
2. **A shared gold base (375 real pairs) is required** and lifts both arms above floor. It
   also means SIEVE's 16K-synthetic endpoint is the WRONG regime here: 16K noisy synthetic
   vs 375 gold (~44:1) would re-drown the anchor. We ran the working regime: n=800 synth +
   gold base.

## Result (n=800 synth + gold base, 3 seeds, closed-book ke chrF)

| arm | s0 | s1 | s2 | mean | stdev |
|-----|----|----|----|------|-------|
| sieve_gen  | 18.61 | 19.95 | 19.75 | 19.43 | 0.59 |
| feynman_gen| 20.20 | 18.91 | 16.68 | 18.60 | 1.45 |
| gold_only  | 19.53 | 18.95 | 18.81 | 19.10 | 0.31 |

Paired seed diffs: feyn-sieve -0.84 (+1.59/-1.04/-3.07), feyn-gold -0.50, sieve-gold +0.34.

**Honest negative result.** With 3 seeds and the gold_only control:
1. **Neither synthetic arm beats the gold-only baseline** -- the synthetic data adds ~nothing
   over the anchor under hard-label SFT (sieve +0.34, feynman -0.50).
2. **Feynman is worse than SIEVE and 2.5x noisier** (stdev 1.45 vs 0.59). Selecting the
   hardest sentences selects the noisiest teacher labels -> variance, not signal.
3. The de-risk's 1-seed +1.3 Feynman "win" was noise -- did not survive 3 seeds. **Third
   time this project that a 1-2 seed lean vanished under more seeds.**

## What this establishes (and its scope)

- In matched-SFT, the synthetic-translation data engine (SIEVE-GEN's or Feynman's) does not
  beat a simple gold anchor, and failure-targeting adds noise. The gold base does the work.
- This does NOT test SIEVE's actual regime (soft distillation), where SIEVE-GEN reaches
  24.48 -- that needs the 8-GPU infra we could not run. Whether Feynman's targeting helps
  under soft distillation is untested.
- Consistent with the whole project: Feynman's data-engine advantage is small-at-best and
  does not survive proper controls (seeds + a no-op baseline).

---

# Soft-distillation head-to-head (SIEVE's real regime, feasible slice)

Hard-label SFT can't reach SIEVE's regime (it's noise-fragile; trains below floor on
synthetic-only). SIEVE's 24.48 uses SOFT distillation. We built a feasible version:
learner/soft_distill.py -- LoRA KL context-distillation on ONE GPU (teacher = frozen
base WITH the applicable grammar pieces; student = base+LoRA WITHOUT context; match
distributions over the answer tokens; adapter toggled between passes). Exact (same
model+tokenizer), no logit storage, no 8-GPU ZeRO-3.

## Result (synthetic-only, n=800, 1 seed, 3 epochs, closed-book ke chrF)

| method | sieve_gen | feynman_gen |
|--------|-----------|-------------|
| hard-label SFT | 14.6 | 11.1 |
| soft distillation | 15.83 | 12.36 |

floor 15.9.

1. **Soft distillation beats hard-label SFT** on the same data (sieve +1.2, feyn +1.3) --
   the predicted noise-robustness is real.
2. **But it only reaches the floor** (sieve 15.83 ~= 15.9); it does NOT reproduce SIEVE's
   24.48 at this slice. Two reasons separate them: SCALE (800 vs SIEVE's 16K) and TEACHER
   STRENGTH (per-example grammar *pieces* here vs the full book). So this is "synthetic-only
   barely clears floor at feasible scale," not "soft distillation fails."
3. **Feynman still loses to SIEVE** under soft distillation too (12.36 vs 15.83) -- the
   failure-targeting keeps selecting the least-learnable teacher signal, in EVERY regime tried.

# 16K + section teacher: chasing SIEVE, and the cheatsheet reversal

We scaled to SIEVE's budget and a strong teacher, and added the Feynman *cheatsheet* arm
(the loop's failure-targeted study notes) next to the two selection arms. Four data engines,
ONE closed-book chrF axis. Only the data engine differs.

- **Section teacher.** The full book is 40K tok > the 16K endpoint, so the teacher conditions
  on one of 4 FIXED ~10K-tok book sections (prefix-cached in vLLM). Far stronger than SIEVE's
  6-8 pieces; a feasible stand-in for "full-book teacher."
- **KV-reuse soft distillation.** A 10K-tok teacher forward per example would be ~20h/arm at
  16K. Instead each section's KV is cached once (GQA -> ~6 GB for all 4) and only the short
  `kal`+answer suffix is forwarded. Verified: cached-suffix logits == a full forward (top-1
  token identical, KL ~1e-2 bf16 noise). This made 16K x 3-epoch soft distillation run in
  ~15 min/arm on one GPU.
- Pool: 24,576 candidates in 94 min. Selection contrast is real -- median student-chrF 15.2
  (sieve random sample) vs 12.5 (feynman hardest tail).

## Result (16K, 1 seed, 3 epochs, closed-book ke chrF; floor 15.9, SIEVE 24.48)

| arm | training | data | examples | chrF |
|-----|----------|------|----------|------|
| **feynman_cheatsheet** | hard SFT | gold + failure-notes | 787 | **20.93** |
| sieve_gen | hard SFT | gold + synthetic | 16,759 | 19.09 |
| feynman_gen | hard SFT | gold + synthetic | 16,759 | 17.52 |
| gold_only | hard SFT | gold pairs only | 375 | 16.32 |
| sieve_gen | soft distill | synthetic only | 16,384 | 18.08 |
| feynman_gen | soft distill | synthetic only | 16,384 | 16.21 |

cheatsheet other readings: base+cheatsheet (weight-free context) 19.96, SFT+cheatsheet 20.57
-- both BELOW the closed-book SFT 20.93, i.e. the notes bake into weights better than they
serve as amortised context here.

1. **The NOTES arm is the first real positive.** The cheatsheet beats the gold anchor by +4.6
   (16.32 -> 20.93), SIEVE-GEN by +1.85, and Feynman-selection by +3.4 -- with **~21x fewer
   examples** (787 vs 16,759). Gold alone is only 16.32, so this is the NOTES, not the anchor.
   **But the follow-up extraction control (next section) shows the driver is the notes FORMAT,
   not the Feynman failure-loop** -- plain uniform-notes extraction matches or beats it.
2. **Feynman SELECTION still loses.** feynman_gen < sieve_gen in BOTH regimes (hard 17.52 <
   19.09; soft 16.21 < 18.08). The hardest-for-student tail is worse training data than
   diverse coverage -- consistent with every prior experiment.
3. **We still don't reach SIEVE's 24.48.** Best sieve_gen is 19.09 (hard) / 18.08 (soft),
   ~5 short. The remaining gap is our LoRA vs SIEVE's full-FT + top-100-logit distillation on
   8 GPUs -- a training-method gap, not a data-engine gap.
4. **Caveat: 1 seed.** This project's recurring lesson is that 1-2 seed leans vanish under
   replication. The cheatsheet gap (+1.85 to +4.6) is larger than any prior lean, but it is
   NOT yet seed-confirmed. Treat as a strong signal to replicate, not a settled result.

# Notes >> synthetic pairs — and the driver is the notes format, not Feynman

The "cheatsheet wins" headline was missing a control: the 16K run never ran the EXTRACTION
arm (uniform grammar notes + vocab, no failure loop). Adding it, plus an inference-scaling
eval and a budget frontier, settles what actually drives the win.

## Extraction control (60K-token notes budget, closed-book ke chrF)

| data engine (hard SFT, gold anchor + engine data) | seed 0 | seed 1 | examples |
|---|---|---|---|
| extraction (uniform grammar notes + vocab) | 21.88 | 20.38 | ~1,847 |
| feynman cheatsheet (failure-targeted notes) | 20.93 | (deferred) | 787 |
| SIEVE-GEN (synthetic translation pairs) | 19.09 | | 16,759 |
| gold anchor | 16.32 | | 375 |

1. **The winner is the NOTES format, not Feynman.** Plain extraction (21.9 / 20.4) matches or
   beats the failure-targeted cheatsheet (20.9), with a simpler, cheaper engine (no student
   probe, no diagnosis). Consistent with the earlier 4-seed grid where extraction ~= feynman.
2. **Notes >> synthetic pairs** (~21 vs 19.1), robust across both seeds.
3. So "pursue the cheatsheet" is really "pursue notes-as-data." The Feynman failure-loop is
   not the active ingredient.

## Studying vs cramming (inference-budget scaling, seed 0, closed-book chrF)

Self-consistency budget K, MBR-selected (reference-free chrF centroid):

| K | extraction | cheatsheet | SIEVE pairs | gold |
|---|---|---|---|---|
| 1  | 20.85 | 20.46 | 18.84 | 17.80 |
| 4  | 22.06 | 22.05 | 19.75 | 19.55 |
| 16 | 23.40 | 23.09 | 20.27 | 19.70 |
| **slope K1->K16** | **+2.55** | **+2.62** | +1.43 | +1.90 |

Notes-trained models **STUDY** -- chrF rises steeply with inference compute (+2.6, to ~23,
nearing the ICL-gold ceiling ~27). Synthetic-pairs **CRAM** -- the flattest slope (+1.43),
worse even than the gold anchor (+1.90). Pairs lose on BOTH level and inference-scalability:
they teach surface patterns that don't compound with reasoning. (The scalar `expertise`
metric is exact-match-based and degenerate on continuous chrF; the curve is the signal.)

## Budget frontier (reported with its confound)

SIEVE-pairs subsampled to matched token budgets, FIXED 3 epochs:

| budget | 15K | 60K | 150K | 400K | 1.13M (full) |
|--------|-----|-----|------|------|------|
| chrF   | 16.32 | 15.35 | 14.47 | 16.58 | 19.09 |
| ~steps | 28 | 58 | 119 | 290 | 785 |

Non-monotonic because FIXED epochs makes step-count scale with budget -- the curve conflates
data budget with training compute (few steps -> stays near the gold init; a little noisy-pair
training HURTS; only full budget recovers). So this is NOT a clean data-budget frontier, and
a clean one must fix training STEPS -- which then trades in an overfitting confound on the
small-data points (intrinsic to the budget axis). What survives cleanly:
- **matched 60K budget: notes 21.9 / 20.9 >> pairs 15.3** (+6 chrF).
- **notes @ 60K (21.9) > pairs @ full 1.13M (19.09)** -- 20x less data AND less compute.
Notes dominate pairs at every point measured.

## Why notes win: the pairs hallucinate the language

Digging into WHY sieve underperforms extraction. It is NOT crude noise -- only 6.6% of sieve
pairs are surface-degenerate, lengths match gold, and the pairs DO help (gold-only 16.32 ->
+16K pairs 19.09, i.e. +2.8). The real mechanism is specific to low-resource GENERATION:

- **47% of the synthetic Kalamang sentences contain >=1 non-Kalamang word**; 16.8% of all
  Kalamang tokens are unattested in the book/wordlist/gold -- vs **0%** for real gold sentences.
- The contamination is diagnostic: Indonesian/Malay leakage (`makan` eat, `kopi` coffee,
  `taman` garden, `balik` return), proper names (`Kawe`, `Rina`), and invented morphology.
- Cause: Kalamang is genuinely low-resource, so the 8B generator does NOT know it. Asked to
  WRITE a Kalamang sentence it drifts to the nearest high-resource neighbour (Indonesian) and
  invents words; the teacher then translates the half-hallucinated sentence into a fluent but
  spurious pair. Surface-degeneracy filters miss it -- the output is well-formed, just
  wrong-language.
- **Identical for sieve and feynman** (16.8% vs 16.9% unattested tokens, ~47% of sentences
  both) -- it is a property of the shared generated POOL, not the selection, and it does not
  track difficulty (unattested-rate by student band is flat/U-shaped: 20/16/15/22%). So it is
  NOT what separates the two selection arms.
- **Intrinsic to SIEVE's design, not our reimplementation.** The hallucination is in the
  sentence-GENERATION step ("write a Kalamang sentence"), which is SIEVE-GEN's recipe followed
  faithfully. Stronger training (their full-FT + top-100-logit soft distill) may tolerate the
  noisy LABELS better, but does not un-hallucinate the INPUTS.
- **Extraction escapes it by grounding.** Reading a real book chunk and explaining it needs
  only reading comprehension of provided text; it never generates target-language content from
  absent knowledge, so it cannot hallucinate vocabulary.

**Principle:** for low-resource knowledge injection, GENERATING synthetic examples is bottle-
necked by the model's knowledge of the target -- exactly what is missing -- so ~half the
sentences are contaminated; EXTRACTING/explaining the source is not. **Testable prediction:**
on a HIGH-resource target the generator would not hallucinate and pairs should close much of
the gap -- a clean way to bound where each engine wins.

Budget note: the headline (extraction 21.9 @ 60K tok vs sieve 19.1 @ 1.13M tok) is NOT
token-matched -- extraction wins with ~19x fewer tokens. At matched 60K, extraction 21.9 vs
sieve 15.3 (sieve undertrained there, steps-confounded); extraction @ 60K beats sieve at EVERY
budget measured, including sieve's full 1.13M. A second, compounding factor is gold dilution:
the reliable gold anchor is 2.2% of the sieve set vs 20.3% of the extraction set.

Caveats: our sieve is a weakened SIEVE (19.1 vs their 24.48, LoRA vs full-FT); "attested"
includes the book's English text, so 17% / 47% are LOWER bounds on non-Kalamang content.
Untested causal capstones (cheap, not run): filter sieve to attested-vocab-only pairs and
retrain; upsample the gold anchor from 2.2% -> 20% in the sieve set.

# RuleArena @ 8B — still capability-bound (revisit of the 3B null)

phase0 headroom gate at Qwen3-8B, L0 airline fees, n=100, thinking ON, max 4000 tok:

| condition | exact | within10 | within20 | median rel err |
|-----------|-------|----------|----------|----------------|
| closed-book | 0.00 | 0.08 | 0.13 | 0.90 |
| ICL-gold (rules in context) | 0.03 | 0.17 | 0.27 | 0.69 |

Going 3B -> 8B does NOT open a usable exact-match ceiling: even given all the rules and full
reasoning, 8B computes the exact fee only 3% of the time. There is a real but weak GRADED
headroom the 3B null lacked (within-20% doubles 0.13 -> 0.27; median error 0.90 -> 0.69), but
a 27% within-20% ceiling is too low/noisy to be a clean data-engine testbed. MTOB stays the
substrate with signal.

## Project-wide verdict — it's the notes format, not Feynman

Both Feynman operationalisations are now settled, and neither is the story:
- **Feynman-as-selection** (keep the hardest-for-student tail): does NOT help. <= SIEVE /
  <= baseline across RuleArena@3B, MTOB notes-grid (4 seeds), matched-SFT, soft distillation,
  and the 16K hard+soft runs.
- **Feynman-as-cheatsheet** (failure-targeted notes): beats synthetic pairs, but NOT because
  of the failure-targeting -- plain uniform-notes extraction matches or beats it (n=2). The
  Feynman-specific machinery adds nothing over ordinary note extraction.

The real, robust, engine-agnostic finding: **for knowledge injection, distilling the corpus
into explanatory notes (a compact "textbook": grammar explanations + vocab) is a strictly
better SFT target than synthetic input-output pairs** -- higher chrF at matched or 20x-smaller
budget, AND more inference-scalable (notes "study", pairs "cram"). The direction to pursue is
the notes/textbook data engine, not the Feynman loop.

Still open: n=2+ on the main 16K arms (seed-1 training deferred on GPU contention; extraction
already replicated at n=2); a clean fixed-STEP frontier if a figure is wanted; SIEVE's exact
full-FT + top-100-logit regime (what separates our 19.1 from their 24.48); and a second
knowledge-bound substrate to show notes-as-data generalises beyond MTOB.

Still untested: SIEVE's exact full-FT + top-100-logit regime (needs ~8 GPUs), which is what
separates our 19.09 from their 24.48.
