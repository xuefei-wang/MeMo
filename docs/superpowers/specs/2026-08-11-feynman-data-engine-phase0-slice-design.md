# Feynman Data Engine — Phase 0 + Thin Vertical Slice

**Date:** 2026-08-11
**Status:** Design approved; autonomous implementation authorized (8h window).
**Scope of THIS spec:** Phase 0 headroom gate + a thin end-to-end slice on **one** benchmark. Later phases (RuleArena, Arm B, incumbents, cross-learner transfer, full ablation ladder) get their own specs.

---

## 1. The program this slice belongs to

**Hypothesis under test (whole program):** Feynman-style teach-back with *failure-driven repair* is a better synthetic-data recipe than one-shot extraction, at matched budget, for injecting a small corpus's knowledge into a model's parameters.

**Feynman decomposes into 5 mechanisms; we implement the "novel-signal core":**

| Mechanism | Role | In scope? |
|---|---|---|
| M1 protégé (teach-back) | source-blind explanation `E` | ✅ |
| M2 simplicity critic | jargon/circularity = intrinsic gap signal | ✅ |
| M3 recall-first delta | teach from memory, diff vs source = corpus's genuine addition | ✅ |
| M4 elaborative "why" chains | relational knowledge | ❌ (EntiGraph territory) |
| M5 analogy / multi-representation | reformulation | ❌ (WRAP territory) |
| Closed failure loop (source-blind student) | extrinsic gap signal | ✅ |

**Fixed across the program:** base model, learner, eval, corpora. **Varied:** the data engine, the budget.
**Primary artifact:** accuracy-vs-budget curves per engine, on **two** budget axes (emitted training tokens; total generator compute/$).
**Substrate (program):** MTOB (primary), RuleArena (secondary, matches SIEVE). Floor = closed-book; ceiling = ICL-gold. Incumbents = SIEVE-GEN, Cartridges, WRAP-style paraphrase (fairness control).

---

## 2. Why this slice exists

Two go/no-go questions that gate the entire program, answerable in days not weeks:

- **G1 (headroom):** On MTOB with the chosen base, is the band between closed-book and ICL-gold wide enough to measure engine differences? If closed-book ≈ ICL-gold, no engine comparison is meaningful — change the base.
- **G2 (mechanism has a pulse):** Does adding the Feynman failure loop move test chrF *at all* above a one-shot extraction engine at equal emitted-token budget? If not, the core bet is dead and we learn it cheaply.

**Non-goals for this slice:** RuleArena, Arm B (learner-in-loop), cross-learner transfer, SIEVE/Cartridges reimplementation, M4/M5, statistical rigor beyond a couple of seeds. Those are later specs.

---

## 3. What gets built

### 3.1 Substrate: MTOB
- **Corpus:** the Kalamang grammar book (medium/long-context reference), bilingual word list, and 375 parallel sentences (the "teaching material" a Feynman engine explains).
- **Eval:** held-out test sentences, **chrF** (continuous → high statistical power per cell). Kalamang↔English both directions; slice may start with **Kalamang→English** only (easier reference grading).
- **Data access risk:** MTOB Kalamang data may be download-gated. First implementation task verifies availability; if gated/unavailable within the window, fall back the slice to **RuleArena** (ungated GitHub) and record the pivot. The *design* is unchanged either way — only the loader differs.

### 3.2 Fixed learner
- **LoRA-SFT on a small instruct base.** Default **Qwen2.5-3B-Instruct** (consistent with the MeMo repo's Qwen family). Phase 0 may swap the base if headroom is inadequate.
- Consumes engine-emitted, task-shaped SFT examples (translation pairs for MTOB). **Not** MeMo's QA schema.

### 3.3 Token-accounting harness (built first — everything logs through it)
A single wrapper around every generator/examiner LLM call that records, per call: prompt tokens, completion tokens, wall-clock, model, and a `purpose` tag (`teach` / `critic` / `probe` / `grade` / `diagnose` / `emit`). Two ledgers per engine run:
- **emitted-training-tokens** = tokens in the SFT examples actually written to disk.
- **total-generator-compute** = all completion tokens spent, including loop calls that never become training data.
Both are the x-axes of the curve. Logged from the first call; non-negotiable.

### 3.4 Engines (this slice: two only)
Both read the MTOB corpus, both emit SFT examples in the same format, both log through the harness.

1. **`extraction_only`** (bottom rung / SIEVE-shaped control): one pass — decompose corpus into concepts, emit an explanation/example per concept. No student, no loop. This is "assume success."
2. **`feynman_core` (Arm A)**: the loop of §1 —
   - M3 recall-first: teach `c` from parametric memory, diff vs source span.
   - M1 teach: write `E`.
   - M2 simplicity critic: flag jargon/circularity, force reduction.
   - probe + **source-blind proxy student** examine (frozen small LLM, `E` in context) on a task-shaped probe.
   - grade mechanically (chrF vs the parallel-sentence reference).
   - diagnose + re-teach; loop to a round cap (e.g. 3).
   - emit converged `E` + repair deltas as SFT examples.

The two share ~everything except the loop, so `extraction_only` is literally `feynman_core` with the examiner forced to "pass" — implement as one engine with a flag where practical.

### 3.5 The run grid (this slice)
`{extraction_only, feynman_core} × {≥3 emitted-token budget points} × {≥2 seeds}` → LoRA-SFT → test chrF.
Plus the two Phase-0 anchors: **closed-book** and **ICL-gold**.

---

## 4. Success criteria for the slice

- **G1 answered:** closed-book and ICL-gold chrF measured; band reported. Go if band is wide (target ≳15 chrF); else swap base and re-measure.
- **G2 answered:** at ≥1 matched emitted-token budget, `feynman_core` test chrF vs `extraction_only` reported with seed variance. A positive, above-noise gap = green light. A null or negative = documented negative result + hypotheses.
- **Both budget ledgers populated** for every engine run (proves the accounting works before the program scales).
- Everything reproducible from one shell script per stage; jobs run in background with progress monitorable.

---

## 5. Build order (de-risked, thin vertical first)

1. **Environment + data availability** — GPU/env discovery; verify MTOB data (else pivot to RuleArena, record it).
2. **Token-accounting harness** — the LLM-call wrapper + two ledgers. Unit-tested on a stub.
3. **MTOB loader + eval** — corpus/word-list/parallel-sentence loading; chrF scorer; closed-book + ICL-gold anchors (**answers G1**).
4. **`extraction_only` engine** — one-pass emit; end-to-end to SFT examples.
5. **`feynman_core` engine (Arm A)** — the loop; emit.
6. **LoRA-SFT learner + eval glue** — train on emitted data, score test chrF.
7. **Run grid + curves** — launch background jobs, collect, plot both-axis curves (**answers G2**).

Each step is independently runnable and logs through the harness. Steps 4–5 and 3 (anchors) are parallelizable once 1–2 land.

---

## 6. Open assumptions (resolve by proceeding; revisit if they bite)

- Base = Qwen2.5-3B-Instruct until Phase 0 says otherwise.
- Generator/examiner served locally via vLLM (repo already ships a Qwen serve script); a stronger generator than the learner is allowed (data-engine convention).
- Kalamang→English direction first; add reverse if time permits.
- 3 budget points × 2 seeds is enough to *see a signal*, not to publish — statistical rigor is a later spec.
- If MTOB is gated, RuleArena carries the slice; the design holds.
