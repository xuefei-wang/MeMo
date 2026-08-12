"""Head-to-head: SIEVE-GEN vs Feynman-GEN as SELECTION policies over one shared pool
of synthetic MTOB translation queries (matched-SFT).

A single candidate factory (SIEVE-GEN's exact 3-step recipe) builds a scored pool:
  base model selects grammar/vocab pieces -> instruction model writes a Kalamang
  sentence (anchored on real gold sentences so it is grammatical) -> teacher
  translates it WITH the applicable pieces (pseudo-gold answer + SFT target) -> a
  partially-competent student (30 gold examples, no grammar book) translates it; the
  student-vs-teacher chrF scores its difficulty.

The two engines then differ ONLY in which N they keep from the SAME pool:
  sieve_gen   : a representative random sample  (diverse coverage -- SIEVE's principle).
  feynman_gen : the lowest-student-chrF tail    (the frontier -- Feynman's principle).

Both emit identical (closed-book translate prompt -> teacher answer) SFT examples, so
feeding them through the same LoRA-SFT + closed-book chrF eval makes the chrF gap the
pure selection-policy effect. Matched-SFT (hard labels), NOT SIEVE soft distillation.
"""
from __future__ import annotations

import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.llm import LLM  # noqa: E402
from eval.chrf import sentence_chrf  # noqa: E402
from eval.translate import build_prompt, _clean  # noqa: E402

# SIEVE-GEN's exact base-model selection prompt (mtob/synthetic_data_gen.py).
SELECT_PROMPT = (
    "Task: Select 6-8 pieces of information from the corpus of knowledge that could "
    "be used to construct a single question about translating from Kalamang to "
    "English. Your job is to only select pieces of knowledge, not construct a "
    "question yourself.\n\nInformation:\n{feedback}\n\nSelected information:\n-"
)
SENTENCE_SYS = ("You are a fluent Kalamang speaker helping build a translation "
                "dataset. You write natural, grammatical Kalamang sentences.")
SENTENCE_USER = (
    "Here are real Kalamang sentences with English translations, to show you what "
    "grammatical Kalamang looks like:\n{exemplars}\n\n"
    "Now, using some of these Kalamang grammar and vocabulary facts:\n{pieces}\n\n"
    "Write ONE new natural, grammatical Kalamang sentence in the same style as the "
    "examples above. It must read like a real Kalamang sentence, not a list of words. "
    "Output ONLY the Kalamang sentence on a single line, nothing else.")

ECHO_CHRF = 65.0  # english this close to the kalamang = teacher echoed, not translated
STUDENT_SHOTS = 30  # gold examples that give the student partial competence


def chunk_feedback(text: str, chunk_chars: int = 8000) -> list[str]:
    """Chunk the grammar book on line boundaries into ~chunk_chars feedback blocks."""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    chunks, buf = [], ""
    for ln in lines:
        if len(buf) + len(ln) > chunk_chars and buf:
            chunks.append(buf); buf = ln
        else:
            buf = (buf + "\n" + ln) if buf else ln
    if buf:
        chunks.append(buf)
    return chunks


def book_sections(book: str, n_sections: int = 4) -> list[str]:
    """Split the book into n_sections contiguous, roughly-equal sections on line
    boundaries. FIXED sections (few, reused across all candidates) so vLLM prefix-
    caches each section's ~12K-token prefill -- a per-candidate window would defeat
    the cache and make a section-level teacher 10-20x slower. Each section is the
    'full-book teacher' condition at a granularity that fits the 16K endpoint."""
    lines = [ln for ln in book.split("\n") if ln.strip()]
    target = sum(len(ln) for ln in lines) / n_sections  # balance by chars, not #lines
    secs, buf, buf_chars = [], [], 0
    for ln in lines:
        buf.append(ln); buf_chars += len(ln)
        if buf_chars >= target and len(secs) < n_sections - 1:
            secs.append("\n".join(buf)); buf, buf_chars = [], 0
    if buf:
        secs.append("\n".join(buf))
    return secs


def _exemplar_block(gold_pairs, rng, k: int = 4) -> str:
    picks = rng.sample(gold_pairs, min(k, len(gold_pairs)))
    return "\n".join(f"- Kalamang: {k_}\n  English: {e}" for k_, e in picks)


def _select_pieces(base: LLM, chunk: str, seed: int) -> str:
    out = base.complete(SELECT_PROMPT.format(feedback=chunk), purpose="select",
                        temperature=1.0, max_tokens=400, seed=seed, stop=["\n\n"])
    return ("-" + out).strip()


def _gen_sentence(instr: LLM, pieces: str, gold_pairs, rng) -> str:
    user = SENTENCE_USER.format(exemplars=_exemplar_block(gold_pairs, rng),
                                pieces=pieces)
    out = instr.chat([{"role": "system", "content": SENTENCE_SYS},
                      {"role": "user", "content": user}],
                     purpose="gen_sentence", temperature=0.7, max_tokens=64, think=False)
    line = next((l.strip() for l in out.splitlines() if l.strip()), "")
    for pre in ("Kalamang:", "Sentence:", "-"):
        if line.startswith(pre):
            line = line[len(pre):].strip()
    return line.strip().strip('"')


def _teacher_translate(teacher: LLM, kal: str, pieces: str) -> str:
    out = teacher.chat(build_prompt(kal, "ke", pieces), purpose="teacher",
                       temperature=0.0, max_tokens=128, think=False)
    return _clean(out)


def _student_translate(student: LLM, kal: str, student_ctx: str) -> str:
    out = student.chat(build_prompt(kal, "ke", student_ctx), purpose="probe",
                       temperature=0.0, max_tokens=128, think=False)
    return _clean(out)


def _degenerate(kal: str) -> bool:
    toks = kal.split()
    if len(toks) < 3 or len(toks) > 40:
        return True
    if len(set(toks)) / len(toks) < 0.6:
        return True
    return max((toks.count(t) for t in set(toks)), default=0) > 3


def _usable(kal: str, eng: str) -> bool:
    if not kal or not eng or len(eng) < 3:
        return False
    if _degenerate(kal):
        return False
    return sentence_chrf(eng, kal) < ECHO_CHRF


def _sft_example(kal: str, eng: str) -> dict:
    """Training example in the EXACT closed-book format eval_mtob.py evaluates."""
    return {"messages": build_prompt(kal, "ke", None)
            + [{"role": "assistant", "content": eng}]}


def _one_candidate(base, instr, teacher, student, windows, sections, gold_pairs,
                   student_ctx, seed: int, teacher_mode: str) -> dict | None:
    """Full factory pass for one candidate; returns a scored dict or None if unusable.

    Pieces are selected from a fine 8K-char window that lies inside ONE fixed book
    section; the teacher conditions on that whole section ("full-book teacher") when
    teacher_mode=="section", else just on the 6-8 pieces (SIEVE-faithful, per-example)."""
    rng = random.Random(seed)
    sec_idx, chunk = windows[rng.randrange(len(windows))]
    try:
        pieces = _select_pieces(base, chunk, seed=rng.randrange(1 << 30))
        kal = _gen_sentence(instr, pieces, gold_pairs, rng)
        ctx = sections[sec_idx] if teacher_mode == "section" else pieces
        eng = _teacher_translate(teacher, kal, ctx) if kal else ""
        if not _usable(kal, eng):
            return None
        hyp = _student_translate(student, kal, student_ctx)
        # keep `pieces` AND `sec_idx` (the teacher's applicable context) for soft
        # distillation: the teacher forward conditions on it; the student does not.
        return {"kal": kal, "eng": eng, "pieces": pieces, "sec_idx": sec_idx,
                "student_chrf": sentence_chrf(hyp, eng)}
    except Exception:  # noqa: BLE001 -- a flaky candidate shouldn't kill the pool
        return None


def build_pool(base, instr, teacher, student, ledger, m_target, seed, sections,
               gold_pairs, concurrency: int = 32, max_factor: int = 4,
               teacher_mode: str = "section") -> list[dict]:
    """Generate m_target usable scored candidates concurrently (shared by both arms).

    `sections` = the few FIXED book sections (prefix-cached teacher context). Fine 8K
    windows for piece-selection are derived from each section so pieces + teacher
    section are always consistent."""
    windows = [(i, c) for i, sec in enumerate(sections)
               for c in chunk_feedback(sec, 8000)]
    student_ctx = ("Here are example Kalamang-English translations:\n"
                   + _exemplar_block(gold_pairs, random.Random(seed), k=STUDENT_SHOTS))
    pool, issued = [], 0
    cap = m_target * max_factor
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        while len(pool) < m_target and issued < cap:
            batch = min(concurrency * 2, cap - issued)
            seeds = [seed * 1_000_003 + issued + i for i in range(batch)]
            issued += batch
            for r in ex.map(lambda s: _one_candidate(
                    base, instr, teacher, student, windows, sections, gold_pairs,
                    student_ctx, s, teacher_mode), seeds):
                if r is not None:
                    pool.append(r)
                    if len(pool) >= m_target:
                        break
    ledger.pool_size = len(pool)
    ledger.pool_issued = issued
    return pool


def select_sieve(pool: list[dict], n: int, seed: int) -> list[dict]:
    """Diverse coverage: a representative random sample of the pool."""
    rng = random.Random(seed ^ 0x51E7E)
    return rng.sample(pool, min(n, len(pool)))


def select_feynman(pool: list[dict], n: int) -> list[dict]:
    """The frontier: the lowest-student-chrF tail (hardest for the student)."""
    return sorted(pool, key=lambda c: c["student_chrf"])[:n]


def to_sft(selected: list[dict], tok, ledger) -> list[dict]:
    out = []
    for c in selected:
        ex = _sft_example(c["kal"], c["eng"])
        out.append(ex)
        ntok = len(tok.encode("\n".join(m["content"] for m in ex["messages"])))
        ledger.record_emitted(ntok, 1)
    return out
