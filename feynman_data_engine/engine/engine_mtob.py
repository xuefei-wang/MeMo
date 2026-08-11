"""MTOB data engine: turn the Kalamang grammar book + wordlist + 375 gold parallel
sentences into SFT translation examples for a Qwen3-8B learner.

Same two modes as the RuleArena engine, differing only in the loop:

  extraction_only : emit gold pairs + uniform vocab + one-pass grammar notes.
  feynman_core    : a source-blind student translates a sampled gold sentence with
                    only the running cheat-sheet; chrF grades it (no LLM judge);
                    PASSES are skipped; for a FAILURE the generator diagnoses the
                    missed words/grammar (M1 teach + M2 critic) and emits a targeted
                    note + the gold pair. Budget concentrates on the student's
                    translation-failure frontier.

Both emit the SAME primary unit (Kalamang->English pair) so the learner sees one
format; the DATA DISTRIBUTION + the attached notes differ. Every LLM call logs
through the ledger; emitted tokens counted with the learner tokenizer.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.llm import LLM  # noqa: E402
from data import mtob  # noqa: E402
from eval.chrf import sentence_chrf  # noqa: E402
from eval.translate import build_prompt, _clean  # noqa: E402

PASS_CHRF = 40.0  # a source-blind translation this good needs no more teaching
TRANSLATE_SYS = ("You are an expert Kalamang translator. Translate the given "
                 "Kalamang sentence into English. Output ONLY the English "
                 "translation on one line.")


def _strip_think(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


def _pair_example(kal: str, eng: str) -> dict:
    return {"messages": [
        {"role": "system", "content": TRANSLATE_SYS},
        {"role": "user", "content": f"Kalamang: {kal}\nEnglish:"},
        {"role": "assistant", "content": eng},
    ]}


def _note_example(title: str, note: str) -> dict:
    return {"messages": [
        {"role": "system", "content": "You are learning Kalamang grammar and vocabulary."},
        {"role": "user", "content": f"Explain this Kalamang point: {title}"},
        {"role": "assistant", "content": note},
    ]}


def _vocab_example(word: str, gloss: str, pos: str) -> dict:
    return {"messages": [
        {"role": "system", "content": "Kalamang-English vocabulary."},
        {"role": "user", "content": f"What does the Kalamang word '{word}' mean?"},
        {"role": "assistant", "content": f"'{word}' ({pos}) means: {gloss}."},
    ]}


def _count(tok, ex: dict) -> int:
    return len(tok.encode("\n".join(m["content"] for m in ex["messages"])))


def _wordlist_examples(wordlist: dict, rng, k: int) -> list[dict]:
    items = list(wordlist.items())
    rng.shuffle(items)
    out = []
    for word, val in items[:k]:
        pos, gloss = (val[0], val[1]) if len(val) >= 2 else ("", val[0] if val else "")
        out.append(_vocab_example(word, gloss, pos))
    return out


def _student_translate(student: LLM, kal: str, cheatsheet: str) -> str:
    user = (f"Reference notes:\n{cheatsheet}\n\n" if cheatsheet else "") + \
           f"Kalamang: {kal}\nEnglish:"
    # thinking REQUIRED for translation (no-think Qwen3 echoes the source)
    out = student.chat([{"role": "system", "content": TRANSLATE_SYS},
                        {"role": "user", "content": user}],
                       purpose="probe", temperature=0.0, max_tokens=2048, think=True)
    return _clean(_strip_think(out))


def _diagnose_note(gen: LLM, kal: str, eng: str, student_hyp: str,
                   book_ctx: str) -> str:
    """M1 teach + M2 critic: name the words/grammar the student missed, plainly."""
    msgs = [
        {"role": "system", "content":
            "You teach Kalamang. Given a sentence a student mistranslated, state in "
            "1-3 plain sentences the specific vocabulary and grammar needed to "
            "translate it correctly (actual Kalamang words -> English meanings, and "
            "the construction). Be concrete; no vague advice."},
        {"role": "user", "content":
            f"=== GRAMMAR/VOCAB REFERENCE ===\n{book_ctx}\n=== END ===\n\n"
            f"Kalamang: {kal}\nCorrect English: {eng}\n"
            f"Student's wrong translation: {student_hyp}\n\n"
            f"What vocabulary and grammar did the student miss? State it plainly."},
    ]
    return _strip_think(gen.chat(msgs, purpose="diagnose", temperature=0.5,
                                 max_tokens=2048))


# --------------------------------------------------------------------------- #
def run_extraction(gen, tok, ledger, budget_tokens, seed, book_ctx, wordlist,
                   gold_pairs, concepts) -> list[dict]:
    rng = random.Random(seed)
    out = []
    # seed with all gold translation pairs (the strongest direct signal)
    for kal, eng in gold_pairs:
        if ledger.emitted_training_tokens >= budget_tokens:
            return out
        ex = _pair_example(kal, eng)
        out.append(ex); ledger.record_emitted(_count(tok, ex), 1)
    # uniform vocab + one-pass grammar notes until budget
    ci = 0
    vocab = _wordlist_examples(wordlist, rng, k=10 ** 6)
    vi = 0
    while ledger.emitted_training_tokens < budget_tokens:
        if ci < len(concepts):
            c = concepts[ci]; ci += 1
            note = _strip_think(gen.chat(
                [{"role": "system", "content":
                  "You teach Kalamang. Explain the grammar point plainly with "
                  "concrete Kalamang->English examples, for a translator."},
                 {"role": "user", "content":
                  f"=== REFERENCE ===\n{book_ctx}\n=== END ===\n\n"
                  f"Explain this section for translation:\n{c['text'][:1500]}"}],
                purpose="teach", temperature=0.7, max_tokens=2048))
            ex = _note_example(c["title"], note)
        elif vi < len(vocab):
            ex = vocab[vi]; vi += 1
        else:
            break
        out.append(ex); ledger.record_emitted(_count(tok, ex), 1)
    return out


def run_feynman(gen, student, tok, ledger, budget_tokens, seed, book_ctx,
                wordlist, gold_pairs, concepts, max_pass_streak: int = 30) -> list[dict]:
    rng = random.Random(seed)
    out = []
    cheatsheet = ""
    pass_streak = 0
    pool = list(gold_pairs)
    while ledger.emitted_training_tokens < budget_tokens:
        kal, eng = pool[rng.randrange(len(pool))]
        # M3 examine: source-blind student translates with only the cheat-sheet
        hyp = _student_translate(student, kal, cheatsheet)
        score = sentence_chrf(hyp, eng)
        if score >= PASS_CHRF:
            pass_streak += 1
            if pass_streak < max_pass_streak:
                continue  # already translatable -> don't spend emit budget
        pass_streak = 0
        # failure -> diagnose missed vocab/grammar (M1/M2), grow cheatsheet, emit
        note = _diagnose_note(gen, kal, eng, hyp, book_ctx)
        if note and note not in cheatsheet:
            cheatsheet = (cheatsheet + "\n- " + note.strip())[-4000:]
        note_ex = _note_example(f"translate: {kal[:40]}", note)
        pair_ex = _pair_example(kal, eng)
        for ex in (note_ex, pair_ex):
            out.append(ex); ledger.record_emitted(_count(tok, ex), 1)
    return out
