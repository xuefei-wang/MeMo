"""The data engine: turns the airline rule corpus into SFT training examples.

Two modes, differing ONLY in how examples are chosen and taught:

  extraction_only : uniform random problems, one-pass correct worked solution.
                    (SIEVE / MeMo-shaped control -- "assume success".)

  feynman_core    : the loop --
                    M3 recall/examine: a SOURCE-BLIND student attempts the problem
                      with only the engine's running rule cheat-sheet; passes are
                      SKIPPED (no budget spent on what's already learned).
                    diagnose + M1 teach + M2 simplicity-critic: for a FAILURE, name
                      the misapplied rule, (re)teach a grounded explanation, and
                      emit a worked solution that recaps that rule.
                    Budget therefore concentrates on the student's failure frontier.

Both emit the SAME example type (problem -> worked solution -> FINAL: gold) so the
learner sees one format; only the DATA DISTRIBUTION differs. Every LLM call logs
through the ledger; emitted-training-tokens are counted with the learner tokenizer.
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.llm import LLM  # noqa: E402
from data import rulearena as ra  # noqa: E402
from eval.rulearena_eval import extract_total, SYS as EVAL_SYS  # noqa: E402

TOL = 1.0


def split_concepts(corpus: str) -> list[dict]:
    """Split the corpus into rule concepts by markdown headers."""
    concepts, cur_title, cur_body = [], "Preamble", []
    for line in corpus.splitlines():
        if re.match(r"^#{1,4}\s", line):
            if cur_body:
                concepts.append({"title": cur_title, "text": "\n".join(cur_body)})
            cur_title, cur_body = line.lstrip("# ").strip(), [line]
        else:
            cur_body.append(line)
    if cur_body:
        concepts.append({"title": cur_title, "text": "\n".join(cur_body)})
    # keep concepts with substantive bodies
    return [c for c in concepts if len(c["text"]) > 40]


def _worked_solution(gen: LLM, prompt: str, gold: float, rules_ctx: str,
                     rule_recap: str | None = None) -> str:
    """Generator writes a correct step-by-step solution ending FINAL: <gold>."""
    extra = f"\n\nFocus especially on correctly applying: {rule_recap}" if rule_recap else ""
    msgs = [
        {"role": "system", "content":
            "You are an expert airline fare tutor writing a worked solution for a "
            "student. Using the rules provided, show clear step-by-step reasoning "
            "that arrives at the correct total. The correct total is given to you; "
            "your reasoning MUST end with a line 'FINAL: <integer>' equal to it."},
        {"role": "user", "content":
            f"=== AIRLINE FEE RULES ===\n{rules_ctx}\n=== END RULES ===\n\n"
            f"Problem:\n{prompt}\n\nThe correct total is {int(gold)}. "
            f"Write the worked solution ending with 'FINAL: {int(gold)}'.{extra}"},
    ]
    sol = gen.chat(msgs, purpose="teach", temperature=0.7, max_tokens=900)
    # guarantee the emitted label is correct regardless of generator slips
    if not re.search(rf"FINAL:\s*\$?\s*{int(gold)}\b", sol):
        sol = sol.rstrip() + f"\nFINAL: {int(gold)}"
    return sol


def _sft_example(prompt: str, solution: str) -> dict:
    return {"messages": [
        {"role": "system", "content": EVAL_SYS},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": solution},
    ]}


def _count_tokens(tok, ex: dict) -> int:
    text = "\n".join(m["content"] for m in ex["messages"])
    return len(tok.encode(text))


# --------------------------------------------------------------------------- #
def run_extraction(gen: LLM, tok, ledger, budget_tokens: int, seed: int,
                   rules_ctx: str) -> list[dict]:
    rng = random.Random(seed)
    out = []
    while ledger.emitted_training_tokens < budget_tokens:
        info = ra.sample_problem(rng)
        gold = ra.gold_for(info)
        prompt = ra.render_prompt(info, rng.choice(ra._NAMES))
        sol = _worked_solution(gen, prompt, gold, rules_ctx)
        ex = _sft_example(prompt, sol)
        out.append(ex)
        ledger.record_emitted(_count_tokens(tok, ex), 1)
    return out


def _student_solve(student: LLM, prompt: str, cheatsheet: str) -> float | None:
    msgs = [
        {"role": "system", "content": EVAL_SYS},
        {"role": "user", "content":
            (f"Rules summary:\n{cheatsheet}\n\n" if cheatsheet else "") + prompt},
    ]
    out = student.chat(msgs, purpose="probe", temperature=0.0, max_tokens=700)
    return extract_total(out)


def _diagnose_and_recap(gen: LLM, prompt: str, gold: float, student_pred, concepts,
                        rules_ctx: str) -> str:
    """M1 teach + M2 critic: name the misapplied rule and give a grounded recap."""
    titles = ", ".join(c["title"] for c in concepts)
    msgs = [
        {"role": "system", "content":
            "You diagnose which airline fee rule a student misapplied, then state "
            "that rule in one or two plain, self-contained sentences with concrete "
            "numbers (no vague references, no jargon)."},
        {"role": "user", "content":
            f"=== RULES ===\n{rules_ctx}\n=== END ===\n\nProblem:\n{prompt}\n\n"
            f"Correct total: {int(gold)}. Student answered: {student_pred}. "
            f"Rule topics: {titles}.\nWhich single rule did the student most likely "
            f"get wrong? State it plainly with numbers."},
    ]
    return gen.chat(msgs, purpose="diagnose", temperature=0.5, max_tokens=200)


def run_feynman(gen: LLM, student: LLM, tok, ledger, budget_tokens: int, seed: int,
                rules_ctx: str, concepts: list[dict],
                pass_frac: float = 0.25, max_pass_streak: int = 15) -> list[dict]:
    rng = random.Random(seed)
    out = []
    cheatsheet = ""  # the engine's running, student-facing rule summary (grows)
    pass_streak = 0
    while ledger.emitted_training_tokens < budget_tokens:
        info = ra.sample_problem(rng)
        gold = ra.gold_for(info)
        prompt = ra.render_prompt(info, rng.choice(ra._NAMES))
        # M3 examine: source-blind student attempts with only the cheat-sheet.
        # "Pass" uses a GRADED band so the weak student can discriminate easy from
        # hard problems -> the loop concentrates emit budget on the hard frontier.
        pred = _student_solve(student, prompt, cheatsheet)
        passed = pred is not None and abs(pred - gold) <= max(TOL, pass_frac * gold)
        if passed:
            pass_streak += 1
            if pass_streak < max_pass_streak:
                continue  # already learnable -> don't spend emit budget here
        pass_streak = 0
        # failure -> diagnose (M1/M2) + emit a worked solution recapping that rule
        recap = _diagnose_and_recap(gen, prompt, gold, pred, concepts, rules_ctx)
        if recap and recap not in cheatsheet:
            cheatsheet = (cheatsheet + "\n- " + recap.strip())[-4000:]  # bounded
        sol = _worked_solution(gen, prompt, gold, rules_ctx, rule_recap=recap)
        ex = _sft_example(prompt, sol)
        out.append(ex)
        ledger.record_emitted(_count_tokens(tok, ex), 1)
    return out
