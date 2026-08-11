"""Translation eval: run a model over MTOB test pairs, score chrF.

Shared by the Phase-0 anchors (closed-book, ICL-gold) and post-SFT learner eval.
Uses concurrent requests against an OpenAI-compatible endpoint.
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.ledger import Ledger  # noqa: E402
from common.llm import LLM, Endpoint  # noqa: E402
from data import mtob  # noqa: E402
from eval.chrf import corpus_chrf, sentence_chrf  # noqa: E402


def build_prompt(src: str, direction: str, context: str | None) -> list[dict]:
    src_lang, tgt_lang = mtob.dir_label(direction)
    sys_msg = (
        f"You are an expert translator for {src_lang}. "
        f"Translate the given {src_lang} sentence into {tgt_lang}. "
        f"Output ONLY the {tgt_lang} translation on a single line, with no "
        f"explanation, quotes, or extra text."
    )
    user = ""
    if context:
        user += (
            "Use the following reference material about the language to help you "
            "translate.\n\n=== REFERENCE MATERIAL ===\n" + context +
            "\n=== END REFERENCE MATERIAL ===\n\n"
        )
    user += f"{src_lang}: {src}\n{tgt_lang}:"
    return [{"role": "system", "content": sys_msg},
            {"role": "user", "content": user}]


def _clean(out: str) -> str:
    line = out.strip().splitlines()[0].strip() if out.strip() else ""
    for pref in ("English:", "Kalamang:", "Translation:"):
        if line.lower().startswith(pref.lower()):
            line = line[len(pref):].strip()
    return line.strip().strip('"').strip()


def run_eval(llm: LLM, pairs: list[mtob.Pair], direction: str,
             context: str | None = None, purpose: str = "eval",
             max_workers: int = 32, max_tokens: int = 256,
             temperature: float = 0.0) -> dict:
    def one(p: mtob.Pair) -> tuple[str, str]:
        msgs = build_prompt(p.source, direction, context)
        out = llm.chat(msgs, purpose=purpose, temperature=temperature,
                       max_tokens=max_tokens)
        return _clean(out), p.target

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(one, pairs))
    hyps = [h for h, _ in results]
    refs = [r for _, r in results]
    score = corpus_chrf(hyps, refs)
    sent = [sentence_chrf(h, r) for h, r in results]
    samples = [{"src": p.source, "hyp": h, "ref": r, "chrf": round(s, 2)}
               for p, (h, r), s in zip(pairs, results, sent)]
    return {"corpus_chrf": round(score, 3), "n": len(pairs), "samples": samples}
