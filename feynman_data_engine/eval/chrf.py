"""chrF scoring for MTOB translation (the continuous metric that gives the
efficiency curve its statistical power)."""
from __future__ import annotations

from sacrebleu.metrics import CHRF

_chrf = CHRF()  # chrF (not chrF++), default word_order=0, matches MTOB convention


def corpus_chrf(hyps: list[str], refs: list[str]) -> float:
    assert len(hyps) == len(refs), (len(hyps), len(refs))
    return _chrf.corpus_score(hyps, [refs]).score


def sentence_chrf(hyp: str, ref: str) -> float:
    return _chrf.sentence_score(hyp, [ref]).score


if __name__ == "__main__":
    # sanity: identical strings -> 100, unrelated -> low
    print("identical:", round(sentence_chrf("the cat sat", "the cat sat"), 2))
    print("unrelated:", round(sentence_chrf("xyz qrs", "the cat sat"), 2))
    print("corpus:", round(corpus_chrf(["the cat", "a dog"], ["the cat", "a dog"]), 2))
