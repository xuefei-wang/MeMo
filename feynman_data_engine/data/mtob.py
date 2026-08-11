"""MTOB (Machine Translation from One Book) data loader.

Corpus = grammar book + wordlist + train parallel sentences (the teaching
material a data engine explains). Test sentences are held out.

Directions:
  ke : Kalamang -> English   (input=original Kalamang, target=ground_truth English)
  ek : English  -> Kalamang   (input=original English,  target=ground_truth Kalamang)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent
RES = _HERE / "mtob_resources"
SPL = _HERE / "mtob_splits"
CANARY = "big-bench-canary"


@dataclass
class Pair:
    source: str
    target: str
    direction: str  # 'ke' or 'ek'


def _drop_canary(rows: list[dict]) -> list[dict]:
    return [r for r in rows if CANARY not in r and "original" in r]


def load_train_pairs() -> list[dict]:
    """375 parallel sentences: {'original': English, 'translation': Kalamang}."""
    rows = json.loads((SPL / "train_examples.json").read_text())
    return _drop_canary(rows)


def load_test(direction: str) -> list[Pair]:
    assert direction in ("ke", "ek")
    rows = json.loads((SPL / f"test_examples_{direction}.json").read_text())
    rows = _drop_canary(rows)
    out = []
    for r in rows:
        # 'original' is the input side; 'ground_truth' the reference output.
        out.append(Pair(source=r["original"], target=r["ground_truth"],
                        direction=direction))
    return out


def train_as_pairs(direction: str) -> list[Pair]:
    """Turn the train parallel sentences into directional Pairs (gold refs for probing)."""
    rows = load_train_pairs()
    out = []
    for r in rows:
        eng, kal = r["original"], r["translation"]
        if direction == "ke":
            out.append(Pair(source=kal, target=eng, direction="ke"))
        else:
            out.append(Pair(source=eng, target=kal, direction="ek"))
    return out


def load_wordlist() -> dict[str, list[str]]:
    """ke wordlist: Kalamang token -> [POS, English gloss]."""
    wl = json.loads((RES / "wordlist.json").read_text())
    return wl.get("ke", {})


def load_grammar_book(size: str = "medium") -> str:
    fname = {
        "medium": "grammar_book_for_claude_medium.txt",
        "long": "grammar_book_for_claude_long.txt",
        "full": "grammar_book.txt",
    }[size]
    return (RES / fname).read_text()


def dir_label(direction: str) -> tuple[str, str]:
    return ("Kalamang", "English") if direction == "ke" else ("English", "Kalamang")


if __name__ == "__main__":
    tr = load_train_pairs()
    print("train pairs:", len(tr), "| example:", tr[0])
    for d in ("ke", "ek"):
        t = load_test(d)
        print(f"test {d}: {len(t)} | {t[0]}")
    wl = load_wordlist()
    print("wordlist ke entries:", len(wl))
    print("grammar medium chars:", len(load_grammar_book("medium")))
