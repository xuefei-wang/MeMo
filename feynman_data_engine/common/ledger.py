"""Token-accounting harness.

Two budget axes are tracked, per the Phase-0 spec:

  * total_generator_compute -- every completion/prompt token spent by ANY engine
    LLM call (teach / critic / probe / grade / diagnose / emit), including loop
    calls that never become training data.
  * emitted_training_tokens -- tokens in the SFT examples actually written to disk
    (set explicitly at emit time via `record_emitted`).

Both are logged from the first call. This module has no LLM dependency so it can
be unit-tested on stubs.
"""
from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Ledger:
    """Accumulates LLM-call usage, tagged by purpose."""

    run_name: str = "unnamed"
    # per-purpose -> dict of counters
    by_purpose: dict = field(default_factory=lambda: defaultdict(lambda: {
        "calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "wall_s": 0.0,
    }))
    emitted_training_tokens: int = 0
    emitted_examples: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_call(self, purpose: str, prompt_tokens: int,
                    completion_tokens: int, wall_s: float) -> None:
        with self._lock:
            b = self.by_purpose[purpose]
            b["calls"] += 1
            b["prompt_tokens"] += int(prompt_tokens)
            b["completion_tokens"] += int(completion_tokens)
            b["wall_s"] += float(wall_s)

    def record_emitted(self, n_tokens: int, n_examples: int = 1) -> None:
        with self._lock:
            self.emitted_training_tokens += int(n_tokens)
            self.emitted_examples += int(n_examples)

    # --- the two headline axes ---------------------------------------------
    @property
    def total_generator_completion_tokens(self) -> int:
        return sum(b["completion_tokens"] for b in self.by_purpose.values())

    @property
    def total_generator_tokens(self) -> int:  # prompt + completion, all calls
        return sum(b["prompt_tokens"] + b["completion_tokens"]
                   for b in self.by_purpose.values())

    def summary(self) -> dict:
        return {
            "run_name": self.run_name,
            "axis_emitted_training_tokens": self.emitted_training_tokens,
            "axis_total_generator_completion_tokens":
                self.total_generator_completion_tokens,
            "axis_total_generator_tokens": self.total_generator_tokens,
            "emitted_examples": self.emitted_examples,
            "total_calls": sum(b["calls"] for b in self.by_purpose.values()),
            "by_purpose": {k: dict(v) for k, v in self.by_purpose.items()},
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(), indent=2))
