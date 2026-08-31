"""Split a generated trace into the units that get resampled.

Thought Anchors resamples at sentence boundaries.  The CoT arm keeps that.  For
the code arm the analogous unit is the statement/line: it is the smallest piece
the model emits that can independently change the executed result.

Both return (text, start_char, end_char) so a prefix can be reconstructed
exactly -- resampling continues from `trace[:start]`, so the boundaries must be
byte-exact against the original string.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class Step:
    idx: int
    text: str
    start: int
    end: int
    kind: str          # "sentence" | "line"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Sentence end: ., !, ? or newline, not inside a decimal or a common abbrev.
_SENT_END = re.compile(r"""
    (?<![A-Z][a-z]\.)          # not 'Dr.'-like
    (?<!\d)                    # not a decimal point
    (?: (?<=[.!?])\s+ | \n+ )
""", re.VERBOSE)


def split_sentences(trace: str) -> list[Step]:
    steps, pos, idx = [], 0, 0
    for m in _SENT_END.finditer(trace):
        end = m.start() + 1 if trace[m.start(): m.start() + 1] in ".!?" else m.start()
        chunk = trace[pos:m.end()]
        if chunk.strip():
            steps.append(Step(idx, trace[pos:m.end()].strip(), pos, m.end(), "sentence"))
            idx += 1
        pos = m.end()
    if trace[pos:].strip():
        steps.append(Step(idx, trace[pos:].strip(), pos, len(trace), "sentence"))
    return steps


def split_code_lines(trace: str) -> list[Step]:
    """One step per non-blank line. Blank lines and pure-comment lines are
    attached to the following line rather than becoming steps of their own, so
    that every step is something that can actually change the result."""
    steps: list[Step] = []
    pos = idx = 0
    pending_start = None
    for line in trace.splitlines(keepends=True):
        start, end = pos, pos + len(line)
        pos = end
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            if pending_start is None:
                pending_start = start
            continue
        s = pending_start if pending_start is not None else start
        steps.append(Step(idx, trace[s:end].strip(), s, end, "line"))
        idx += 1
        pending_start = None
    if pending_start is not None and trace[pending_start:].strip():
        steps.append(Step(idx, trace[pending_start:].strip(), pending_start,
                          len(trace), "line"))
    return steps


def split(trace: str, condition: str) -> list[Step]:
    return split_sentences(trace) if condition == "cot" else split_code_lines(trace)


def prefix_for(trace: str, steps: list[Step], i: int) -> str:
    """Everything strictly before step i -- the context resampling continues from."""
    return trace[: steps[i].start]
