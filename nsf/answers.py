"""Extraction, normalisation and equivalence for free-form math answers.

Two jobs:
  1. Pull the model's final answer out of a CoT completion or a PoT stdout.
  2. Decide whether two answers are the same, so we can label correctness.

Normalisation follows the usual MATH conventions (Hendrycks et al.; Minerva).
It is deliberately conservative: when in doubt we return the string unchanged
and let string equality decide, rather than risk collapsing distinct answers.
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Optional

ANSWER_RE = re.compile(r"Answer:\s*(.+?)\s*$", re.MULTILINE)


def extract_answer(text: str) -> Optional[str]:
    """Return the value after the LAST `Answer:` line, or None if absent.

    The last match, not the first: a CoT may restate the format instruction or
    talk about the answer line before committing to one.
    """
    if not text:
        return None
    matches = ANSWER_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
PHRASE_RE = re.compile(
    r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|:|=)\s*[:=]?\s*"
    r"\$?\\?\$?\s*(-?[\d,]+(?:\.\d+)?|\\frac\{[^{}]+\}\{[^{}]+\})",
    re.IGNORECASE,
)
# Must contain at least one digit: a bare "[\\d,]+" happily matches a lone
# comma, which then normalises to the empty string.
NUMBER_RE = re.compile(r"-?\$?(?:\d[\d,]*(?:\.\d+)?|\.\d+)")

# Ordered most- to least-trustworthy.  The tier that fired is reported alongside
# the value so extraction quality stays visible instead of averaging away.
EXTRACTION_TIERS = ("answer_line", "boxed", "phrase", "last_number", "none")


def extract_answer_lenient(text: str) -> tuple[Optional[str], str]:
    """Recover a CoT answer, falling back through progressively looser rules.

    Models that ignore the `Answer:` instruction are not rare -- a 0.5B Qwen
    complied on 6/60 GSM8K items, writing "Therefore, the answer is: $4500"
    instead.  Parsing strictly does not make that go away; it silently narrows
    the CoT arm to format-compliant completions, which is a biased subset
    (short, confident, disproportionately easy problems).  Falling back keeps
    coverage comparable to PoT, whose format compliance is high because the
    model is copying it into a print() call rather than obeying prose.

    Returns (value, tier).  Always check the tier distribution before trusting
    a run: a CoT arm resting mostly on `last_number` is measuring the parser.
    """
    if not text:
        return None, "none"

    strict = extract_answer(text)
    if strict is not None:
        return strict, "answer_line"

    boxed = BOXED_RE.findall(text)
    if boxed:
        return boxed[-1].strip(), "boxed"

    phrase = PHRASE_RE.findall(text)
    if phrase:
        return phrase[-1].strip(), "phrase"

    # Last resort: the final number anywhere in the closing lines.  Weak, and
    # the tier label says so.
    tail = "\n".join(text.strip().splitlines()[-3:])
    nums = NUMBER_RE.findall(tail)
    if nums:
        return nums[-1].strip(), "last_number"

    return None, "none"


# --- normalisation ----------------------------------------------------------

# Math-mode delimiters a model may wrap its answer in.  Stripped repeatedly
# because they nest: Qwen-7B answered `\(\text{2}\)` on a MATH item, which is
# simply 2 -- but leaving `\(` in place scored it wrong AND gave the probe the
# label '\\' instead of '2'.
_WRAPPERS = [
    re.compile(r"^\\boxed\{(.*)\}$", re.DOTALL),
    re.compile(r"^\\fbox\{(.*)\}$", re.DOTALL),
    re.compile(r"^\\\((.*)\\\)$", re.DOTALL),
    re.compile(r"^\\\[(.*)\\\]$", re.DOTALL),
    re.compile(r"^\$\$(.*)\$\$$", re.DOTALL),
    re.compile(r"^\$(.*)\$$", re.DOTALL),
]


def _strip_wrappers(s: str, max_depth: int = 6) -> str:
    s = s.strip()
    for _ in range(max_depth):
        before = s
        for pat in _WRAPPERS:
            m = pat.match(s)
            if m:
                s = m.group(1).strip()
        if s == before:
            return s
    return s


_LATEX_STRIP = [
    (r"\\left", ""),
    (r"\\right", ""),
    (r"\\!", ""),
    (r"\\,", ""),
    (r"\;", ""),
    (r"\\ ", " "),
    (r"\\\$", ""),
    (r"\\%", "%"),
    (r"\\dfrac", r"\\frac"),
    (r"\\tfrac", r"\\frac"),
    (r"\\cdot", "*"),
]

_TEXT_WRAPPER = re.compile(r"\\(?:text|mbox|mathrm)\{([^{}]*)\}")
_FRAC = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
_SQRT_BARE = re.compile(r"\\sqrt(?!\{)\s*([0-9a-zA-Z])")


def normalize(ans: Optional[str]) -> Optional[str]:
    """Canonicalise an answer string for equivalence checking."""
    if ans is None:
        return None
    s = ans.strip()
    if not s:
        return None

    s = _strip_wrappers(s)

    s = _TEXT_WRAPPER.sub(r"\1", s)
    for pat, rep in _LATEX_STRIP:
        s = re.sub(pat, rep, s)
    s = _SQRT_BARE.sub(r"\\sqrt{\1}", s)
    s = _FRAC.sub(r"\1/\2", s)

    # Units and trailing punctuation that never change the value.  Degree marks
    # must go before the bare \\circ strip, or the exponent caret is orphaned.
    s = re.sub(r"\^\s*\{?\s*\\(?:circ|degree)\s*\}?", "", s)
    s = re.sub(r"\\(?:degree|circ)\b", "", s)
    s = s.rstrip(".").strip()

    # Numeric surface noise: thousands separators, currency, percent, spaces.
    s = s.replace(",", "").replace("$", "").replace(" ", "")
    if s.endswith("%"):
        s = s[:-1]
    if not s:
        # Stripping consumed everything (e.g. the input was just "," or "$").
        return None

    # Canonicalise pure numbers so "3", "3.0", "007" and "3/1" agree.
    canon = _canonical_number(s)
    if canon is not None:
        return canon

    return s


def _as_number(s: str) -> Optional[Fraction]:
    try:
        return Fraction(s)  # handles "3", "-4", "7/2"
    except (ValueError, ZeroDivisionError):
        pass
    try:
        return Fraction(float(s))  # handles "3.0", "1e3"
    except (ValueError, OverflowError, ZeroDivisionError):
        return None


_INT_RE = re.compile(r"[+-]?\d+")
_DEC_RE = re.compile(r"([+-]?)(\d*)\.(\d+)")
_FRAC_RE = re.compile(r"([+-]?\d+)/([+-]?\d+)")


def _canonical_number(s: str) -> Optional[str]:
    """Canonicalise a numeric string, PRESERVING its representation class.

    Decimals stay decimal and fractions stay fractional.  This matters because
    `answer_class("first_token")` reads the leading character, and it is only
    meaningful as a stand-in for the model's own first answer token if the
    canonical form keeps the shape the model wrote.

    Routing a decimal through Fraction does not do that: Fraction("307.20") is
    exactly 1536/5, so "307.20" would canonicalise to "1536/5" and label as
    '1' rather than '3'.  Correctness is unharmed either way -- is_correct
    falls back to numeric comparison -- but the probe label would be silently
    wrong for every decimal answer.

    Consequence, accepted deliberately: equivalent values written differently
    (0.5 vs 1/2) still receive different labels.  That is noise across items,
    not bias between conditions, and it preserves the surface-form semantics
    that justify the first_token scheme in the first place.
    """
    if _INT_RE.fullmatch(s):
        return str(int(s))

    m = _DEC_RE.fullmatch(s)
    if m:
        sign, whole, frac = m.groups()
        frac = frac.rstrip("0")
        sign = "-" if sign == "-" else ""
        whole_i = int(whole) if whole else 0
        if not frac:  # 5.00 -> 5
            return f"{sign}{whole_i}" if (whole_i or sign == "") else "0"
        return f"{sign}{whole_i}.{frac}"

    m = _FRAC_RE.fullmatch(s)
    if m:
        try:
            f = Fraction(int(m.group(1)), int(m.group(2)))
        except ZeroDivisionError:
            return None
        return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"

    # Scientific notation and anything else numeric-but-unusual.
    num = _as_number(s)
    if num is not None and num.denominator == 1:
        return str(num.numerator)
    return None


def is_correct(predicted: Optional[str], gold: Optional[str]) -> bool:
    """Equivalence between a model answer and the reference answer."""
    p, g = normalize(predicted), normalize(gold)
    if p is None or g is None:
        return False
    if p == g:
        return True
    # Fall back to numeric comparison with tolerance for decimal answers.
    pn, gn = _as_number(p), _as_number(g)
    if pn is not None and gn is not None:
        return abs(float(pn) - float(gn)) < 1e-6
    return False


def answer_class(ans: Optional[str], scheme: str = "first_token") -> Optional[str]:
    """Map an answer to a small-support class label for probing.

    Free-form answers have unbounded support, which makes a k-way probe
    ill-posed.  These schemes project onto a tractable label set:

      first_token  leading character of the normalised answer (digit or sign).
                   ~11 classes on GSM8K, roughly the model's first answer token.
      parity       even/odd, integers only.  Binary, near-balanced, and cheap.
      magnitude    order of magnitude bucket.  Coarse but defined for all reals.

    A probe that recovers `first_token` from pre-reasoning activations is
    evidence the specific answer was already determined.  `parity` and
    `magnitude` are weaker but better balanced, so they are the more honest
    headline when the first-token distribution is skewed.
    """
    n = normalize(ans)
    if n is None:
        return None
    if scheme == "first_token":
        return n[0]
    num = _as_number(n)
    if num is None:
        return None
    if scheme == "parity":
        if num.denominator != 1:
            return None
        return "even" if num.numerator % 2 == 0 else "odd"
    if scheme == "magnitude":
        v = abs(float(num))
        if v == 0:
            return "0"
        import math
        return str(int(math.floor(math.log10(v))))
    raise ValueError(f"unknown scheme {scheme!r}")
