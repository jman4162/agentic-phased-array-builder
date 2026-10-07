"""Score ablation answers against reference values.

Pure functions: no LLM, no tools. ``classify`` maps one final answer to an
outcome:

- ``correct``          the number matches ``reference`` (or a refusal where
                       ``expect: none`` / ``accept_none``, or where the
                       scored surface is listed in ``expect_none_on``)
- ``silently_wrong``   the number matches ``ignored_reference``: the agent
                       reported the result of a computation that dropped
                       part of the request
- ``wrong``            any other number
- ``refused``          ``ANSWER: NONE`` where a valid answer exists
- ``no_answer``        no ``ANSWER:`` line

The tolerance on a reported number is the task's ``tol`` plus half a unit
in the last digit the agent reported, so rounding to fewer decimals than
the tolerance assumes is not penalized.
"""

from __future__ import annotations

import re
from typing import Any

ANSWER_RE = re.compile(
    r"ANSWER:\s*\**\s*(NONE|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)
OUTCOMES = ("correct", "silently_wrong", "wrong", "refused", "no_answer")


def parse_answer(text: str) -> tuple[str, float | None, int]:
    """Return (kind, value, decimals) from the last ANSWER line.

    kind is "none", "number" or "missing"; decimals is the number of digits
    after the decimal point in the reported number (for rounding tolerance).
    """
    matches = ANSWER_RE.findall(text or "")
    if not matches:
        return "missing", None, 0
    raw = matches[-1]
    if raw.upper() == "NONE":
        return "none", None, 0
    mantissa = re.split(r"[eE]", raw)[0]
    decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
    exponent = int(re.split(r"[eE]", raw)[1]) if re.search(r"[eE]", raw) else 0
    return "number", float(raw), max(decimals - exponent, 0)


def _matches(value: float, target: Any, tol: float, decimals: int) -> bool:
    if not isinstance(target, (int, float)):
        return False
    return abs(value - float(target)) <= tol + 0.5 * 10.0 ** (-decimals) + 1e-12


def classify(
    task: dict[str, Any],
    text: str,
    reference: float | None,
    ignored: float | None,
    surface: str | None = None,
) -> dict[str, Any]:
    """Classify one final answer. ``reference``/``ignored`` are numeric values or None.

    ``expect_none_on`` lists surfaces that cannot express the request; on
    those, the task is scored as ``expect: none``.
    """
    kind, value, decimals = parse_answer(text)
    tol = float(task["answer"]["tol"])
    expect_none = task.get("expect") == "none" or surface in task.get("expect_none_on", ())

    if kind == "missing":
        outcome = "no_answer"
    elif kind == "none":
        outcome = "correct" if (expect_none or task.get("accept_none")) else "refused"
    else:
        assert value is not None
        if not expect_none and reference is not None and _matches(value, reference, tol, decimals):
            outcome = "correct"
        elif ignored is not None and _matches(value, ignored, tol, decimals):
            outcome = "silently_wrong"
        else:
            outcome = "wrong"
    return {"outcome": outcome, "answer_kind": kind, "answer_value": value}


def tool_call_stats(audit: list[dict[str, Any]]) -> dict[str, int]:
    """Count tool calls and failed tool results in an audit log."""
    # Failed calls return {"error": ...} (tool-level failure or argument
    # validation in the dispatcher); summaries are str(dict), so match the key.
    failed = sum(
        1
        for e in audit
        if re.match(r"""\{\s*['"]error['"]\s*:""", str(e.get("result_summary", "")))
    )
    return {"tool_calls": len(audit), "failed_tool_calls": failed}
