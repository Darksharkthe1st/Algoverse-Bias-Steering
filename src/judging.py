"""Explicit judgement extraction with a failure channel.

``get_judgements_legacy`` preserves the historical notebook's case-sensitive,
last-match-wins behavior for audits.  New measurements should use
``parse_judgement`` and keep no-match and ambiguity separate from behavioral
labels such as nonsense/degenerate.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal, Sequence


ParseStatus = Literal["ok", "no_match", "ambiguous"]


@dataclass(frozen=True)
class ParseResult:
    label: str | None
    status: ParseStatus
    n_matches: int
    match_span: tuple[int, int] | None


def _validated_options(options: Sequence[str], *, tolerant: bool) -> tuple[str, ...]:
    if not options:
        raise ValueError("at least one judgement option is required")
    if any(not option for option in options):
        raise ValueError("judgement options must be non-empty")

    normalized = [option.casefold() if tolerant else option for option in options]
    if len(set(normalized)) != len(normalized):
        raise ValueError("judgement options must be unique after normalization")
    return tuple(options)


def parse_judgement(
    text: str, options: Sequence[str], *, tolerant: bool = False
) -> ParseResult:
    """Extract an ``ANSWER: option`` judgement and report extraction defects.

    Strict mode reproduces the historical answer token's case and spacing while
    surfacing all matches.  Tolerant mode additionally accepts case differences,
    flexible whitespace, and Markdown fences around ``ANSWER:`` or the label.
    Multiple occurrences of one label remain usable (``status='ok'``); matches
    for distinct labels are ambiguous and return no label.
    """

    option_tuple = _validated_options(options, tolerant=tolerant)
    matches: list[tuple[int, int, str]] = []

    # Longest first is defensive if a future option is a prefix of another.
    for option in sorted(option_tuple, key=len, reverse=True):
        if tolerant:
            pattern = re.compile(
                rf"(?:\*\*)?ANSWER\s*:(?:\*\*)?\s*"
                rf"(?P<label>{re.escape(option)})(?![\w-])(?:\*\*)?",
                re.IGNORECASE,
            )
        else:
            # Deliberately no boundary: this is the notebook's substring rule.
            pattern = re.compile(re.escape(f"ANSWER: {option}"))
        for match in pattern.finditer(text):
            matches.append((match.start(), match.end(), option))

    matches.sort(key=lambda item: (item[0], item[1]))
    if not matches:
        return ParseResult(None, "no_match", 0, None)

    distinct = {label for _, _, label in matches}
    if len(distinct) > 1:
        return ParseResult(None, "ambiguous", len(matches), None)

    last = matches[-1]
    return ParseResult(last[2], "ok", len(matches), (last[0], last[1]))


def get_judgements_legacy(
    responses: Sequence[str], options: Sequence[str]
) -> list[str]:
    """Behavior-exact port of the post-68ac661 notebook extractor.

    Matching is case-sensitive ``ANSWER: <option>`` anywhere in the response,
    scanning left-to-right and retaining the last match.  The historical string
    sentinel ``"None"`` is retained solely for artifact reconstruction.
    """

    option_tuple = _validated_options(options, tolerant=False)
    judgements: list[str] = []
    for response in responses:
        result = "None"
        for index in range(len(response)):
            for option in option_tuple:
                token = f"ANSWER: {option}"
                if response.startswith(token, index):
                    result = option
                    break
        judgements.append(result)
    return judgements
