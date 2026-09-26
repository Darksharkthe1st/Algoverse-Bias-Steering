"""LLM-as-a-judge: the judge functions (`base`) and the judge v2.1 spec (`v2`).

Importing this package runs `base`'s registry side effects (registers the
`neutrality` and `refusal_substring` judges in JUDGES), so `from . import judges`
is enough to make them available. The public API is re-exported here so callers
say `from bias_steer.judges import parse_verdict, judge_v2_spec` rather than
reaching into submodules.
"""

from . import base, v2  # noqa: F401  (import triggers judge registration)
from .base import (  # noqa: F401
    parse_verdict,
    neutrality_judge,
    UNMATCHED,
    is_refusal,
    refusal_substring_judge,
    REFUSAL_PREFIXES,
    REFUSAL,
    COMPLIANCE,
)
from .v2 import (  # noqa: F401
    V21_LABELS,
    V21_RUBRIC,
    COLLAPSE_TO_IGNORED,
    COLLAPSED_LABELS,
    BEHAVIOR_LABELS,
    collapse,
    judge_v2_spec,
)

__all__ = [
    "base", "v2",
    "parse_verdict", "neutrality_judge", "UNMATCHED",
    "is_refusal", "refusal_substring_judge", "REFUSAL_PREFIXES", "REFUSAL", "COMPLIANCE",
    "V21_LABELS", "V21_RUBRIC", "COLLAPSE_TO_IGNORED", "COLLAPSED_LABELS",
    "BEHAVIOR_LABELS", "collapse", "judge_v2_spec",
]
