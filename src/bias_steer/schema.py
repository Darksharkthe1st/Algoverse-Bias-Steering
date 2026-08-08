"""Canonical data contract shared across the pipeline.

See docs/02-architecture-roadmap.md §3.2. Every dataset maps its raw files into
`Example`s; every downstream stage speaks `Example`/`Result` and never needs to
know which dataset produced them. The open `metadata` dict is the escape hatch:
dataset-specific fields (BBQ options, a gold answer, a category) live there and
are ignored by generic code, read only by code that cares.
"""

from dataclasses import dataclass, field

# Steering-eval conditions (arch roadmap §5.2 / §7). A `Result` is produced once
# per (example, condition).
INITIAL = "initial"          # no steering
STEERED_POS = "steered_pos"  # steered toward the positive pole (e.g. opinionated)
STEERED_NEG = "steered_neg"  # steered toward the negative pole (e.g. neutral)
CONDITIONS = (INITIAL, STEERED_POS, STEERED_NEG)


@dataclass
class Example:
    """One prompt fed to a model.

    `prompt` is the USER-turn text only — chat-template wrapping and the system
    instruction are applied later (by the model layer, from config), so the same
    Example feeds a chat model and a base model unchanged.
    """

    id: str
    prompt: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Result:
    """One judged model response for a given (example, condition)."""

    example_id: str
    condition: str  # one of CONDITIONS
    response: str
    verdict: str    # a label from the judge's label set (JudgeSpec.labels)
    metadata: dict = field(default_factory=dict)
