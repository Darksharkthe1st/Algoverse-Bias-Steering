"""Strict, dependency-free parser for the repository's archived steering logs.

The archive contains model-generated text, so delimiters are recognized only when
an entire line matches the historical format.  Judgements are data, not trusted
instructions, and this module never deserializes the adjacent pickle files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable


PROMPT_RE = re.compile(r"^\*\*Prompt\*+$")
ARM_RE = re.compile(r"^==(?P<arm>INITIAL|OPINION|NEUTRAL)_RESPONSE=+$")
# The writer emits exactly 44 stars.  Shorter all-star lines occur in model
# output, so ``^\*+$`` would silently truncate real records.
END_RE = re.compile(r"^\*{44}$")
JUDGEMENT_RE = re.compile(r"^\*\*JUDGEMENT:(?P<label>.*)\*\*$")
ARM_NAMES = ("initial", "opinion", "neutral")
LOG_PATTERNS = ("*_steered_responses.txt", "*_steered.txt")


@dataclass(frozen=True)
class ArmResponse:
    text: str
    label: str | None
    judgement_count: int


@dataclass(frozen=True)
class SteeredRecord:
    prompt: str
    initial: ArmResponse | None
    opinion: ArmResponse | None
    neutral: ArmResponse | None

    def arm(self, name: str) -> ArmResponse | None:
        if name not in ARM_NAMES:
            raise KeyError(f"unknown arm: {name}")
        return getattr(self, name)


@dataclass(frozen=True)
class LogParse:
    source: Path
    records: tuple[SteeredRecord, ...]
    labels_seen: frozenset[str]
    warnings: tuple[str, ...]

    @property
    def n_records(self) -> int:
        return len(self.records)


def find_steered_log(log_dir: str | Path) -> Path:
    """Return the sole archived text log in *log_dir*, or fail closed."""

    directory = Path(log_dir)
    matches = sorted(
        {path for pattern in LOG_PATTERNS for path in directory.glob(pattern)}
    )
    if not matches:
        raise FileNotFoundError(f"no steering text log found in {directory}")
    if len(matches) != 1:
        joined = ", ".join(str(path) for path in matches)
        raise ValueError(f"multiple steering text logs found in {directory}: {joined}")
    return matches[0]


def _finish_record(
    source: Path,
    record_number: int,
    prompt_lines: list[str],
    sections: dict[str, list[str]],
    warnings: list[str],
) -> SteeredRecord:
    arms: dict[str, ArmResponse | None] = {}
    for arm_name in ARM_NAMES:
        lines = sections.get(arm_name)
        if lines is None:
            arms[arm_name] = None
            warnings.append(
                f"{source}: record {record_number} is missing the {arm_name} arm"
            )
            continue

        labels: list[str] = []
        text_lines: list[str] = []
        for line in lines:
            match = JUDGEMENT_RE.fullmatch(line)
            if match:
                labels.append(match.group("label"))
            else:
                text_lines.append(line)
        if not labels:
            warnings.append(
                f"{source}: record {record_number} {arm_name} arm has no judgement"
            )
        elif len(labels) > 1:
            warnings.append(
                f"{source}: record {record_number} {arm_name} arm has "
                f"{len(labels)} judgements; using the last"
            )
        arms[arm_name] = ArmResponse(
            text="\n".join(text_lines).strip("\n"),
            label=labels[-1] if labels else None,
            judgement_count=len(labels),
        )

    return SteeredRecord(
        prompt="\n".join(prompt_lines).strip("\n"),
        initial=arms["initial"],
        opinion=arms["opinion"],
        neutral=arms["neutral"],
    )


def parse_steered_lines(lines: Iterable[str], *, source: str | Path = "<memory>") -> LogParse:
    """Parse steering-log lines without interpreting any model-produced text."""

    source_path = Path(source)
    records: list[SteeredRecord] = []
    warnings: list[str] = []
    prompt_lines: list[str] | None = None
    sections: dict[str, list[str]] = {}
    current_arm: str | None = None
    orphan_judgement_lines: list[int] = []
    malformed_prompt_lines: list[int] = []

    def finish() -> None:
        nonlocal prompt_lines, sections, current_arm
        if prompt_lines is None:
            return
        records.append(
            _finish_record(
                source_path, len(records) + 1, prompt_lines, sections, warnings
            )
        )
        prompt_lines = None
        sections = {}
        current_arm = None

    for line_number, raw_line in enumerate(lines, 1):
        line = raw_line.rstrip("\r\n")

        if PROMPT_RE.fullmatch(line):
            if prompt_lines is not None:
                warnings.append(
                    f"{source_path}: record {len(records) + 1} ended at a new prompt "
                    "without a closing delimiter"
                )
                finish()
            prompt_lines = []
            sections = {}
            current_arm = None
            continue

        if prompt_lines is None:
            # Some malformed coefficient-sweep logs concatenate metadata and a
            # prompt delimiter.  Do not guess where the record starts, but do
            # make the resulting orphan labels impossible to miss.
            if JUDGEMENT_RE.fullmatch(line):
                orphan_judgement_lines.append(line_number)
            elif "**Prompt" in line:
                malformed_prompt_lines.append(line_number)
            continue  # timestamp and cumulative summaries between records

        arm_match = ARM_RE.fullmatch(line)
        if arm_match:
            arm_name = arm_match.group("arm").lower()
            if arm_name in sections:
                warnings.append(
                    f"{source_path}: line {line_number} repeats the {arm_name} delimiter"
                )
            sections.setdefault(arm_name, [])
            current_arm = arm_name
            continue

        if END_RE.fullmatch(line):
            finish()
            continue

        if current_arm is None:
            prompt_lines.append(line)
        else:
            sections[current_arm].append(line)

    if prompt_lines is not None:
        warnings.append(
            f"{source_path}: record {len(records) + 1} reaches end of file "
            "without a closing delimiter"
        )
        finish()

    if malformed_prompt_lines:
        warnings.append(
            f"{source_path}: {len(malformed_prompt_lines)} prompt-like line(s) do not "
            f"exactly match the delimiter; first at line {malformed_prompt_lines[0]}"
        )
    if orphan_judgement_lines:
        warnings.append(
            f"{source_path}: {len(orphan_judgement_lines)} judgement marker(s) occur "
            f"outside parsed records; first at line {orphan_judgement_lines[0]}"
        )

    labels = frozenset(
        response.label
        for record in records
        for arm_name in ARM_NAMES
        if (response := record.arm(arm_name)) is not None
        and response.label is not None
    )
    return LogParse(source_path, tuple(records), labels, tuple(warnings))


def parse_steered_log(path: str | Path) -> LogParse:
    log_path = Path(path)
    with log_path.open(encoding="utf-8", errors="replace") as handle:
        return parse_steered_lines(handle, source=log_path)
