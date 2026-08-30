"""Recompute archived count tables from text logs, never from pickle files.

The historical CSV headings are misleading: each triple is a marginal label
histogram for one arm, not a transition.  This module treats CSV rows as
untrusted comparators and derives counts and label semantics from text logs.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

from .textlog_parse import ARM_NAMES, find_steered_log, parse_steered_log


KNOWN_LABEL_PAIRS: tuple[tuple[str, str], ...] = (
    ("opinionated", "neutral"),
    ("unsafe", "safe"),
)
UNPARSED_LABELS = frozenset({"none"})


class RecountError(ValueError):
    """Base class for fail-closed recount errors."""


class AmbiguousLabelSpace(RecountError):
    pass


class IncompleteLog(RecountError):
    pass


class MalformedLegacyCSV(RecountError):
    pass


@dataclass(frozen=True)
class LabelMapping:
    positive: str
    complement: str


@dataclass(frozen=True)
class ArmCounts:
    positive: int
    complement: int
    other: int
    raw: Mapping[str, int]

    def triple(self) -> tuple[int, int, int]:
        return self.positive, self.complement, self.other


@dataclass(frozen=True)
class Recount:
    source_path: Path
    source_sha256: str
    denominator: int
    label_space: frozenset[str]
    mapping: LabelMapping
    arms: Mapping[str, ArmCounts]
    n_unparsed: int
    warnings: tuple[str, ...]

    def triples(self) -> tuple[int, ...]:
        return tuple(value for arm in ARM_NAMES for value in self.arms[arm].triple())


@dataclass(frozen=True)
class LegacyCountRow:
    csv_path: Path
    line_number: int
    metadata: tuple[str, ...]
    counts: tuple[int, ...]
    denominator: int
    coeff_repair: bool

    @property
    def file_hint(self) -> str:
        return self.metadata[5]

    @property
    def key(self) -> str:
        return f"{self.csv_path.as_posix()}:{self.line_number}"


@dataclass(frozen=True)
class Discrepancy:
    expected: tuple[int, ...]
    observed: tuple[int, ...]


@dataclass(frozen=True)
class AuditEntry:
    row: LegacyCountRow
    log_dir: Path
    recount: Recount
    discrepancy: Discrepancy | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_label_mapping(labels: Iterable[str]) -> LabelMapping:
    label_set = set(labels)
    candidates = [
        LabelMapping(positive, complement)
        for positive, complement in KNOWN_LABEL_PAIRS
        if positive in label_set and complement in label_set
    ]
    if len(candidates) != 1:
        raise AmbiguousLabelSpace(
            "cannot derive CSV columns from labels "
            f"{sorted(label_set)!r}; expected exactly one complete known pair"
        )
    return candidates[0]


def recount_from_logs(log_dir: str | Path) -> Recount:
    """Recount one log directory (or one log file), failing on missing markers."""

    supplied = Path(log_dir)
    source = supplied if supplied.is_file() else find_steered_log(supplied)
    parsed = parse_steered_log(source)
    mapping = derive_label_mapping(parsed.labels_seen)

    raw_by_arm = {arm: Counter() for arm in ARM_NAMES}
    marker_count = 0
    defects: list[str] = []
    for record_number, record in enumerate(parsed.records, 1):
        for arm_name in ARM_NAMES:
            response = record.arm(arm_name)
            if response is None or response.label is None:
                defects.append(f"record {record_number} {arm_name} has no label")
                continue
            marker_count += response.judgement_count
            if response.judgement_count != 1:
                defects.append(
                    f"record {record_number} {arm_name} has "
                    f"{response.judgement_count} judgement markers"
                )
            raw_by_arm[arm_name][response.label] += 1

    expected_markers = 3 * parsed.n_records
    if marker_count != expected_markers:
        defects.append(
            f"found {marker_count} judgement markers; expected {expected_markers} "
            f"for {parsed.n_records} records"
        )
    if defects:
        raise IncompleteLog(f"{source}: " + "; ".join(defects))

    arms: dict[str, ArmCounts] = {}
    for arm_name, raw in raw_by_arm.items():
        positive = raw[mapping.positive]
        complement = raw[mapping.complement]
        arms[arm_name] = ArmCounts(
            positive=positive,
            complement=complement,
            other=sum(raw.values()) - positive - complement,
            raw=dict(sorted(raw.items())),
        )
        if sum(raw.values()) != parsed.n_records:
            raise IncompleteLog(
                f"{source}: {arm_name} denominator {sum(raw.values())} does not "
                f"equal record count {parsed.n_records}"
            )

    return Recount(
        source_path=source,
        source_sha256=_sha256(source),
        denominator=parsed.n_records,
        label_space=parsed.labels_seen,
        mapping=mapping,
        arms=arms,
        n_unparsed=sum(
            count
            for raw in raw_by_arm.values()
            for label, count in raw.items()
            if label.casefold() in UNPARSED_LABELS
        ),
        warnings=parsed.warnings,
    )


def _repair_legacy_fields(fields: list[str]) -> tuple[list[str], bool]:
    repaired = False
    # Historical tuple coefficients containing a comma were not CSV-quoted.
    if len(fields) >= 5 and fields[3].startswith("(") and fields[4].endswith(")"):
        fields = fields[:3] + [fields[3] + "," + fields[4]] + fields[5:]
        repaired = True
    return fields, repaired


def load_count_rows(path: str | Path) -> tuple[LegacyCountRow, ...]:
    """Load only the fixed six-metadata + nine-count legacy row shape.

    Repeated/malformed headings and all-empty filler rows are skipped.  Data
    arity and per-arm denominators are validated without trusting header names.
    """

    csv_path = Path(path)
    rows: list[LegacyCountRow] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        for line_number, fields in enumerate(csv.reader(handle), 1):
            if not fields or all(not field.strip() for field in fields):
                continue
            fields = [field.strip() for field in fields]
            if fields[0].casefold() == "model name":
                continue
            fields, coeff_repair = _repair_legacy_fields(fields)
            if len(fields) != 15:
                raise MalformedLegacyCSV(
                    f"{csv_path}:{line_number}: expected 15 data fields after "
                    f"repair, found {len(fields)}"
                )
            try:
                counts = tuple(int(value) for value in fields[6:15])
            except ValueError as error:
                raise MalformedLegacyCSV(
                    f"{csv_path}:{line_number}: non-integer result count"
                ) from error
            denominators = tuple(sum(counts[index : index + 3]) for index in (0, 3, 6))
            if len(set(denominators)) != 1:
                raise MalformedLegacyCSV(
                    f"{csv_path}:{line_number}: arm denominators disagree: "
                    f"{denominators}"
                )
            rows.append(
                LegacyCountRow(
                    csv_path=csv_path,
                    line_number=line_number,
                    metadata=tuple(fields[:6]),
                    counts=counts,
                    denominator=denominators[0],
                    coeff_repair=coeff_repair,
                )
            )
    return tuple(rows)


def compare_to_csv_row(recount: Recount, row: LegacyCountRow) -> Discrepancy | None:
    observed = recount.triples()
    if observed == row.counts:
        return None
    return Discrepancy(expected=row.counts, observed=observed)


def resolve_log_dir(row: LegacyCountRow) -> Path:
    """Resolve a historical ``farhan_logs/...`` hint relative to its CSV."""

    match = re.search(r"(?:^|/)(Log_\d+_[^/]+)", row.file_hint.replace("\\", "/"))
    if not match:
        raise FileNotFoundError(f"{row.key}: no Log_NNN directory in {row.file_hint!r}")
    directory_name = match.group(1)
    candidates = sorted(
        path for path in row.csv_path.parent.glob(directory_name) if path.is_dir()
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"{row.key}: expected one {directory_name!r} beside CSV, found "
            f"{len(candidates)}"
        )
    return candidates[0]


def audit_csv(path: str | Path) -> tuple[AuditEntry, ...]:
    entries: list[AuditEntry] = []
    for row in load_count_rows(path):
        log_dir = resolve_log_dir(row)
        recount = recount_from_logs(log_dir)
        entries.append(
            AuditEntry(row, log_dir, recount, compare_to_csv_row(recount, row))
        )
    return tuple(entries)


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def audit_manifest(path: str | Path, *, repo_root: str | Path = ".") -> dict[str, object]:
    """Return a path-keyed, JSON-serializable recount manifest."""

    root = Path(repo_root)
    csv_path = Path(path)
    entries = audit_csv(csv_path)
    manifest_entries: dict[str, object] = {}
    for entry in entries:
        recount = entry.recount
        key = _display_path(entry.log_dir, root)
        manifest_entries[key] = {
            "csv_row": entry.row.key,
            "source_log": _display_path(recount.source_path, root),
            "source_sha256": recount.source_sha256,
            "denominator": recount.denominator,
            "label_space": sorted(recount.label_space),
            "column_mapping": {
                "column_1": recount.mapping.positive,
                "column_2": recount.mapping.complement,
                "column_3": "all_other_labels",
            },
            "arms": {
                arm: {
                    "raw": recount.arms[arm].raw,
                    "csv_triple": recount.arms[arm].triple(),
                }
                for arm in ARM_NAMES
            },
            "n_unparsed": recount.n_unparsed,
            "csv_match": entry.discrepancy is None,
            "discrepancy": (
                None
                if entry.discrepancy is None
                else {
                    "csv": entry.discrepancy.expected,
                    "recount": entry.discrepancy.observed,
                }
            ),
            "warnings": recount.warnings,
        }
    return {
        "schema_version": 1,
        "csv": _display_path(csv_path, root),
        "rows": len(entries),
        "all_match": all(entry.discrepancy is None for entry in entries),
        "entries": manifest_entries,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="legacy CSV to compare")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    manifest = audit_manifest(args.csv, repo_root=args.repo_root)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["all_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
