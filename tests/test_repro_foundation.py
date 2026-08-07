from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from src.judging import get_judgements_legacy, parse_judgement
from src.recount import (
    AmbiguousLabelSpace,
    IncompleteLog,
    audit_csv,
    audit_manifest,
    derive_label_mapping,
    load_count_rows,
    recount_from_logs,
)
from src.textlog_parse import parse_steered_lines


ROOT = Path(__file__).resolve().parents[1]
REFUSAL = (
    ROOT
    / "experiments/past_logs/refusal_experiments/official_refusal_to_opinion"
)
BBQ_NORMAL = ROOT / "experiments/past_logs/bbq_experiments/bbq_on_normal"


def record(prompt: str, labels: tuple[str, str, str]) -> str:
    sections = []
    for arm, label in zip(("INITIAL", "OPINION", "NEUTRAL"), labels):
        sections.append(
            f"=={arm}_RESPONSE================\n"
            f"model text with ** and == substrings\n"
            f"*\n"  # short all-star model output is not the 44-star record end
            f"**JUDGEMENT:{label}**\n"
        )
    return (
        "**Prompt************************************\n"
        + prompt
        + "\n"
        + "\n".join(sections)
        + "********************************************\n"
    )


class TextLogParserTests(unittest.TestCase):
    def test_exact_delimiters_and_last_judgement(self) -> None:
        text = record("Which one?", ("safe", "unsafe", "safe")).replace(
            "**JUDGEMENT:unsafe**",
            "**JUDGEMENT:safe**\nordinary output\n**JUDGEMENT:unsafe**",
        )
        parsed = parse_steered_lines(text.splitlines(keepends=True))
        self.assertEqual(parsed.n_records, 1)
        self.assertEqual(parsed.records[0].prompt, "Which one?")
        self.assertEqual(parsed.records[0].opinion.label, "unsafe")
        self.assertEqual(parsed.records[0].opinion.judgement_count, 2)
        self.assertIn("model text with ** and == substrings", parsed.records[0].initial.text)
        self.assertEqual(parsed.labels_seen, frozenset({"safe", "unsafe"}))
        self.assertEqual(len(parsed.warnings), 1)

    def test_truncated_record_warns(self) -> None:
        parsed = parse_steered_lines(
            [
                "**Prompt********\n",
                "p\n",
                "==INITIAL_RESPONSE====\n",
                "text\n",
                "**JUDGEMENT:neutral**\n",
            ]
        )
        self.assertEqual(parsed.n_records, 1)
        self.assertIsNone(parsed.records[0].opinion)
        self.assertGreaterEqual(len(parsed.warnings), 3)

    def test_orphan_judgement_is_not_silently_ignored(self) -> None:
        parsed = parse_steered_lines(
            ["COEFF: 0**Prompt********\n", "**JUDGEMENT:None**\n"]
        )
        self.assertEqual(parsed.n_records, 0)
        self.assertEqual(len(parsed.warnings), 2)
        self.assertIn("outside parsed records", parsed.warnings[1])


class JudgementParserTests(unittest.TestCase):
    def test_legacy_is_case_sensitive_last_match_wins(self) -> None:
        responses = [
            "ANSWER: safe then ANSWER: unsafe",
            "answer: safe",
            "ANSWER: safe then ANSWER: safe",
        ]
        self.assertEqual(
            get_judgements_legacy(responses, ["safe", "unsafe"]),
            ["unsafe", "None", "safe"],
        )

    def test_new_parser_separates_ambiguity_and_no_match(self) -> None:
        ambiguous = parse_judgement(
            "ANSWER: safe then ANSWER: unsafe", ["safe", "unsafe"]
        )
        self.assertEqual(ambiguous.status, "ambiguous")
        self.assertIsNone(ambiguous.label)
        self.assertEqual(ambiguous.n_matches, 2)

        missing = parse_judgement("answer: SAFE", ["safe", "unsafe"])
        self.assertEqual(missing.status, "no_match")
        self.assertIsNone(missing.label)

    def test_tolerant_markdown_form(self) -> None:
        result = parse_judgement(
            "reasoning\n**ANSWER:** Neutral", ["neutral", "opinionated"], tolerant=True
        )
        self.assertEqual((result.status, result.label), ("ok", "neutral"))


class RecountUnitTests(unittest.TestCase):
    def test_recount_and_unparsed_bucket(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "example_steered.txt").write_text(
                record("a", ("unsafe", "safe", "None"))
                + record("b", ("safe", "unsafe", "safe")),
                encoding="utf-8",
            )
            recount = recount_from_logs(directory)
            self.assertEqual(recount.denominator, 2)
            self.assertEqual(recount.mapping.positive, "unsafe")
            self.assertEqual(recount.arms["initial"].triple(), (1, 1, 0))
            self.assertEqual(recount.arms["neutral"].triple(), (0, 1, 1))
            self.assertEqual(recount.n_unparsed, 1)

    def test_recount_fails_on_missing_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            text = record("a", ("unsafe", "safe", "safe")).replace(
                "**JUDGEMENT:safe**", "", 1
            )
            (directory / "example_steered_responses.txt").write_text(text)
            with self.assertRaises(IncompleteLog):
                recount_from_logs(directory)

    def test_unknown_label_pair_is_ambiguous(self) -> None:
        with self.assertRaises(AmbiguousLabelSpace):
            derive_label_mapping({"yes", "no"})

    def test_csv_loader_skips_bad_headers_and_repairs_coeff(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy.csv"
            path.write_text(
                "Model name,Model Size,Layer,Coeff,Max Tokens,File PathInit->Opin,x\n"
                "m,s,14,(14,15),128,farhan_logs/Log_1_X/base,1,1,0,0,2,0,2,0,0\n"
                ",,,,,,,,,,,,,,,\n",
                encoding="utf-8",
            )
            rows = load_count_rows(path)
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0].coeff_repair)
            self.assertEqual(rows[0].metadata[3], "(14,15)")
            self.assertEqual(rows[0].denominator, 2)


class ArchivedIntegrationTests(unittest.TestCase):
    def test_refusal_csv_is_exactly_recomputed(self) -> None:
        entries = audit_csv(REFUSAL / "Refusal_To_Opinion.csv")
        self.assertEqual(len(entries), 5)
        self.assertTrue(all(entry.discrepancy is None for entry in entries))
        self.assertTrue(all(entry.recount.denominator == 99 for entry in entries))

        llama = next(entry for entry in entries if "Log_213_" in entry.log_dir.name)
        self.assertEqual(llama.recount.arms["initial"].triple(), (1, 98, 0))
        self.assertEqual(llama.recount.arms["opinion"].triple(), (27, 72, 0))
        self.assertEqual(llama.recount.arms["neutral"].triple(), (21, 78, 0))

    def test_bbq_csv_is_exactly_recomputed_despite_headers(self) -> None:
        entries = audit_csv(BBQ_NORMAL / "BBQ_On_Normal.csv")
        self.assertEqual(len(entries), 7)
        self.assertTrue(all(entry.discrepancy is None for entry in entries))
        self.assertTrue(all(entry.recount.denominator == 99 for entry in entries))

    def test_manifest_is_keyed_by_full_path(self) -> None:
        manifest = audit_manifest(
            REFUSAL / "Refusal_To_Opinion.csv", repo_root=ROOT
        )
        self.assertTrue(manifest["all_match"])
        self.assertEqual(manifest["rows"], 5)
        keys = tuple(manifest["entries"])
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(all(key.startswith("experiments/past_logs/") for key in keys))
        self.assertTrue(
            all(len(value["source_sha256"]) == 64 for value in manifest["entries"].values())
        )


if __name__ == "__main__":
    unittest.main()
