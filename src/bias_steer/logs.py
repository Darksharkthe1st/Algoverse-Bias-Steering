"""Plaintext, human-readable run logs (arch roadmap §7.2).

Mirrors the notebook's `_pre-steering.txt` / `_steered.txt` habit: every prompt +
response recorded verbatim, written incrementally so `tail -f` is a live view. A
separate concern from the tidy `results.csv` (that's for pandas; this is for eyes).
Quarantined here so the science functions stay clean.

Evaluation output is split PER CONDITION under `logs/by_condition/<cond>.txt` (one
arm per file, e.g. `steered_pos.txt`) so a human can read a single arm end to end
instead of scrolling one interleaved `eval.txt`. Every arm actually present in the
run is written -- including the prompt-baseline arms (`prompt_pos`/`prompt_neg`),
which the old single-file writer silently dropped.
"""

from collections import Counter
from pathlib import Path

from ..utils import get_current_time_str
from .schema import CONDITIONS


def _append(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def _degenerate_flag(text: str) -> str:
    """A cheap readability hint for obviously degenerate output -- repetition loops
    or near-zero lexical diversity (the c40-style "coherence ceiling" failure). This
    is NOT the coherence gate; it only marks blatant mush so a human skimming a
    `by_condition` file isn't fooled by a confident verdict on broken text."""
    words = text.split()
    if len(words) < 20:
        return ""
    distinct_ratio = len(set(words)) / len(words)
    trigrams = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
    top_share = (Counter(trigrams).most_common(1)[0][1] / len(trigrams)) if trigrams else 0.0
    if distinct_ratio < 0.30 or top_share > 0.15:
        return "  [!! repeat-loop]"
    return ""


class RunLogger:
    """Writes `logs/run.log` (events), `logs/train.txt` (vector-building pass), and
    `logs/by_condition/<cond>.txt` (one steered/prompt eval arm per file) under a
    run directory."""

    def __init__(self, run_dir):
        self.dir = Path(run_dir) / "logs"
        self.dir.mkdir(parents=True, exist_ok=True)

    def event(self, msg: str) -> None:
        _append(self.dir / "run.log", f"[{get_current_time_str()}] {msg}\n")

    def train(self, example, response: str, verdict: str) -> None:
        _append(
            self.dir / "train.txt",
            f"=== {example.id} ===\n"
            f"PROMPT:   {example.prompt}\n"
            f"RESPONSE: {response}\n"
            f"VERDICT:  {verdict}\n\n",
        )

    def eval(self, example, results) -> None:
        """Append each condition's (prompt, verdict, output) to its own
        `logs/by_condition/<cond>.txt`, ordered by `CONDITIONS`; only arms present
        in `results` are written. Verdicts on blatantly degenerate output are flagged
        so a reader isn't misled by a confident label on a repetition loop."""
        by_dir = self.dir / "by_condition"
        by_dir.mkdir(exist_ok=True)
        by_cond = {r.condition: r for r in results}
        for cond in CONDITIONS:
            r = by_cond.get(cond)
            if r is None:
                continue
            _append(
                by_dir / f"{cond}.txt",
                f"=== {example.id} ===\n"
                f"PROMPT:  {example.prompt}\n"
                f"VERDICT: {r.verdict}{_degenerate_flag(r.response)}\n"
                f"OUTPUT:  {r.response}\n\n",
            )
