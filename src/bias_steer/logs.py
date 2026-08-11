"""Plaintext, human-readable run logs (arch roadmap §7.2).

Mirrors the notebook's `_pre-steering.txt` / `_steered.txt` habit: every prompt +
response recorded verbatim, written incrementally so `tail -f` is a live view. A
separate concern from the tidy `results.csv` (that's for pandas; this is for eyes).
Quarantined here so the science functions stay clean.
"""

from pathlib import Path

from ..utils import get_current_time_str
from .schema import INITIAL, STEERED_POS, STEERED_NEG


def _append(path: Path, text: str) -> None:
    with path.open("a") as f:
        f.write(text)


class RunLogger:
    """Writes `logs/run.log` (events), `logs/train.txt` (vector-building pass), and
    `logs/eval.txt` (steered evaluation) under a run directory."""

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
        by_cond = {r.condition: r for r in results}
        lines = [f"=== {example.id} ===", f"PROMPT: {example.prompt}"]
        for cond, title in (
            (INITIAL, "INITIAL"),
            (STEERED_POS, "STEERED+ (opinion)"),
            (STEERED_NEG, "STEERED- (neutral)"),
        ):
            r = by_cond.get(cond)
            if r is not None:
                lines.append(f"  [{title}] ({r.verdict}) {r.response}")
        _append(self.dir / "eval.txt", "\n".join(lines) + "\n\n")
