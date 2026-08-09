"""Parity ladder, step 1: re-judge archived responses (no GPU, no model load).

Feeds the *archived* Log_103 train responses back through the refactored
`neutrality_judge` and compares the labels to the ones recorded in 2025. Because
the responses are fixed, generation is entirely out of the picture — any
disagreement is the judge and only the judge.

That buys two things at once:

1. **Port validation.** The refactor rewrote the notebook's exact-case
   ``ANSWER: <label>`` substring scan as `judge.parse_verdict`, and moved the
   rubric into `JudgeSpec`. If those are faithful, labels should mostly agree.
2. **The §0.2 measurement.** `docs/needed-experiments.md` §0.2 says the unpinned
   `gpt-4o-mini` judge drifts on re-runs and that nothing downstream is
   trustworthy until that drift is quantified. Running k trials over identical
   inputs measures it directly.

Those two are deliberately confounded in the *aggregate* number and separated by
the k-trial structure: judge self-disagreement across trials is pure drift, while
a label that all k trials agree on but the archive disagrees with is a genuine
behavioral difference between old and new judge.

Lives in tools/ rather than analysis/ because it imports the engine, which
`analysis/` is contractually forbidden from doing (arch roadmap §7.1).

Usage:
    python tools/parity_rejudge.py [--k 3] [--limit N] [--log PATH]
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bias_steer.cli import _load_env
from src.bias_steer.config import JudgeSpec
from src.bias_steer.schema import Example

# The parity anchor: the one archived run that kept its dataset, vector, and
# per-example verdicts together (see docs/03-gpu-bringup.md §4).
DEFAULT_LOG = (
    "experiments/past_logs/methodology_experiments/misc_coeff_tests/"
    "new_older_worse_coeffs/Log_103_Automated_Test_Qwen1.5-1.8B-Chat/"
    "log_103_Qwen1.5-1.8B-Chat_pre-steering_responses.txt"
)

_SEP = "======================================================"
_JUDGEMENT = re.compile(r"^\*\*JUDGEMENT:(?P<label>\w+)\*\*$")


def parse_archive(path: Path) -> list[tuple[str, str, str]]:
    """Return [(prompt, response, archived_verdict)] from a textlog dump.

    Block format written by the notebook's `textlog_initial_responses`::

        ======================================================
        PROMPT:<prompt>
        <response, possibly many lines>
        **JUDGEMENT:<label>**
        **Progress: Neutral ( n ) + Opinion ( n ) + Nonsense ( n ) => Tn

    Blocks without a JUDGEMENT line are skipped (a crashed run can leave one).
    """
    out: list[tuple[str, str, str]] = []
    for block in path.read_text().split(_SEP):
        lines = block.strip().splitlines()
        if not lines or not lines[0].startswith("PROMPT:"):
            continue

        prompt = lines[0][len("PROMPT:"):].strip()
        response_lines, verdict = [], None
        for line in lines[1:]:
            m = _JUDGEMENT.match(line.strip())
            if m:
                verdict = m.group("label")
                break
            response_lines.append(line)

        if verdict is None:
            continue
        out.append((prompt, "\n".join(response_lines).strip(), verdict))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--log", default=DEFAULT_LOG, help="archived textlog to re-judge")
    p.add_argument("--k", type=int, default=3, help="judge trials per response (§0.2)")
    p.add_argument("--limit", type=int, help="only the first N responses (smoke test)")
    p.add_argument("--model", default="gpt-4o-mini",
                   help="judge model; the archive used the unpinned gpt-4o-mini alias")
    args = p.parse_args(argv)

    _load_env()
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("error: OPENAI_API_KEY not set (see .env.example)")
        return 2

    path = Path(args.log)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    rows = parse_archive(path)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print(f"error: no judged blocks parsed from {path}")
        return 1

    prompts, responses, archived = zip(*rows)
    examples = [Example(id=f"log103-{i}", prompt=q) for i, q in enumerate(prompts)]
    spec = JudgeSpec(name="neutrality", model=args.model)

    print(f"re-judging {len(rows)} archived responses, k={args.k}, model={args.model}")
    print(f"  archive tally: {dict(Counter(archived))}\n")

    from src.bias_steer.judge import neutrality_judge

    trials = []
    for t in range(args.k):
        labels = neutrality_judge(list(responses), examples, spec)
        agree = sum(a == b for a, b in zip(labels, archived))
        trials.append(labels)
        print(f"  trial {t + 1}: {dict(Counter(labels))}  "
              f"agreement vs archive {agree}/{len(rows)} ({agree / len(rows):.1%})")

    # Majority label per example, and how often the k trials agreed with themselves.
    majority, unanimous = [], 0
    for i in range(len(rows)):
        votes = Counter(trials[t][i] for t in range(args.k))
        majority.append(votes.most_common(1)[0][0])
        unanimous += len(votes) == 1

    maj_agree = sum(a == b for a, b in zip(majority, archived))
    print(f"\n  majority vs archive: {maj_agree}/{len(rows)} ({maj_agree / len(rows):.1%})")
    print(f"  judge self-consistency (all {args.k} trials agree): "
          f"{unanimous}/{len(rows)} ({unanimous / len(rows):.1%})   <- §0.2 drift")

    print("\n  confusion (archive -> new majority):")
    confusion = Counter(zip(archived, majority))
    for (old, new), n in sorted(confusion.items(), key=lambda kv: -kv[1]):
        mark = "" if old == new else "   <-- flipped"
        print(f"    {old:>12} -> {new:<12} {n:>4}{mark}")

    disagreements = [i for i in range(len(rows)) if majority[i] != archived[i]]
    if disagreements:
        print(f"\n  first disagreements (of {len(disagreements)}):")
        for i in disagreements[:3]:
            print(f"    [{i}] archive={archived[i]} new={majority[i]}")
            print(f"        prompt:   {prompts[i][:100]}")
            print(f"        response: {responses[i][:160]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
