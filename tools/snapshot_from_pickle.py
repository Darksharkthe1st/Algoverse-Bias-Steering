"""Convert an archived `*_dataset.pkl` prompt set into a JSON snapshot.

Archived runs pinned their prompt set with `pickle`, but unpickling executes
arbitrary code and the format is version-fragile (`02-architecture-roadmap.md` §8.2).
This lifts the prompt list out once, into JSON that configs can point at safely.

Reads a *copy* — nothing under `experiments/` is modified, per the arch roadmap's
"legacy is frozen" rule (§12).

Usage:
    python tools/snapshot_from_pickle.py <in.pkl> <out.json>
"""

import json
import pickle
import sys
from pathlib import Path


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    if len(argv) != 2:
        print(__doc__)
        return 2

    src, dst = Path(argv[0]), Path(argv[1])
    with open(src, "rb") as f:
        prompts = pickle.load(f)

    if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
        print(f"error: {src} does not contain a list[str] (got {type(prompts).__name__})")
        return 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(prompts, indent=2))
    print(f"wrote {len(prompts)} prompts -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
