"""Dataset loaders (raw files -> list[Example]) and representative sampling.

See docs/02-architecture-roadmap.md §3.3. Each loader maps one dataset format
into the canonical `Example`, stashing format-specific fields in `metadata` so
generic downstream code never has to know which dataset it came from. `sample`
is dataset-agnostic: it filters/stratifies over `metadata`, so adding a dataset
never touches sampling.

Loaders are registered in `DATASETS`. This module is stdlib-only (no torch), so
it imports and runs anywhere.
"""

import json
import random
from pathlib import Path

from ..utils import get_repo_root
from .config import DatasetSpec, SampleSpec
from .registry import register, DATASETS
from .schema import Example


def _resolve(path: str) -> Path:
    """Resolve a config path: absolute as-is, else relative to the repo root
    (so configs are portable across machines and cwds)."""
    p = Path(path)
    return p if p.is_absolute() else get_repo_root() / p


@register(DATASETS, "bbq")
def load_bbq(spec: DatasetSpec) -> list[Example]:
    """BBQ jsonl -> Examples. Prompt format matches the legacy
    `src.data.load_bbq_dataset`; `metadata` preserves category/label/answers so
    the data supports per-category sampling."""
    examples: list[Example] = []
    with open(_resolve(spec.path)) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            prompt = (
                f"{o['context']} {o['question']} Pick one of three options: "
                f"{o['ans0']}, {o['ans1']}, {o['ans2']}"
            )
            examples.append(Example(
                id=f"bbq-{o.get('category', 'NA')}-{o.get('example_id', i)}",
                prompt=prompt,
                metadata={
                    "category": o.get("category"),
                    "label": o.get("label"),
                    "question_polarity": o.get("question_polarity"),
                    "context_condition": o.get("context_condition"),
                    "answers": [o["ans0"], o["ans1"], o["ans2"]],
                },
            ))
    return examples


@register(DATASETS, "plain")
def load_plain(spec: DatasetSpec) -> list[Example]:
    """One prompt per line (wraps `src.data.load_plain_dataset`)."""
    from src.data import load_plain_dataset
    rows = load_plain_dataset(str(_resolve(spec.path)))
    return [Example(id=f"plain-{i}", prompt=s) for i, s in enumerate(rows)]


@register(DATASETS, "crows")
def load_crows(spec: DatasetSpec) -> list[Example]:
    """CrowS-Pairs CSV cells (wraps `src.data.load_crows_pairs`)."""
    from src.data import load_crows_pairs
    cells = load_crows_pairs(str(_resolve(spec.path)))
    return [
        Example(id=f"crows-{i}", prompt=s)
        for i, s in enumerate(cells)
        if isinstance(s, str) and s.strip()
    ]


@register(DATASETS, "hidden_bias")
def load_hidden_bias(spec: DatasetSpec) -> list[Example]:
    """Hidden-bias CSV -> two-option questions (wraps
    `src.data.load_hidden_bias_dataset`)."""
    from src.data import load_hidden_bias_dataset
    rows = load_hidden_bias_dataset(str(_resolve(spec.path)))
    return [Example(id=f"hidden-{i}", prompt=s) for i, s in enumerate(rows)]


def sample(examples: list[Example], spec: SampleSpec) -> list[Example]:
    """Filter + stratify + cap, deterministically by `spec.seed` (arch §3.3).

    Order: (1) keep Examples whose `metadata[k]` is in `spec.filter[k]` for every
    key; (2) if `per_group=(key, n)`, keep up to `n` random Examples per distinct
    `metadata[key]` (balanced/representative); (3) if `limit` is set, randomly cap
    the total. All randomness is seeded, so the same spec yields the same subset.
    """
    rng = random.Random(spec.seed)
    out = examples

    if spec.filter:
        def keep(ex: Example) -> bool:
            return all(ex.metadata.get(k) in vals for k, vals in spec.filter.items())
        out = [e for e in out if keep(e)]

    if spec.per_group:
        key, n = spec.per_group
        groups: dict = {}
        for e in out:
            groups.setdefault(e.metadata.get(key), []).append(e)
        picked: list[Example] = []
        for g in sorted(groups, key=lambda x: (x is None, str(x))):
            items = groups[g][:]
            rng.shuffle(items)
            picked.extend(items[:n])
        out = picked

    if spec.limit is not None and len(out) > spec.limit:
        idx = list(range(len(out)))
        rng.shuffle(idx)
        keep_idx = set(idx[: spec.limit])
        out = [e for i, e in enumerate(out) if i in keep_idx]

    return out
