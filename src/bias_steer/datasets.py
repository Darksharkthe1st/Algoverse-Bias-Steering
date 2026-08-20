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
    the data supports per-category sampling.

    `answer_groups` and `stereotyped_groups` are additive fields carried through
    for the bias-taxonomy workstream (JZ-1/JZ-2). BBQ's `answer_info` maps each
    answer to the demographic group it names ("Muslim", "unknown", ...), and
    `stereotyped_groups` says which group the row's stereotype targets. Together
    they let `bias_taxonomy.resolve_answer_roles` decide, deterministically,
    which option counts as the biased choice — which is what lets this dataset
    be scored WITHOUT an LLM judge, so no number derived from it carries a judge
    version. Nothing downstream reads these keys unless it asks for them.
    """
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
            info = o.get("answer_info") or {}
            # answer_info["ansN"] is [answer_text, group_label]; keep the group.
            answer_groups = [
                (info.get(f"ans{n}") or [None, None])[-1] for n in range(3)
            ]
            examples.append(Example(
                id=f"bbq-{o.get('category', 'NA')}-{o.get('example_id', i)}",
                prompt=prompt,
                metadata={
                    "category": o.get("category"),
                    "label": o.get("label"),
                    "question_polarity": o.get("question_polarity"),
                    "context_condition": o.get("context_condition"),
                    "answers": [o["ans0"], o["ans1"], o["ans2"]],
                    "answer_groups": answer_groups,
                    "stereotyped_groups": (
                        (o.get("additional_metadata") or {}).get("stereotyped_groups") or []
                    ),
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


@register(DATASETS, "crows_q")
def load_crows_questions(spec: DatasetSpec) -> list[Example]:
    """CrowS-Pairs sentences turned into questions — the `farhan-fixed-crows` fix.

    `needed-experiments.md` §2 records that a bare CrowS sentence is a poor prompt:
    the pipeline's system instruction asks for "the clear, definitive answer", and
    a declarative statement gives the model nothing to answer. Appending a question
    turns each sentence into something a neutrality judge can meaningfully score.
    That run crashed in the judge before writing a transition matrix, so this
    loader exists to complete it.

    Unlike `crows`, this preserves the pair structure: each CSV row is
    (more-stereotypical, less-stereotypical), recorded in `metadata["stereo"]`, so
    results can be split by pole after the fact.

    Caveat worth knowing before designing an analysis: this anonymized CSV has
    only those two columns — **no `bias_type`** — so the per-category breakdown
    (race / gender / religion / ...) that §2 asks for is NOT derivable from this
    file. It needs the full CrowS-Pairs release.
    """
    import csv

    suffix = spec.__dict__.get("question_suffix", " Why does this make sense?")
    examples: list[Example] = []
    with open(_resolve(spec.path), newline="") as f:
        for i, row in enumerate(csv.reader(f)):
            for pole, cell in zip(("more", "less"), row):
                cell = (cell or "").strip()
                if not cell:
                    continue
                examples.append(Example(
                    id=f"crows-{i}-{pole}",
                    prompt=f"{cell}{suffix}",
                    metadata={"stereo": pole, "pair_id": i},
                ))
    return examples


@register(DATASETS, "hidden_bias")
def load_hidden_bias(spec: DatasetSpec) -> list[Example]:
    """Hidden-bias CSV -> two-option questions (wraps
    `src.data.load_hidden_bias_dataset`)."""
    from src.data import load_hidden_bias_dataset
    rows = load_hidden_bias_dataset(str(_resolve(spec.path)))
    return [Example(id=f"hidden-{i}", prompt=s) for i, s in enumerate(rows)]


@register(DATASETS, "stereoset")
def load_stereoset(spec: DatasetSpec) -> list[Example]:
    """StereoSet intrasentence JSON -> Examples (a fill-in-the-blank pick-one).

    `metadata["category"]` is the StereoSet `bias_type` (race/profession/gender/
    religion) — deliberately the same key BBQ uses, so `sample(per_group=("category",
    N))` works identically across both. This loader is the Phase-3 "new dataset":
    it is one function + one registry line, and nothing downstream changed.
    """
    with open(_resolve(spec.path)) as f:
        data = json.load(f)
    examples: list[Example] = []
    for ex in data["data"]["intrasentence"]:
        sents = ex["sentences"]
        options = ", ".join(s["sentence"] for s in sents)
        prompt = f"{ex['context']} Which option best completes the sentence? Pick one: {options}"
        examples.append(Example(
            id=f"stereoset-{ex['id']}",
            prompt=prompt,
            metadata={
                "category": ex["bias_type"],
                "target": ex["target"],
                "gold_labels": [s["gold_label"] for s in sents],
            },
        ))
    return examples


@register(DATASETS, "snapshot")
def load_snapshot(spec: DatasetSpec) -> list[Example]:
    """A frozen prompt list from a prior run -> Examples, in recorded order.

    Formalizes the notebook's `get_any_variable(..._dataset.pkl)` habit of pinning
    the exact prompt set a historical run used (`01-feature-roadmap.md` §1.2). This
    is what makes an archived run reproducible when its prompts were never a file
    in `datasets/` — as with the Log_103 parity anchor, whose 200 prompts exist
    only inside its pickle.

    Accepts `.json` (a list of strings) or `.pkl`. Order is preserved exactly,
    because a snapshot's whole point is reproducing a specific train/test split.

    NOTE on `.pkl`: unpickling executes arbitrary code, so this path is only for
    our own archived artifacts under `experiments/`. Prefer converting once to
    JSON (`tools/snapshot_from_pickle.py`) and pointing configs at that.
    """
    path = _resolve(spec.path)
    if path.suffix == ".json":
        prompts = json.loads(path.read_text())
    else:
        import pickle
        with open(path, "rb") as f:
            prompts = pickle.load(f)

    if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
        raise ValueError(f"{path} must contain a list[str] of prompts")

    stem = path.stem
    return [Example(id=f"{stem}-{i}", prompt=p) for i, p in enumerate(prompts)]


@register(DATASETS, "refusal_eval")
def load_refusal_eval(spec: DatasetSpec) -> list[Example]:
    """Refusal-repro eval prompts (arXiv:2406.11717), read from the paper's own
    committed baseline completions so our prompts are byte-identical to theirs.

    Reads from `third_party/refusal_direction/runs/<source_model>/completions/`
    (fetched by scripts/fetch_refusal_artifacts.py; resolved worktree-safely via
    `refusal.artifact_dir`, not `_resolve`). Ad-hoc spec fields:
      - `source_model`: run-dir or catalog key to read from (default qwen-1_8b-chat).
        jailbreakbench harmful is a fixed 100-prompt benchmark; for faithful
        per-model comparison, point this at the model being evaluated.
      - `harm`: "harmful" (jailbreakbench), "harmless" (alpaca), or "both" (default).

    `metadata`: {harm, category, source_model}. Prompt = the raw user instruction.
    """
    from . import refusal

    source = getattr(spec, "source_model", None) or "qwen-1_8b-chat"
    run_dir = source if source in refusal.RUN_DIR_TO_MODEL else refusal.MODEL_TO_RUN_DIR.get(source, source)
    which = getattr(spec, "harm", "both")
    wanted = ["harmful", "harmless"] if which == "both" else [which]
    files = {
        "harmful": "jailbreakbench_baseline_completions.json",
        "harmless": "harmless_baseline_completions.json",
    }

    comp_dir = refusal.artifact_dir(run_dir) / "completions"
    examples: list[Example] = []
    for harm in wanted:
        path = comp_dir / files[harm]
        if not path.exists():
            raise FileNotFoundError(
                f"missing {path}\nFetch it first:\n"
                f"    python scripts/fetch_refusal_artifacts.py --model {run_dir}"
            )
        for i, rec in enumerate(json.loads(path.read_text())):
            examples.append(Example(
                id=f"{harm}-{i}",
                prompt=rec["prompt"],
                metadata={"harm": harm, "category": rec.get("category"), "source_model": run_dir},
            ))
    return examples


@register(DATASETS, "refusal_contrast")
def load_refusal_contrast(spec: DatasetSpec) -> list[Example]:
    """Harmful + harmless instructions for one split, combined into one Example
    list — the TRAIN set for reproducing a refusal vector with OUR native
    `mean_diff` pipeline (bucketed by the refusal judge, not by label).

    Unlike `refusal` (one split file per call), this loads both poles so the
    pipeline sees a mix that produces refusals and compliances. `metadata["label"]`
    ("harmful"/"harmless") is preserved so `sample(per_group=("label", N))` can
    balance the two — recommended, since `harmless_train` is ~70x larger.

    Ad-hoc `spec.split` in {train, val, test} (default "train").
    """
    split = getattr(spec, "split", "train")
    examples: list[Example] = []
    for label in ("harmful", "harmless"):
        path = _refusal_split_path(f"{label}_{split}.json")
        if not path.exists():
            raise FileNotFoundError(
                f"missing refusal split {path}\nFetch it first:\n"
                f"    python scripts/fetch_refusal_artifacts.py"
            )
        for i, row in enumerate(json.loads(path.read_text())):
            examples.append(Example(
                id=f"refusal-{label}-{split}-{i}",
                prompt=row["instruction"],
                metadata={"label": label, "split": split, "category": row.get("category")},
            ))
    return examples


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


# --------------------------------------------------------------------------- #
# Refusal-direction repro (arXiv:2406.11717): harmful/harmless instruction splits
# from andyrdt/refusal_direction (fetched by scripts/fetch_refusal_artifacts.py
# into third_party/refusal_direction/dataset/splits/). Each split file is a JSON
# list of {"instruction", "category"}. The loader tags each Example with its
# label ("harmful"/"harmless") and split, parsed from the filename, so the
# extraction path can bucket BY LABEL (not by a judge verdict) — see
# refusal_extract.run_extraction and docs/06-refusal-generation.md.
# --------------------------------------------------------------------------- #

# Worktree root (src/bias_steer/datasets.py -> src -> root). NOT get_repo_root(),
# which follows a `.git` *directory* and would escape a git worktree (there `.git`
# is a file). Mirrors refusal.py's package-relative resolution.
_REFUSAL_SPLITS_DIR = (
    Path(__file__).resolve().parents[2]
    / "third_party" / "refusal_direction" / "dataset" / "splits"
)

_REFUSAL_LABELS = ("harmful", "harmless")
_REFUSAL_SPLITS = ("train", "val", "test")


def _parse_label_split(stem: str) -> tuple[str, str]:
    """"harmful_train" -> ("harmful", "train"). Validates against the known sets."""
    label, _, split = stem.partition("_")
    if label not in _REFUSAL_LABELS or split not in _REFUSAL_SPLITS:
        raise ValueError(
            f"refusal split filename {stem!r} must be '<label>_<split>' with "
            f"label in {_REFUSAL_LABELS} and split in {_REFUSAL_SPLITS}"
        )
    return label, split


def _refusal_split_path(path: str) -> Path:
    """Resolve a refusal split path (worktree-safe):

    - absolute            -> as-is
    - contains a '/'      -> worktree_root / path
    - bare filename       -> third_party/.../dataset/splits/<filename>
    """
    if not path:
        raise ValueError(
            "refusal dataset needs a split file, e.g. DatasetSpec(name='refusal', "
            "path='harmful_train.json'). Fetch splits with "
            "scripts/fetch_refusal_artifacts.py"
        )
    p = Path(path)
    if p.is_absolute():
        return p
    if len(p.parts) > 1:
        return Path(__file__).resolve().parents[2] / p
    return _REFUSAL_SPLITS_DIR / p


@register(DATASETS, "refusal")
def load_refusal(spec: DatasetSpec) -> list[Example]:
    """One refusal split JSON ('<label>_<split>.json') -> Examples.

    `prompt` is the bare instruction (USER turn only). Refusal-specific chat-
    template formatting (upstream literal template, system=None) is applied later
    by the extraction path, NOT by models.render_prompts — matching this is what
    makes our extracted grid match the paper's mean_diffs.pt. `metadata` carries
    the label ("harmful"/"harmless") and split (parsed from the filename) plus the
    upstream category, so downstream code buckets by known label.
    """
    path = _refusal_split_path(spec.path)
    if not path.exists():
        raise FileNotFoundError(
            f"missing refusal split {path}\nFetch it first:\n"
            f"    python scripts/fetch_refusal_artifacts.py"
        )
    label, split = _parse_label_split(path.stem)
    rows = json.loads(path.read_text())
    return [
        Example(
            id=f"refusal-{label}-{split}-{i}",
            prompt=row["instruction"],
            metadata={"label": label, "split": split, "category": row.get("category")},
        )
        for i, row in enumerate(rows)
    ]
