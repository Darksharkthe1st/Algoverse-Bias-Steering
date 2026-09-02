"""Dataset loaders (raw files -> list[Example]) and representative sampling.

See docs/02-architecture-roadmap.md §3.3. Each loader maps one dataset format
into the canonical `Example`, stashing format-specific fields in `metadata` so
generic downstream code never has to know which dataset it came from. `sample`
is dataset-agnostic: it filters/stratifies over `metadata`, so adding a dataset
never touches sampling.

Loaders are registered in `DATASETS`. This module is stdlib-only (no torch), so
it imports and runs anywhere.
"""

import csv
import json
import random
from pathlib import Path

from ..utils import get_repo_root
from .config import DatasetSpec, SampleSpec
from .registry import register, DATASETS
from .schema import Example


def _resolve(path: str) -> Path:
    """Resolve a config path: absolute as-is, else relative to the repo root
    (so configs are portable across machines and cwds).

    Fails loud and early if the resolved path doesn't exist. The ambiguity is
    fundamental — a bare relative string can't distinguish "relative to root" from
    "relative to root's parent" — so the natural mistake (a path that starts with the
    repo dir name, e.g. copied from a file browser) silently doubles the repo name
    (`<root>/Algoverse-Bias-Steering/...`) and would otherwise surface as an opaque
    FileNotFoundError deep in a loader's open(), with no hint that resolution was the
    culprit. We name the input, the resolved path, and the root instead of guessing."""
    p = Path(path)
    resolved = p if p.is_absolute() else get_repo_root() / p
    if not resolved.exists():
        raise FileNotFoundError(
            f"dataset path does not exist: {resolved}\n"
            f"  (from config path {path!r}, resolved under repo root {get_repo_root()})\n"
            f"  a bare parent-relative path like 'Algoverse-Bias-Steering/...' doubles "
            f"the repo name — use a path relative to the repo root, or an absolute path."
        )
    return resolved


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
    """One prompt per line.

    Inlined from legacy `src.data.load_plain_dataset` (arch §3.3: legacy loaders
    *become the bodies*, not imports). Byte-for-byte: strip every line, blanks kept.
    """
    with open(_resolve(spec.path)) as f:
        rows = [line.strip() for line in f]
    return [Example(id=f"plain-{i}", prompt=s) for i, s in enumerate(rows)]


@register(DATASETS, "crows")
def load_crows(spec: DatasetSpec) -> list[Example]:
    """CrowS-Pairs CSV cells.

    Inlined from legacy `src.data.load_crows_pairs` (flatten every cell of every
    row into one list, `newline=''` for the csv reader). The non-empty filter drops
    the blank cells that flattening produces.
    """
    cells: list = []
    with open(_resolve(spec.path), newline="") as f:
        for row in csv.reader(f):
            cells += row
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
    """Hidden-bias CSV -> two-option questions.

    Inlined from legacy `src.data.load_hidden_bias_dataset`; the prompt string is
    byte-identical to that loader's `get_question`. Note the csv reader here has NO
    `newline=''` (legacy differs from crows on this) — preserved deliberately.
    """
    rows: list[str] = []
    with open(_resolve(spec.path)) as f:
        for r in csv.reader(f, delimiter=",", quotechar='"'):
            rows.append(
                f"{r[0]} What happened? Pick one of two options: "
                f"(1) {r[1]}, (2) {r[2]}"
            )
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


# --------------------------------------------------------------------------- #
# IssueBench (Röttger et al., 2025; arXiv:2502.08395): realistic writing-
# assistance prompts (template × political issue) for measuring issue bias.
# Fetched from hf.co/datasets/Paul/IssueBench by scripts/fetch_issuebench.py into
# third_party/issuebench/prompts/ as parquet. FK-5 in the fk task-list: the
# boundary check for the single-direction story.
# --------------------------------------------------------------------------- #

# Worktree root -> the fetched parquet. Same rationale as _REFUSAL_SPLITS_DIR:
# parents[2] (not get_repo_root()) stays inside a git worktree, where `.git` is a
# file rather than a directory.
_ISSUEBENCH_DIR = (
    Path(__file__).resolve().parents[2] / "third_party" / "issuebench"
)
_ISSUEBENCH_SPLITS = ("debug", "sample", "full")


@register(DATASETS, "issuebench")
def load_issuebench(spec: DatasetSpec) -> list[Example]:
    """IssueBench prompt split -> Examples (arXiv:2502.08395).

    `spec.path` names the split: "debug" (150 prompts, the default), "sample"
    (636k), or "full" (2.49m, sharded across two parquet files). The split must
    already be fetched:

        python scripts/fetch_issuebench.py --split <split>

    Each row's `prompt_text` (the fully-materialised user prompt) becomes
    `Example.prompt`; the rest of the release's schema is preserved in
    `metadata`. `category` is set to `topic_polarity` (neutral/pro/con) so the
    generic `sample(per_group=("category", n))` stratifier balances across issue
    framings without any IssueBench-specific code.

    Ad-hoc `spec.max_rows` (int, optional) caps how many rows are materialised at
    load time — the full split is 2.49m prompts, and a pilot rarely wants every
    Example object in memory. It slices in file order *before* any `SampleSpec`
    sampling; leave it None to load the whole split.

    pandas is imported lazily (like torch in steering.py) so this module still
    imports without it; only actually loading IssueBench pays for pandas.
    """
    split = (spec.path or "debug").strip()
    if split not in _ISSUEBENCH_SPLITS:
        raise ValueError(
            f"issuebench split {split!r} must be one of {_ISSUEBENCH_SPLITS}; "
            f"pass it as DatasetSpec(name='issuebench', path='debug')"
        )

    shards = sorted(_ISSUEBENCH_DIR.glob(f"prompts/prompts_{split}-*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"no IssueBench '{split}' parquet under {_ISSUEBENCH_DIR / 'prompts'}\n"
            f"Fetch it first:\n"
            f"    python scripts/fetch_issuebench.py --split {split}"
        )

    import pandas as pd  # lazy: keeps datasets.py importable without pandas

    frames = [pd.read_parquet(s) for s in shards]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    max_rows = getattr(spec, "max_rows", None)
    if max_rows is not None:
        df = df.head(int(max_rows))

    examples: list[Example] = []
    for i, row in enumerate(df.itertuples(index=False)):
        r = row._asdict()
        examples.append(Example(
            id=f"issuebench-{split}-{i}",
            prompt=r["prompt_text"],
            metadata={
                "category": r.get("topic_polarity"),  # neutral/pro/con — stratify key
                "topic_id": r.get("topic_id"),
                "topic_text": r.get("topic_text"),
                "topic_polarity": r.get("topic_polarity"),
                "template_id": r.get("template_id"),
                "split": split,
            },
        ))
    return examples


# --------------------------------------------------------------------------- #
# AxBench Concept500 (Wu et al., 2025; arXiv:2501.17148): per-concept labelled
# positive (concept-present) / negative (unsteered) instruction+response pairs.
# Fetched from hf.co/datasets/pyvene/axbench-concept500 by scripts/fetch_axbench.py
# into third_party/axbench/<variant>/<split>/data.parquet. FK-5: run OUR
# difference-of-means steering on it and measure single-direction effectiveness.
# --------------------------------------------------------------------------- #

_AXBENCH_DIR = (
    Path(__file__).resolve().parents[2] / "third_party" / "axbench"
)
_AXBENCH_VARIANTS = ("2b/l10", "2b/l20", "9b/l20", "9b/l31")
_AXBENCH_SPLITS = ("train", "test")


@register(DATASETS, "axbench")
def load_axbench(spec: DatasetSpec) -> list[Example]:
    """AxBench Concept500 -> Examples (arXiv:2501.17148).

    `spec.path` selects the released copy to read as ``"<variant>/<split>"`` —
    e.g. ``"2b/l20/train"`` (the default when `spec.path` is empty is
    ``"2b/l20/train"``). Fetch it first:

        python scripts/fetch_axbench.py --variant 2b/l20

    Mapping (see third_party/axbench/README.md): `input` -> `Example.prompt`;
    `metadata` carries **`label`** = `category` (``positive``/``negative`` — the
    DiffMean contrast, so a label-bucketed extraction can group residuals by it),
    **`category`** = `concept_genre` (the coarse stratify key, matching how other
    loaders use `category`), plus `concept_id`, `output_concept`, and the
    reference `output`. Filter to one concept with
    ``SampleSpec(filter={"concept_id": [<id>]})``.

    Two intended uses, per FK-5 (run separately):
      - **label-bucketed (AxBench-native)**: bucket residuals by `metadata["label"]`
        (``positive``/``negative``) to build a diff-of-means direction the way
        AxBench's DiffMean does. Contrast = ``("positive", "negative")``.
      - **judge-bucketed (ours)**: feed the prompts through the standard
        judge-bucketed pipeline, same as the bias battery.

    Ad-hoc `spec.max_rows` caps rows at load time (Concept500 is large). pandas is
    imported lazily so this module still imports without it.
    """
    sel = (spec.path or "2b/l20/train").strip().strip("/")
    parts = sel.split("/")
    if len(parts) != 3 or f"{parts[0]}/{parts[1]}" not in _AXBENCH_VARIANTS or parts[2] not in _AXBENCH_SPLITS:
        raise ValueError(
            f"axbench path {spec.path!r} must be '<variant>/<split>' with variant in "
            f"{_AXBENCH_VARIANTS} and split in {_AXBENCH_SPLITS}, e.g. '2b/l20/train'"
        )
    variant, split = f"{parts[0]}/{parts[1]}", parts[2]

    path = _AXBENCH_DIR / variant / split / "data.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"missing AxBench parquet {path}\nFetch it first:\n"
            f"    python scripts/fetch_axbench.py --variant {variant}"
        )

    import pandas as pd  # lazy: keeps datasets.py importable without pandas

    df = pd.read_parquet(path)
    max_rows = getattr(spec, "max_rows", None)
    if max_rows is not None:
        df = df.head(int(max_rows))

    tag = sel.replace("/", "-")
    examples: list[Example] = []
    for i, row in enumerate(df.itertuples(index=False)):
        r = row._asdict()
        examples.append(Example(
            id=f"axbench-{tag}-{i}",
            prompt=r["input"],
            metadata={
                "label": r.get("category"),          # positive/negative — DiffMean contrast
                "category": r.get("concept_genre"),  # text/code/math — coarse stratify key
                "concept_id": r.get("concept_id"),
                "output_concept": r.get("output_concept"),
                "output": r.get("output"),           # reference (concept-bearing if positive)
                "variant": variant,
                "split": split,
            },
        ))
    return examples


def sample(examples: list[Example], spec: SampleSpec) -> list[Example]:
    """Filter + stratify + cap, deterministically by `spec.seed` (arch §3.3).

    Order: (1) keep Examples whose `metadata[k]` is in `spec.filter[k]` for every
    key; (2) if `per_group=(key, n)`, keep up to `n` random Examples per distinct
    `metadata[key]` (balanced/representative); (3) if `limit` is set, randomly cap
    the total; (4) shuffle so the result is de-blocked (interleaved), not grouped by
    category. All randomness is seeded, so the same spec yields the same subset.

    The final shuffle is part of the contract: `sample()` returns a *randomly-ordered*
    representative subset, so a positional train/test slice over it is balanced without
    the caller having to know to shuffle first. It uses a fresh `Random(spec.seed)` (not
    the stream already advanced by steps 2-3) so the order is bit-identical to the
    historical caller-side shuffle it replaces — a pure refactor, reproducible splits.
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

    # De-block: a fresh Random(seed) so the permutation matches the caller-side
    # shuffle this replaces (see docstring), keeping historical splits reproducible.
    if out is examples:
        out = list(out)  # never shuffle the caller's list in place
    random.Random(spec.seed).shuffle(out)
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
