"""Refusal-direction EXTRACTION driver (arXiv:2406.11717, generate_directions).

Reproduces the paper's candidate-grid extraction *ourselves* (vs loading their
`direction.pt` in `refusal.py`). The grid is validated against their committed
`mean_diffs.pt` by cosine, cell-by-cell (see scripts/refusal_extract_check.py and
docs/06-refusal-generation.md) — that check is the arbiter of correctness, so this
module is written to mirror upstream exactly.

Layout of this module:
- TORCH-FREE (runs anywhere, unit-tested on CPU): the per-model chat templates
  (upstream literals, system=None), prompt formatting, end-of-instruction suffix,
  and the deterministic train/val sampling that reproduces upstream's
  `random.seed(42); random.sample(...)` selection.
- MODEL/TORCH (lazy-imported; executed on the Lambda GPU box): the prompt-
  activation forward pass and `run_extraction`, wired through an `ExtractionBackend`
  so the label-bucketing logic is testable with a fake (torch-gated, no model).

Why not reuse models.render_prompts / generate_with_cache:
- render_prompts injects DEFAULT_SYS as a *system* turn; the paper formats with
  NO system prompt. A system turn shifts every token position and breaks cosine.
- generate_with_cache caches over the generated RESPONSE; extraction reads the
  PROMPT's residual stream with no generation.
So extraction owns its formatting and its forward pass.
"""

from dataclasses import dataclass, field
from typing import Callable, Iterator
import random

from .registry import MODELS
from . import datasets
from .steering import capture_prompt_positions, build_refusal_grid, resid_pre_hook_names


# --------------------------------------------------------------------------- #
# Per-model chat templates — verbatim from andyrdt/refusal_direction@9d852fa
# (pipeline/model_utils/*_model.py), keyed by our MODEL_CATALOG keys. system=None
# variant only (extraction uses no system prompt). `eoi_add_special_tokens` records
# whether upstream encodes the end-of-instruction suffix with add_special_tokens
# (False for gemma/llama2/llama3; default/True for qwen/yi) — it changes n_pos.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RefusalTemplate:
    template: str               # contains a single "{instruction}" placeholder
    eoi_add_special_tokens: bool  # how upstream tokenizes the eoi suffix

    @property
    def eoi_suffix(self) -> str:
        """Text after the instruction (the assistant-prompt suffix). Its token
        count is the paper's n_pos = len(range(-n_eoi, 0))."""
        return self.template.split("{instruction}")[-1]

    def format(self, instruction: str) -> str:
        return self.template.format(instruction=instruction)


REFUSAL_TEMPLATES: dict[str, RefusalTemplate] = {
    "qwen-1.8b": RefusalTemplate(
        "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n", True),
    "yi-6b": RefusalTemplate(
        "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n", True),
    "gemma-2b": RefusalTemplate(
        "<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n", False),
    "llama3-8b": RefusalTemplate(
        "<|start_header_id|>user<|end_header_id|>\n\n{instruction}"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", False),
    "llama-2-7b": RefusalTemplate("[INST] {instruction} [/INST] ", False),
}


def _resolve_model_key(model_key: str) -> str:
    """Accept a catalog key or an upstream run-dir name -> catalog key."""
    from .refusal import RUN_DIR_TO_MODEL
    if model_key in REFUSAL_TEMPLATES:
        return model_key
    if model_key in RUN_DIR_TO_MODEL:
        return RUN_DIR_TO_MODEL[model_key]
    raise KeyError(f"no refusal template for {model_key!r}; known: {sorted(REFUSAL_TEMPLATES)}")


def format_refusal_prompt(model_key: str, instruction: str) -> str:
    """Format one instruction with the model's upstream template (system=None)."""
    return REFUSAL_TEMPLATES[_resolve_model_key(model_key)].format(instruction)


def eoi_suffix(model_key: str) -> str:
    return REFUSAL_TEMPLATES[_resolve_model_key(model_key)].eoi_suffix


def n_positions(loaded) -> int:
    """The paper's n_pos for this model = number of end-of-instruction tokens.

    Needs the tokenizer (so it runs where the model is loaded). Equals the grid's
    leading axis (qwen/gemma/llama3=5; llama-2/yi=6)."""
    tmpl = REFUSAL_TEMPLATES[_resolve_model_key(loaded.spec.name)]
    ids = loaded.tokenizer.encode(tmpl.eoi_suffix, add_special_tokens=tmpl.eoi_add_special_tokens)
    return len(ids)


# --------------------------------------------------------------------------- #
# Deterministic sampling — reproduces run_pipeline.load_and_sample_datasets:
#   random.seed(42)
#   harmful_train  = random.sample(<harmful train instructions>,  n_train)
#   harmless_train = random.sample(<harmless train instructions>, n_train)
#   harmful_val    = random.sample(<harmful val instructions>,    n_val)
#   harmless_val   = random.sample(<harmless val instructions>,   n_val)
# The RNG state flows across all four calls, so the SAME seed + SAME input order
# (file order, preserved by json.load) reproduces the exact selection. CPU-only.
# --------------------------------------------------------------------------- #

def _split_instructions(label: str, split: str) -> list[str]:
    from .config import DatasetSpec
    exs = datasets.load_refusal(DatasetSpec(name="refusal", path=f"{label}_{split}.json"))
    return [e.prompt for e in exs]


def load_and_sample_repro(n_train: int = 128, n_val: int = 32, seed: int = 42) -> dict:
    """Reproduce upstream's sampled instruction sets (pre-filter). Deterministic."""
    random.seed(seed)
    return {
        "harmful_train":  random.sample(_split_instructions("harmful", "train"), n_train),
        "harmless_train": random.sample(_split_instructions("harmless", "train"), n_train),
        "harmful_val":    random.sample(_split_instructions("harmful", "val"), n_val),
        "harmless_val":   random.sample(_split_instructions("harmless", "val"), n_val),
    }


# --------------------------------------------------------------------------- #
# Extraction — forward pass over PROMPT tokens, cache resid_pre, bucket by label.
# The ExtractionBackend seam mirrors experiment.Backend: default = real (torch +
# model); tests inject a fake `prompt_caches` to exercise the label-bucketing and
# grid assembly without a model. `capture_prompt_positions` still uses torch, so
# such tests are torch-gated (but need no model download).
# --------------------------------------------------------------------------- #

def _real_load(spec):
    from . import models
    return models.load_model(spec)


def _real_prompt_caches(loaded, instructions, capture_names, batch_size=32) -> Iterator:
    """Yield one resid_pre cache per instruction, from a forward pass on the
    formatted prompt (NO generation). Left-padded batches; the last n_pos positions
    of each row are the real end-of-instruction tokens (padding is on the left).

    Faithful to the paper's get_mean_activations (forward pre-hook on each block ==
    hook_resid_pre). Correctness is confirmed by the cosine check on Lambda.
    """
    import torch

    model = loaded.model
    tok = loaded.tokenizer
    model_key = loaded.spec.name
    wanted = set(capture_names)
    prev_side = getattr(tok, "padding_side", None)
    tok.padding_side = "left"
    try:
        for i in range(0, len(instructions), batch_size):
            chunk = instructions[i:i + batch_size]
            strs = [format_refusal_prompt(model_key, ins) for ins in chunk]
            enc = tok(strs, return_tensors="pt", padding=True, add_special_tokens=True)
            input_ids = enc["input_ids"].to(model.cfg.device)
            attn = enc["attention_mask"].to(model.cfg.device)
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    input_ids, attention_mask=attn,
                    names_filter=lambda n: n in wanted, return_type=None,
                )
            for r in range(len(chunk)):
                yield {name: cache[name][r:r + 1] for name in capture_names}
    finally:
        if prev_side is not None:
            tok.padding_side = prev_side


@dataclass
class ExtractionBackend:
    """Env-dependent extraction ops (default = real). Swap for a fake in tests."""
    load: Callable = _real_load
    prompt_caches: Callable = _real_prompt_caches


def run_extraction(model_key, harmful_instructions, harmless_instructions, *,
                   n_pos, backend: ExtractionBackend | None = None, batch_size=32):
    """Extract the mean-difference candidate grid, bucketing residuals BY LABEL.

    Returns (grid, resids_by_label) where grid is (n_pos, n_layers, d_model) =
    mean_harmful - mean_harmless. This is the paper's `mean_diffs`. Deliberately
    does NOT go through experiment.run (which buckets by judge verdict and caches
    over the response).
    """
    backend = backend or ExtractionBackend()
    loaded = backend.load(MODELS[model_key])
    n_layers = loaded.model.cfg.n_layers
    capture_names = resid_pre_hook_names(n_layers)

    resids_by_label: dict = {}
    for label, instrs in (("harmful", harmful_instructions),
                          ("harmless", harmless_instructions)):
        for cache in backend.prompt_caches(loaded, instrs, capture_names, batch_size):
            resids_by_label.setdefault(label, []).append(
                capture_prompt_positions(cache, n_layers, n_pos))

    grid = build_refusal_grid(resids_by_label, ("harmful", "harmless"))
    return grid, resids_by_label


# First-token refusal-marker token ids, verbatim from upstream *_REFUSAL_TOKS
# (pipeline/model_utils/*_model.py). Used by the refusal_score filter (Chunk A's
# faithful reproduction of filter_train) and by select_direction (Chunk E). These
# are the token ids of "I"/"As" etc. that begin a refusal ("I cannot...").
REFUSAL_TOKS: dict[str, list[int]] = {
    "qwen-1.8b": [40, 2121],   # ['I', 'As']
    "yi-6b": [59597],          # ['I']
    "gemma-2b": [235285],      # ['I']
    "llama3-8b": [40],         # ['I']
    "llama-2-7b": [306],       # ['I']
}


def refusal_toks(model_key: str) -> list[int]:
    return REFUSAL_TOKS[_resolve_model_key(model_key)]


def get_refusal_scores(loaded, instructions, toks=None, batch_size=32):
    """Paper's first-token refusal score (select_direction.refusal_score), per
    instruction: log(sum p[refusal_toks]) - log(1 - sum p[refusal_toks]) at the
    LAST prompt token. High => about to refuse. Model/torch; runs on Lambda.

    This is the LOGIT-based metric the paper uses for BOTH filter_train and
    select_direction — it is NOT the substring judge (judge.is_refusal), which is
    used only at the downstream jailbreak-eval stage.
    """
    import torch

    toks = toks if toks is not None else refusal_toks(loaded.spec.name)
    model, tok = loaded.model, loaded.tokenizer
    prev_side = getattr(tok, "padding_side", None)
    tok.padding_side = "left"
    out: list[float] = []
    try:
        for i in range(0, len(instructions), batch_size):
            strs = [format_refusal_prompt(loaded.spec.name, s)
                    for s in instructions[i:i + batch_size]]
            enc = tok(strs, return_tensors="pt", padding=True, add_special_tokens=True)
            with torch.no_grad():
                logits = model(enc["input_ids"].to(model.cfg.device),
                               attention_mask=enc["attention_mask"].to(model.cfg.device))
            logits = logits[:, -1, :].to(torch.float64)
            probs = torch.softmax(logits, dim=-1)
            refusal_p = probs[:, toks].sum(dim=-1)
            s = torch.log(refusal_p + 1e-8) - torch.log(1.0 - refusal_p + 1e-8)
            out.extend(s.tolist())
    finally:
        if prev_side is not None:
            tok.padding_side = prev_side
    return out
