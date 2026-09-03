"""Model loading, prompting, and generation (arch roadmap §3.4).

Models are declared as data (`ModelSpec` / `MODEL_CATALOG`); one loader handles
every HookedTransformer model. torch / transformer_lens are imported lazily, so
this module imports without the ML stack — only `load_model`/`generate` need it.

Faithful ports of the notebook: greedy generation, prompt+BOS stripping, and the
per-model chat-template flag (chat models get system+user; base models get the
raw prompt).
"""

from dataclasses import dataclass

from .config import ModelSpec
from .registry import register, MODELS


_THINK_CLOSE = "</think>"


def answer_text(response: str) -> str:
    """The model's answer with any reasoning trace stripped.

    Reasoning models (qwen3) emit ``<think>...</think>answer``; the answer is
    everything after the LAST ``</think>``. Responses without the marker are
    returned unchanged, so this is safe to apply to any model's output. Used to
    feed the JUDGE the answer rather than the reasoning trace (config
    ``strip_reasoning``); residual capture still uses the full response.
    """
    i = response.rfind(_THINK_CLOSE)
    return response[i + len(_THINK_CLOSE):].strip() if i != -1 else response


def get_device() -> str:
    """CUDA (RunPod/Lambda) > MPS (Apple) > CPU. Ports notebook `getDevice`."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class LoadedModel:
    """A loaded model plus the spec it came from and the device it's on."""

    model: object
    tokenizer: object
    spec: ModelSpec
    device: str


def load_model(spec: ModelSpec, device: str | None = None) -> LoadedModel:
    """Load a HookedTransformer in inference mode. Ports notebook `get_model`."""
    import torch
    from transformer_lens import HookedTransformer

    device = device or get_device()
    # Pin the weights when the spec names an immutable revision. Passed through
    # to the HF fetch, so the run loads the exact commit PREREG §3b froze rather
    # than whatever the branch points at today.
    extra = {"revision": spec.revision} if spec.revision else {}
    model = HookedTransformer.from_pretrained_no_processing(
        spec.hf_id,
        device=device,
        dtype=torch.float16,
        default_padding_side="left",
        output_hidden_states=True,
        **extra,
    )
    model.eval()
    model.to(device)
    if model.cfg.positional_embedding_type == "rotary" and model.cfg.n_ctx < 4096:
        # TransformerLens leaves `n_ctx` at its library default (2048) for some
        # models (observed on Qwen3-8B) rather than the model's real context
        # window. `apply_rotary`'s dynamic cache-extension clamps to `n_ctx`
        # instead of growing past it, so any prompt+generation over 2048 tokens
        # hits an out-of-bounds CUDA assert. Rotary caches are cheap (a
        # (n_ctx, rotary_dim) sin/cos table), so raise the ceiling well above
        # this repo's `max_tokens` budgets rather than truncate generations.
        model.cfg.n_ctx = 4096
    return LoadedModel(model=model, tokenizer=model.tokenizer, spec=spec, device=device)


def build_chat_messages(system: str, user: str) -> list[dict]:
    """System + user turns for a chat template (torch-free)."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def render_prompts(loaded: LoadedModel, prompts, system_prompt, *, template=None):
    """Return (token_lists, prompt_strs). Chat models get the system+user chat
    template; base models (`chat_template=False`) get the raw prompt.

    NOTE: this mirrors the notebook, where gemma/llama-3 ran with
    `apply_chat_template=False` (no system instruction). Revisit per-model if that
    turns out to matter — it's one `ModelSpec.chat_template` flag.

    `template` overrides both branches with a literal format string containing
    `{instruction}`, rendered verbatim with NO system turn. The refusal repro
    (arXiv:2406.11717) needs this: the paper formats with the model's raw chat
    template and no system turn, and neither branch above reproduces that for
    Qwen — `build_chat_messages` emits an *empty* system turn
    (`<|im_start|>system\\n<|im_end|>`, 20 tok), while dropping the system message
    makes HF inject its default "You are a helpful assistant." (26 tok). The
    paper's own string is 15 tok. That gap moved harmful/baseline refusal by
    -0.33 (see docs/05-refusal-repro.md §3).
    """
    tok = loaded.tokenizer
    token_lists, strs = [], []
    for p in prompts:
        if template is not None:
            s = template.format(instruction=p)
            token_lists.append(tok(s).input_ids)
            strs.append(s)
        elif loaded.spec.chat_template:
            msg = build_chat_messages(system_prompt, p)
            token_lists.append(tok.apply_chat_template(msg, tokenize=True, add_generation_prompt=True))
            strs.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
        else:
            token_lists.append(tok(p).input_ids)
            strs.append(p)
    return token_lists, strs


def _strip(loaded: LoadedModel, out_tokens, n_input: int) -> list[str]:
    """Decode only the newly generated tokens of each row.

    Replaces the notebook's character-based slice
    (``s[len(prompt_str) + len(bos):]``), which is wrong under left padding: a
    batch is padded to its longest member, so the decoded string is
    ``<pad>... <bos> <prompt> <response>`` and cutting `len(prompt)` *characters*
    off the front lands in the middle of the padding, leaving prompt tail and
    chat-template markup (``<|im_end|><|im_start|>assistant``) glued to the front
    of every response. Measured on the Log_103 anchor before this fix: 0/8
    responses shared even a single leading character with the archive
    (docs/04-parity.md, rung 2).

    Slicing by token index is exact instead of approximate: left padding makes the
    prompt occupy the same width `n_input` in *every* row, so `row[n_input:]` is
    precisely the generated continuation regardless of prompt length.
    """
    return [
        loaded.tokenizer.decode(row[n_input:], skip_special_tokens=True)
        for row in out_tokens
    ]


def generate(loaded: LoadedModel, prompts, max_new_tokens, system_prompt, *, template=None) -> list[str]:
    """Greedy generation. Ports notebook `normal_generation`."""
    _, strs = render_prompts(loaded, prompts, system_prompt, template=template)
    tokens = loaded.model.to_tokens(strs)  # left-padded to a uniform width
    out = loaded.model.generate(tokens, max_new_tokens=max_new_tokens,
                                do_sample=False, return_type="tokens")
    return _strip(loaded, out, tokens.shape[1])


def generate_with_cache(loaded: LoadedModel, prompts, max_new_tokens, system_prompt,
                        capture_names=None):
    """Return (responses, caches). The cache is taken over the *response* text —
    faithful to the notebook, where `batch_resids` calls `run_with_cache` on the
    stripped output. Feed each cache to `steering.capture_*`.

    `capture_names` is the list of hook points to retain. It matters a lot: an
    unfiltered `run_with_cache` keeps *every* hook point at every layer (~15x the
    tensors actually read) for every example in a batch simultaneously, which is
    what the notebook did and why it could not scale past small models. Filtering
    to the handful of names `capture` reads cuts both memory and time by an order
    of magnitude and is what lets a 14B model run at a usable batch size.

    Defaults to the `resid_pre` names used by `capture_mean` / `capture_last`; a
    method reading other hook points passes its own (see `SteeringMethod.names`).
    """
    import torch

    responses = generate(loaded, prompts, max_new_tokens, system_prompt)
    if capture_names is None:
        n_layers = loaded.model.cfg.n_layers
        capture_names = [f"blocks.{i}.hook_resid_pre" for i in range(n_layers)]
    wanted = set(capture_names)
    # `run_with_cache` (unlike `generate`, which is `@torch.inference_mode()`)
    # builds a full autograd graph by default — every layer's activations kept
    # alive for a `.backward()` we never call. For a long response over an 8B
    # model that alone can burn tens of GB independent of batch size, which is
    # what caused repeated OOMs here regardless of how small `batch_size` got.
    with torch.no_grad():
        caches = [
            loaded.model.run_with_cache(r, names_filter=lambda n: n in wanted)[1]
            for r in responses
        ]
    return responses, caches


def generate_with_hooks(loaded: LoadedModel, prompts, fwd_hooks, max_new_tokens, system_prompt,
                        *, template=None) -> list[str]:
    """Steered generation under `fwd_hooks` (build them with `steering.apply_*`).
    Ports notebook `batched_generation`."""
    _, strs = render_prompts(loaded, prompts, system_prompt, template=template)
    tokens = loaded.model.to_tokens(strs)
    with loaded.model.hooks(fwd_hooks):
        # temperature=0 is greedy: sample_logits early-returns argmax on 0.0, so
        # this matches `generate`'s do_sample=False. Verified against
        # transformer_lens 3.7 (docs/04-parity.md).
        out = loaded.model.generate(tokens, max_new_tokens=max_new_tokens, temperature=0)
    return _strip(loaded, out, tokens.shape[1])


# Catalog of known models (pure data). Adding a HookedTransformer model = one
# entry here; chat_template flags match the notebook (qwen/yi True; gemma/llama3 False).
MODEL_CATALOG = {
    "qwen-1.8b": ModelSpec("qwen-1.8b", "Qwen/Qwen1.5-1.8B-Chat", True, "1.8B", ["qwen"]),
    "qwen-7b":   ModelSpec("qwen-7b", "Qwen/Qwen1.5-7B-Chat", True, "7B", ["qwen"]),
    "qwen-14b":  ModelSpec("qwen-14b", "Qwen/Qwen1.5-14B-Chat", True, "14B", ["qwen"]),
    "yi-6b":     ModelSpec("yi-6b", "01-ai/Yi-6B-Chat", True, "6B"),
    "gemma-2b":  ModelSpec("gemma-2b", "google/gemma-2b-it", False, "2B"),
    "gemma-7b":  ModelSpec("gemma-7b", "google/gemma-7b-it", False, "7B"),
    "llama3-8b": ModelSpec("llama3-8b", "meta-llama/Meta-Llama-3-8B-Instruct", False, "8B"),
    # Added for the refusal-direction repro (arXiv:2406.11717); not used by the
    # legacy bias runs. chat_template=True because the refusal direction is defined
    # at post-instruction template positions (see src/bias_steer/refusal.py).
    "llama-2-7b": ModelSpec("llama-2-7b", "meta-llama/Llama-2-7b-chat-hf", True, "7B"),
    # The frozen submission model (docs/PREREG.md §3b; contract §12 A4). Arditi
    # ships no per-model artifact for it, which is exactly why G1 is defined
    # model-internally (contract §12 A6): the only thing G1 needs from
    # third_party/ is the prompt splits, and those are model-independent.
    "qwen3-8b": ModelSpec("qwen3-8b", "Qwen/Qwen3-8B", True, "8B", ["qwen"],
                          revision="b968826d9c46"),
}

for _name, _spec in MODEL_CATALOG.items():
    register(MODELS, _name, _spec)
