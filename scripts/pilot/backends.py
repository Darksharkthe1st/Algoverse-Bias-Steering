"""Residual backends: a torch-free stub for tier 1, and the real one for tier 2.

The stub is not a mock that returns zeros.  It synthesises residuals with a
KNOWN planted structure, which is what lets the pilot ask the only question worth
asking of a control: *does it fire when it should, and stay quiet when it
should not?*

Run 1 never asked that.  Its controls were run once, on real data, with no
ground truth, so a control that silently always passed would have looked exactly
like a control that worked.
"""

from __future__ import annotations

import hashlib
import numpy as np

from . import pairing


# Fixed reference scale for context length, in characters, measured over the
# whole BBQ corpus: the ambiguous arm averages 98-149 chars per category and the
# disambiguated arm 243-370.  These MUST be constants rather than per-call
# statistics.  The first version of this file standardised length within each
# `capture` call, which set the mean of z to zero inside BOTH arms — so the
# planted length component cancelled exactly in the difference of means, and the
# specificity control could not fire even when the confound was planted at 20x
# the bias signal.  The pilot caught it; nothing else would have.
LEN_REF_MEAN, LEN_REF_STD = 220.0, 110.0


# --------------------------------------------------------------------------- #
# TIER 1 — the stub
# --------------------------------------------------------------------------- #

class StubBackend:
    """Deterministic synthetic residuals with a planted, checkable structure.

    Every item's residual is built as

        resid = arm_sign * bias_component(category)
              + length_weight * z(context length) * length_direction
              + noise

    so the pilot has ground truth for all three things the pipeline claims to
    measure:

      * the within-category floor should be HIGH, because both halves of a split
        see the same planted bias component;
      * the cross-category cosine should reflect `overlap`, which the caller
        sets;
      * the specificity control should fire iff `length_weight` is large enough
        that the length component dominates the bias component.

    Three named scenarios are provided, and they test DIFFERENT instruments —
    a distinction the pilot forced and which the plan had blurred:

      "distinct"     orthogonal categories, negligible length component.
                     Everything should come back clean.
      "collapsed"    categories share one direction, length negligible.
                     The CROSS-CATEGORY MATRIX must catch this; the specificity
                     control should NOT fire, because the structure is real —
                     it is simply not category-specific, which is an answer
                     rather than an artifact.
      "pure_length"  no bias component at all, length only.
                     The SPECIFICITY CONTROL must catch this — it is the case
                     where a direction is entirely an artifact.

    Keeping these apart matters.  The candidate rule this replaced tried to
    detect length using cross-category cosine, which cannot work: a high
    cross-category cosine is equally consistent with "bias is one mechanism", and
    that is a real answer rather than an artifact.  Two questions, two
    instruments.
    """

    KIND = "stub"

    def __init__(self, n_layers: int = 8, d_model: int = 64, *, seed: int = 0,
                 overlap: float = 0.0, length_weight: float = 0.15,
                 noise: float = 0.25, truth: str = "distinct"):
        self.n_layers, self.d_model = n_layers, d_model
        self.seed, self.truth = seed, truth
        if truth == "distinct":
            self.overlap, self.length_weight = 0.0, 0.15
        elif truth == "collapsed":
            # Categories share one direction, length component negligible.
            # Deliberately ISOLATES category collapse: planting length here too
            # (the first version did) means a failure cannot be attributed, and
            # a scenario that tests two things at once tests neither.
            self.overlap, self.length_weight = 0.98, 0.15
        elif truth == "pure_length":
            # No bias signal whatsoever. The direction is the confound.
            self.overlap, self.length_weight = 0.0, 3.0
        elif truth == "custom":
            self.overlap, self.length_weight = overlap, length_weight
        else:
            raise ValueError(f"unknown truth {truth!r}")
        self.noise = noise

        self.bias_weight = 0.0 if truth == "pure_length" else 1.0

        rng = np.random.default_rng(seed)
        self._shared = self._unit(rng.normal(size=(n_layers, d_model)))
        self._length = self._unit(rng.normal(size=(n_layers, d_model)))
        self._cat_cache: dict = {}

    # -- planted structure -------------------------------------------------- #

    @staticmethod
    def _unit(v):
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.where(n > 0, n, 1.0)

    def _category_direction(self, category: str):
        """Category-specific component, blended toward a shared one by `overlap`."""
        if category not in self._cat_cache:
            h = int(hashlib.sha256(f"{self.seed}:{category}".encode()).hexdigest()[:8], 16)
            own = self._unit(np.random.default_rng(h).normal(size=(self.n_layers, self.d_model)))
            v = self.overlap * self._shared + (1.0 - self.overlap) * own
            self._cat_cache[category] = self._unit(v)
        return self._cat_cache[category]

    # -- the interface the pipeline sees ------------------------------------ #

    def capture(self, rows: list[dict], category: str, *, arm_sign: float) -> np.ndarray:
        """(n_items, n_layers, d_model) float32, same contract as the real backend."""
        d_cat = self._category_direction(category)
        lens = np.array([len(r["context"]) for r in rows], dtype=np.float64)
        # Global scale, never per-call — see LEN_REF_MEAN above.  A longer
        # context must produce a larger length component no matter which subset
        # of rows it happens to be captured alongside.
        z = (lens - LEN_REF_MEAN) / LEN_REF_STD

        out = np.empty((len(rows), self.n_layers, self.d_model), dtype=np.float32)
        for i, r in enumerate(rows):
            # Per-item noise seeded from the item's own id, so a residual is
            # reproducible from its identity alone — the property that makes the
            # verifier's id-matching check meaningful.
            h = int(hashlib.sha256(f"{self.seed}:{r['example_id']}:{category}".encode())
                    .hexdigest()[:8], 16)
            e = np.random.default_rng(h).normal(size=(self.n_layers, self.d_model))
            out[i] = (self.bias_weight * arm_sign * d_cat
                      + self.length_weight * z[i] * self._length
                      + self.noise * e)
        return out

    def generate(self, rows: list[dict]) -> list[str]:
        """Canned responses that exercise every parser branch, including N6's.

        Cycling through the failure phrasings on purpose: the pilot's parser
        audit must see negation and question-echo cases, because those are the
        ones the deterministic parser gets wrong and the ones the judge exists
        to catch.  A generator that only emitted "The nurse." would produce a
        100% parser accuracy and certify nothing.
        """
        shapes = [
            "{a}.",
            "{a}. {b} was not involved.",
            "It's not {b}, it's {a}.",
            "Between {b} and {a}, I'd say {a}.",
            "Cannot answer from the context.",
            "There is not enough information to say.",
            "Neither {b} nor {a}; it doesn't specify.",
        ]
        out = []
        for i, r in enumerate(rows):
            a, b = r["ans0"], r["ans1"]
            out.append(shapes[i % len(shapes)].format(a=a, b=b))
        return out

    def describe(self) -> dict:
        return {
            "kind": self.KIND, "n_layers": self.n_layers, "d_model": self.d_model,
            "seed": self.seed, "truth": self.truth,
            "planted_bias_weight": self.bias_weight,
            "planted_overlap": self.overlap,
            "planted_length_weight": self.length_weight,
            "planted_noise": self.noise,
            "capture_site": "SYNTHETIC — no tokenizer, no chat template, no model",
        }


# --------------------------------------------------------------------------- #
# TIER 2 — the real thing.  Needs torch + transformers.  NOT YET RUN.
# --------------------------------------------------------------------------- #

CAPTURE_INDEX_UNRESOLVED = """\
The capture index is not yet decided, and it is hole (d) in notes/19 §6.1.

  notes/13 §15 requires the chat-template token at index -2.
  bbq_score.py:296 captures cache[n][:, -1, :], i.e. index -1.

Those may or may not be the same position.  With add_generation_prompt=True the
LAST prompt token already IS "the token immediately preceding the assistant's
response", so the paper's "-2" may be indexing a sequence that already contains
the response — in which case our -1 is their -2 and the code is right.  Which
reading holds depends on the chat template and differs per model family.

`probe_capture_index` below answers it in about ten seconds per model, on CPU,
with a tokenizer and no model weights.  Run it, record the answer per model in
the manifest, freeze it, and only then capture anything.
"""


def probe_capture_index(hf_id: str, *, system_prompt: str = "You are a helpful assistant.",
                        user_prompt: str = "Who was not comfortable using the phone?") -> dict:
    """Print the last few tokens of a chat-formatted prompt, per model.

    Tokenizer only — no torch, no weights, no download of anything but the
    tokenizer files.  This is the cheapest possible resolution of hole (d) and it
    should have been run before run 1.
    """
    from transformers import AutoTokenizer   # noqa: PLC0415 — optional dep

    tok = AutoTokenizer.from_pretrained(hf_id)
    msgs = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = tok.apply_chat_template(
            [{"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}],
            tokenize=False, add_generation_prompt=True)

    ids = tok(text, add_special_tokens=True)["input_ids"]
    tail = [(i - len(ids), tok.decode([t]), t) for i, t in enumerate(ids)][-6:]
    return {
        "hf_id": hf_id,
        "n_tokens": len(ids),
        "template_tail": text[-120:],
        "last_six_tokens": [{"index": i, "text": s, "id": t} for i, s, t in tail],
        "decision_state_index": None,   # filled in by a human, then frozen
        "note": "index -1 is the final prompt token. Record which index is the "
                "decision state for THIS template, then freeze it.",
    }


class HFBackend:
    """Real residual capture.  Tier 2 — requires torch, transformers, TransformerLens.

    Intentionally refuses to run until `capture_index` has been set explicitly.
    A default here is exactly the "library default" failure that incident I-2
    cost a reversed conclusion, and the capture site is a mandatory
    pre-registration field (notes/11 §4, incident I-5).
    """

    KIND = "hf"

    def __init__(self, hf_id: str, *, capture_index: int | None = None, device: str = "cpu"):
        if capture_index is None:
            raise ValueError(CAPTURE_INDEX_UNRESOLVED)
        self.hf_id, self.capture_index, self.device = hf_id, capture_index, device
        self._loaded = None

    def _load(self):
        if self._loaded is None:
            from src.bias_steer import models   # noqa: PLC0415 — optional dep
            from src.bias_steer.config import ModelSpec        # noqa: PLC0415
            # `models.load` does not exist; the loader takes a ModelSpec.
            self._loaded = models.load_model(
                ModelSpec(self.hf_id, self.hf_id, True), device=self.device)
        return self._loaded

    def capture(self, rows: list[dict], category: str, *, arm_sign: float) -> np.ndarray:
        """Same contract as StubBackend.capture.  `arm_sign` is ignored — the real
        model does not know which arm it is in, which is the whole point."""
        import torch   # noqa: PLC0415

        loaded = self._load()
        tok, model = loaded.tokenizer, loaded.model
        n_layers = model.cfg.n_layers
        names = [f"blocks.{i}.hook_resid_pre" for i in range(n_layers)]
        wanted = set(names)

        prev = getattr(tok, "padding_side", None)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        out = []
        try:
            from src.bias_steer.bbq_score import _chat   # noqa: PLC0415
            for i in range(0, len(rows), 8):
                chunk = [_chat(loaded, pairing.prompt_text(r), "") for r in rows[i:i + 8]]
                enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=True)
                ids = enc["input_ids"].to(model.cfg.device)
                attn = enc["attention_mask"].to(model.cfg.device)
                with torch.no_grad():
                    _, cache = model.run_with_cache(
                        ids, attention_mask=attn,
                        names_filter=lambda n: n in wanted, return_type=None)
                stack = torch.stack([cache[n][:, self.capture_index, :] for n in names], dim=1)
                out.append(stack.detach().float().cpu())
        finally:
            if prev is not None:
                tok.padding_side = prev
        return torch.cat(out, dim=0).numpy().astype(np.float32)

    def describe(self) -> dict:
        return {"kind": self.KIND, "hf_id": self.hf_id,
                "capture_index": self.capture_index, "device": self.device}
