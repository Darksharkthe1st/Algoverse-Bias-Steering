"""Tensor persistence (safetensors). Isolated here so the rest of the package
imports without torch/safetensors; these functions lazy-import them and are only
reached on a machine with the ML stack.

See docs/02-architecture-roadmap.md §8.2: tensors -> safetensors, not pickle.
"""


def save_vector(path, vector) -> None:
    """Save a steering vector (n_layers, d_model) — the committed deliverable."""
    from safetensors.torch import save_file
    save_file({"vector": vector.contiguous()}, str(path))


def load_vector(path):
    from safetensors.torch import load_file
    return load_file(str(path))["vector"]


def load_pt_tensor(path, *, device: str = "cpu"):
    """Load a single tensor from a PyTorch `.pt` (pickle) file.

    Generic counterpart to `load_vector` (which reads this repo's own
    safetensors). Use it for externally-produced `.pt` vectors — e.g. the
    published refusal directions in `third_party/`. `weights_only=True` so no
    pickle code executes; falls back for torch versions predating the kwarg.
    """
    import torch

    try:
        return torch.load(str(path), map_location=device, weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location=device)


def save_residuals(path, resids_by_label: dict) -> None:
    """Save per-verdict residual stacks (bulky; git-ignored). One tensor per label,
    each (n_examples, n_layers, d_model)."""
    import torch
    from safetensors.torch import save_file

    tensors = {
        label: torch.stack(items).contiguous()
        for label, items in resids_by_label.items()
        if items
    }
    if tensors:
        save_file(tensors, str(path))
