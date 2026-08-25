"""Tensor persistence (safetensors). Isolated here so the rest of the package
imports without torch/safetensors; these functions lazy-import them and are only
reached on a machine with the ML stack.

See docs/02-architecture-roadmap.md §8.2: tensors -> safetensors, not pickle.
"""


def save_vector(path, vector, *, n_layers, d_model) -> None:
    """Save a steering vector (n_layers, d_model) — the committed deliverable.

    Asserts the shape against the model's ground-truth dims at the persistence
    boundary (CLAUDE.md §6): the retracted DC-offset bug was a silently mis-shaped
    vector, so we refuse to persist one whose shape we can't vouch for."""
    assert tuple(vector.shape) == (n_layers, d_model), (
        f"steering vector: expected (n_layers, d_model) = {(n_layers, d_model)}, "
        f"got {tuple(vector.shape)}"
    )
    from safetensors.torch import save_file
    save_file({"vector": vector.contiguous()}, str(path))


def load_vector(path):
    from safetensors.torch import load_file
    return load_file(str(path))["vector"]


def save_residuals(path, resids_by_label: dict, *, n_layers, d_model) -> None:
    """Save per-verdict residual stacks (bulky; git-ignored). One tensor per label,
    each (n_examples, n_layers, d_model).

    Asserts every per-example residual is (n_layers, d_model) before `torch.stack`
    (CLAUDE.md §6): a mis-shaped residual would otherwise stack into a wrong-shaped
    tensor and surface far from its cause — this class of bug is silent. The check
    runs before `import torch` so it validates inputs ahead of the heavy import."""
    for label, items in resids_by_label.items():
        for i, item in enumerate(items):
            assert tuple(item.shape) == (n_layers, d_model), (
                f"residual {label}[{i}]: expected (n_layers, d_model) = "
                f"{(n_layers, d_model)}, got {tuple(item.shape)}"
            )

    import torch
    from safetensors.torch import save_file

    tensors = {
        label: torch.stack(items).contiguous()
        for label, items in resids_by_label.items()
        if items
    }
    if tensors:
        save_file(tensors, str(path))
