"""Is the cosine shortfall (0.90 vs target 0.999) caused by the fp16 forward?

Re-extract the candidate grid with a float32 forward and recompute cosine vs the
paper's mean_diffs.pt. No filtering (it moved the selected cell by <0.01).
"""
import sys; sys.path.insert(0, "/lambda/nfs/farhan-algoverse-summer26/Algoverse-Bias-Steering")
import torch
from src.bias_steer.registry import MODELS
from src.bias_steer import models, refusal
from src.bias_steer import refusal_extract as rx

DTYPE = getattr(torch, sys.argv[1]) if len(sys.argv) > 1 else torch.float32
_orig = models.load_model

def load_f32(spec, device=None):
    from transformer_lens import HookedTransformer
    dev = device or models.get_device()
    m = HookedTransformer.from_pretrained_no_processing(
        spec.hf_id, device=dev, dtype=DTYPE,
        default_padding_side="left", output_hidden_states=True)
    m.eval(); m.to(dev)
    return models.LoadedModel(model=m, tokenizer=m.tokenizer, spec=spec, device=dev)

models.load_model = load_f32

key = "qwen-1.8b"
loaded = load_f32(MODELS[key])
n_pos, n_layers = rx.n_positions(loaded), loaded.model.cfg.n_layers
s = rx.load_and_sample_repro(n_train=128, n_val=32)
grid, _ = rx.run_extraction(key, s["harmful_train"], s["harmless_train"], n_pos=n_pos, batch_size=16)
grid = grid.to(torch.float64).cpu()
theirs = refusal.artifacts.load_pt_tensor(
    refusal.artifact_dir(refusal.MODEL_TO_RUN_DIR[key]) / "generate_directions" / "mean_diffs.pt"
).to(torch.float64).cpu()
cos = torch.nn.functional.cosine_similarity(
    grid.reshape(-1, grid.shape[-1]), theirs.reshape(-1, theirs.shape[-1]), dim=-1).reshape(n_pos, n_layers)
nz = cos[:, 1:]  # layer 0 is zero in THEIR grid too
print(f"\n=== dtype={DTYPE} (no filter) ===")
print(f"selected cell (pos=-2, layer=15) cosine = {cos[3,15].item():.5f}")
print(f"excluding layer 0:  min={nz.min():.4f}  mean={nz.mean():.4f}  max={nz.max():.4f}")
print("  fp16 reference:   selected 0.89165, mean(all incl L0) 0.6465")
