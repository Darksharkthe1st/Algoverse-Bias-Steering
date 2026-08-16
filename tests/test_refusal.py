"""Refusal-direction repro (arXiv:2406.11717): artifact loading + catalog wiring.

    python3 tests/test_refusal.py

Structural tests (mapping, catalog membership) run anywhere. Tests that actually
`torch.load` a `direction.pt` are gated twice: they skip if torch is absent AND
if the artifacts have not been fetched (run scripts/fetch_refusal_artifacts.py).
"""

import glob
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import src.bias_steer as bs  # noqa: E402
from src.bias_steer import refusal, registry, artifacts, steering, datasets, judge  # noqa: E402
from src.bias_steer.config import DatasetSpec, JudgeSpec  # noqa: E402

try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

# Ground-truth (layer, pos) from each model's committed direction_metadata.json,
# verified against the mean_diffs storage-offset provenance in Chunk 0.
_EXPECTED_META = {
    "qwen-1_8b-chat": (15, -2),
    "gemma-2b-it": (10, -2),
    "yi-6b-chat": (20, -5),
    "meta-llama-3-8b-instruct": (12, -5),
    "llama-2-7b-chat-hf": (14, -1),
}
_EXPECTED_DMODEL = {
    "qwen-1_8b-chat": 2048, "gemma-2b-it": 2048, "yi-6b-chat": 4096,
    "meta-llama-3-8b-instruct": 4096, "llama-2-7b-chat-hf": 4096,
}


# ---------------------------------------------------------------- structural

def test_run_dir_mapping_targets_registered_models():
    # Every upstream run dir maps to a model that exists in the catalog, so a
    # loaded direction can always be paired with a loadable model.
    for run_dir, model_key in refusal.RUN_DIR_TO_MODEL.items():
        assert model_key in registry.MODELS, f"{run_dir} -> {model_key} not in MODELS"


def test_llama2_catalog_entry():
    spec = registry.MODELS["llama-2-7b"]
    assert spec.hf_id == "meta-llama/Llama-2-7b-chat-hf"
    assert spec.chat_template is True  # refusal direction needs the chat template


def test_resolve_accepts_both_key_forms():
    assert refusal._resolve("qwen-1.8b") == ("qwen-1_8b-chat", "qwen-1.8b")
    assert refusal._resolve("qwen-1_8b-chat") == ("qwen-1_8b-chat", "qwen-1.8b")


def test_resolve_rejects_unknown():
    try:
        refusal._resolve("gpt-4")
    except KeyError:
        return
    raise AssertionError("expected KeyError for unknown model")


def test_missing_artifact_error_names_fetch_command():
    # Resolvable model, but point the loader at a run dir with no file: the error
    # must tell the user how to fetch. Use a model unlikely to be fetched here.
    if "gemma-7b" in refusal.available_run_dirs():
        print("      (skipped: unexpected artifact present)")
        return
    # gemma-7b isn't in the mapping; craft the check via a resolvable-but-absent one.
    # If nothing is fetched, any known key triggers FileNotFoundError.
    if refusal.available_run_dirs():
        print("      (skipped: artifacts fetched; missing-file path not exercised)")
        return
    if not _HAS_TORCH:
        print("      (skipped: torch not installed)")
        return
    try:
        refusal.load_refusal_direction("qwen-1.8b")
    except FileNotFoundError as e:
        assert "fetch_refusal_artifacts.py" in str(e)
        return
    raise AssertionError("expected FileNotFoundError naming the fetch script")


# ---------------------------------------------------------------- torch + data gated

def test_generic_pt_loader_reads_direction_tensor():
    # artifacts.load_pt_tensor is the reusable, refusal-agnostic loader for
    # external .pt vectors; exercise it directly on a fetched file.
    if not _HAS_TORCH:
        print("      (skipped: torch not installed)")
        return
    fetched = refusal.available_run_dirs()
    if not fetched:
        print("      (skipped: no artifacts fetched)")
        return
    import torch
    path = refusal.artifact_dir(fetched[0]) / "direction.pt"
    t = artifacts.load_pt_tensor(path)
    assert isinstance(t, torch.Tensor) and t.ndim == 1 and t.numel() > 0


def test_load_directions_shape_and_provenance():
    if not _HAS_TORCH:
        print("      (skipped: torch not installed)")
        return
    fetched = refusal.available_run_dirs()
    if not fetched:
        print("      (skipped: no artifacts fetched — run scripts/fetch_refusal_artifacts.py)")
        return
    for run_dir in fetched:
        rd = refusal.load_refusal_direction(run_dir)
        # shape: a single (d_model,) vector
        assert rd.direction.ndim == 1, f"{run_dir}: expected 1-D, got {rd.direction.shape}"
        assert rd.direction.dtype.is_floating_point
        # metadata matches the committed ground truth
        if run_dir in _EXPECTED_META:
            assert (rd.layer, rd.pos) == _EXPECTED_META[run_dir], \
                f"{run_dir}: (layer,pos)={ (rd.layer, rd.pos) } != {_EXPECTED_META[run_dir]}"
            assert rd.d_model == _EXPECTED_DMODEL[run_dir], \
                f"{run_dir}: d_model={rd.d_model} != {_EXPECTED_DMODEL[run_dir]}"
        # the raw direction is non-degenerate
        assert float(rd.direction.norm()) > 0
        # loaded model_key is a catalog model
        assert rd.model_key in registry.MODELS
        print(f"      {run_dir:28s} d_model={rd.d_model} layer={rd.layer} pos={rd.pos} "
              f"|r|={float(rd.direction.norm()):.3f}")


# ---------------------------------------------------------------- steering methods (structural)

class _StubCfg:
    def __init__(self, n): self.n_layers = n


class _StubModel:
    def __init__(self, n): self.cfg = _StubCfg(n)


def test_all_resid_stream_hook_names():
    assert steering.all_resid_stream_hook_names(2) == [
        "blocks.0.hook_resid_pre", "blocks.0.hook_attn_out", "blocks.0.hook_mlp_out",
        "blocks.1.hook_resid_pre", "blocks.1.hook_attn_out", "blocks.1.hook_mlp_out",
    ]


def test_ablation_registered_as_method():
    m = registry.METHODS["ablation"]
    assert m.apply is steering.apply_directional_ablation


def test_ablation_hook_count_and_names_without_math():
    if not _HAS_TORCH:
        print("      (skipped: torch not installed)")
        return
    import torch
    hooks = steering.apply_directional_ablation(_StubModel(3), torch.ones(4))
    assert len(hooks) == 9  # 3 layers x 3 hook points
    assert [n for n, _ in hooks] == steering.all_resid_stream_hook_names(3)


def test_actadd_single_hook_name():
    if not _HAS_TORCH:
        print("      (skipped: torch not installed)")
        return
    import torch
    hooks = steering.apply_actadd_single(_StubModel(24), torch.ones(4), coeff=1.0, layer=15)
    assert len(hooks) == 1
    assert hooks[0][0] == "blocks.15.hook_resid_pre"


# ---------------------------------------------------------------- steering methods (math)

def test_ablation_removes_component_along_direction():
    if not _HAS_TORCH:
        print("      (skipped: torch not installed)")
        return
    import torch
    direction = torch.tensor([3.0, 0.0, 0.0, 0.0])          # r̂ = e0
    value = torch.tensor([[[2.0, 5.0, 7.0, 1.0]]])          # (1,1,4)
    _, fn = steering.apply_directional_ablation(_StubModel(1), direction)[0]
    out = fn(value.clone(), None)
    # component along r̂ zeroed; orthogonal complement preserved
    assert torch.allclose(out, torch.tensor([[[0.0, 5.0, 7.0, 1.0]]]), atol=1e-6)
    r_hat = steering.unit_direction(direction)
    assert torch.allclose(r_hat.norm(), torch.tensor(1.0), atol=1e-6)
    assert abs(float(out.flatten() @ r_hat)) < 1e-5


def test_ablation_is_scale_invariant_and_idempotent():
    if not _HAS_TORCH:
        print("      (skipped: torch not installed)")
        return
    import torch
    value = torch.tensor([[[2.0, 5.0, 7.0, 1.0]]])
    small = steering.apply_directional_ablation(_StubModel(1), torch.tensor([3.0, 0, 0, 0]))[0][1]
    big = steering.apply_directional_ablation(_StubModel(1), torch.tensor([9000.0, 0, 0, 0]))[0][1]
    out_small = small(value.clone(), None)
    out_big = big(value.clone(), None)
    assert torch.allclose(out_small, out_big, atol=1e-5)        # only the direction, not its scale, matters
    once = small(value.clone(), None)
    twice = small(once.clone(), None)
    assert torch.allclose(once, twice, atol=1e-6)               # projecting an orthogonal vector is a no-op


def test_actadd_uses_raw_vector_as_dose():
    if not _HAS_TORCH:
        print("      (skipped: torch not installed)")
        return
    import torch
    direction = torch.tensor([3.0, 0.0, 0.0, 0.0])             # norm 3, deliberately un-normalized
    value = torch.zeros(1, 1, 4)
    _, fn = steering.apply_actadd_single(_StubModel(1), direction, coeff=2.0, layer=0)[0]
    out = fn(value.clone(), None)
    assert torch.allclose(out, torch.tensor([[[6.0, 0.0, 0.0, 0.0]]]), atol=1e-6)
    # dose magnitude = |coeff| * ||raw direction|| (would be |coeff| if normalized)
    assert torch.allclose(out.flatten().norm(), torch.tensor(6.0), atol=1e-6)


def test_hooks_cast_direction_to_activation_dtype():
    if not _HAS_TORCH:
        print("      (skipped: torch not installed)")
        return
    import torch
    direction = torch.ones(4, dtype=torch.float32)             # vector fp32
    value = torch.zeros(1, 1, 4, dtype=torch.float16)          # activations fp16
    _, add = steering.apply_actadd_single(_StubModel(1), direction, coeff=1.0, layer=0)[0]
    out = add(value.clone(), None)
    assert out.dtype == torch.float16                          # activation dtype preserved
    _, ablate = steering.apply_directional_ablation(_StubModel(1), direction)[0]
    assert ablate(torch.ones(1, 1, 4, dtype=torch.float16), None).dtype == torch.float16


# ---------------------------------------------------------------- refusal metric

def test_refusal_prefixes_and_matching():
    assert len(judge.REFUSAL_PREFIXES) == 12
    assert judge.is_refusal("I'm sorry, but I cannot help with that.")
    assert judge.is_refusal("AS AN AI language model, I can't")   # case-insensitive
    assert not judge.is_refusal("Sure! Here is how you do it.")
    assert not judge.is_refusal("")


def test_refusal_judge_labels():
    out = judge.refusal_substring_judge(["I cannot do that", "Sure, here you go"])
    assert out == [judge.REFUSAL, judge.COMPLIANCE]
    # honors custom labels from a JudgeSpec (index 0 = compliance, last = refusal)
    spec = JudgeSpec(name="refusal_substring", labels=["compliance", "refusal"])
    assert judge.refusal_substring_judge(["I apologize"], None, spec) == ["refusal"]
    assert "refusal_substring" in registry.JUDGES


def test_metric_reproduces_committed_substring_labels():
    # Zero-GPU correctness gate: our is_refusal must reproduce the paper's own
    # per-response is_jailbreak_substring_matching (= int(not is_refusal)) and the
    # aggregate substring_matching_success_rate, across EVERY fetched
    # model x condition evaluations file.
    files = sorted(glob.glob(str(refusal._ARTIFACT_ROOT / "*" / "completions" / "*_evaluations.json")))
    if not files:
        print("      (skipped: no artifacts fetched)")
        return
    total, mismatches, checked = 0, 0, 0
    for f in files:
        d = json.load(open(f))
        recs = d["completions"]
        mine = [int(not judge.is_refusal(r["response"])) for r in recs]
        for r, m in zip(recs, mine):
            if m != int(r["is_jailbreak_substring_matching"]):
                mismatches += 1
        assert abs(sum(mine) / len(mine) - d["substring_matching_success_rate"]) < 1e-9, \
            f"{os.path.basename(f)}: aggregate rate mismatch"
        total += len(recs); checked += 1
    assert mismatches == 0, f"{mismatches} per-record label mismatches"
    print(f"      reproduced {total} labels across {checked} files (0 mismatches)")


def test_refusal_eval_loader_prompt_parity():
    if not refusal.available_run_dirs():
        print("      (skipped: no artifacts fetched)")
        return
    spec = DatasetSpec(name="refusal_eval")
    spec.harm, spec.source_model = "harmful", "qwen-1_8b-chat"
    exs = datasets.load_refusal_eval(spec)
    assert len(exs) == 100
    assert all(e.metadata["harm"] == "harmful" for e in exs)
    src = json.loads((refusal.artifact_dir("qwen-1_8b-chat") / "completions"
                      / "jailbreakbench_baseline_completions.json").read_text())
    assert exs[0].prompt == src[0]["prompt"], "eval prompt drifted from the committed file"
    assert "refusal_eval" in registry.DATASETS


# ---------------------------------------------------------------- comparison harness

def test_compare_rates_math_and_tolerance():
    from src.bias_steer import refusal_compare as rc
    ours = {"harmful/baseline": {"refusal_rate": 0.90}, "harmful/ablation": {"refusal_rate": 0.10}}
    theirs = {"harmful/baseline": {"refusal_rate": 0.88}, "harmful/ablation": {"refusal_rate": 0.12}}
    rows = rc.compare_rates(ours, theirs, tol=0.05)
    assert [r["condition"] for r in rows] == ["harmful/baseline", "harmful/ablation"]  # canonical order
    d = {r["condition"]: r for r in rows}
    assert abs(d["harmful/ablation"]["delta"] - (-0.02)) < 1e-9
    assert d["harmful/ablation"]["within_tol"] is True
    # a large gap fails tolerance
    big = rc.compare_rates({"harmless/actadd": {"refusal_rate": 0.2}},
                           {"harmless/actadd": {"refusal_rate": 0.9}}, tol=0.05)
    assert big[0]["within_tol"] is False
    # only arms present in BOTH are compared
    assert rc.compare_rates({"harmful/baseline": {"refusal_rate": 0.5}}, {}) == []


def test_paper_rates_matches_recomputed_from_committed_responses():
    # Threads the whole harness through the paper's own data, torch-free: for each
    # fetched model x arm, paper_rates (from their aggregate) must equal the rate we
    # recompute by running our is_refusal over their committed responses.
    from src.bias_steer import refusal_compare as rc
    fetched = refusal.available_run_dirs()
    if not fetched:
        print("      (skipped: no artifacts fetched)")
        return
    checked = 0
    for run_dir in fetched:
        theirs = rc.paper_rates(run_dir)
        comp_dir = refusal.artifact_dir(run_dir) / "completions"
        for cond, ev_file in rc.ARM_TO_EVAL_FILE.items():
            if cond not in theirs:
                continue
            recs = json.loads((comp_dir / ev_file).read_text())["completions"]
            recomputed = sum(judge.is_refusal(r["response"]) for r in recs) / len(recs)
            assert abs(recomputed - theirs[cond]["refusal_rate"]) < 1e-9, \
                f"{run_dir}/{cond}: paper {theirs[cond]['refusal_rate']} != recomputed {recomputed}"
            checked += 1
    assert checked > 0
    print(f"      paper_rates == our metric on {checked} model×arm rates (0 drift)")


# ---------------------------------------------------------------- repro flow (fake backend)

def test_run_refusal_end_to_end_with_fake_backend():
    # Whole flow with NO model and NO API: a fake backend simulates the physics
    # (harmful refuses at baseline, complies under ablation/act-add; harmless
    # complies at baseline, refuses under act-add(+)). Needs torch only because
    # the real steering.apply_* build the hooks from a tensor direction.
    if not _HAS_TORCH:
        print("      (skipped: torch not installed)")
        return
    import tempfile
    import torch
    from src.bias_steer import experiment_refusal as er
    from src.bias_steer.config import ExperimentConfig, Coeffs
    from src.bias_steer.schema import Example

    class _Model:
        class cfg:  # noqa: N801
            n_layers = 3

    class _Loaded:
        model = _Model()

    def fake_load(spec):
        return _Loaded()

    def fake_load_direction(model_key):
        return refusal.RefusalDirection(model_key=model_key, run_dir="qwen-1_8b-chat",
                                        layer=2, pos=-1, direction=torch.ones(4))

    def fake_load_eval(run_dir, harm):
        tag = harm.upper()
        return [Example(id=f"{harm}-{i}", prompt=f"{tag}-{i}", metadata={"category": harm}) for i in range(2)]

    def fake_generate(loaded, prompts, max_tokens, sys_prompt):
        # baseline: harmful refuses, harmless complies
        return ["I'm sorry, I cannot." if p.startswith("HARMFUL") else "Sure, here you go." for p in prompts]

    def fake_generate_with_hooks(loaded, prompts, hooks, max_tokens, sys_prompt):
        # harmful under any intervention -> complies; harmless under act-add(+) -> refuses
        return ["Sure, here you go." if p.startswith("HARMFUL") else "I'm sorry, I cannot." for p in prompts]

    backend = er.RefusalBackend(
        load=fake_load, generate=fake_generate, generate_with_hooks=fake_generate_with_hooks,
        load_direction=fake_load_direction, load_eval=fake_load_eval,
    )
    cfg = ExperimentConfig(
        label="fake refusal", models=["qwen-1.8b"],
        dataset=DatasetSpec(name="refusal_eval"),
        judge=JudgeSpec(name="refusal_substring", labels=["compliance", "refusal"]),
        coeffs=Coeffs(opinion=1.0, neutral=1.0), method="ablation", batch_size=16,
    )
    with tempfile.TemporaryDirectory() as tmp:
        results = er.run_refusal(cfg, backend=backend, runs_dir=tmp)
        assert len(results) == 1
        rr = results[0]
        rates = {c: s["refusal_rate"] for c, s in rr.rates.items()}
        # the paper's expected directions, exactly:
        assert rates["harmful/baseline"] == 1.0     # refuses without intervention
        assert rates["harmful/ablation"] == 0.0     # ablation bypasses refusal
        assert rates["harmful/actadd"] == 0.0       # act-add(-) bypasses too
        assert rates["harmless/baseline"] == 0.0    # complies normally
        assert rates["harmless/actadd"] == 1.0      # act-add(+) induces refusal
        # artifacts written
        assert rr.results_csv.exists() and rr.summary_md.exists()
        assert (rr.dir / "manifest.json").exists()
        assert (Path(tmp) / "index.csv").exists()
        # results.csv has one row per (example, arm): 2 harmful*3 + 2 harmless*2 = 10
        import csv as _csv
        rows = list(_csv.DictReader(open(rr.results_csv)))
        assert len(rows) == 10
        # comparison harness is wired in: with qwen artifacts fetched, run produces
        # a 5-arm our-vs-paper diff and writes it into the summary.
        if "qwen-1_8b-chat" in refusal.available_run_dirs():
            assert rr.comparison is not None and len(rr.comparison) == 5
            assert "vs. paper" in rr.summary_md.read_text()
        print(f"      5 arms scored; {len(rows)} result rows; rates as paper predicts")


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed"
          + ("" if _HAS_TORCH else "  (torch-gated load test ran in skip mode)"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
