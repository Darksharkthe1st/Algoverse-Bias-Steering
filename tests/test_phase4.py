"""Phase 4 verification: the batch coordinator (§10).

Driven against a REAL temporary git repo with a FAKE runner, so commits / checkout
/ soft-landing push / batch-restart are exercised for real — no GPU, no remote.
Also verifies the phase-signal plumbing (engine -> sentinel -> coordinator).

    python3 tests/test_phase4.py
"""

import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import src.bias_steer as bs  # noqa: E402
from src.bias_steer import registry, experiment, cli  # noqa: E402
from src.bias_steer.coordinator import Coordinator, GitOps, RouteEntry, _PHASE_RE  # noqa: E402
from src.bias_steer.config import (  # noqa: E402
    ExperimentConfig, DatasetSpec, SampleSpec, JudgeSpec, Coeffs, ModelSpec,
)
from src.bias_steer.schema import Example  # noqa: E402


# ------------------------------------------------------------------ git helpers

def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)


def _make_repo(tmp) -> Path:
    repo = Path(tmp)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "README.md").write_text("seed\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    _git(repo, "branch", "exp")
    return repo


def _commit_count(repo) -> int:
    return int(_git(repo, "rev-list", "--count", "HEAD").stdout.strip())


def _coordinator(repo, runner, push=True, configs=("c1.py",)):
    (repo / "_coordinator").mkdir(exist_ok=True)
    (repo / "_coordinator" / "route.json").write_text(
        json.dumps([{"branch": "exp", "configs": list(configs), "push": push}])
    )
    return Coordinator(repo_dir=repo, coord_dir=repo / "_coordinator",
                       runner=runner, git=GitOps(repo), runs_dir="runs")


def _good_runner(config, runs_dir, on_phase, repo):
    run_dir = Path(repo) / runs_dir / f"run_{Path(config).stem}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "steering_vector.safetensors").write_text("v")
    on_phase("vector", run_dir.name)
    (run_dir / "results.csv").write_text("example_id,condition,verdict\n")
    on_phase("eval", run_dir.name)
    return 0


# ------------------------------------------------------------------ GitOps units

def test_gitops_add_commit_and_push_softland():
    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_repo(tmp)
        g = GitOps(repo)
        assert g.current_branch() in ("main", "master")
        # nothing staged -> commit returns False (not an error)
        assert g.add_commit("runs", "empty") is False
        (repo / "runs").mkdir()
        (repo / "runs" / "x.txt").write_text("hi")
        assert g.add_commit("runs", "add x") is True
        # no 'origin' remote -> push soft-lands to False, never raises
        assert g.push("main") is False


# ------------------------------------------------------------------ full loop

def test_coordinator_runs_commits_per_phase_and_marks_done():
    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_repo(tmp)
        before = _commit_count(repo)  # on default branch
        coord = _coordinator(repo, _good_runner)

        coord.run()

        # ended up on the route branch, with commits added there
        assert GitOps(repo).current_branch() == "exp"
        assert _commit_count(repo) >= before + 2  # vector + eval (+ finalize)
        # done marker written; nothing failed
        done = list((repo / "_coordinator" / "queue" / "done").iterdir())
        assert len(done) == 1
        assert not list((repo / "_coordinator" / "queue" / "failed").iterdir())
        # status.json reflects completion; the run's files are committed on exp
        status = json.loads((repo / "_coordinator" / "status.json").read_text())
        assert status["state"] == "done"
        assert _git(repo, "ls-files", "runs").stdout.strip() != ""


def test_coordinator_batch_restart_skips_done():
    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_repo(tmp)
        _coordinator(repo, _good_runner).run()
        after_first = _commit_count(repo)
        # a second drain should skip the completed config -> no new commits
        _coordinator(repo, _good_runner).run()
        assert _commit_count(repo) == after_first


def test_coordinator_soft_lands_failures():
    calls = {"n": 0}

    def flaky_runner(config, runs_dir, on_phase, repo):
        calls["n"] += 1
        if Path(config).stem == "bad":
            raise RuntimeError("boom (simulated OOM)")
        return _good_runner(config, runs_dir, on_phase, repo)

    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_repo(tmp)
        coord = _coordinator(repo, flaky_runner, configs=("bad.py", "good.py"))
        coord.run()
        # the batch continued past the crash
        assert calls["n"] == 2
        failed = [p.name for p in (repo / "_coordinator" / "queue" / "failed").iterdir()]
        done = [p.name for p in (repo / "_coordinator" / "queue" / "done").iterdir()]
        assert any("bad" in k for k in failed)
        assert any("good" in k for k in done)


def test_coordinator_stop_control_halts():
    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_repo(tmp)
        coord = _coordinator(repo, _good_runner)
        (repo / "_coordinator" / "control.json").write_text(json.dumps({"command": "stop"}))
        coord.run()
        assert not list((repo / "_coordinator" / "queue" / "done").iterdir())
        assert json.loads((repo / "_coordinator" / "status.json").read_text())["state"] == "stopped"


# ------------------------------------------------------------------ phase-signal plumbing

def test_phase_sentinel_format_roundtrip(capsys=None):
    # cli._emit_phase prints exactly what the coordinator's regex parses.
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        cli._emit_phase("vector", "20260101-120000_lbl_model")
    line = buf.getvalue().strip()
    m = _PHASE_RE.match(line)
    assert m and m.group(1) == "vector" and m.group(2) == "20260101-120000_lbl_model"


def test_experiment_emits_phase_signals():
    _register_p4_fakes()
    seen = []
    with tempfile.TemporaryDirectory() as tmp:
        experiment.run(_p4_config(), backend=_p4_backend(), runs_dir=tmp,
                       on_phase=lambda phase, rid: seen.append(phase))
    assert seen == ["vector", "eval"]  # one model -> vector then eval


# --- tiny fakes for the phase-signal test (mirrors earlier phases) ---

def _register_p4_fakes():
    if "p4ds" not in registry.DATASETS:
        registry.register(registry.DATASETS, "p4ds",
                          lambda spec: [Example(f"e{i}", "p", {"category": "X"}) for i in range(4)])
    if "p4model" not in registry.MODELS:
        registry.register(registry.MODELS, "p4model", ModelSpec("p4model", "fake/m", True, "S"))
    if "p4method" not in registry.METHODS:
        m = types.SimpleNamespace(
            name="p4method",
            capture=lambda cache, n: ("r", n),
            build=lambda rbl, contrast: "VEC",
            apply=lambda model, vec, coeff: [("sign", 1 if coeff >= 0 else -1)],
            names=lambda n: [f"blocks.{i}.hook_resid_pre" for i in range(n)],
        )
        registry.register(registry.METHODS, "p4method", m)
    if "p4judge" not in registry.JUDGES:
        registry.register(registry.JUDGES, "p4judge", lambda resp, ex, spec: list(resp))


def _p4_backend():
    return experiment.Backend(
        load=lambda spec: types.SimpleNamespace(
            model=types.SimpleNamespace(cfg=types.SimpleNamespace(n_layers=2)),
            tokenizer=None, spec=spec, device="cpu"),
        generate=lambda l, p, m, s: ["opinionated"] * len(p),
        generate_with_cache=lambda l, p, m, s, capture_names=None: (
            ["opinionated" if i % 2 == 0 else "neutral" for i in range(len(p))], [None] * len(p)),
        generate_with_hooks=lambda l, p, h, m, s: [
            ("opinionated" if (h and h[0][1] > 0) else "neutral")] * len(p),
        save_vector=lambda p, v: Path(p).write_text("v"),
        save_residuals=lambda p, r: Path(p).write_text("r"),
    )


def _p4_config():
    return ExperimentConfig(
        label="p4", models=["p4model"],
        dataset=DatasetSpec(name="p4ds", path="x", train_split=0.5),
        judge=JudgeSpec(name="p4judge", labels=["neutral", "opinionated"]),
        coeffs=Coeffs(opinion=5, neutral=7), sample=SampleSpec(seed=0), method="p4method",
    )


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
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
