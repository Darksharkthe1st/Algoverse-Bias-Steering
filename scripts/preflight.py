"""Step zero on the box: prove the environment before spending the window.

    python3 -m scripts.preflight              # ~30 s, no GPU work
    python3 -m scripts.preflight --load-model qwen-1.8b   # + one real forward pass

Exits non-zero if anything would break a run. Run it FIRST, every time.

This exists because of incident I-10 in notes/11: three environment collisions
(numpy 2.x against a numpy-1.x-compiled torch, Pillow too old for transformers,
jinja2 too old for apply_chat_template) were each discovered by running and
failing, one at a time, on a metered box. The rule that came out of it is
"environment smoke test as step zero" (notes/11 §9.5), and this is that test.

The machine is wiped at expiry and there is no SSH, so a debugging hour is an
hour of the window. Everything here is checkable in seconds and every failure
has a stated fix.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OK, WARN, FAIL = "ok", "warn", "FAIL"
results: list = []


def check(name, status, detail="", fix=""):
    results.append((name, status, detail, fix))
    mark = {OK: "  ok  ", WARN: " warn ", FAIL: " FAIL "}[status]
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""))
    if status != OK and fix:
        print(f"          fix: {fix}")


# --------------------------------------------------------------------------- #

def check_python():
    v = sys.version_info
    check("python >= 3.10", OK if v >= (3, 10) else FAIL,
          f"{v.major}.{v.minor}.{v.micro}",
          "the package declares python_requires>=3.10")


def check_imports(load_model):
    """The three collisions from I-10, plus what the runs actually import."""
    need = ["numpy", "scipy", "sklearn"]
    if load_model:
        need += ["torch", "transformers", "transformer_lens", "PIL", "jinja2"]
    for mod in need:
        try:
            m = importlib.import_module(mod)
            check(f"import {mod}", OK, getattr(m, "__version__", "?"))
        except Exception as e:
            check(f"import {mod}", FAIL, f"{type(e).__name__}: {e}",
                  f"pip install {mod}")

    # I-10, collision 1: numpy 2.x against a numpy-1.x-compiled torch
    try:
        import numpy as np
        major = int(np.__version__.split(".")[0])
        check("numpy < 2 (Lambda/A100 torch is built against 1.x)",
              OK if major < 2 else FAIL, np.__version__,
              "pip install 'numpy<2'  — do this BEFORE importing torch")
    except Exception:
        pass

    # I-10, collision 2 and 3
    if load_model:
        try:
            from PIL import Image
            check("pillow has Image.Resampling",
                  OK if hasattr(Image, "Resampling") else FAIL, "",
                  "pip install 'pillow>=9.1'")
        except Exception:
            pass
        try:
            import jinja2
            ok = tuple(int(x) for x in jinja2.__version__.split(".")[:2]) >= (3, 1)
            check("jinja2 >= 3.1 (apply_chat_template needs it)",
                  OK if ok else FAIL, jinja2.__version__,
                  "pip install 'jinja2>=3.1'")
        except Exception:
            pass


def check_gpu(load_model):
    if not load_model:
        check("GPU check", WARN, "skipped (pass --load-model to run it)")
        return
    try:
        import torch
        if not torch.cuda.is_available():
            check("torch.cuda.is_available()", FAIL, "no GPU visible",
                  "wrong instance type, or the driver is not loaded")
            return
        name = torch.cuda.get_device_name(0)
        gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        enough = gb >= 38
        check("GPU present", OK if enough else WARN, f"{name}, {gb:.0f} GB",
              "qwen-14b fp16 is ~28 GB of weights before activations; "
              "under ~38 GB expect OOM on the 14B runs")
    except Exception as e:
        check("GPU check", FAIL, f"{type(e).__name__}: {e}")


def check_data():
    bbq = os.path.join(ROOT, "datasets", "BBQ_Prompt_Sets")
    n = len([f for f in os.listdir(bbq)]) if os.path.isdir(bbq) else 0
    check("BBQ category files", OK if n >= 10 else FAIL, f"{n} files in {bbq}",
          "the repo ships these; a shallow clone may have missed them")

    md = os.path.join(ROOT, "third_party", "bbq", "additional_metadata.csv")
    check("BBQ answer key (additional_metadata.csv)",
          OK if os.path.exists(md) else FAIL, "",
          "needed for target_loc; without it the answer key is reconstructed")


def check_margin_cache():
    """The runs skip margin scoring if this is present. It is the single
    biggest time saver on the box and the easiest thing to lose in a clone."""
    d = os.path.join(ROOT, "runs", "_margins_cache")
    if not os.path.isdir(d):
        check("margins cache", FAIL, "missing",
              "runs/_margins_cache is committed; without it every run "
              "re-scores margins (three forward passes per item)")
        return
    files = [f for f in os.listdir(d) if f.endswith(".json")]
    models = sorted({f.split("_")[0] for f in files})
    check("margins cache", OK if len(files) >= 30 else WARN,
          f"{len(files)} files, models: {', '.join(models)}")


def check_p3_manifest():
    import hashlib
    p = os.path.join(ROOT, "runs", "_p3_manifest.json")
    if not os.path.exists(p):
        check("P3 manifest", WARN, "absent",
              "python3 -m scripts.p3_subgroup_manifest  (only needed for P3)")
        return
    with open(p, "rb") as f:
        raw = f.read()
    digest = hashlib.sha256(raw).hexdigest()[:16]
    man = json.loads(raw)
    crlf = b"\r\n" in raw
    check("P3 manifest hash reproduces", FAIL if crlf else OK,
          f"sha256[:16]={digest}, {man.get('n_tested_subsets')} subsets"
          + (" — CRLF present, hash will not match" if crlf else ""),
          "git checkout the file; .gitattributes marks it -text")


def check_disk():
    free = shutil.disk_usage(ROOT).free / 1e9
    # P1 persists ~250 MB/category fp16; 10 categories plus headroom.
    check("free disk >= 20 GB", OK if free >= 20 else WARN, f"{free:.0f} GB free",
          "--save-residuals writes ~2.5 GB; model weights need more")


def check_tests():
    print("\nrunning the test suite (torch-free, ~5 s)...")
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q",
                        "--no-header", "-p", "no:cacheprovider"],
                       cwd=ROOT, capture_output=True, text=True)
    tail = [ln for ln in r.stdout.strip().splitlines() if ln.strip()][-1:]
    line = tail[0] if tail else "(no output)"
    passed = "passed" in line and r.returncode == 0
    check("test suite", OK if passed else FAIL, line,
          "do not spend GPU time until this is green")
    if "xfailed" not in line:
        check("N6 xfail markers present", WARN, "expected 4 xfailed",
              "the parser-defect markers are missing — wrong branch?")


def check_hf_token():
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    check("HF_TOKEN set", OK if tok else WARN,
          f"{len(tok)} chars" if tok else "not set",
          "export HF_TOKEN=...  (gemma is gated; use your ROTATED token)")


def check_model_loads(model):
    print(f"\nloading {model} (this is the expensive check)...")
    try:
        from src.bias_steer import models as M
        from src.bias_steer.registry import MODELS
        if model not in MODELS:
            check("model registered", FAIL, f"{model} not in registry",
                  f"known: {sorted(MODELS)}")
            return
        loaded = M.load(model)
        n_l = loaded.model.cfg.n_layers
        d_m = loaded.model.cfg.d_model
        check("model loads", OK, f"{model}: {n_l} layers, d_model {d_m}")

        from src.bias_steer import bbq_score as bs
        from src.bias_steer.config import DEFAULT_SYS
        items = bs.load_scoreable("Religion", "ambig", 2, 0)
        e, r = items[0]
        s = bs.score_answers(loaded, bs.bare_prompt(e),
                             [e.metadata["answers"][r.biased]], DEFAULT_SYS)
        check("one real forward pass", OK if s and s[0] == s[0] else FAIL,
              f"logprob {s[0]:.4f}")
    except Exception as ex:
        check("model loads", FAIL, f"{type(ex).__name__}: {ex}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--load-model", default=None, metavar="MODEL",
                    help="also import torch, check the GPU, and load this model "
                         "for one real forward pass (e.g. qwen-1.8b)")
    args = ap.parse_args(argv)
    heavy = args.load_model is not None

    print("=" * 70)
    print("PREFLIGHT — prove the environment before spending the window")
    print("=" * 70)
    check_python()
    check_imports(heavy)
    check_gpu(heavy)
    check_data()
    check_margin_cache()
    check_p3_manifest()
    check_disk()
    check_hf_token()
    check_tests()
    if heavy:
        check_model_loads(args.load_model)

    fails = [r for r in results if r[1] == FAIL]
    warns = [r for r in results if r[1] == WARN]
    print("\n" + "=" * 70)
    print(f"{len(results)} checks · {len(fails)} FAIL · {len(warns)} warn")
    if fails:
        print("\nDO NOT START A RUN. Fix these first:")
        for n, _s, d, fix in fails:
            print(f"  - {n}: {d}")
            if fix:
                print(f"      {fix}")
    else:
        print("\nGreen. Safe to start P0.")
        if not heavy:
            print("Re-run with --load-model qwen-1.8b before the first real run;")
            print("that is the check that catches a broken torch/GPU stack.")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
