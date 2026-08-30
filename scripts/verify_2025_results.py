"""Recompute every number in docs/VERIFICATION_2026-08-07.md from committed artifacts.

CPU only, no GPU, no API calls. Run from the repo root:

    python3 scripts/verify_2025_results.py

Exits non-zero if the headline CSV no longer reproduces from the raw response
pickles, so this doubles as a regression guard on the archive.

Two gotchas this script encodes, both discovered on 2026-08-07 and both easy to
get wrong if you write the loader yourself:

1. The response pickles are CUMULATIVE. `log_231_...responses.pkl` contains the
   1.8B run followed by the 7B run; `log_236` contains all seven. Only the LAST
   `RECORDS_PER_RUN` records belong to the model named in the filename. Counting
   the whole file silently blends models and produces numbers matching no row.

2. The pickles reference `SteeredResponses` / `Response` from `src.data` as if
   they were defined in `__main__` (they were, in the notebook). They will not
   unpickle until those names are aliased into `__main__` — see `_install()`.

Torch tensors inside the vector pickles were saved from CUDA; `_CPUUnpickler`
maps them back to CPU so this runs on a laptop.
"""
import csv
import glob
import io
import os
import pickle
import re
import sys

RECORDS_PER_RUN = 96  # the scored-set size; NOT ~100 — see the doc
BATCHED_DIR = "experiments/past_logs/methodology_experiments/batched_tests"
BEST_VECS = "experiments/best_vecs"


def _install():
    """Alias the project's data classes into __main__ so the pickles load."""
    sys.path.insert(0, os.getcwd())
    import src.data as sd
    for name in ("SteeredResponses", "Response", "ModelResiduals"):
        if hasattr(sd, name):
            setattr(sys.modules["__main__"], name, getattr(sd, name))


class _CPUUnpickler(pickle.Unpickler):
    """Load pickles containing CUDA-saved torch tensors on a CPU-only machine."""

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            import torch
            return lambda b: torch.load(
                io.BytesIO(b), map_location="cpu", weights_only=False
            )
        return super().find_class(module, name)


def load_run(pkl_path, records_per_run=RECORDS_PER_RUN):
    """Return ONLY the records belonging to the model named in the filename."""
    with open(pkl_path, "rb") as f:
        records = pickle.load(f)
    return records[-records_per_run:]


def count_opinionated(records, field):
    return sum(1 for r in records
               if getattr(getattr(r, field), "neutrality") == "opinionated")


def verify_headline():
    """Recount Batched_Gen.csv from the raw response pickles. Returns ok, rows."""
    csv_path = os.path.join(BATCHED_DIR, "Batched_Gen.csv")
    rows = [r for r in csv.DictReader(open(csv_path))
            if r.get("Model name") and r["Model name"] != "Model name"]
    dirs = sorted(
        glob.glob(os.path.join(BATCHED_DIR, "Log_*")),
        key=lambda d: int(re.search(r"Log_(\d+)_", os.path.basename(d)).group(1)),
    )
    print(f"{'log':<5}{'model':<26}{'recs':>6}{'runs':>5}  "
          f"{'recount I/O/N':<20}{'CSV I/O/N':<20}verdict")
    all_ok = True
    for d, row in zip(dirs, rows):
        num = int(re.search(r"Log_(\d+)_", os.path.basename(d)).group(1))
        pkls = glob.glob(os.path.join(d, "*_responses.pkl"))
        if not pkls:
            continue
        with open(pkls[0], "rb") as f:
            total = len(pickle.load(f))
        recs = load_run(pkls[0])
        got = tuple(count_opinionated(recs, f) for f in
                    ("initial_resp", "opinion_resp", "neutral_resp"))
        want = (int(row["Init->Opin"]), int(row["Opin->Opin"]),
                int(row["Neut->Opin"]))
        ok = got == want
        all_ok &= ok
        print(f"{num:<5}{row['Model name'].split('/')[-1]:<26}{total:>6}"
              f"{total // RECORDS_PER_RUN:>5}  {str(got):<20}{str(want):<20}"
              f"{'REPRODUCES' if ok else 'MISMATCH'}")

        denom = sum(int(row[k]) for k in
                    ("Init->Opin", "Init->Neut", "Init->Nons"))
        if denom != RECORDS_PER_RUN:
            print(f"      ! denominator is {denom}, expected {RECORDS_PER_RUN}")
    return all_ok


def report_norm_profiles():
    """Per-layer norm profile of each committed vector — the coefficient story."""
    import torch
    print(f"\n{'vector':<44}{'L':>4}{'last¼':>8}{'first¼':>8}{'max/min':>10}")
    for path in sorted(glob.glob(os.path.join(BEST_VECS, "*_steer_vec.pkl"))):
        vec = _CPUUnpickler(open(path, "rb")).load()
        if not hasattr(vec, "shape") or vec.dim() < 2:
            continue
        norms = torch.tensor([vec[i].float().norm() for i in range(vec.shape[0])])
        n_layers = vec.shape[0]
        q = max(1, n_layers // 4)
        print(f"{os.path.basename(path)[:42]:<44}{n_layers:>4}"
              f"{norms[-q:].sum() / norms.sum():>7.1%}"
              f"{norms[:q].sum() / norms.sum():>8.1%}"
              f"{norms.max() / norms.min():>9.0f}x")
    print("  Steering adds (coeff/n_layers)*vec[layer] with ONE scalar coeff, so a")
    print("  steep profile means 'all layers' is in practice 'the last few layers'.")
    print("  Compare directions unit-normalized; report the norm profile separately.")


def report_scaffold_pollution():
    """How much template/scaffold noise is in the stored response text."""
    d = glob.glob(os.path.join(BATCHED_DIR, "Log_230_*"))
    if not d:
        return
    pkls = glob.glob(os.path.join(d[0], "*_responses.pkl"))
    if not pkls:
        return
    recs = load_run(pkls[0])
    fields = ("initial_resp", "opinion_resp", "neutral_resp")
    total = len(recs) * len(fields)
    print(f"\nscaffold markers in stored response text (n={total}):")
    for marker in ("PROMPT:", "OUTPUT:", "<|im_start|>", "<|im_end|>"):
        c = sum(1 for r in recs for f in fields
                if marker in getattr(getattr(r, f), "resp"))
        print(f"  {marker!r:<16}{c:>5}/{total}  ({c / total:.0%})")
    print("  Re-judging must strip these AND the echoed prompt span first.")


if __name__ == "__main__":
    _install()
    print("=== headline: Batched_Gen.csv vs raw response pickles ===")
    ok = verify_headline()
    report_norm_profiles()
    report_scaffold_pollution()
    print("\n" + ("ALL ROWS REPRODUCE." if ok else "!! MISMATCH — investigate before building on these numbers."))
    sys.exit(0 if ok else 1)
