"""Run 2, minimal scope: the annotation-derived contrast, one model.

    # on the box, GPU:
    python -m scripts.run2_annotation_contrast capture \
        --model qwen-14b --capture-index -1 --n-per-arm 200 \
        --out runs/run2/qwen-14b

    # on the laptop, CPU, from the synced residuals:
    python -m scripts.run2_annotation_contrast analyse \
        --out runs/run2/qwen-14b

WHY THIS EXISTS
---------------
Run 1 built bias directions from the extremes of the model's own stereotype
margin and got split-half floors of -0.45 to +0.82, mostly below 0.5. The
reference paper (Joad et al.) builds directions from DATASET ANNOTATIONS and gets
0.95-0.99 from 32 items per class. Sample size cannot explain that gap -- we used
up to 20x more data and did worse. The contrast can, and it is the only thing
left that differs.

This script measures the annotation-derived contrast on the same items, the same
models and the same pipeline, so the comparison is controlled:

    direction_C = mean(resid | category C, ambiguous)
                - mean(resid | category C, disambiguated)

matched on the full BBQ scenario key. `context_condition` ships with the dataset
and never consults the model, which is what closes defect M1.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
No generation, no judge, no steering, no margins. Those are the expensive,
defect-prone parts and none of them is needed for this comparison. At
n=200 per arm this is ~4,000 forward passes per model -- about 17 minutes at
run 1's measured throughput on qwen-14b -- and ~3.3 GB of residuals.

ROBUST TO THE TWO OPEN HOLES IN notes/19
----------------------------------------
* The capture index (-1 vs -2) is unresolved, but BOTH contrasts are measured at
  the same site, so the comparison between them holds whichever index is used.
  It is required as an explicit flag and recorded in every sidecar, so the
  choice is never implicit.
* The 114 GB storage problem disappears at n=200.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.pilot import analysis, pairing, verifier          # noqa: E402
from scripts.pilot.queue import Step, run_queue                # noqa: E402

MODELS = {
    "qwen-1.8b": "Qwen/Qwen1.5-1.8B-Chat",
    "gemma-2b":  "google/gemma-2b-it",
    "yi-6b":     "01-ai/Yi-6B-Chat",
    "qwen-7b":   "Qwen/Qwen1.5-7B-Chat",
    "qwen-14b":  "Qwen/Qwen1.5-14B-Chat",
}

# Verbatim from src/bias_steer/config.py, so run 2 scores the same way run 1 did.
DEFAULT_SYS = ("You are to follow the instructions given in the question. First give the "
               "clear, definitive answer and then explain your answers very briefly")


# --------------------------------------------------------------------------- #
# capture (GPU)
# --------------------------------------------------------------------------- #

def _load(hf_id, device):
    import torch                                                # noqa: PLC0415
    from transformer_lens import HookedTransformer              # noqa: PLC0415
    from transformers import AutoTokenizer                      # noqa: PLC0415

    tok = AutoTokenizer.from_pretrained(hf_id)
    model = HookedTransformer.from_pretrained_no_processing(
        hf_id, device=device, dtype=torch.float16)
    model.eval()
    return tok, model


def _chat(tok, prompt, system_prompt):
    """Chat template, with gemma's no-system-turn fallback (bbq_score.py:76)."""
    msgs = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return tok.apply_chat_template(
            [{"role": "user", "content": f"{system_prompt}\n\n{prompt}"}],
            tokenize=False, add_generation_prompt=True)


def probe_tokens(tok, system_prompt):
    """Print the tail of a chat-formatted prompt. Hole (d), notes/19 §6.1.

    Run this and LOOK at it before capturing anything. The capture index is the
    one parameter that cannot be fixed after the fact.
    """
    text = _chat(tok, "Who was not comfortable using the phone?", system_prompt)
    ids = tok(text, add_special_tokens=True)["input_ids"]
    tail = [(i - len(ids), tok.decode([t])) for i, t in enumerate(ids)][-6:]
    return {"n_tokens": len(ids), "template_tail": text[-140:],
            "last_six": [{"index": i, "text": s} for i, s in tail]}


def capture_arm(tok, model, rows, *, capture_index, system_prompt, batch_size=8):
    """(n_items, n_layers, d_model) float32 at `capture_index`."""
    import torch                                                # noqa: PLC0415

    n_layers = model.cfg.n_layers
    names = [f"blocks.{i}.hook_resid_pre" for i in range(n_layers)]
    wanted = set(names)

    prev = getattr(tok, "padding_side", None)
    tok.padding_side = "left"          # so the last position is the real final token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    out = []
    try:
        for i in range(0, len(rows), batch_size):
            chunk = [_chat(tok, pairing.prompt_text(r), system_prompt)
                     for r in rows[i:i + batch_size]]
            enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=True)
            ids = enc["input_ids"].to(model.cfg.device)
            attn = enc["attention_mask"].to(model.cfg.device)
            with torch.no_grad():
                _logits, cache = model.run_with_cache(
                    ids, attention_mask=attn,
                    names_filter=lambda n: n in wanted, return_type=None)
            stack = torch.stack([cache[n][:, capture_index, :] for n in names], dim=1)
            out.append(stack.detach().float().cpu())
            del cache
    finally:
        if prev is not None:
            tok.padding_side = prev
    return torch.cat(out, dim=0).numpy().astype(np.float32)


def persist(out_dir, category, arm, rows, resid, meta_extra):
    d = os.path.join(out_dir, "residuals")
    os.makedirs(d, exist_ok=True)
    stem = os.path.join(d, f"{category}__{arm}")
    np.save(stem + ".npy", resid)
    meta = {
        "category": category, "arm": arm,
        "item_ids": [pairing.item_key(r) for r in rows],
        "n_items": len(rows), "n_layers": int(resid.shape[1]),
        "d_model": int(resid.shape[2]), "dtype": "float32",
        **meta_extra,
    }
    with open(stem + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return stem + ".npy"


def cmd_capture(args):
    hf_id = MODELS[args.model]
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    print(f"loading {hf_id} on {args.device} ...", flush=True)
    tok, model = _load(hf_id, args.device)
    print(f"  n_layers={model.cfg.n_layers} d_model={model.cfg.d_model}", flush=True)

    probe = probe_tokens(tok, args.system_prompt)
    print("\n  CAPTURE SITE -- look at this before trusting anything downstream:")
    for e in probe["last_six"]:
        print(f"    index {e['index']:>3}  {e['text']!r}")
    print(f"    capturing at index {args.capture_index}\n", flush=True)
    with open(os.path.join(args.out, "capture_site.json"), "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "hf_id": hf_id,
                   "capture_index": args.capture_index, **probe}, f, indent=2)

    cats = pairing.load_pilot_categories(
        args.categories or pairing.categories(), limit_pairs=args.n_per_arm)

    meta_extra = {
        "model": args.model, "hf_id": hf_id,
        "capture_site": f"resid_pre, chat-template token index {args.capture_index}",
        "capture_index": args.capture_index,
        "system_prompt": args.system_prompt,
        "contrast": "context_condition (ambig minus disambig), annotation-derived",
        "scenario_key": "(question_index, question_polarity, ans0, ans1, ans2)",
    }

    steps, produced = [], []
    for c in cats:
        def _make(c=c):
            def _run():
                a_rows, b_rows = pairing.arms(c.pairs)
                for arm, rows in (("a", a_rows), ("b", b_rows)):
                    stem = os.path.join(args.out, "residuals", f"{c.category}__{arm}.npy")
                    if os.path.exists(stem) and not args.force:
                        print(f"  {c.category}/{arm}: cached, skipping", flush=True)
                        continue
                    t = time.time()
                    r = capture_arm(tok, model, rows,
                                    capture_index=args.capture_index,
                                    system_prompt=args.system_prompt,
                                    batch_size=args.batch_size)
                    produced.append(persist(args.out, c.category, arm, rows, r, meta_extra))
                    print(f"  {c.category}/{arm}: {r.shape} in {time.time()-t:.0f}s",
                          flush=True)
            return _run

        steps.append(Step(
            name=f"capture_{c.category}",
            produces=[os.path.join(args.out, "residuals", f"{c.category}__{arm}.npy")
                      for arm in ("a", "b")],
            fn=_make()))

    # prompts, verbatim -- notes/13 §13
    def _prompts():
        p = os.path.join(args.out, "prompts.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for c in cats:
                for arm, rows in zip(("a", "b"), pairing.arms(c.pairs)):
                    for r in rows:
                        f.write(json.dumps({
                            "item_id": pairing.item_key(r), "category": c.category,
                            "arm": arm, "context_condition": r["context_condition"],
                            "question_polarity": r["question_polarity"],
                            "prompt": pairing.prompt_text(r),
                            "chat_formatted": _chat(tok, pairing.prompt_text(r),
                                                    args.system_prompt),
                            "answers": [r["ans0"], r["ans1"], r["ans2"]],
                        }) + "\n")
    steps.append(Step(name="write_prompts",
                      produces=[os.path.join(args.out, "prompts.jsonl")], fn=_prompts))

    m = run_queue(steps, out_dir=args.out)
    with open(os.path.join(args.out, "pairing_report.json"), "w", encoding="utf-8") as f:
        json.dump({c.category: c.report for c in cats}, f, indent=2)

    print(f"\ncapture finished in {(time.time()-t0)/60:.1f} min; all_ok={m['all_ok']}")
    if not m["all_ok"]:
        for s in m["steps"]:
            if s["status"] != "OK":
                print(f"  {s['name']}: {s['status']} {s.get('error','')}")
        return 1
    print("\nSYNC TO THE LAPTOP AND RUN THE VERIFIER BEFORE TERMINATING THE BOX.")
    return 0


# --------------------------------------------------------------------------- #
# analyse (CPU, on the laptop, from cached residuals)
# --------------------------------------------------------------------------- #

def cmd_analyse(args):
    c = verifier.verify(args.out)
    print(f"verifier: {c.checked} checks, {len(c.failures)} failures")
    for f in c.failures[:10]:
        print(f"  FAIL {f}")
    if not c.passed and not args.force:
        print("\nverifier failed -- refusing to analyse. Pass --force to override.")
        return 1

    resid_dir = os.path.join(args.out, "residuals")
    cats = sorted({f.split("__")[0] for f in os.listdir(resid_dir) if f.endswith(".npy")})
    print(f"\nanalysing {len(cats)} categories from cached residuals (no GPU)\n")

    from scripts.pilot.run_pilot import cached_capture
    cd = pairing.load_pilot_categories(cats, limit_pairs=args.n_per_arm)
    by_cat = {x.category: x for x in cd}

    directions, floors, negs, verdicts, per_rows = {}, {}, {}, {}, {}
    caps = {}
    for name in cats:
        cap = cached_capture(args.out, [name], None)
        caps[name] = cap
        pairs = by_cat[name].pairs
        per_rows[name] = pairing.arms(pairs)[0]
        directions[name] = analysis.extract_from_pairs(pairs, cap)
        floors[name] = analysis.floor(pairs, cap, n_splits=args.n_splits, seed=0)
        negs[name] = analysis.negative_control_floor(pairs, cap,
                                                     n_splits=args.n_splits, seed=0)
        verdicts[name] = analysis.reproduces(floors[name], negs[name])
        f, n = floors[name], negs[name]
        print(f"  {name:<22} floor {f['mean']:+.3f} [{f['ci_lo']:+.3f},{f['ci_hi']:+.3f}]"
              f"   control {n['mean']:+.3f} [{n['ci_lo']:+.3f},{n['ci_hi']:+.3f}]"
              f"   -> {verdicts[name]}")

    pooled = cached_capture(args.out, cats, None)
    d_len = analysis.pooled_length_direction(per_rows, pooled)
    selfchk = analysis.length_direction_selfcheck(per_rows, pooled, seed=0)
    spec = analysis.specificity_control(
        {k: by_cat[k].pairs for k in cats}, caps, directions, floors, negs,
        d_len, selfchk, n_splits=args.n_splits, seed=0)
    cross = analysis.cross_category(directions)

    print(f"\n  cross-category median |cos| = {cross['median_offdiagonal']:+.3f}")
    print(f"  specificity control         = {spec['overall']} "
          f"({spec['n_failing']}/{spec['n_categories']} read as LENGTH)")
    print(f"  length-direction self-check = {selfchk['mean']:+.3f} "
          f"(usable={selfchk['usable']})")

    report = {
        "contrast": "annotation-derived (context_condition), run-2 primary",
        "n_splits": args.n_splits,
        "observed_floor": floors, "negative_control_floor": negs,
        "reproduces": verdicts, "cross_category": cross,
        "specificity_control": spec, "length_direction_selfcheck": selfchk,
        "capture_site": json.load(open(os.path.join(args.out, "capture_site.json"),
                                       encoding="utf-8"))
        if os.path.exists(os.path.join(args.out, "capture_site.json")) else None,
    }
    p = os.path.join(args.out, "report_annotation_contrast.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {p}")

    n_repro = sum(1 for v in verdicts.values() if v == "YES")
    print(f"\n  {n_repro}/{len(cats)} categories beat their own negative control.")
    print("  Run 1's behaviour-derived contrast cleared 0.50 in 10 of 46 cells.")
    print("  That is the comparison the paper turns on -- same items, same pipeline.")
    return 0


def cmd_control(args):
    """Topic-identity POSITIVE control, through the IDENTICAL pipeline.

    Without this a null result is uninterpretable: you cannot tell "the
    annotation contrast does not recover a direction" from "our code is broken."

    Run 1 had this control and notes/11 calls it the single most valuable
    artifact of that session -- but it validated the OLD pipeline. This runs it
    through the new one: same capture site, same split-by-pair, same bootstrap,
    same negative control, same decision rule.

    The contrast is two different BBQ CATEGORIES rather than two arms of one
    scenario -- race-themed prompts against gender-themed prompts. Topic identity
    is linearly present if anything is, so this direction has to reproduce. If it
    does not, stop: nothing downstream can be read.
    """
    hf_id = MODELS[args.model]
    os.makedirs(args.out, exist_ok=True)
    tok, model = _load(hf_id, args.device)

    report = {
        "model": args.model, "hf_id": hf_id,
        "capture_index": args.capture_index,
        "purpose": "topic-identity positive control through the R1 pipeline",
        "n_splits": args.n_splits,
        "pairs": {},
    }

    hdr = "{:<40}{:>7}{:>9}{:>18}{:>10}  verdict".format(
        "topic contrast", "n/arm", "floor", "95% CI", "control")
    print("\n" + hdr)
    print("-" * 94)

    for spec in args.pairs:
        a_cat, b_cat = spec.split(":")
        a_rows = [p.a for p in pairing.load_pilot_categories(
            [a_cat], limit_pairs=args.n_per_arm)[0].pairs]
        b_rows = [p.a for p in pairing.load_pilot_categories(
            [b_cat], limit_pairs=args.n_per_arm)[0].pairs]
        n = min(len(a_rows), len(b_rows))
        a_rows, b_rows = a_rows[:n], b_rows[:n]

        ra = capture_arm(tok, model, a_rows, capture_index=args.capture_index,
                         system_prompt=args.system_prompt)
        rb = capture_arm(tok, model, b_rows, capture_index=args.capture_index,
                         system_prompt=args.system_prompt)

        # Reuse the R1 machinery exactly: one synthetic Pair per index, so the
        # same split-by-pair, floor and negative-control code runs unchanged.
        pairs = [pairing.Pair(key=(i,), category=spec, a=a_rows[i], b=b_rows[i])
                 for i in range(n)]
        # Key by ITEM ONLY, never by item+arm. negative_control_floor swaps the
        # two members of a pair, so a row that started in arm A is looked up as
        # arm B on the shuffled pass -- an arm-qualified key raises KeyError
        # there. The two arms come from different BBQ categories, so item_key
        # ("<category>:<example_id>") is already unique across them.
        store = {}
        for i in range(n):
            store[pairing.item_key(a_rows[i])] = ra[i]
            store[pairing.item_key(b_rows[i])] = rb[i]

        def cap(rows, arm_sign=1.0, _s=store):
            return np.stack([_s[pairing.item_key(r)] for r in rows])

        fl = analysis.floor(pairs, cap, n_splits=args.n_splits, seed=0)
        ng = analysis.negative_control_floor(pairs, cap,
                                             n_splits=args.n_splits, seed=0)
        verdict = analysis.reproduces(fl, ng)
        report["pairs"][spec] = {"n_per_arm": n, "floor": fl,
                                 "negative_control": ng, "reproduces": verdict}
        ci = "[{:+.3f},{:+.3f}]".format(fl["ci_lo"], fl["ci_hi"])
        print("{:<40}{:>7}{:>+9.3f}{:>18}{:>+10.3f}  {}".format(
            spec, n, fl["mean"], ci, ng["mean"], verdict))

    path = os.path.join(args.out, "positive_control.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("\nwrote " + path)

    ok = [v for v in report["pairs"].values() if v["reproduces"] == "YES"]
    print("{}/{} topic contrasts reproduce.".format(len(ok), len(report["pairs"])))
    if len(ok) < len(report["pairs"]):
        print("")
        print("  *** STOP. The pipeline failed to recover a direction that must")
        print("  *** exist. Do not read any bias result until this passes.")
        return 1
    print("  The pipeline recovers a direction that must exist, so bias nulls")
    print("  measured below this line are interpretable.")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    cap = sub.add_parser("capture", help="GPU: capture and persist residuals")
    cap.add_argument("--model", required=True, choices=sorted(MODELS))
    cap.add_argument("--capture-index", type=int, required=True,
                     help="REQUIRED and never defaulted: notes/19 hole (d). "
                          "Look at the printed token tail before choosing.")
    cap.add_argument("--n-per-arm", type=int, default=200)
    cap.add_argument("--out", required=True)
    cap.add_argument("--categories", nargs="*", default=None)
    cap.add_argument("--device", default="cuda")
    cap.add_argument("--batch-size", type=int, default=8)
    cap.add_argument("--system-prompt", default=DEFAULT_SYS)
    cap.add_argument("--force", action="store_true")
    cap.set_defaults(func=cmd_capture)

    ct = sub.add_parser(
        "control",
        help="GPU: topic-identity POSITIVE control through the same pipeline")
    ct.add_argument("--model", required=True, choices=sorted(MODELS))
    ct.add_argument("--capture-index", type=int, required=True)
    ct.add_argument("--out", required=True)
    ct.add_argument("--pairs", nargs="+",
                    default=["Race_ethnicity:Gender_identity",
                             "Religion:Age",
                             "Nationality:Sexual_orientation"],
                    help="topic contrasts as CategoryA:CategoryB")
    ct.add_argument("--n-per-arm", type=int, default=200)
    ct.add_argument("--n-splits", type=int, default=100)
    ct.add_argument("--device", default="cuda")
    ct.add_argument("--system-prompt", default=DEFAULT_SYS)
    ct.set_defaults(func=cmd_control)

    an = sub.add_parser("analyse", help="CPU: floors and controls from cached residuals")
    an.add_argument("--out", required=True)
    an.add_argument("--n-splits", type=int, default=400)
    an.add_argument("--n-per-arm", type=int, default=200)
    an.add_argument("--force", action="store_true")
    an.set_defaults(func=cmd_analyse)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
