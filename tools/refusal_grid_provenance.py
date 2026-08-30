#!/usr/bin/env python3
"""Chunk A0 — grid-layout provenance check for the refusal-direction artifacts.

Proves, WITHOUT torch or any model, that we understand the exact layout of the
paper's committed `mean_diffs.pt` candidate grid and that `direction.pt` is a
storage-view into it at the `(pos, layer)` given in `direction_metadata.json`.

A `.pt` file is a zip whose `*/data.pkl` member is a pickle describing the tensor
(`torch._utils._rebuild_tensor_v2(storage, storage_offset, shape, stride, ...)`).
We disassemble that pickle with the stdlib (`zipfile` + `pickletools`) and read:

  - the candidate grid shape/dtype/stride  (mean_diffs.pt)
  - direction.pt's storage_offset into the SAME underlying storage

Then we decode the offset back to `(pos_index, layer)` and check it against the
committed metadata. Because `direction.pt = mean_diffs[pos, layer]` is saved as a
view, both files carry the whole grid storage (same `data/0` blob size), and

    storage_offset / d_model == pos_index * n_layers + layer

with the position axis mapping array index i -> token position (i - n_pos).

Run:  python3 tools/refusal_grid_provenance.py            # all fetched models
      python3 tools/refusal_grid_provenance.py qwen-1_8b-chat

Exit 0 iff every fetched model's decoded (pos, layer) matches its metadata.
Stdlib only; needs only the fetched artifacts (scripts/fetch_refusal_artifacts.py).
"""

import io
import json
import pickletools
import sys
import zipfile
from pathlib import Path

_ARTIFACT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "third_party" / "refusal_direction" / "runs"
)


def _read_pickle_ops(pt_path: Path, member_suffix="data.pkl"):
    """Return the list of (opcode_name, arg) tuples for a .pt's pickle member."""
    z = zipfile.ZipFile(pt_path)
    name = next(n for n in z.namelist() if n.endswith(member_suffix))
    data = z.read(name)
    return [(op.name, arg) for op, arg, _pos in pickletools.genops(io.BytesIO(data))]


def parse_rebuild_tensor(pt_path: Path) -> dict:
    """Extract {storage_dtype, storage_numel, storage_offset, shape, stride} from a
    `.pt` saving a single tensor via `_rebuild_tensor_v2`. Pure stdlib.

    The pickle stream (protocol 2) lays these out positionally after the
    persistent-id (storage) tuple: storage_offset, shape (a tuple of ints), stride
    (a tuple of ints). We walk the opcodes and pick them off in order.
    """
    ops = _read_pickle_ops(pt_path)

    dtype = None
    storage_numel = None
    for i, (name, arg) in enumerate(ops):
        if name == "GLOBAL" and isinstance(arg, str) and "Storage" in arg:
            dtype = arg.split()[-1]  # e.g. "DoubleStorage"
        if name == "BINPERSID":
            # the int just before BINPERSID's closing is the storage numel; find the
            # last BININT/BININT1/BININT2 before this op
            for backname, backarg in reversed(ops[:i]):
                if backname in ("BININT", "BININT1", "BININT2"):
                    storage_numel = backarg
                    break
            persid_index = i
            break

    # After BINPERSID come: storage_offset (int), then the shape tuple, then the
    # stride tuple. Collect the integer args following the persid.
    ints_after = [
        arg for (name, arg) in ops[persid_index + 1:]
        if name in ("BININT", "BININT1", "BININT2")
    ]
    # Determine tuple sizes from the TUPLE* opcodes following persid.
    tuple_ops = [
        name for (name, _arg) in ops[persid_index + 1:]
        if name in ("TUPLE1", "TUPLE2", "TUPLE3", "TUPLE")
    ]
    storage_offset = ints_after[0]
    rest = ints_after[1:]
    ndim = {"TUPLE1": 1, "TUPLE2": 2, "TUPLE3": 3}.get(tuple_ops[0]) if tuple_ops else None
    if ndim is None:
        raise ValueError(f"could not determine tensor rank for {pt_path}")
    shape = tuple(rest[:ndim])
    stride = tuple(rest[ndim:2 * ndim])
    return {
        "dtype": dtype,
        "storage_numel": storage_numel,
        "storage_offset": storage_offset,
        "shape": shape,
        "stride": stride,
    }


def check_model(run_dir: Path) -> bool:
    name = run_dir.name
    grid_pt = run_dir / "generate_directions" / "mean_diffs.pt"
    dir_pt = run_dir / "direction.pt"
    meta_json = run_dir / "direction_metadata.json"
    if not (grid_pt.exists() and dir_pt.exists() and meta_json.exists()):
        print(f"  {name:28s} SKIP (missing artifacts)")
        return True

    grid = parse_rebuild_tensor(grid_pt)
    direc = parse_rebuild_tensor(dir_pt)
    meta = json.loads(meta_json.read_text())

    n_pos, n_layers, d_model = grid["shape"]
    exp_layer, exp_pos = int(meta["layer"]), int(meta["pos"])

    # direction.pt must be a (d_model,) view sharing the grid's storage.
    assert direc["shape"] == (d_model,), f"{name}: direction shape {direc['shape']} != ({d_model},)"
    assert direc["storage_numel"] == grid["storage_numel"], (
        f"{name}: direction storage numel {direc['storage_numel']} != grid {grid['storage_numel']}"
    )

    # Decode the flat offset (in units of d_model rows) back to (pos_index, layer).
    assert direc["storage_offset"] % d_model == 0, f"{name}: offset not a multiple of d_model"
    row = direc["storage_offset"] // d_model
    pos_index, layer = divmod(row, n_layers)
    pos_token = pos_index - n_pos  # array index i -> token position (i - n_pos)

    ok = (layer == exp_layer) and (pos_token == exp_pos)
    status = "OK " if ok else "FAIL"
    print(
        f"  {name:28s} {status} grid={grid['shape']} {grid['dtype']} "
        f"| dir offset={direc['storage_offset']} -> row={row} "
        f"-> (pos={pos_token}, layer={layer})  meta=(pos={exp_pos}, layer={exp_layer})"
    )
    return ok


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not _ARTIFACT_ROOT.exists():
        print(f"no artifacts under {_ARTIFACT_ROOT}\n"
              f"fetch first: python scripts/fetch_refusal_artifacts.py")
        return 2
    run_dirs = (
        [_ARTIFACT_ROOT / a for a in argv] if argv
        else sorted(d for d in _ARTIFACT_ROOT.iterdir() if d.is_dir())
    )
    print(f"provenance check over {len(run_dirs)} model(s):")
    all_ok = all(check_model(d) for d in run_dirs)
    print("PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
