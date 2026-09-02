import csv
import sys
from collections import defaultdict

LABELS = ["neutral", "opinionated"]


def load(path):
    by_ex = defaultdict(dict)
    with open(path) as f:
        for row in csv.DictReader(f):
            by_ex[row["example_id"]][row["condition"]] = row["verdict"]
    return by_ex


def transition_matrix(by_ex, from_cond, to_cond):
    mat = defaultdict(lambda: defaultdict(int))
    unmatched = 0
    n = 0
    for ex_id, conds in by_ex.items():
        a = conds.get(from_cond)
        b = conds.get(to_cond)
        if a is None or b is None:
            continue
        n += 1
        if a not in LABELS or b not in LABELS:
            unmatched += 1
        mat[a][b] += 1
    return mat, n, unmatched


def print_matrix(name, mat, n, unmatched):
    print(f"\n{name} (n={n}, unmatched/none={unmatched})")
    all_labels = LABELS + sorted(set(k for row in mat.values() for k in row if k not in LABELS) |
                                  set(k for k in mat if k not in LABELS))
    header = "init\\steered".ljust(14) + "".join(l.rjust(14) for l in all_labels)
    print(header)
    for a in all_labels:
        row = "".join(str(mat[a].get(b, 0)).rjust(14) for b in all_labels)
        print(a.ljust(14) + row)


if __name__ == "__main__":
    path = sys.argv[1]
    by_ex = load(path)
    for to_cond, label in [("steered_pos", "INITIAL -> STEERED_POS"), ("steered_neg", "INITIAL -> STEERED_NEG")]:
        mat, n, unmatched = transition_matrix(by_ex, "initial", to_cond)
        print_matrix(label, mat, n, unmatched)
