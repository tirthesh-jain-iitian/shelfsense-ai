import json, difflib
from itertools import zip_longest
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

def _cy(d):
    return (d["bbox"][1] + d["bbox"][3]) / 2

def group_rows(dets):
    if not dets:
        return []
    dets = sorted(dets, key=_cy)
    tol = 0.5 * np.median([d["bbox"][3] - d["bbox"][1] for d in dets])
    rows, cur = [], [dets[0]]
    for d in dets[1:]:
        if abs(_cy(d) - np.mean([_cy(c) for c in cur])) <= tol:
            cur.append(d)
        else:
            rows.append(cur)
            cur = [d]
    rows.append(cur)
    return [sorted(r, key=lambda d: d["bbox"][0]) for r in rows]

def find_gaps(rows, factor=1.5):
    """Empty spaces: horizontal gap between neighbours much wider than a normal product."""
    gaps = []
    for r, row in enumerate(rows, start=1):
        if len(row) < 2:
            continue
        med_w = np.median([d["bbox"][2] - d["bbox"][0] for d in row])
        for a, b in zip(row, row[1:]):
            gap = b["bbox"][0] - a["bbox"][2]
            if gap > factor * med_w:
                gaps.append({"row": r, "x1": a["bbox"][2], "x2": b["bbox"][0],
                             "y1": min(a["bbox"][1], b["bbox"][1]),
                             "y2": max(a["bbox"][3], b["bbox"][3])})
    return gaps

def compare_row(expected, actual):
    sm = difflib.SequenceMatcher(None, expected, actual, autojunk=False)
    matched, issues = 0, []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            matched += i2 - i1
        elif tag == "replace":
            for k in range(max(i2 - i1, j2 - j1)):
                e = expected[i1 + k] if i1 + k < i2 else None
                a = actual[j1 + k] if j1 + k < j2 else None
                if e and a: issues.append(("misplaced", e, a))
                elif e:     issues.append(("missing", e, None))
                else:       issues.append(("extra", None, a))
        elif tag == "delete":
            issues += [("missing", e, None) for e in expected[i1:i2]]
        elif tag == "insert":
            issues += [("extra", None, a) for a in actual[j1:j2]]
    return matched, issues

def evaluate_planogram(rows, planogram_path="data/planogram.json",
                       catalogue_path="data/catalogue.csv"):
    plano = json.load(open(planogram_path))
    cat = pd.read_csv(catalogue_path).set_index("sku_id")
    def nm(s):
        if s == "unknown": return "Unrecognised item"
        return cat.loc[s]["name"] if s in cat.index else s

    total = matched = 0
    violations = []
    for r, (shelf, row) in enumerate(zip_longest(plano["shelves"], rows), start=1):
        if shelf and shelf.get("mode") == "ignore":
            continue
        if shelf and shelf.get("mode") == "zone":
            required = shelf.get("stock", [])
            seen = [d["sku"] for d in row if d["sku"] != "unknown"] if row else []
            total += len(required)
            if not seen:
                violations.append(("missing", f"Shelf {r}: no planned products detected (empty shelf, restock needed)"))
                continue
            for s in required:
                if s in seen:
                    matched += 1
                else:
                    violations.append(("missing", f"Shelf {r}: no {nm(s)} detected on this shelf"))
            for s in sorted(set(seen) - set(required)):
                violations.append(("misplaced", f"Shelf {r}: {nm(s)} found here but planned for another shelf"))
            continue
        expected = shelf["slots"] if shelf else []
        actual = [d["sku"] for d in row if d["sku"] != "unknown"] if row else []
        total += len(expected)
        m, issues = compare_row(expected, actual)
        matched += m
        for kind, e, a in issues:
            if kind == "misplaced":
                violations.append((kind, f"Shelf {r}: misplaced. Found {nm(a)}, expected {nm(e)}"))
            elif kind == "missing":
                violations.append((kind, f"Shelf {r}: missing {nm(e)} (out of stock or not detected)"))
            else:
                violations.append((kind, f"Shelf {r}: unplanned item {nm(a)}"))
    score = int(100 * matched / max(1, total))
    return score, violations


def group_rows_k(dets, k):
    if not dets:
        return []
    k = max(1, min(k, len(dets)))
    ys = np.array([[_cy(d)] for d in dets])
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(ys)
    centers = {c: ys[labels == c].mean() for c in set(labels)}
    rows = []
    for c in sorted(centers, key=centers.get):
        row = [d for d, l in zip(dets, labels) if l == c]
        rows.append(sorted(row, key=lambda d: d["bbox"][0]))
    return rows
