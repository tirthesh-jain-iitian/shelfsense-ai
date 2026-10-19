def best_match(sims, labels, thr=0.82, margin=0.04):
    per = {}
    for lab, v in zip(labels, sims.tolist()):
        per[lab] = max(per.get(lab, -1.0), v)
    top = sorted(per.items(), key=lambda kv: -kv[1])
    best_lab, best = top[0]
    second = top[1][1] if len(top) > 1 else 0.0
    if best < thr or (best - second) < margin:
        return "unknown", best
    return best_lab, best
