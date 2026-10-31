import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
TASKS = ["tw_sentiment", "tw_topic", "bitext", "banking77", "clinc150", "clinc_k10", "clinc_k25", "clinc_k50", "clinc_k100"]
rep = []


def log(s):
    rep.append(s)
    print(s, flush=True)


def jl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


REC = {t: {r["id"]: r for r in jl(ROOT / f"data/records/{t}/test.jsonl")} for t in TASKS}


def parse_key(key):
    if key.endswith("_readout"):
        return key[:-8], "vtrie"
    if key.endswith("_gen"):
        return key[:-4], "gen"
    if "-" in key:
        return tuple(key.rsplit("-", 1))
    return key, "reference"


def load(run):
    data, nrow, nerr, ndup = {}, 0, 0, 0
    for t in TASKS:
        recs = REC[t]
        for f in sorted((ROOT / f"results/{run}/raw").glob(f"*/{t}/test.jsonl")):
            rows = jl(f)
            nrow += len(rows)
            nerr += sum(bool(x.get("error")) for x in rows)
            ids_all = [x["id"] for x in rows]
            ndup += len(ids_all) - len(set(ids_all))
            last = {}
            for x in rows:
                if not x.get("error"):
                    last[x["id"]] = x
            ids = [i for i in recs if i in last]
            if not ids:
                continue
            P = np.array([[last[i]["probs"].get(o, 0.0) for o in recs[i]["options"]] for i in ids], float)
            P = P / np.clip(P.sum(1, keepdims=True), 1e-12, None)
            y = np.array([recs[i]["options"].index(recs[i]["gold"]) for i in ids])
            m, s = parse_key(f.parent.parent.name)
            data[(m, s, t)] = {"ids": ids, "P": P, "y": y, "k": len(recs[ids[0]]["options"]),
                               "sec": np.array([last[i]["latency_s"] or 0.0 for i in ids]),
                               "tok": np.array([last[i]["input_tokens"] or 0 for i in ids], float)}
    return data, nrow, nerr, ndup


def f1_macro(y, pred, k):
    f = []
    for c in range(k):
        tp = int(np.sum((pred == c) & (y == c)))
        fp = int(np.sum((pred == c) & (y != c)))
        fn = int(np.sum((pred != c) & (y == c)))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def ece(P, y, nb=15):
    conf = P.max(1)
    cor = (P.argmax(1) == y).astype(float)
    order = np.argsort(conf)
    tot = 0.0
    for b in np.array_split(order, nb):
        if len(b):
            tot += len(b) / len(conf) * abs(cor[b].mean() - conf[b].mean())
    return tot


def rc(P, y):
    conf = P.max(1)
    err = (P.argmax(1) != y).astype(float)
    o = np.argsort(-conf, kind="stable")
    k = np.arange(1, len(y) + 1)
    return k / len(y), np.cumsum(err[o]) / k


def cov(P, y, r):
    c, risk = rc(P, y)
    okk = np.where(risk <= r)[0]
    return float(c[okk].max()) if len(okk) else 0.0


def quality(data):
    rows = []
    for (m, s, t), d in data.items():
        P, y = d["P"], d["y"]
        Y = np.zeros_like(P)
        Y[np.arange(len(y)), y] = 1
        rows.append({"model": m, "scorer": s, "task": t, "k": d["k"], "n": len(y), "accuracy": float(np.mean(P.argmax(1) == y)),
                     "macro_f1": f1_macro(y, P.argmax(1), P.shape[1]), "ece": ece(P, y), "brier": float(np.mean(((P - Y) ** 2).sum(1))),
                     "aurc": float(rc(P, y)[1].mean()), "coverage_at_1pct": cov(P, y, 0.01), "coverage_at_2pct": cov(P, y, 0.02),
                     "sec_per_decision": float(d["sec"].mean()), "forward_tokens": float(d["tok"].mean())})
    return pd.DataFrame(rows)


def aligned(a, b):
    ib = {i: j for j, i in enumerate(b["ids"])}
    idx = [(j, ib[i]) for j, i in enumerate(a["ids"]) if i in ib]
    return [x for x, _ in idx], [x for _, x in idx]


def exactness(data):
    rows = []
    for (m, s, t), d in data.items():
        if s != "trie":
            continue
        for other in ("cached", "vtrie"):
            o = data.get((m, other, t))
            if o is None:
                continue
            ja, jb = aligned(d, o)
            diff = np.abs(d["P"][ja] - o["P"][jb])
            rows.append({"model": m, "task": t, "k": d["k"], "compare": f"{other}_vs_trie", "n": len(ja), "max_abs_diff": float(diff.max()),
                         "mean_abs_diff": float(diff.mean()), "argmax_agree": float((d["P"][ja].argmax(1) == o["P"][jb].argmax(1)).mean())})
    return pd.DataFrame(rows)


def first_bias(data):
    return pd.DataFrame([{"model": m, "scorer": s, "task": t, "k": d["k"], "mass_on_options_ge10": float(d["P"][:, 9:].sum(1).mean()),
                          "gold_share_ge10": float((d["y"] >= 9).mean())} for (m, s, t), d in data.items() if d["k"] >= 10 and s in ("first", "cached", "trie")])


def boot(d, n_boot=10000):
    rng = np.random.default_rng(0)
    n = len(d)
    bs = np.empty(n_boot)
    for b in range(n_boot):
        bs[b] = d[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def mcn(a, b):
    n01, n10 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n = n01 + n10
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, j) for j in range(min(n01, n10) + 1)) / 2 ** n)


def comparisons(data):
    rows = []
    for (m, s, t), d in data.items():
        if s != "cached":
            continue
        for other in ("first", "letters", "gen", "vtrie"):
            o = data.get((m, other, t))
            if o is None:
                continue
            ja, jb = aligned(d, o)
            ca = d["P"][ja].argmax(1) == d["y"][ja]
            cb = o["P"][jb].argmax(1) == o["y"][jb]
            diff, lo, hi = boot(ca.astype(float) - cb.astype(float))
            rows.append({"model": m, "task": t, "k": d["k"], "a": "cached", "b": other, "n": len(ja), "acc_diff": diff, "lo": lo, "hi": hi,
                         "p_mcnemar": mcn(ca, cb), "coverage1_a": cov(d["P"][ja], d["y"][ja], 0.01), "coverage1_b": cov(o["P"][jb], o["y"][jb], 0.01)})
    out = pd.DataFrame(rows)
    if len(out):
        p = out.p_mcnemar.to_numpy()
        o = np.argsort(p, kind="stable")
        adj, run = np.empty(len(p)), 0.0
        for r_, i in enumerate(o):
            run = max(run, min(1.0, (len(p) - r_) * p[i]))
            adj[i] = run
        out["p_holm"] = adj
    return out


def latency(run):
    rows = []
    for f in sorted((ROOT / f"results/{run}/latency").glob("*.jsonl")) if (ROOT / f"results/{run}/latency").exists() else []:
        rows += jl(f)
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    g = d.groupby(["model", "scorer", "task"]).ms_per_decision
    return pd.DataFrame({"mean": g.mean(), "std": g.std(), "size": g.size()}).reset_index()


def compare(run, name, mine, keys, tol=1e-9):
    p = ROOT / f"results/{run}/tables/{name}.csv"
    ref = pd.read_csv(p) if p.stat().st_size > 1 else pd.DataFrame()
    if ref.empty and mine.empty:
        log(f"[OK  ] {run}/{name}: both empty")
        return
    cols = [c for c in ref.columns if c not in keys]
    mm = ref.merge(mine, on=keys, how="outer", suffixes=("_ref", "_me"), indicator=True)
    unmatched = int((mm._merge != "both").sum())
    bad, maxd = 0, 0.0
    for c in cols:
        a, b = mm[f"{c}_ref"], mm[f"{c}_me"]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            av, bv = a.to_numpy(float), b.to_numpy(float)
            dd = np.where(np.isnan(av) & np.isnan(bv), 0.0, np.abs(av - bv))
            dd = np.where(np.isnan(dd), np.inf, dd)
            maxd = max(maxd, float(dd.max()))
            bad += int((dd > tol).sum())
        else:
            bad += int((a.astype(str) != b.astype(str)).sum())
    log(f"[{'OK  ' if bad == 0 and unmatched == 0 else 'FAIL'}] {run}/{name}: {len(ref)} rows (unmatched {unmatched}), {len(cols)} cols, max |diff| {maxd:.2e}, bad cells {bad}")


for run in ("main", "fp32"):
    data, nrow, nerr, ndup = load(run)
    log(f"{run}: {nrow:,} raw rows, {nerr} errors, {ndup} duplicate ids, {len(data)} model x scorer x task cells ({time.time() - T0:.0f} s)")
    q = quality(data)
    tabs = {"exactness": (exactness(data), ["model", "task", "compare"]), "quality": (q, ["model", "scorer", "task"]),
            "cost": (q[["model", "scorer", "task", "k", "sec_per_decision", "forward_tokens"]], ["model", "scorer", "task"]),
            "first_bias": (first_bias(data), ["model", "scorer", "task"]), "comparisons": (comparisons(data), ["model", "task", "a", "b"]),
            "latency": (latency(run), ["model", "scorer", "task"])}
    for name, (df, keys) in tabs.items():
        compare(run, name, df, keys)
        df.to_csv(OUT / f"{run}_{name}.csv", index=False)
    np.savez_compressed(OUT / f"{run}_arrays.npz", **{f"{m}|{s}|{t}|{k}": d[k] for (m, s, t), d in data.items() for k in ("P", "y", "sec", "tok")},
                        **{f"{m}|{s}|{t}|ids": np.array(d["ids"]) for (m, s, t), d in data.items()})
    log(f"{run} done ({time.time() - T0:.0f} s)")
(OUT / "verify_report.txt").write_text("\n".join(rep) + "\n")
