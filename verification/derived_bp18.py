import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RC = Path(sys.argv[1])
F = {}
TASKS = ["tw_sentiment", "tw_topic", "bitext", "banking77", "clinc150", "clinc_k10", "clinc_k25", "clinc_k50", "clinc_k100"]
K = {"tw_sentiment": 3, "tw_topic": 20, "bitext": 27, "banking77": 77, "clinc150": 151, "clinc_k10": 10, "clinc_k25": 25, "clinc_k50": 50, "clinc_k100": 100}
SMALL, LARGE = ["gemmaE2b", "qwen2b"], ["qwen", "gemma"]


def put(k, v):
    F[k] = v
    print(f"{k}: {str(v)[:400]}", flush=True)


def js(o):
    if isinstance(o, dict):
        return {("/".join(map(str, k)) if isinstance(k, tuple) else str(k)): js(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [js(v) for v in o]
    return o.item() if hasattr(o, "item") else o


Q = pd.read_csv(RC / "main_quality.csv")
E = pd.read_csv(RC / "main_exactness.csv")
E32 = pd.read_csv(RC / "fp32_exactness.csv")
C = pd.read_csv(RC / "main_comparisons.csv")
L = pd.read_csv(RC / "main_latency.csv")
FB = pd.read_csv(RC / "main_first_bias.csv")
X = json.loads((RC / "extras_facts.json").read_text())
A = np.load(RC / "main_arrays.npz")
cells = sorted({tuple(k.split("|")[:3]) for k in A.files})
q = Q.set_index(["model", "scorer", "task"])


def ece_order(conf, cor, order, nb=15):
    return sum(len(b) / len(conf) * abs(cor[b].mean() - conf[b].mean()) for b in np.array_split(order, nb) if len(b))


def cov_thr(conf, cor, r):
    conf = np.round(conf, 9)
    best = 0.0
    for t in np.unique(conf):
        acc = conf >= t
        if (1 - cor[acc].mean()) <= r + 1e-12:
            best = max(best, acc.mean())
    return best


def aurc_tie(conf, cor):
    conf = np.round(conf, 9)
    o = np.argsort(-conf, kind="stable")
    c, e = conf[o], (1 - cor[o])
    n = len(c)
    risk_sum, done, errs = 0.0, 0, 0.0
    i = 0
    while i < n:
        j = i
        while j < n and c[j] == c[i]:
            j += 1
        g, eg = j - i, e[i:j].sum()
        for m in range(1, g + 1):
            risk_sum += (errs + eg * m / g) / (done + m)
        done, errs, i = j, errs + eg, j
    return risk_sum / n


rng = np.random.default_rng(12345)
rows = []
for m, s, t in cells:
    P, y = A[f"{m}|{s}|{t}|P"], A[f"{m}|{s}|{t}|y"]
    conf, cor = P.max(1), (P.argmax(1) == y).astype(float)
    ck = np.round(conf, 9)
    st = np.argsort(ck, kind="stable")
    eces = []
    for _ in range(200):
        perm = rng.permutation(len(conf))
        eces.append(ece_order(conf, cor, perm[np.argsort(ck[perm], kind="stable")]))
    nties = len(ck) - len(np.unique(ck))
    rows.append({"model": m, "scorer": s, "task": t, "k": K[t], "accuracy": cor.mean(), "ece_stable": ece_order(conf, cor, st),
                 "ece_tie_min": min(eces), "ece_tie_max": max(eces), "ece_pipeline": q.loc[(m, s, t), "ece"],
                 "cov_thr_1": cov_thr(conf, cor, 0.01), "cov_thr_2": cov_thr(conf, cor, 0.02), "cov_pipe_1": q.loc[(m, s, t), "coverage_at_1pct"],
                 "cov_pipe_2": q.loc[(m, s, t), "coverage_at_2pct"], "aurc_tie": aurc_tie(conf, cor), "aurc_pipe": q.loc[(m, s, t), "aurc"],
                 "brier": q.loc[(m, s, t), "brier"], "macro_f1": q.loc[(m, s, t), "macro_f1"], "n_tied": nties,
                 "share_conf_ge_0_999": float((conf >= 0.999).mean())})
MET = pd.DataFrame(rows)
MET.to_csv(RC / "metrics.csv", index=False)
mt = MET.set_index(["model", "scorer", "task"])
put("ece_pipeline_outside_tie_range", MET[(MET.ece_pipeline < MET.ece_tie_min - 1e-12) | (MET.ece_pipeline > MET.ece_tie_max + 1e-12)][["model", "scorer", "task"]].values.tolist())
put("ece_tie_spread_max", float((MET.ece_tie_max - MET.ece_tie_min).max()))
put("ece_tie_spread_max_cell", MET.loc[(MET.ece_tie_max - MET.ece_tie_min).idxmax(), ["model", "scorer", "task"]].tolist())
put("ece_stable_minus_pipeline_absmax", float((MET.ece_stable - MET.ece_pipeline).abs().max()))
d2 = (MET.cov_thr_2 - MET.cov_pipe_2)
put("cov2_thr_minus_pipe_range", [float(d2.min()), float(d2.max())])
put("cov2_thr_differs_cells", MET[d2.abs() > 1e-9][["model", "scorer", "task", "cov_pipe_2", "cov_thr_2"]].values.tolist())
put("aurc_tie_minus_pipe_absmax", float((MET.aurc_tie - MET.aurc_pipe).abs().max()))
ex = E[E["compare"] == "cached_vs_trie"]
put("bf16_cached_vs_trie", {m: {"max": g.max_abs_diff.max(), "mean_of_task_means": g.mean_abs_diff.mean(), "task_mean_range": [g.mean_abs_diff.min(), g.mean_abs_diff.max()],
                                "argmax_min": g.argmax_agree.min()} for m, g in ex.groupby("model")})
acc_ct = {m: max(abs(q.loc[(m, "cached", t), "accuracy"] - q.loc[(m, "trie", t), "accuracy"]) for t in TASKS) for m in SMALL}
put("bf16_cached_vs_trie_accuracy_absmax", acc_ct)
ev = E[E["compare"] == "vtrie_vs_trie"]
put("vtrie_vs_trie", {m: {"max": g.max_abs_diff.max(), "mean_of_task_means": g.mean_abs_diff.mean(), "task_mean_range": [g.mean_abs_diff.min(), g.mean_abs_diff.max()],
                          "argmax_min": g.argmax_agree.min()} for m, g in ev.groupby("model")})
cv = C[C.b == "vtrie"]
put("cached_vs_vtrie_accuracy", {"absmax": cv.acc_diff.abs().max(), "n_sig_holm": int((cv.p_holm < 0.05).sum()), "n_sig_raw": int((cv.p_mcnemar < 0.05).sum())})
e32 = E32.set_index(["model", "task"])
win = ["tw_sentiment", "tw_topic", "bitext", "clinc_k10", "clinc_k25", "clinc_k50"]
put("fp32_qwen", {"max": E32[E32.model == "qwen2b"].max_abs_diff.max(), "mean_max": E32[E32.model == "qwen2b"].mean_abs_diff.max(), "argmax_min": E32[E32.model == "qwen2b"].argmax_agree.min()})
put("fp32_gemma_window", {"max": max(e32.loc[("gemmaE2b", t), "max_abs_diff"] for t in win), "argmax_min": min(e32.loc[("gemmaE2b", t), "argmax_agree"] for t in win)})
put("fp32_gemma_beyond", {t: [e32.loc[("gemmaE2b", t), "max_abs_diff"], e32.loc[("gemmaE2b", t), "argmax_agree"]] for t in ("banking77", "clinc_k100", "clinc150")})
fx = X["fp32_exception"]
put("fp32_items_gt_1e-3", {k: v["n_diff_gt_1e-3"] for k, v in fx.items() if v["n_diff_gt_1e-3"]})
put("fp32_items_gt_0_1", {k: sum(i["fp32_diff"] > 0.1 for i in v["items"]) for k, v in fx.items() if v["n_diff_gt_1e-3"]})
its = [i for v in fx.values() for i in v["items"]]
put("fp32_exception_bf16_agree", {"argmax_all_same": all(i["argmax_bf16_all_same"] for i in its), "max_bf16_trie_vs_cached": max(i["bf16_trie_vs_cached"] for i in its),
                                  "max_bf16_trie_vs_vtrie": max(i["bf16_trie_vs_vtrie"] for i in its),
                                  "trie_only": sum(i["fp32_trie_vs_bf16_trie"] > 0.1 and i["fp32_cached_vs_bf16_trie"] <= 0.1 for i in its),
                                  "cached_only": sum(i["fp32_cached_vs_bf16_trie"] > 0.1 and i["fp32_trie_vs_bf16_trie"] <= 0.1 for i in its),
                                  "both": sum(i["fp32_cached_vs_bf16_trie"] > 0.1 and i["fp32_trie_vs_bf16_trie"] > 0.1 for i in its), "n": len(its)})
pt = X["prompt_tokens"]
put("gemma_prompt_tokens_window_tasks_max", max(pt[f"gemmaE2b-first/{t}"][2] for t in win))
put("gemma_prompt_tokens_beyond_min", min(pt[f"gemmaE2b-first/{t}"][0] for t in ("banking77", "clinc_k100", "clinc150")))
tok = {(m, s): {K[t]: q.loc[(m, s, t), "forward_tokens"] for t in TASKS if (m, s, t) in q.index} for m in SMALL for s in ("trie", "cached", "first", "letters") if (m, s, TASKS[0]) in q.index}
put("forward_tokens", {f"{m}/{s}": {k: v[k] for k in sorted(v) if not np.isnan(v[k])} for (m, s), v in tok.items()})
put("cached_over_prompt", {m: [q.loc[(m, "cached", t), "forward_tokens"] / q.loc[(m, "first", t), "forward_tokens"] for t in TASKS] for m in SMALL})
put("trie_calls", {k: v for k, v in X["trie_calls_mean"].items()})
lat = L.set_index(["model", "scorer", "task"])["mean"]
put("latency_ms", {f"{m}/{s}/{t}": lat[(m, s, t)] for m in SMALL for s in ("trie", "cached", "first", "letters") for t in ("tw_sentiment", "tw_topic", "bitext", "banking77", "clinc150")})
put("speedup_151", {m: lat[(m, "trie", "clinc150")] / lat[(m, "cached", "clinc150")] for m in SMALL})
put("speedup_77", {m: lat[(m, "trie", "banking77")] / lat[(m, "cached", "banking77")] for m in SMALL})
put("first_latency_range", [min(lat[(m, "first", t)] for m in SMALL for t in ("tw_sentiment", "tw_topic", "bitext", "banking77", "clinc150")),
                            max(lat[(m, "first", t)] for m in SMALL for t in ("tw_sentiment", "tw_topic", "bitext", "banking77", "clinc150"))])
put("latency_letters_empty_rows", L[L["size"] == 3][["model", "scorer", "task", "mean"]][L["mean"] < 1].values.tolist())
put("vtrie_sec", {f"{m}/{t}": q.loc[(m, "vtrie", t), "sec_per_decision"] for m in SMALL + LARGE for t in TASKS})
put("gen_sec", {f"{m}/{t}": q.loc[(m, "gen", t), "sec_per_decision"] for m in SMALL + LARGE for t in TASKS})
for b in ("first", "letters", "gen", "vtrie"):
    c = C[C.b == b]
    put(f"cached_minus_{b}", {"mean": c.acc_diff.mean(), "min": c.acc_diff.min(), "max": c.acc_diff.max(), "absmax": c.acc_diff.abs().max(),
                              "n": len(c), "n_sig_holm": int((c.p_holm < 0.05).sum()), "n_sig_raw": int((c.p_mcnemar < 0.05).sum()),
                              "sig_cells": c[c.p_holm < 0.05][["model", "task", "acc_diff"]].values.tolist(), "nonsig_cells": c[c.p_holm >= 0.05][["model", "task", "acc_diff"]].values.tolist()})
put("holm_family_size_small", len(C))
put("accuracy", {f"{m}/{s}/{t}": q.loc[(m, s, t), "accuracy"] for (m, s, t) in q.index})
put("first_mass_ge10", FB[FB.scorer == "first"].set_index(["model", "task"]).mass_on_options_ge10.to_dict())
put("cached_mass_ge10", FB[FB.scorer == "cached"].set_index(["model", "task"]).mass_on_options_ge10.to_dict())
put("gold_share_ge10", FB[FB.scorer == "first"].set_index(["model", "task"]).gold_share_ge10.to_dict())
fsh = {}
for m in SMALL:
    for t in TASKS:
        if K[t] >= 10:
            P = A[f"{m}|first|{t}|P"]
            fsh[f"{m}/{t}"] = float((P.argmax(1) < 9).mean())
put("first_pred_option_le9", fsh)
ids = lambda m, s, t: A[f"{m}|{s}|{t}|ids"]
lt = []
for m in LARGE:
    for t in TASKS:
        a, b = A[f"{m}|vtrie|{t}|P"], A[f"{m}|gen|{t}|P"]
        assert (ids(m, "vtrie", t) == ids(m, "gen", t)).all()
        y = A[f"{m}|vtrie|{t}|y"]
        ca, cb = a.argmax(1) == y, b.argmax(1) == y
        n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
        n = n01 + n10
        p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, j) for j in range(min(n01, n10) + 1)) / 2 ** n)
        lt.append({"model": m, "task": t, "k": K[t], "diff": ca.mean() - cb.mean(), "p": p})
LT = pd.DataFrame(lt)
o = np.argsort(LT.p.to_numpy(), kind="stable")
adj, run = np.empty(len(LT)), 0.0
for r_, i in enumerate(o):
    run = max(run, min(1.0, (len(LT) - r_) * LT.p.iloc[i]))
    adj[i] = run
LT["p_holm"] = adj
LT.to_csv(RC / "large_vtrie_vs_gen.csv", index=False)
put("large_vtrie_minus_gen", {"mean": LT["diff"].mean(), "min": LT["diff"].min(), "max": LT["diff"].max(), "n_sig_holm": int((LT.p_holm < 0.05).sum()),
                              "sig": LT[LT.p_holm < 0.05][["model", "task", "diff"]].values.tolist()})
for m in LARGE + SMALL:
    for s in ("vtrie", "gen", "cached", "first"):
        if (m, s, TASKS[0]) in mt.index:
            put(f"ece_{m}_{s}", {t: mt.loc[(m, s, t), "ece_stable"] for t in TASKS})
put("ece_ref", {f"{m}/{t}": mt.loc[(m, "reference", t), "ece_stable"] for m in ("clef", "jev") for t in TASKS})
for c_ in ("cov_thr_1", "cov_thr_2", "aurc_tie"):
    put(c_, {f"{m}/{s}/{t}": mt.loc[(m, s, t), c_] for (m, s, t) in mt.index})
put("share_conf_ge_0_999", {f"{m}/{s}/{t}": mt.loc[(m, s, t), "share_conf_ge_0_999"] for (m, s, t) in mt.index})
def cov_raw(m, s, t, r=0.02):
    P, y = A[f"{m}|{s}|{t}|P"], A[f"{m}|{s}|{t}|y"]
    conf, cor = P.max(1), P.argmax(1) == y
    best = 0.0
    for th_ in np.unique(conf):
        acc_ = conf >= th_
        if (1 - cor[acc_].mean()) <= r + 1e-12:
            best = max(best, acc_.mean())
    return best
put("cov2_unrounded", {f"{m}/{s}/{t}": cov_raw(m, s, t) for (m, s, t) in mt.index})
put("cov1_unrounded", {f"{m}/{s}/{t}": cov_raw(m, s, t, 0.01) for (m, s, t) in mt.index})
put("cov2_rounding_sensitive", [k for k in mt.index if abs(cov_raw(*k) - mt.loc[k, "cov_thr_2"]) > 1e-9])
put("share_conf_within_1e9_of_1", {f"{m}/{s}/{t}": float((A[f"{m}|{s}|{t}|P"].max(1) > 1 - 1e-9).mean()) for (m, s, t) in mt.index})
put("n_tied", {f"{m}/{s}/{t}": int(mt.loc[(m, s, t), "n_tied"]) for (m, s, t) in mt.index})
(RC / "derived_facts.json").write_text(json.dumps(js(F), indent=1, default=str))
