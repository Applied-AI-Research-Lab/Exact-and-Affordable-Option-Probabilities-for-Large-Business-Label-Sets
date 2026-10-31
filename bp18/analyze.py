from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from . import stats as S
from .config import BOOTSTRAP_N, TASKS, records_path, results_dir
from .utils import get_logger, iter_jsonl, latest_ok, read_jsonl

log = get_logger("analyze")


def parse_key(key: str) -> tuple[str, str]:
    if key.endswith("_readout"):
        return key[:-8], "vtrie"
    if key.endswith("_gen"):
        return key[:-4], "gen"
    if "-" in key:
        m, s = key.rsplit("-", 1)
        return m, s
    return key, "reference"


def load(run: str):
    data = {}
    raw = results_dir(run) / "raw"
    for task in TASKS:
        rp = records_path(task, "test")
        if not rp.exists():
            continue
        recs = {r["id"]: r for r in read_jsonl(rp)}
        for f in raw.glob(f"*/{task}/test.jsonl") if raw.exists() else []:
            preds = latest_ok(f)
            ids = [i for i in recs if i in preds]
            if not ids:
                continue
            P = np.array([[preds[i]["probs"].get(o, 0.0) for o in recs[i]["options"]] for i in ids], float)
            P = P / np.clip(P.sum(1, keepdims=True), 1e-12, None)
            y = np.array([recs[i]["options"].index(recs[i]["gold"]) for i in ids])
            opts = recs[ids[0]]["options"]
            model, scorer = parse_key(f.parent.parent.name)
            data[(model, scorer, task)] = {"ids": ids, "P": P, "y": y, "k": len(opts),
                                           "sec": np.array([preds[i]["latency_s"] or 0.0 for i in ids]),
                                           "tok": np.array([preds[i]["input_tokens"] or 0 for i in ids], float)}
    return data


def aligned(a, b):
    ib = {i: j for j, i in enumerate(b["ids"])}
    idx = [(j, ib[i]) for j, i in enumerate(a["ids"]) if i in ib]
    ja, jb = zip(*idx) if idx else ([], [])
    return list(ja), list(jb)


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
            rows.append({"model": m, "task": t, "k": d["k"], "compare": f"{other}_vs_trie", "n": len(ja),
                         "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
                         "argmax_agree": float((d["P"][ja].argmax(1) == o["P"][jb].argmax(1)).mean())})
    return pd.DataFrame(rows)


def quality(data):
    rows = []
    for (m, s, t), d in data.items():
        P, y = d["P"], d["y"]
        rows.append({"model": m, "scorer": s, "task": t, "k": d["k"], "n": len(y), "accuracy": M.accuracy(P, y),
                     "macro_f1": M.macro_f1(P, y), "ece": M.ece_equal_mass(P, y), "brier": M.brier(P, y),
                     "aurc": M.aurc(P, y), "coverage_at_1pct": M.coverage_at_risk(P, y, 0.01),
                     "coverage_at_2pct": M.coverage_at_risk(P, y, 0.02),
                     "sec_per_decision": float(d["sec"].mean()), "forward_tokens": float(d["tok"].mean())})
    return pd.DataFrame(rows)


def first_bias(data):
    rows = []
    for (m, s, t), d in data.items():
        if d["k"] < 10 or s not in ("first", "cached", "trie"):
            continue
        rows.append({"model": m, "scorer": s, "task": t, "k": d["k"],
                     "mass_on_options_ge10": float(d["P"][:, 9:].sum(1).mean()),
                     "gold_share_ge10": float((d["y"] >= 9).mean())})
    return pd.DataFrame(rows)


def comparisons(data, n_boot):
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
            diff, lo, hi = S.bootstrap_diff_ci(ca, cb, n_boot=n_boot)
            rows.append({"model": m, "task": t, "k": d["k"], "a": "cached", "b": other, "n": len(ja), "acc_diff": diff,
                         "lo": lo, "hi": hi, "p_mcnemar": S.mcnemar_exact(ca, cb),
                         "coverage1_a": M.coverage_at_risk(d["P"][ja], d["y"][ja], 0.01),
                         "coverage1_b": M.coverage_at_risk(o["P"][jb], o["y"][jb], 0.01)})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = S.holm(out.p_mcnemar.tolist())
    return out


def latency(run):
    rows = []
    for f in (results_dir(run) / "latency").glob("*.jsonl"):
        rows += list(iter_jsonl(f))
    d = pd.DataFrame(rows)
    return d.groupby(["model", "scorer", "task"]).ms_per_decision.agg(["mean", "std", "size"]).reset_index() if len(d) else d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="main")
    ap.add_argument("--n-boot", type=int, default=BOOTSTRAP_N)
    args = ap.parse_args()
    out_t, out_f = results_dir(args.run_name) / "tables", results_dir(args.run_name) / "figures"
    out_t.mkdir(parents=True, exist_ok=True)
    out_f.mkdir(parents=True, exist_ok=True)
    data = load(args.run_name)
    if not data:
        log.warning("no predictions")
        return
    q = quality(data)
    tables = {"exactness": exactness(data), "quality": q, "cost": q[["model", "scorer", "task", "k", "sec_per_decision",
                                                                         "forward_tokens"]],
              "first_bias": first_bias(data), "comparisons": comparisons(data, args.n_boot), "latency": latency(args.run_name)}
    for name, t in tables.items():
        t.to_csv(out_t / f"{name}.csv", index=False)
    log.info("tables written to %s", out_t)
    make_figures(q, out_f)


def make_figures(q, out_f: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for (m, s), g in q.groupby(["model", "scorer"]):
        g = g.sort_values("k")
        axes[0].plot(g.k, g.forward_tokens.clip(lower=1), marker="o", label=f"{m}/{s}")
        axes[1].plot(g.k, g.ece, marker="o", label=f"{m}/{s}")
    axes[0].set_xscale("log"); axes[0].set_yscale("log"); axes[0].set_xlabel("number of options")
    axes[0].set_ylabel("forward tokens per decision"); axes[1].set_xscale("log"); axes[1].set_xlabel("number of options")
    axes[1].set_ylabel("ECE"); axes[0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out_f / "cost_and_calibration.png", dpi=200); plt.close(fig)
    log.info("figures written to %s", out_f)


if __name__ == "__main__":
    main()
