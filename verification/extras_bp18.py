import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
F = {}
BASE = ["tw_sentiment", "tw_topic", "bitext", "banking77", "clinc150"]
SWEEP = ["clinc_k10", "clinc_k25", "clinc_k50", "clinc_k100"]
TASKS = BASE + SWEEP


def put(k, v):
    F[k] = v
    print(f"{k}: {str(v)[:400]}", flush=True)


def jl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


REC = {t: jl(ROOT / f"data/records/{t}/test.jsonl") for t in TASKS}
put("items", {t: len(REC[t]) for t in TASKS})
put("options", {t: sorted({len(r["options"]) for r in REC[t]}) for t in TASKS})
put("gold_in_options", all(r["gold"] in r["options"] for t in TASKS for r in REC[t]))
put("ids_unique", len({r["id"] for t in TASKS for r in REC[t]}) == sum(len(REC[t]) for t in TASKS))
put("labels_used", {t: len({r["gold"] for r in REC[t]}) for t in TASKS})
put("label_share_max", {t: Counter(r["gold"] for r in REC[t]).most_common(1)[0][1] / len(REC[t]) for t in TASKS})
c150 = {r["id"].split("-")[1]: r for r in REC["clinc150"]}
full_opts = REC["clinc150"][0]["options"]
put("clinc150_oos_option", full_opts[-1])
put("clinc150_oos_items", sum(r["gold"] == full_opts[-1] for r in REC["clinc150"]))
nested, canon, same_text = True, True, True
for i, k in enumerate(SWEEP):
    for r in REC[k]:
        base = c150[r["id"].split("-")[1]]
        same_text &= r["text"] == base["text"] and r["gold"] == base["gold"]
        canon &= r["options"] == [o for o in full_opts if o in set(r["options"])]
        if i + 1 < len(SWEEP):
            nxt = {x["id"].split("-")[1]: x for x in REC[SWEEP[i + 1]]}[r["id"].split("-")[1]]
            nested &= set(r["options"]) <= set(nxt["options"])
put("sweep_same_texts_and_gold", same_text)
put("sweep_canonical_order", canon)
put("sweep_nested", nested)
put("sweep_distinct_taxonomies", {k: len({tuple(r["options"]) for r in REC[k]}) for k in SWEEP})
gold_pos = {t: [r["options"].index(r["gold"]) for r in REC[t]] for t in TASKS}
put("gold_share_option_ge10", {t: float(np.mean([p >= 9 for p in gold_pos[t]])) for t in TASKS})
RAW = ROOT / "results/main/raw"
keys = sorted(p.name for p in RAW.iterdir() if p.is_dir())
put("keys", keys)
rows_per_key, nrow, nerr = {}, 0, 0
valid, conf_vals, calls, mass, ptok = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
jtok, jreq = 0, 0
for key in keys:
    n = 0
    for f in sorted((RAW / key).glob("*/test.jsonl")):
        t = f.parent.name
        for r in jl(f):
            n += 1
            nerr += bool(r.get("error"))
            me = r.get("meta") or {}
            if key.endswith("_gen"):
                valid[key].append(bool(me.get("valid")))
                conf_vals[(key, t)].append(max(r["probs"].values()))
            if key.endswith("-trie") or key.endswith("_readout"):
                if "calls" in me:
                    calls[(key, t)].append(me["calls"])
                if "mass" in me:
                    mass[(key, t)].append(me["mass"])
            if key.endswith("-first"):
                ptok[(key, t)].append(r["input_tokens"])
            if key == "jev":
                jtok += r.get("input_tokens") or 0
                jreq += 1
    rows_per_key[key] = n
    nrow += n
put("rows_per_key", rows_per_key)
put("raw_rows", nrow)
put("error_rows", nerr)
put("gen_valid_share", {k: float(np.mean(v)) for k, v in valid.items()})
put("gen_invalid_count", {k: int(len(v) - sum(v)) for k, v in valid.items()})
put("gen_top_conf_distinct_values", {f"{k}/{t}": len(set(np.round(v, 6))) for (k, t), v in conf_vals.items()})
put("gen_share_conf_ge_0_99", {f"{k}/{t}": float(np.mean(np.array(v) >= 0.99)) for (k, t), v in conf_vals.items()})
put("trie_calls_mean", {f"{k}/{t}": float(np.mean(v)) for (k, t), v in calls.items()})
put("trie_calls_max", {f"{k}/{t}": int(np.max(v)) for (k, t), v in calls.items()})
put("readout_mass_min", {f"{k}/{t}": float(np.min(v)) for (k, t), v in mass.items()})
put("readout_mass_mean", {f"{k}/{t}": float(np.mean(v)) for (k, t), v in mass.items()})
put("prompt_tokens", {f"{k}/{t}": [int(np.min(v)), float(np.mean(v)), int(np.max(v))] for (k, t), v in ptok.items()})
qp = {}
for t in TASKS:
    v = [r["input_tokens"] for r in jl(RAW / f"qwen_readout/{t}/test.jsonl")]
    qp[t] = [int(min(v)), float(np.mean(v)), int(max(v)), int(sum(x >= 784 for x in v))]
put("qwen38_prompt_tokens", qp)
dec = Counter()
for k in keys:
    if k.endswith("_gen"):
        for f in (RAW / k).glob("*/test.jsonl"):
            dec.update((r.get("meta") or {}).get("decoding") for r in jl(f))
put("gen_decoding", dict(dec))
put("jev_requests_main", jreq)
put("jev_input_tokens_main", jtok)
put("jev_usd_main", jtok * 0.042 / 1e6)
if (ROOT / "results/pilot/raw").exists():
    pj, pt = 0, 0
    for f in (ROOT / "results/pilot/raw/jev").glob("*/test.jsonl"):
        rr = jl(f)
        pj += len(rr)
        pt += sum(r.get("input_tokens") or 0 for r in rr)
    put("jev_requests_pilot", pj)
    put("jev_input_tokens_pilot", pt)
    put("pilot_rows_per_file", sorted({len(jl(f)) for f in (ROOT / "results/pilot/raw").glob("*/*/test.jsonl")}))
    put("pilot_keys", sorted({f.parent.parent.name for f in (ROOT / "results/pilot/raw").glob("*/*/test.jsonl")}))
def load_probs(run, key, t):
    p = ROOT / f"results/{run}/raw/{key}/{t}/test.jsonl"
    rec = {r["id"]: r for r in REC[t]}
    out = {}
    for r in jl(p):
        if not r.get("error"):
            v = np.array([r["probs"].get(o, 0.0) for o in rec[r["id"]]["options"]])
            out[r["id"]] = v / v.sum()
    return out


fp = {}
for m in ("gemmaE2b", "qwen2b"):
    for t in TASKS:
        a, b = load_probs("fp32", f"{m}-trie", t), load_probs("fp32", f"{m}-cached", t)
        bt, bc = load_probs("main", f"{m}-trie", t), load_probs("main", f"{m}-cached", t)
        vt = load_probs("main", f"{m}_readout", t)
        big = [i for i in a if np.abs(a[i] - b[i]).max() > 1e-3]
        info = []
        for i in big:
            info.append({"id": i, "fp32_diff": float(np.abs(a[i] - b[i]).max()),
                         "bf16_trie_vs_cached": float(np.abs(bt[i] - bc[i]).max()), "bf16_trie_vs_vtrie": float(np.abs(bt[i] - vt[i]).max()),
                         "fp32_trie_vs_bf16_trie": float(np.abs(a[i] - bt[i]).max()), "fp32_cached_vs_bf16_trie": float(np.abs(b[i] - bt[i]).max()),
                         "argmax_bf16_all_same": bool(bt[i].argmax() == bc[i].argmax() == vt[i].argmax()),
                         "argmax_fp32": [int(a[i].argmax()), int(b[i].argmax())], "argmax_bf16": int(bt[i].argmax())})
        fp[f"{m}/{t}"] = {"n": len(a), "n_diff_gt_1e-3": len(big), "items": info}
put("fp32_exception", fp)
lat = []
for f in sorted((ROOT / "results/main/latency").glob("*.jsonl")):
    lat += jl(f)
put("latency_rows", len(lat))
put("latency_n_values", sorted({r["n"] for r in lat}))
put("latency_rep0_minus_mean_rep12", {f"{r0['model']}/{r0['scorer']}/{r0['task']}": r0["ms_per_decision"] - np.mean([r["ms_per_decision"] for r in lat if (r["model"], r["scorer"], r["task"]) == (r0["model"], r0["scorer"], r0["task"]) and r["rep"] > 0])
                                     for r0 in lat if r0["rep"] == 0})
if (ROOT / "logs").exists() and (ROOT / "results/main/CODE_MANIFEST.txt").exists():
    spans = {}
    for f in sorted((ROOT / "logs").glob("*.log")):
        ts = re.findall(r"20\d\d-\d\d-\d\d \d\d:\d\d:\d\d", f.read_text(errors="ignore"))
        if ts:
            spans[f.name] = [ts[0], ts[-1]]
    put("log_spans", spans)
    q = {}
    for f in sorted((ROOT / "logs").glob("full_gpu*_queue.log")):
        for ln in f.read_text().splitlines():
            m_ = re.match(r"(\w{3} \w{3} \d+ \d\d:\d\d:\d\d [AP]M) EEST (\d{4}) (GPU\d) (claims shard|shard) (\d)/8(?: exit (\d+))?", ln)
            if m_:
                ts = datetime.strptime(f"{m_.group(1)} {m_.group(2)}", "%a %b %d %I:%M:%S %p %Y")
                s = m_.group(5)
                if m_.group(4) == "claims shard":
                    q.setdefault(s, {})["start"], q[s]["gpu"] = ts, m_.group(3)
                else:
                    q.setdefault(s, {})["end"], q[s]["exit"] = ts, int(m_.group(6))
    put("qwen_shards", {s: {"gpu": v["gpu"], "hours": (v["end"] - v["start"]).total_seconds() / 3600, "exit": v["exit"]} for s, v in sorted(q.items())})
    put("qwen_shard_gpu_hours", sum((v["end"] - v["start"]).total_seconds() for v in q.values()) / 3600)
    man = (ROOT / "results/main/CODE_MANIFEST.txt").read_text().splitlines()
    launch = datetime.strptime(" ".join(man[0].split()[1:5] + man[0].split()[6:]), "%b %d %I:%M:%S %p %Y")
    put("manifest_launch", man[0])
    gpu_end = {}
    for v in q.values():
        gpu_end[v["gpu"]] = max(gpu_end.get(v["gpu"], launch), v["end"])
    put("gpu_last_end", {g: str(e) for g, e in sorted(gpu_end.items())})
    put("gpu_hours_full_wall", sum((e - launch).total_seconds() for e in gpu_end.values()) / 3600)
    put("full_end", str(max(gpu_end.values())))
    dur = lambda a, b: (datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds() / 3600
    put("gpu_hours_pilot_logs", sum(dur(*v) for k, v in spans.items() if k.startswith("pilot_gpu")))
    import hashlib
    mf = [ln.split() for ln in man[1:] if ln.strip()]
    put("manifest_files", len(mf))
    put("manifest_files_changed", [f for h, f in mf if hashlib.sha256((ROOT / f).read_bytes()).hexdigest() != h])
    blk = Counter(re.findall(r"Setting attention block size to (\d+) tokens", "".join((ROOT / "logs" / n).read_text(errors="ignore") for n in ["full_gpu1.log"] + [f"main_qwen_readout_shard{i}.log" for i in range(8)])))
    put("qwen38_attention_block_size", dict(blk))
    vv = re.findall(r"Initializing a V1 LLM engine \((v[\d.]+)\)", (ROOT / "logs/full_gpu1.log").read_text(errors="ignore"))
    put("vllm_version", sorted(set(vv)))
(OUT / "extras_facts.json").write_text(json.dumps(F, indent=1, default=str))
