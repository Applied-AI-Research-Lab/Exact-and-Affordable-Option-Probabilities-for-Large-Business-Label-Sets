from __future__ import annotations

import argparse
import json
import time

from .config import BASE_TASKS, HF_SCORERS, LATENCY_ITEMS, REPEATS, SMALL_MODELS, TASKS, raw_path, records_path, results_dir
from .readout import char_token_ids
from .runners.common import pred_row
from .scoring import Scorer
from .student import load_model
from .utils import append_jsonl, done_ids, get_logger, read_jsonl

log = get_logger("score")


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(SMALL_MODELS), required=True)
    ap.add_argument("--scorers", default=",".join(HF_SCORERS))
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--run-name", default="main")
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16",
                    help="model precision on GPU (fp32 isolates numerical rounding in the cached-vs-trie check)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = load_model(SMALL_MODELS[args.model], device, torch.bfloat16 if device == "cuda" and args.dtype == "bf16" else torch.float32)
    model.eval()
    sc = Scorer(model, tok, char_token_ids(tok), device)
    tasks = TASKS if args.tasks == "all" else args.tasks.split(",")
    for scorer in args.scorers.split(","):
        key = f"{args.model}-{scorer}"
        for task in tasks:
            recs = read_jsonl(records_path(task, "test"))[: args.limit or None]
            if args.timing:
                if task not in BASE_TASKS:
                    continue
                for rep in range(REPEATS):
                    t0 = time.perf_counter()
                    n = 0
                    for r in recs[:LATENCY_ITEMS]:
                        p, _ = sc.score(r, scorer)
                        n += p is not None
                    if device == "cuda":
                        torch.cuda.synchronize()
                    sec = (time.perf_counter() - t0) / max(n, 1)
                    out = results_dir(args.run_name) / "latency" / f"{key}.jsonl"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with open(out, "a") as f:
                        f.write(json.dumps({"model": args.model, "scorer": scorer, "task": task, "rep": rep, "n": n,
                                            "ms_per_decision": 1000 * sec}) + "\n")
                continue
            out = raw_path(args.run_name, key, task, "test")
            done = done_ids(out)
            rows = []
            for r in recs:
                if r["id"] in done:
                    continue
                t0 = time.perf_counter()
                p, d = sc.score(r, scorer)
                if p is None:
                    break
                rows.append(pred_row(r, key, dict(zip(r["options"], p)), time.perf_counter() - t0, d.get("forward_tokens"), d))
                if len(rows) >= 100:
                    append_jsonl(out, rows)
                    rows = []
            if rows:
                append_jsonl(out, rows)
            log.info("%s %s done", key, task)


if __name__ == "__main__":
    main()
