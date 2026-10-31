from __future__ import annotations

import argparse
import random

from . import sources as S
from .config import BASE_TASKS, N_TEST, SEED, SWEEP, records_path
from .utils import get_logger, write_jsonl

log = get_logger("data")
SRC = {"tw_sentiment": S.src_tw_sentiment, "tw_topic": S.src_tw_topic, "bitext": S.src_bitext,
       "banking77": S.src_banking77}


def records(task, rows, labels):
    prefix, instr = S.TASK_TEXT[task]
    crit = S.criteria_for(task, labels)
    return [{"id": f"{task}-{i:05d}", "task": task, "condition": "test", "qtype": "choice", "text": t,
             "state": prefix + t, "gold": y, "instructions": instr, "options": list(labels),
             "criteria": {l: crit[l] for l in labels}, "meta": {"k": len(labels)}} for i, (t, y) in enumerate(rows)]


def sweep(recs: list[dict], k: int) -> list[dict]:
    out = []
    for r in recs:
        rng = random.Random(f"{SEED}-{r['id']}")
        others = [o for o in r["options"] if o != r["gold"]]
        rng.shuffle(others)
        keep = set([r["gold"]] + others[: k - 1])
        opts = [o for o in r["options"] if o in keep]
        out.append({**r, "id": r["id"].replace("clinc150", f"clinc_k{k}"), "task": f"clinc_k{k}",
                    "options": opts, "criteria": {o: r["criteria"][o] for o in opts}, "meta": {"k": k}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    n = args.limit or N_TEST
    built = {}
    for task in BASE_TASKS:
        if task == "clinc150":
            rows, labels = S.src_clinc()
        else:
            tr, te, labels = SRC[task]()
            rows = te if te is not None else tr
        built[task] = records(task, S.take(rows, n, f"{SEED}-{task}"), labels)
    for k in SWEEP:
        built[f"clinc_k{k}"] = sweep(built["clinc150"], k)
    for task, recs in built.items():
        write_jsonl(records_path(task, "test"), recs)
        log.info("%s: %d items, %d options", task, len(recs), len(recs[0]["options"]))


if __name__ == "__main__":
    main()
