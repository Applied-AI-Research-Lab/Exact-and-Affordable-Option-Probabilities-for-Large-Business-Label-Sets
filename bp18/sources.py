from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import zipfile

from .config import (CFPB_DATE_MAX, CFPB_DATE_MIN, CFPB_MIRROR, HELD_OUT, IN_DOMAIN, N_TEST, N_TRAIN, N_VAL, POOL, SEED,
                     TASKS_DIR, TEXT_MAX_CHARS)
from .utils import get_logger, write_jsonl

log = get_logger("data")

BANKING_URLS = {s: f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/{s}.csv"
                for s in ("train", "test")}
BITEXT = ("bitext/Bitext-customer-support-llm-chatbot-training-dataset",
          "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")
SENT_REPO, TOPIC_REPO = "zeroshot/twitter-financial-news-sentiment", "zeroshot/twitter-financial-news-topic"
PB_REPO, PB_ZIP = "takala/financial_phrasebank", "data/FinancialPhraseBank-v1.0.zip"
FOMC_REPO, CLINC_REPO = "gtfintechlab/fomc_communication", "clinc/clinc_oos"
TOPIC_NAMES = ["Analyst Update", "Fed | Central Banks", "Company | Product News", "Treasuries | Corporate Debt",
               "Dividend", "Earnings", "Energy | Oil", "Financials", "Currencies", "General News | Opinion",
               "Gold | Metals | Materials", "IPO", "Legal | Regulation", "M&A | Investments", "Macro", "Markets",
               "Politics", "Personnel Change", "Stock Commentary", "Stock Movement"]
CFPB_PRODUCTS = {
    "Checking or savings account": "checking, savings, CDs and other deposit accounts",
    "Credit card": "credit cards and store cards",
    "Credit reporting or other personal consumer reports": "credit reports, credit scores and consumer reports",
    "Debt collection": "attempts to collect a debt",
    "Debt or credit management": "debt settlement, credit repair and debt management services",
    "Money transfer, virtual currency, or money service": "money transfers, virtual currency, check cashing",
    "Mortgage": "mortgages and home equity loans",
    "Payday loan, title loan, personal loan, or advance loan": "payday, title, installment and personal loans",
    "Prepaid card": "prepaid and gift cards",
    "Student loan": "federal and private student loans",
    "Vehicle loan or lease": "auto loans and leases",
}
TASK_TEXT = {
    "banking77": ("Customer message: ", "Which customer-service intent does this bank customer's message express?"),
    "bitext": ("Customer message: ", "Which customer-service request does this message express?"),
    "tw_sentiment": ("Financial news tweet: ", "What is the market sentiment of this financial news tweet?"),
    "tw_topic": ("Financial news tweet: ", "What is the main topic of this financial news tweet?"),
    "cfpb": ("Consumer complaint narrative: ", "Which financial product is this consumer complaint about?"),
    "phrasebank": ("Sentence from a financial news article: ", "What is the sentiment of this sentence for the company's investors?"),
    "fomc": ("Sentence from a Federal Reserve (FOMC) communication: ", "What monetary-policy stance does this sentence express?"),
    "clinc150": ("User request to a virtual assistant: ", "Which intent does this request express? Choose out of scope if none fits."),
}
SPECIAL = {"tw_sentiment": {"bearish": "Bearish: negative for the market or the stock.",
                            "bullish": "Bullish: positive for the market or the stock.",
                            "neutral": "Neutral: neither clearly positive nor negative."},
           "phrasebank": {"negative": "Negative", "neutral": "Neutral", "positive": "Positive"},
           "fomc": {"dovish": "Dovish: favours easier monetary policy.", "hawkish": "Hawkish: favours tighter monetary policy.",
                    "neutral": "Neutral: neither dovish nor hawkish."}}


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower().replace("&", " and ")).strip("_")[:60]


def humanize(label: str) -> str:
    s = label.replace("_", " ").strip()
    return s[:1].upper() + s[1:]


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", str(t)).strip().lower()


def hub(repo, fn):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, fn, repo_type="dataset")


def dedup(rows):
    seen, out = set(), []
    for t, y in rows:
        k = norm(t)
        if k and k not in seen:
            seen.add(k)
            out.append((t, y))
    return out


def take(rows, n, rng_seed, exclude=frozenset()):
    import random
    rows = [r for r in rows if norm(r[0]) not in exclude]
    rng = random.Random(rng_seed)
    by = {}
    for r in rows:
        by.setdefault(r[1], []).append(r)
    for v in by.values():
        rng.shuffle(v)
    if n >= len(rows):
        out = [r for v in by.values() for r in v]
    else:
        quota = {k: max(1, round(n * len(v) / len(rows))) for k, v in by.items()}
        out = [r for k, v in by.items() for r in v[:quota[k]]]
        rest = [r for k, v in by.items() for r in v[quota[k]:]]
        rng.shuffle(rest)
        out = (out + rest)[:n]
    rng.shuffle(out)
    return out


def split_in_domain(train_src, test_src, n_pool, seed):
    if test_src is None:
        test = take(train_src, N_TEST, f"{seed}-test")
    else:
        test = take(test_src, N_TEST, f"{seed}-test")
    used = {norm(t) for t, _ in test}
    train = take(train_src, N_TRAIN, f"{seed}-train", used)
    used |= {norm(t) for t, _ in train}
    val = take(train_src, N_VAL, f"{seed}-val", used)
    used |= {norm(t) for t, _ in val}
    pool = take(train_src, n_pool, f"{seed}-pool", used)
    return {"train": train, "val": val, "test": test, "pool": pool}


def src_banking77():
    import pandas as pd
    tr, te = pd.read_csv(BANKING_URLS["train"]), pd.read_csv(BANKING_URLS["test"])
    labels = sorted(tr.category.unique())
    return dedup(zip(tr.text, tr.category)), dedup(zip(te.text, te.category)), labels


def src_bitext():
    import pandas as pd
    df = pd.read_csv(hub(*BITEXT))[["instruction", "intent"]].dropna()
    return dedup(zip(df.instruction, df.intent)), None, sorted(df.intent.unique())


def src_tw_sentiment():
    import pandas as pd
    names = {0: "bearish", 1: "bullish", 2: "neutral"}
    tr, te = pd.read_csv(hub(SENT_REPO, "sent_train.csv")), pd.read_csv(hub(SENT_REPO, "sent_valid.csv"))
    f = lambda d: dedup((t, names[int(y)]) for t, y in zip(d.text, d.label))
    return f(tr), f(te), ["bearish", "bullish", "neutral"]


def src_tw_topic():
    import pandas as pd
    labels = [slug(n) for n in TOPIC_NAMES]
    tr, te = pd.read_csv(hub(TOPIC_REPO, "topic_train.csv")), pd.read_csv(hub(TOPIC_REPO, "topic_valid.csv"))
    f = lambda d: dedup((t, labels[int(y)]) for t, y in zip(d.text, d.label))
    return f(tr), f(te), labels


def src_cfpb(n_shards):
    import pandas as pd
    import pyarrow.parquet as pq
    from huggingface_hub import list_repo_files
    files = sorted(f for f in list_repo_files(CFPB_MIRROR, repo_type="dataset") if f.endswith(".parquet"))[-n_shards:]
    parts = []
    for f in files:
        t = pq.read_table(hub(CFPB_MIRROR, f), columns=["date_received", "product", "complaint_what_happened"]).to_pandas()
        t = t[(t.date_received >= CFPB_DATE_MIN) & (t.date_received <= CFPB_DATE_MAX + "T23:59:59")]
        t = t[t["product"].isin(CFPB_PRODUCTS) & (t.complaint_what_happened.fillna("").str.strip().str.len() > 50)]
        parts.append(t)
    df = pd.concat(parts)
    rows = dedup((t[:TEXT_MAX_CHARS], slug(p)) for t, p in zip(df.complaint_what_happened, df["product"]))
    return rows, None, [slug(p) for p in CFPB_PRODUCTS]


def src_phrasebank():
    zf = zipfile.ZipFile(hub(PB_REPO, PB_ZIP))
    member = next(m for m in zf.namelist() if m.endswith("Sentences_50Agree.txt"))
    rows = []
    for line in zf.read(member).decode("iso-8859-1").splitlines():
        if "@" in line:
            s, lab = line.rsplit("@", 1)
            rows.append((s.strip(), lab.strip()))
    return dedup(rows), ["negative", "neutral", "positive"]


def src_fomc():
    import pandas as pd
    names = {0: "dovish", 1: "hawkish", 2: "neutral"}
    te = pd.read_csv(hub(FOMC_REPO, "test.csv"), encoding="utf-8-sig")
    return dedup((s, names[int(y)]) for s, y in zip(te.sentence, te.label)), ["dovish", "hawkish", "neutral"]


def src_clinc():
    from datasets import load_dataset
    ds = load_dataset(CLINC_REPO, "plus")
    names = ds["test"].features["intent"].names
    rows = dedup((t, names[y]) for t, y in zip(ds["test"]["text"], ds["test"]["intent"]))
    labels = sorted(n for n in names if n != "oos") + ["oos"]
    return rows, labels


def criteria_for(task, labels):
    if task in SPECIAL:
        return dict(SPECIAL[task])
    if task == "tw_topic":
        return {slug(n): n for n in TOPIC_NAMES}
    if task == "cfpb":
        return {slug(p): f"{p}: {d}." for p, d in CFPB_PRODUCTS.items()}
    if task == "clinc150":
        return {l: ("Out of scope: none of the listed intents." if l == "oos" else humanize(l)) for l in labels}
    return {l: humanize(l) for l in labels}


def save(task, splits, labels, source, limit):
    d = TASKS_DIR / task
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, rows in splits.items():
        if limit:
            rows = rows[: limit * (5 if name in ("train", "pool") else 1)]
        write_jsonl(d / f"{name}.jsonl", [{"id": f"{task}-{name}-{i:05d}", "text": t, "label": y} for i, (t, y) in enumerate(rows)])
        out[name] = len(rows)
    prefix, instr = TASK_TEXT[task]
    meta = {"labels": labels, "criteria": criteria_for(task, labels), "instructions": instr, "prefix": prefix,
            "in_domain": task in IN_DOMAIN, "source": source}
    assert list(meta["criteria"]) == labels or set(meta["criteria"]) == set(labels), task
    meta["criteria"] = {l: meta["criteria"][l] for l in labels}
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    sha = hashlib.sha256((d / "test.jsonl").read_bytes()).hexdigest()
    log.info("%s: %s, %d labels", task, out, len(labels))
    return {"task": task, **out, "n_labels": len(labels), "source": source, "sha256_test": sha}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cfpb-shards", type=int, default=2)
    args = ap.parse_args()
    manifest = []
    srcs = {"banking77": src_banking77, "bitext": src_bitext, "tw_sentiment": src_tw_sentiment, "tw_topic": src_tw_topic,
            "cfpb": lambda: src_cfpb(args.cfpb_shards)}
    for task in IN_DOMAIN:
        tr, te, labels = srcs[task]()
        manifest.append(save(task, split_in_domain(tr, te, POOL[task], SEED), labels, task, args.limit))
    for task, fn in (("phrasebank", src_phrasebank), ("fomc", src_fomc), ("clinc150", src_clinc)):
        rows, labels = fn()
        manifest.append(save(task, {"test": take(rows, N_TEST, f"{SEED}-test")}, labels, task, args.limit))
    (TASKS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))
    assert set(HELD_OUT) <= {m["task"] for m in manifest}


if __name__ == "__main__":
    main()
