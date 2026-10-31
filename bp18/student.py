from __future__ import annotations

import argparse
import json
import math
import random

import numpy as np

from .config import (CKPT_DIR, IN_DOMAIN, LORA, READOUT_PRUNE, SEED, STUDENTS, TASKS, TEXT_MAX_TOKENS, TRAIN, raw_path,
                     records_path, results_dir, run_name)
from .prompts import READOUT_PREFILL, readout_user_prompt
from .readout import char_token_ids, score_options
from .utils import Timer, append_jsonl, done_ids, get_logger, latest_ok, read_jsonl

log = get_logger("student")


def truncate_text(rec: dict, tok, max_tokens: int = TEXT_MAX_TOKENS) -> dict:
    ids = tok(rec["text"], add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return rec
    text = tok.decode(ids[:max_tokens])
    return {**rec, "state": rec["state"][: len(rec["state"]) - len(rec["text"])] + text}


def prompt_ids(tok, rec: dict) -> list[int]:
    text = tok.apply_chat_template([{"role": "user", "content": readout_user_prompt(rec)}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    return tok.encode(text + READOUT_PREFILL, add_special_tokens=False)


def target_ids(rec: dict, label: str, cids: dict) -> list[int]:
    return [cids[c] for c in f"{rec['options'].index(label) + 1}]"]


def teacher_labels(run: str, teacher: str, task: str, split: str) -> dict[str, str]:
    p = raw_path(run, teacher, task, split)
    return {i: r["pred"] for i, r in latest_ok(p).items()} if p.exists() else {}


def training_items(arm: str, teacher: str, run: str, limit=None) -> tuple[list[tuple[dict, str]], dict]:
    items, stats = [], {}
    for task in IN_DOMAIN:
        train = read_jsonl(records_path(task, "train"))[: limit or None]
        pool = read_jsonl(records_path(task, "pool"))[: limit or None] if arm in ("T", "M") else []
        if arm == "G":
            sel = [(r, r["gold"]) for r in train]
        else:
            tl = {**teacher_labels(run, teacher, task, "train"), **teacher_labels(run, teacher, task, "pool")}
            first = [(r, r["gold"]) for r in train] if arm == "M" else [(r, tl[r["id"]]) for r in train if r["id"] in tl]
            sel = first + [(r, tl[r["id"]]) for r in pool if r["id"] in tl]
            missing = (0 if arm == "M" else len(train) - sum(r["id"] in tl for r in train)) + len(pool) - sum(r["id"] in tl for r in pool)
            stats[f"{task}_teacher_missing"] = missing
            stats[f"{task}_teacher_agree_gold"] = float(np.mean([tl[r["id"]] == r["gold"] for r in train + pool if r["id"] in tl])) if tl else None
        stats[f"{task}_n"] = len(sel)
        items += sel
    return items, stats


def load_model(name: str, device: str, dtype):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(name, dtype=dtype)
    model.to(device)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    torch.manual_seed(SEED)
    return tok, model


def logits_at(model, ids, att, positions):
    import torch
    keep = sorted(set(positions))
    try:
        out = model(input_ids=ids, attention_mask=att, logits_to_keep=torch.tensor(keep, device=ids.device))
        if out.logits.shape[1] == len(keep):
            return out.logits, {p: i for i, p in enumerate(keep)}
    except TypeError:
        pass
    out = model(input_ids=ids, attention_mask=att)
    return out.logits, {p: p for p in keep}


def batches(examples, bs, seed):
    rng = random.Random(seed)
    order = sorted(range(len(examples)), key=lambda i: len(examples[i][0]) + rng.random())
    bl = [order[i:i + bs] for i in range(0, len(order), bs)]
    rng.shuffle(bl)
    return bl


def collate(examples, idx, pad_id, device):
    import torch
    seqs = [examples[i][0] + examples[i][1] for i in idx]
    L = max(map(len, seqs))
    ids = torch.full((len(idx), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(idx), L), dtype=torch.long)
    targets = []
    for j, i in enumerate(idx):
        p, t = examples[i]
        ids[j, :len(p) + len(t)] = torch.tensor(p + t)
        att[j, :len(p) + len(t)] = 1
        targets += [(j, len(p) - 1 + k, tok_id) for k, tok_id in enumerate(t)]
    return ids.to(device), att.to(device), targets


def train(model, tok, items, cids, device, seed, max_steps=None):
    import torch
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(r=LORA["r"], lora_alpha=LORA["alpha"], lora_dropout=LORA["dropout"],
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    for prm in model.parameters():
        if prm.requires_grad:
            prm.data = prm.data.float()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    examples = []
    for rec, label in items:
        p = prompt_ids(tok, truncate_text(rec, tok))
        examples.append((p[-(TRAIN["max_len"] - 4):], target_ids(rec, label, cids)))
    bl = batches(examples, TRAIN["batch_size"], seed)
    steps = math.ceil(len(bl) * TRAIN["epochs"])
    steps = min(steps, max_steps) if max_steps else steps
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=TRAIN["lr"],
                            weight_decay=TRAIN["weight_decay"])
    warm = max(1, int(TRAIN["warmup_ratio"] * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) *
                                              0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    model.train()
    losses = []
    with Timer() as t:
        for step in range(steps):
            ids, att, targets = collate(examples, bl[step % len(bl)], tok.pad_token_id, device)
            with torch.autocast(device_type=device if device != "mps" else "cpu", dtype=torch.bfloat16,
                                enabled=device == "cuda"):
                logits, col = logits_at(model, ids, att, [p for _, p, _ in targets])
            rows = torch.tensor([j for j, _, _ in targets], device=logits.device)
            cols = torch.tensor([col[p] for _, p, _ in targets], device=logits.device)
            y = torch.tensor([t for _, _, t in targets], device=logits.device)
            loss = torch.nn.functional.cross_entropy(logits[rows, cols].float(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            losses.append(float(loss.detach()))
            if step % max(1, min(50, steps // 4)) == 0 or step == steps - 1:
                log.info("step %d/%d loss %.4f", step, steps, np.mean(losses[-50:]))
    model.eval()
    return model, {"steps": steps, "train_seconds": t.seconds, "loss_first50": float(np.mean(losses[:50])),
                   "loss_last50": float(np.mean(losses[-50:])), "n_examples": len(examples)}


def expand_cache(cache, n: int):
    import torch
    if hasattr(cache, "reorder_cache"):
        try:
            cache.reorder_cache(torch.zeros(n, dtype=torch.long, device=_cache_device(cache)))
            return cache
        except Exception:
            pass

    def rep(v):
        if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == 1:
            return v.repeat_interleave(n, dim=0)
        if isinstance(v, dict):
            return {k: rep(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return type(v)(rep(x) for x in v)
        return v

    for layer in getattr(cache, "layers", []):
        for name, v in list(vars(layer).items()):
            setattr(layer, name, rep(v))
    return cache


def _cache_device(cache):
    import torch
    for layer in getattr(cache, "layers", []):
        for v in vars(layer).values():
            if torch.is_tensor(v):
                return v.device
            if isinstance(v, dict):
                for x in v.values():
                    if torch.is_tensor(x):
                        return x.device
    return "cpu"


def cached_option_probs(model, prompt: list[int], k: int, cids: dict, device: str, chunk: int = 64):
    import copy

    import torch

    from .readout import label_strings
    labels = label_strings(k)
    close = cids["]"]
    lps = []
    with torch.inference_mode():
        out = model(input_ids=torch.tensor([prompt], device=device), use_cache=True)
        first = torch.log_softmax(out.logits[0, -1].float(), -1)
        past = out.past_key_values
        for a in range(0, len(labels), chunk):
            labs = labels[a:a + chunk]
            L = max(len(x) for x in labs)
            cont = torch.tensor([[cids[c] for c in x] + [cids[x[-1]]] * (L - len(x)) for x in labs], device=device)
            pc = expand_cache(copy.deepcopy(past), len(labs))
            att = torch.ones((len(labs), len(prompt) + L), dtype=torch.long, device=device)
            lo = torch.log_softmax(model(input_ids=cont, past_key_values=pc, attention_mask=att, use_cache=False)
                                   .logits.float(), -1)
            for j, x in enumerate(labs):
                lp = first[cids[x[0]]].item()
                for m in range(1, len(x)):
                    lp += lo[j, m - 1, cids[x[m]]].item()
                lps.append(lp + lo[j, len(x) - 1, close].item())
    lps = np.array(lps)
    mass = float(np.exp(lps).sum())
    p = np.exp(lps - lps.max())
    return (p / p.sum()).tolist(), {"mass": mass, "calls": 1 + math.ceil(len(labels) / chunk), "scorer": "cached"}


def batched_backend(model, cand_ids, device, pad_id, bs=16):
    import torch

    def fn(batch):
        out = []
        order = sorted(range(len(batch)), key=lambda i: len(batch[i]))
        res = [None] * len(batch)
        for k in range(0, len(order), bs):
            idx = order[k:k + bs]
            L = max(len(batch[i]) for i in idx)
            ids = torch.full((len(idx), L), pad_id, dtype=torch.long)
            att = torch.zeros((len(idx), L), dtype=torch.long)
            for j, i in enumerate(idx):
                ids[j, :len(batch[i])] = torch.tensor(batch[i])
                att[j, :len(batch[i])] = 1
            lasts = [len(batch[i]) - 1 for i in idx]
            with torch.inference_mode():
                logits, col = logits_at(model, ids.to(device), att.to(device), lasts)
            cols = torch.tensor([col[p] for p in lasts], device=logits.device)
            lp = torch.log_softmax(logits[torch.arange(len(idx), device=logits.device), cols].float(), dim=-1)
            sel = lp[:, cand_ids].cpu().numpy()
            for j, i in enumerate(idx):
                res[i] = {c: float(sel[j, m]) for m, c in enumerate(cand_ids)}
        out.extend(res)
        return out

    return fn


CACHED_MIN_OPTIONS = 21


def make_scorer(model, tok, cids, device, bs=16):
    backend = batched_backend(model, list(cids.values()), device, tok.pad_token_id, bs)
    try:
        cached_option_probs(model, [cids["1"]] * 4, 2, cids, device)
        cache_ok = True
    except Exception as e:
        log.warning("cached scorer unavailable (%s); trie only", repr(e)[:160])
        cache_ok = False

    def scorer(prefixes, ks):
        probs, diags = [None] * len(ks), [None] * len(ks)
        small = [i for i, k in enumerate(ks) if not cache_ok or k < CACHED_MIN_OPTIONS]
        if small:
            p, d = score_options([prefixes[i] for i in small], [ks[i] for i in small], backend, cids, prune=READOUT_PRUNE)
            for i, pi, di in zip(small, p, d):
                probs[i], diags[i] = pi, {**di, "scorer": "trie"}
        for i, k in enumerate(ks):
            if probs[i] is None:
                probs[i], diags[i] = cached_option_probs(model, prefixes[i], k, cids, device)
        return probs, diags

    return scorer


def evaluate(model, tok, cids, device, model_key, run, splits=("val", "test"), limit=None, bs=16):
    from .runners.common import pred_row
    scorer = make_scorer(model, tok, cids, device, bs)
    for task in TASKS:
        for split in splits:
            rp = records_path(task, split)
            if not rp.exists():
                continue
            out = raw_path(run, model_key, task, split)
            done = done_ids(out)
            recs = [r for r in read_jsonl(rp)[: limit or None] if r["id"] not in done]
            for k in range(0, len(recs), 64):
                chunk = recs[k:k + 64]
                prefixes = [prompt_ids(tok, truncate_text(r, tok)) for r in chunk]
                with Timer() as t:
                    probs, diags = scorer(prefixes, [len(r["options"]) for r in chunk])
                append_jsonl(out, [pred_row(r, model_key, dict(zip(r["options"], p)), t.seconds / len(chunk), len(ids), d)
                                   for r, p, d, ids in zip(chunk, probs, diags, prefixes)])
            log.info("%s %s/%s evaluated", model_key, task, split)


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", choices=list(STUDENTS), required=True)
    ap.add_argument("--arm", choices=["G", "T", "M", "base"], required=True, help="base = zero-shot backbone")
    ap.add_argument("--teacher", choices=["clef", "pplx"], default="clef")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--run-name", default="main")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--eval-bs", type=int, default=16)
    ap.add_argument("--eval-limit", type=int, default=None, help="eval items per task and split (default: --limit)")
    ap.add_argument("--batch-size", type=int, default=None, help="override TRAIN batch size (Mac preflight)")
    ap.add_argument("--max-len", type=int, default=None, help="override TRAIN max_len (Mac preflight)")
    ap.add_argument("--model-override", default=None,
                    help="Mac preflight only: load this HF model instead (e.g. Qwen/Qwen3-0.6B; Qwen3.5's linear "
                         "attention has no fast kernels on the Mac). Results are written under the same run key.")
    args = ap.parse_args()
    if args.batch_size:
        TRAIN["batch_size"] = args.batch_size
    if args.max_len:
        TRAIN["max_len"] = args.max_len
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    r = {"student": args.student, "arm": args.arm, "teacher": args.teacher, "seed": args.seed}
    key = f"{args.student}-base" if args.arm == "base" else run_name(r)
    random.seed(SEED + args.seed)
    np.random.seed(SEED + args.seed)
    torch.manual_seed(SEED + args.seed)
    tok, model = load_model(args.model_override or STUDENTS[args.student], device, dtype)
    cids = char_token_ids(tok)
    info = {"run": key, "device": device}
    if args.arm != "base":
        items, stats = training_items(args.arm, args.teacher, args.run_name, args.limit)
        info.update(stats)
        log.info("%s: %d training examples %s", key, len(items), stats)
        model, tinfo = train(model, tok, items, cids, device, SEED + args.seed, args.max_steps)
        info.update(tinfo)
        (CKPT_DIR / args.run_name / key).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(CKPT_DIR / args.run_name / key)
    evaluate(model, tok, cids, device, key, args.run_name, limit=args.eval_limit or args.limit, bs=args.eval_bs)
    d = results_dir(args.run_name) / "runs"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{key}.json").write_text(json.dumps(info, indent=1))
    log.info("done %s", key)


if __name__ == "__main__":
    main()
