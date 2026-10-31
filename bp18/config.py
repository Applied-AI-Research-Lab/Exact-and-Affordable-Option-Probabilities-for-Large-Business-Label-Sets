from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = PROJECT_ROOT / "data"
TASKS_DIR = DATA_DIR / "tasks"
RECORDS_DIR = DATA_DIR / "records"
RESULTS_ROOT = PROJECT_ROOT / "results"
CKPT_DIR = PROJECT_ROOT / "checkpoints"

SEED = 20261011
BASE_TASKS = ["tw_sentiment", "tw_topic", "bitext", "banking77", "clinc150"]
SWEEP = [10, 25, 50, 100]
SWEEP_TASKS = [f"clinc_k{k}" for k in SWEEP]
TASKS = BASE_TASKS + SWEEP_TASKS
N_TEST = 1000
SCORERS = ["trie", "cached", "first", "letters", "vtrie", "gen"]
HF_SCORERS = ["trie", "cached", "first", "letters"]
SMALL_MODELS = {"qwen2b": "Qwen/Qwen3.5-2B", "gemmaE2b": "google/gemma-4-E2B-it"}
REPEATS = 3
LATENCY_ITEMS = 200

IN_DOMAIN, HELD_OUT = BASE_TASKS, []
N_TRAIN, N_VAL = 0, 0
POOL = {}
STUDENTS = SMALL_MODELS
CFPB_MIRROR = "davidheineman/consumer-finance-complaints-large"
CFPB_DATE_MIN, CFPB_DATE_MAX = "2023-07-01", "2024-05-31"
TEXT_MAX_TOKENS, TEXT_MAX_CHARS = 1500, 6000
LORA = {"r": 16, "alpha": 32, "dropout": 0.05}
TRAIN = {"epochs": 1, "lr": 2e-4, "batch_size": 16, "grad_accum": 1, "max_len": 4096, "warmup_ratio": 0.03,
         "weight_decay": 0.0}

JEV = {
    "base_url": os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai/v1/systemone"),
    "model": os.environ.get("JEV_MODEL", "jev-1.13.0"),
    "api_key_env": "TYPESAFE_API_KEY",
    "usd_per_m_input": 0.042,
    "concurrency": int(os.environ.get("JEV_CONCURRENCY", "12")),
    "timeout_s": 60.0,
    "max_retries": 8,
}
PPLX_API = {
    "base_url": os.environ.get("PPLX_BASE_URL", "https://api.perplexity.ai/v1/decisions"),
    "model": os.environ.get("PPLX_MODEL", "pplx-decider-v1.1-27b"),
    "api_key_env": "PERPLEXITY_API_KEY",
    "usd_per_m_input": 0.02,
    "concurrency": int(os.environ.get("PPLX_CONCURRENCY", "8")),
    "timeout_s": 90.0,
    "max_retries": 8,
}
API_PROVIDERS = {"jev": JEV, "pplx_api": PPLX_API}

HF_MODELS = {
    "clef": "Cloudflare/clef",
    "clef_flash": "Cloudflare/clef-flash",
    "pplx": "perplexity-ai/pplx-decider-v1.1-27b",
    "qwen": "Qwen/Qwen3.8-27B-FP8",
    "gemma": "google/gemma-4-31B-it",
    "mac_readout": "Qwen/Qwen3-0.6B",
    **SMALL_MODELS,
}

READOUT_TOP_LOGPROBS = 64
READOUT_PRUNE = 1e-5
VLLM_MAX_MODEL_LEN = 8192
VLLM_GPU_MEM = float(os.environ.get("VLLM_GPU_MEM", "0.90"))
THINK = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "max_tokens": 4096, "reasoning_effort": "low"}

GPU_HOUR_USD = 2.50
BOOTSTRAP_N = 10000


def results_dir(run_name: str = "main") -> Path:
    return RESULTS_ROOT / run_name


def raw_path(run_name: str, model: str, task: str, condition: str) -> Path:
    return results_dir(run_name) / "raw" / model / task / f"{condition}.jsonl"


def records_path(task: str, condition: str) -> Path:
    return RECORDS_DIR / task / f"{condition}.jsonl"


def run_name(r: dict) -> str:
    return f"{r['student']}-{r['arm']}-s{r['seed']}"


def conditions_for(task: str, model: str) -> list[str]:
    return ["test"]
