#!/usr/bin/env bash
P="$(cd "$(dirname "$0")/.." && pwd)"
GPU=${1:?gpu}; RUN=${2:-main}; N=8
cd "$P"; mkdir -p logs/qshard_locks
for k in $(seq 0 $((N - 1))); do
  if mkdir "logs/qshard_locks/${RUN}_${k}" 2>/dev/null; then
    echo "$(date) GPU$GPU claims shard $k/$N"
    tcsh -c "source activate_project.csh && setenv CUDA_VISIBLE_DEVICES $GPU && setenv VLLM_USE_DEEP_GEMM 0 && setenv VLLM_DISABLE_COMPILE_CACHE 1 && python -m bp18.runners.vllm_llm --llm qwen --mode readout --run-name $RUN --shard $k/$N" >> "logs/${RUN}_qwen_readout_shard$k.log" 2>&1
    echo "$(date) GPU$GPU shard $k/$N exit $?"
  fi
done
echo "$(date) GPU$GPU: no shards left"
