#!/usr/bin/env bash
set -euo pipefail
P="$(cd "$(dirname "$0")/.." && pwd)"
STAGE=${1:?usage: launch_tmux.sh prepare|pilot|full|timing|analyze}
cd "$P"; mkdir -p logs
case "$STAGE" in
  pilot) ARGS="--limit 50 --run-name pilot" ;;
  full|timing) ARGS="--run-name main" ;;
  prepare|analyze) ARGS="" ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
start() {
  local name=$1 cmd=$2
  if tmux has-session -t "$name" 2>/dev/null; then echo "!! session $name exists"; return; fi
  tmux new-session -d -s "$name" tcsh; tmux send-keys -t "$name" "cd $P" Enter; tmux send-keys -t "$name" "$cmd" Enter
  echo "started $name"
}
log() { echo "logs/${STAGE}_$1.log"; }
case "$STAGE" in
  prepare) start bp18-prep "source activate_project.csh && python -m bp18.data |& tee -a $(log data) && echo PREPARE-DONE" ;;
  analyze) start bp18-analyze "source activate_project.csh && python -m bp18.analyze --run-name main |& tee -a $(log analyze) && python -m bp18.analyze --run-name fp32 |& tee -a $(log analyze) && echo ANALYZE-DONE" ;;
  timing)
    start bp18-gpu3 "source activate_train.csh && setenv CUDA_VISIBLE_DEVICES 3 && python -m bp18.score --model qwen2b --timing $ARGS |& tee -a $(log qwen2b); echo DONE"
    start bp18-gpu2 "source activate_train.csh && setenv CUDA_VISIBLE_DEVICES 2 && python -m bp18.score --model gemmaE2b --timing $ARGS |& tee -a $(log gemmaE2b); echo DONE" ;;
  full)
    Q="bash scripts/qwen_shard_queue.sh"
    start bp18-gpu3 "source activate_train.csh && setenv CUDA_VISIBLE_DEVICES 3 && python -m bp18.score --model qwen2b $ARGS |& tee -a $(log gpu3_qwen2b); python -m bp18.score --model qwen2b --scorers trie,cached --dtype fp32 --limit 100 --run-name fp32 |& tee -a $(log gpu3_qwen2b_fp32); python -m bp18.score --model qwen2b --timing $ARGS |& tee -a $(log gpu3_qwen2b_timing); $Q 3 main |& tee -a $(log gpu3_queue); echo GPU3-DONE"
    start bp18-gpu2 "source activate_train.csh && setenv CUDA_VISIBLE_DEVICES 2 && python -m bp18.score --model gemmaE2b $ARGS |& tee -a $(log gpu2_gemmaE2b); python -m bp18.score --model gemmaE2b --scorers trie,cached --dtype fp32 --limit 100 --run-name fp32 |& tee -a $(log gpu2_gemmaE2b_fp32); python -m bp18.score --model gemmaE2b --timing $ARGS |& tee -a $(log gpu2_gemmaE2b_timing); $Q 2 main |& tee -a $(log gpu2_queue); echo GPU2-DONE"
    start bp18-gpu1 "source activate_project.csh && setenv CUDA_VISIBLE_DEVICES 1 && setenv VLLM_USE_DEEP_GEMM 0 && setenv VLLM_DISABLE_COMPILE_CACHE 1 && python -m bp18.runners.vllm_llm --llm qwen2b --mode readout $ARGS |& tee -a $(log gpu1); python -m bp18.runners.vllm_llm --llm qwen2b --mode generative $ARGS |& tee -a $(log gpu1); python -m bp18.runners.vllm_llm --llm qwen --mode generative $ARGS |& tee -a $(log gpu1); $Q 1 main |& tee -a $(log gpu1_queue); echo GPU1-DONE"
    start bp18-gpu0 "source activate_project.csh && setenv CUDA_VISIBLE_DEVICES 0 && setenv VLLM_DISABLE_COMPILE_CACHE 1 && python -m bp18.runners.vllm_llm --llm gemmaE2b --mode readout $ARGS |& tee -a $(log gpu0); python -m bp18.runners.vllm_llm --llm gemmaE2b --mode generative $ARGS |& tee -a $(log gpu0); python -m bp18.runners.vllm_llm --llm gemma --mode readout $ARGS |& tee -a $(log gpu0); python -m bp18.runners.vllm_llm --llm gemma --mode generative $ARGS |& tee -a $(log gpu0); source activate_clef.csh && python -m bp18.runners.clef --variant clef $ARGS |& tee -a $(log gpu0); $Q 0 main |& tee -a $(log gpu0_queue); echo GPU0-DONE"
    start bp18-cpu "source activate_project.csh && python -m bp18.runners.api --provider jev $ARGS |& tee -a $(log cpu_jev); echo CPU-DONE" ;;
  pilot)
    start bp18-gpu3 "source activate_train.csh && setenv CUDA_VISIBLE_DEVICES 3 && python -m bp18.score --model qwen2b $ARGS |& tee -a $(log gpu3_qwen2b); echo GPU3-DONE"
    start bp18-gpu2 "source activate_train.csh && setenv CUDA_VISIBLE_DEVICES 2 && python -m bp18.score --model gemmaE2b $ARGS |& tee -a $(log gpu2_gemmaE2b); echo GPU2-DONE"
    start bp18-gpu1 "source activate_project.csh && setenv CUDA_VISIBLE_DEVICES 1 && setenv VLLM_USE_DEEP_GEMM 0 && setenv VLLM_DISABLE_COMPILE_CACHE 1 && python -m bp18.runners.vllm_llm --llm qwen2b --mode readout $ARGS |& tee -a $(log gpu1); python -m bp18.runners.vllm_llm --llm qwen2b --mode generative $ARGS |& tee -a $(log gpu1); python -m bp18.runners.vllm_llm --llm qwen --mode readout $ARGS |& tee -a $(log gpu1); python -m bp18.runners.vllm_llm --llm qwen --mode generative $ARGS |& tee -a $(log gpu1); echo GPU1-DONE"
    start bp18-gpu0 "source activate_project.csh && setenv CUDA_VISIBLE_DEVICES 0 && setenv VLLM_DISABLE_COMPILE_CACHE 1 && python -m bp18.runners.vllm_llm --llm gemmaE2b --mode readout $ARGS |& tee -a $(log gpu0); python -m bp18.runners.vllm_llm --llm gemmaE2b --mode generative $ARGS |& tee -a $(log gpu0); python -m bp18.runners.vllm_llm --llm gemma --mode readout $ARGS |& tee -a $(log gpu0); python -m bp18.runners.vllm_llm --llm gemma --mode generative $ARGS |& tee -a $(log gpu0); source activate_clef.csh && python -m bp18.runners.clef --variant clef $ARGS |& tee -a $(log gpu0); echo GPU0-DONE"
    start bp18-cpu "source activate_project.csh && python -m bp18.runners.api --provider jev $ARGS |& tee -a $(log cpu_jev); echo CPU-DONE" ;;
esac
