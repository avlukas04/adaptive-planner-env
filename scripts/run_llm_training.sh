#!/bin/bash
# Run LLM training with a project-local HuggingFace cache (avoids permission issues).
# Usage: ./scripts/run_llm_training.sh [episodes] [model]
#   episodes: default 10
#   model: e.g. google/flan-t5-base (optional; default uses fallback chain)
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
mkdir -p "$HF_HOME"
echo "HF cache: $HF_HOME"
MODEL_ARG=""
[[ -n "$2" ]] && MODEL_ARG="--model $2"
.venv/bin/python training/train_rl.py -n "${1:-10}" --agent llm $MODEL_ARG --no-plot
