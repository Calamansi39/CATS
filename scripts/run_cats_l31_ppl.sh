#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-3}"
MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.1-8B}"
DTYPE="${DTYPE:-bfloat16}"
SPARSITY="${SPARSITY:-0.5}"
ARC_SAVED_DIR="${ARC_SAVED_DIR:-}"
ARC_DATASET="${ARC_DATASET:-wikitext2}"
ARC_METRIC="${ARC_METRIC:-max}"
ARC_QUANT_TYPE="${ARC_QUANT_TYPE:-NVFP4}"
LM_EVAL_LIMIT="${LM_EVAL_LIMIT:-1}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
CALIB_BATCH_SIZE="${CALIB_BATCH_SIZE:-1}"

ARGS=(
  --model_path "${MODEL_PATH}"
  --dtype "${DTYPE}"
  --attn_implementation "${ATTN_IMPL}"
  --sparsity "${SPARSITY}"
  --limit "${LM_EVAL_LIMIT}"
  --calib_batch_size "${CALIB_BATCH_SIZE}"
  --skip_lm_eval
)

if [[ -n "${ARC_SAVED_DIR}" ]]; then
  ARGS+=(
    --arc_saved_dir "${ARC_SAVED_DIR}"
    --arc_dataset "${ARC_DATASET}"
    --arc_metric "${ARC_METRIC}"
    --arc_quant_type "${ARC_QUANT_TYPE}"
  )
fi

CUDA_VISIBLE_DEVICES="${GPU}" HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 python -u /mnt/data2/lbc/CATS/scripts/cats_eval_l31.py "${ARGS[@]}"
