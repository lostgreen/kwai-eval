#!/usr/bin/env bash
# ============================================================
#  MVBench - 使用 VLMEvalKit 原生 run.py (torchrun 多卡)
#
#  Usage:
#    bash mvbench/run.sh
#    CUDA_VISIBLE_DEVICES=0,1 bash mvbench/run.sh
# ============================================================
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SELF_DIR}/config.sh"

export CUDA_VISIBLE_DEVICES

mkdir -p "${WORK_DIR}"

echo "============================================================"
echo "  MVBench Evaluation (VLMEvalKit)"
echo "  Model   : ${MODEL_NAME}"
echo "  Dataset : ${DATASET}"
echo "  GPU     : ${CUDA_VISIBLE_DEVICES}"
echo "  nproc   : ${NPROC}"
echo "  Output  : ${WORK_DIR}"
echo "============================================================"

cd "${VLMEVALKIT_REPO}"

torchrun --nproc-per-node="${NPROC}" --master_port="${MASTER_PORT}" \
    run.py \
    --data "${DATASET}" \
    --model "${MODEL_NAME}" \
    --work-dir "${WORK_DIR}"

echo ""
echo "Done! Results in: ${WORK_DIR}"
