#!/usr/bin/env bash
# ============================================================
#  FutureOmni - 使用 FutureOmni 仓库原生 infer_ddp.py
#
#  Usage:
#    bash futureomni/run.sh
#    CUDA_VISIBLE_DEVICES=4,5 bash futureomni/run.sh
# ============================================================
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SELF_DIR}/config.sh"

export CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false

mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "  FutureOmni Evaluation"
echo "  Model  : ${MODEL_PATH}"
echo "  Type   : ${MODEL_TYPE}"
echo "  GPU    : ${CUDA_VISIBLE_DEVICES}"
echo "  DDP    : nproc=${NPROC_PER_NODE}"
echo "  Frames : ${MAX_FRAMES}  Tokens/Frame : ${TOKENS_PER_FRAME}"
echo "  Output : ${OUTPUT_DIR}"
echo "============================================================"

cd "${FUTUREOMNI_REPO}"

torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    eval/infer_ddp.py \
    --model_path "${MODEL_PATH}" \
    --data_file  "${DATA_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --dataset    "futureomni" \
    --model_type "${MODEL_TYPE}" \
    --mode       "video" \
    --root       "${VIDEO_ROOT}" \
    --batch_size "${BATCH_SIZE}" \
    --max_frames "${MAX_FRAMES}" \
    --tokens_per_frame "${TOKENS_PER_FRAME}"

echo ""
echo "Done! Results saved to: ${OUTPUT_DIR}"
