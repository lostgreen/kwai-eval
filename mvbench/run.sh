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
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE_ROOT}"
export HF_HUB_CACHE="${HF_HUB_CACHE_ROOT}"
export HF_DATASETS_OFFLINE=1   # 禁止网络请求, 强制使用本地数据

# VLMEvalKit 内部会将路径拼成 ${HF_HUB_CACHE_ROOT}/hub/datasets--xxx
# 如果数据直接在 hf_cache_temp/datasets--xxx, 自动创建 hub/ 符号链接
HUB_DIR="${HF_HUB_CACHE_ROOT}/hub"
mkdir -p "${HUB_DIR}"
for _d in "${HF_HUB_CACHE_ROOT}"/datasets--*; do
    [ -d "${_d}" ] && ln -sfn "${_d}" "${HUB_DIR}/$(basename "${_d}")"
done

mkdir -p "${WORK_DIR}"

echo "============================================================"
echo "  MVBench Evaluation (VLMEvalKit)"
echo "  Model   : ${MODEL_NAME}"
echo "  Dataset : ${DATASET}"
echo "  Config  : ${EVAL_CONFIG}"
echo "  GPU     : ${CUDA_VISIBLE_DEVICES}"
echo "  nproc   : ${NPROC}"
echo "  Output  : ${WORK_DIR}"
echo "============================================================"

cd "${VLMEVALKIT_REPO}"

# 如果 eval_config.json 存在则用 --config (支持本地模型路径), 否则回落到 --data/--model
if [ -f "${EVAL_CONFIG}" ]; then
    torchrun --nproc-per-node="${NPROC}" --master_port="${MASTER_PORT}" \
        run.py \
        --config "${EVAL_CONFIG}" \
        --work-dir "${WORK_DIR}"
else
    echo "  [WARN] eval_config.json not found, falling back to --data/--model"
    torchrun --nproc-per-node="${NPROC}" --master_port="${MASTER_PORT}" \
        run.py \
        --data "${DATASET}" \
        --model "${MODEL_NAME}" \
        --work-dir "${WORK_DIR}"
fi

echo ""
echo "Done! Results in: ${WORK_DIR}"
