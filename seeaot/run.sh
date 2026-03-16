#!/usr/bin/env bash
# ============================================================
#  seeAoT - 使用 seeAoT 仓库原生 eval/run_qwen3.py (单卡)
#
#  Usage:
#    bash seeaot/run.sh
#    CUDA_VISIBLE_DEVICES=3 bash seeaot/run.sh
# ============================================================
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SELF_DIR}/config.sh"

export CUDA_VISIBLE_DEVICES

echo "============================================================"
echo "  seeAoT Evaluation"
echo "  Model   : ${CKPT}"
echo "  GPU     : ${CUDA_VISIBLE_DEVICES}"
echo "  Frames  : ${NFRAMES}"
echo "  Subsets : ${SUBSETS[*]}"
echo "============================================================"

cd "${SEEAOT_REPO}"

VIDEO_PATH="${DATA_ROOT}"

for SUBSET in "${SUBSETS[@]}"; do
    DATA_FILE="${DATA_ROOT}/data_files/${SUBSET}.json"
    if [ ! -f "${DATA_FILE}" ]; then
        echo "  [Skip] ${SUBSET}: not found at ${DATA_FILE}"
        continue
    fi

    echo ""
    echo "-- [${SUBSET}] --"
    python eval/run_qwen3.py \
        --data_json  "${DATA_FILE}" \
        --video_path "${VIDEO_PATH}" \
        --ckpt       "${CKPT}" \
        --nframes    "${NFRAMES}"
done

echo ""
echo "-- [Accuracy] --"
python eval/read_qa.py --data_root "${DATA_ROOT}"

# ---- 收集结果到统一目录 ----
mkdir -p "${OUTPUT_DIR}"
MODEL_SHORT="$(basename "${CKPT}")"
for SUBSET in "${SUBSETS[@]}"; do
    SRC_DIR="${DATA_ROOT}/data_files/output/${SUBSET}"
    if [ -d "${SRC_DIR}" ]; then
        cp -f "${SRC_DIR}"/*.jsonl "${OUTPUT_DIR}/" 2>/dev/null || true
    fi
done
echo "Results copied to: ${OUTPUT_DIR}"

echo ""
echo "Done!"
