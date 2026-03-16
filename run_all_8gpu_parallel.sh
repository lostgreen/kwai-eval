#!/usr/bin/env bash
# ============================================================
#  8-GPU 一键并行跑四个评测
#
#  默认分配:
#    GPU 0     ->  FutureOmni   (单卡 DDP)
#    GPU 1     ->  seeAoT       (单卡 transformers)
#    GPU 2,3   ->  MVBench      (VLMEvalKit torchrun x2)
#    GPU 4,5   ->  LongVideoBench (VLMEvalKit torchrun x2)
#    GPU 6,7   ->  空闲 / 留给更大模型扩展
#
#  如果模型需要双卡, 可自行调整 config.sh 中 GPU 分配
#
#  Usage:
#    bash run_all_8gpu_parallel.sh
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"

echo "============================================================"
echo "  8-GPU Parallel Evaluation Launcher"
echo "  Time : $(date)"
echo "  Logs : ${LOG_DIR}/"
echo "============================================================"

PIDS=()
NAMES=()
LOGS=()

launch() {
    local name="$1"
    local gpus="$2"
    local script="$3"
    local log="${LOG_DIR}/${name}_${TS}.log"

    echo "  [Launch] ${name}  GPU=${gpus}  -> ${log}"

    (
        export CUDA_VISIBLE_DEVICES="${gpus}"
        bash "${script}" > "${log}" 2>&1
    ) &

    PIDS+=("$!")
    NAMES+=("${name}")
    LOGS+=("${log}")
}

# ---- 启动四个评测 ----
launch "futureomni"     "0"     "${SCRIPT_DIR}/futureomni/run.sh"
launch "seeaot"         "1"     "${SCRIPT_DIR}/seeaot/run.sh"
launch "mvbench"        "2,3"   "${SCRIPT_DIR}/mvbench/run.sh"
launch "longvideobench" "4,5"   "${SCRIPT_DIR}/longvideobench/run.sh"

echo ""
echo "  All 4 benchmarks launched. Waiting..."
echo ""

# ---- 等待所有完成 ----
FAILED=0
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    name="${NAMES[$i]}"
    log="${LOGS[$i]}"

    if wait "${pid}"; then
        echo "  [OK]   ${name}  (${log})"
    else
        echo "  [FAIL] ${name}  (${log})"
        echo "         tail -50 ${log}"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================================"
if [ ${FAILED} -gt 0 ]; then
    echo "  WARNING: ${FAILED} benchmark(s) failed. Check logs."
    exit 1
else
    echo "  All 4 benchmarks completed!"
fi
echo "============================================================"
