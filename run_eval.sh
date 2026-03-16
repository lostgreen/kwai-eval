#!/usr/bin/env bash
# ============================================================
#  Qwen3 Unified Evaluation — One-Click Runner
#
#  Benchmarks: FutureOmni | seeAoT | MVBench | LongVideoBench
#  Backends  : transformers (single-GPU) | vllm (multi-GPU)
#
#  Usage:
#    bash run_eval.sh                  # all four benchmarks
#    bash run_eval.sh futureomni       # FutureOmni only
#    bash run_eval.sh seeaot           # seeAoT (5 subsets)
#    bash run_eval.sh mvbench          # MVBench (20 tasks)
#    bash run_eval.sh longvideobench   # LongVideoBench
#    bash run_eval.sh metrics          # recompute metrics only
# ============================================================
set -euo pipefail
export VLLM_WORKER_MULTIPROC_METHOD=spawn
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
#  Load configuration
# ============================================================
CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/config.sh}"
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "[ERROR] config.sh not found at ${CONFIG_FILE}"
    exit 1
fi
# shellcheck source=config.sh
source "${CONFIG_FILE}"

export CUDA_VISIBLE_DEVICES
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE_ROOT}"
export HF_HUB_CACHE="${HF_HUB_CACHE_ROOT}"

# ============================================================
#  Select inference script based on BACKEND
# ============================================================
EVAL_METRICS="${SCRIPT_DIR}/eval/metrics.py"
BENCHMARK="${1:-all}"

case "${BACKEND}" in
    vllm)
        INFER_SCRIPT="${SCRIPT_DIR}/eval/infer_vllm.py"
        RESULT_SUFFIX="vllm"
        # Required env for multi-GPU vLLM
        export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
        ;;
    transformers)
        INFER_SCRIPT="${SCRIPT_DIR}/eval/infer.py"
        RESULT_SUFFIX="tf"
        ;;
    *)
        echo "[ERROR] Unknown BACKEND='${BACKEND}'. Set to 'transformers' or 'vllm' in config.sh."
        exit 1
        ;;
esac

mkdir -p "${OUTPUT_DIR}"
MODEL_NAME="$(basename "${MODEL_CKPT}")"

# ---- vLLM extra args (only appended when BACKEND=vllm) ----
VLLM_EXTRA_ARGS=""
if [ "${BACKEND}" = "vllm" ]; then
    VLLM_EXTRA_ARGS="--batch_size ${VLLM_BATCH_SIZE} --gpu_mem_util ${VLLM_GPU_MEM_UTIL} --max_num_seqs ${VLLM_MAX_NUM_SEQS} --max_batches ${VLLM_MAX_BATCHES}"
    if [ -n "${VLLM_TP_SIZE}" ]; then
        VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS} --tp_size ${VLLM_TP_SIZE}"
    fi
fi

echo "============================================================"
echo "  Qwen3 Unified Evaluation"
echo "  Model          : ${MODEL_CKPT}"
echo "  Backend        : ${BACKEND}"
echo "  GPUs           : ${CUDA_VISIBLE_DEVICES}"
echo "  Mode           : ${BENCHMARK}"
echo "  Output         : ${OUTPUT_DIR}"
echo "  HF Cache       : ${HUGGINGFACE_HUB_CACHE}"
echo "  FutureOmni     : ${FUTUREOMNI_NFRAMES} frames"
echo "  seeAoT         : ${SEEAOT_NFRAMES} frames"
echo "  MVBench        : ${MVBENCH_NFRAMES} frames"
echo "  MVBench FPS    : ${MVBENCH_FPS:-0}"
echo "  LongVideoBench : ${LONGVIDEOBENCH_NFRAMES} frames"
echo "============================================================"

RESULT_FILES=()

# ============================================================
#  Helpers
# ============================================================

# Register a result file and print its location
_register() {
    local f="$1"
    RESULT_FILES+=("$f")
    echo "  → ${f}"
}

_gpu_list() {
    local devices="${CUDA_VISIBLE_DEVICES:-0}"
    IFS=',' read -r -a GPU_LIST <<< "${devices}"
    printf '%s\n' "${GPU_LIST[@]}"
}

_resolve_shard_workers() {
    local gpu_count="$1"
    if [ "${BACKEND}" != "vllm" ]; then
        echo 1
        return
    fi
    if [ -n "${VLLM_TP_SIZE}" ]; then
        echo 1
        return
    fi
    case "${VLLM_SHARD_WORKERS}" in
        auto)
            echo "${gpu_count}"
            ;;
        false|0)
            echo 1
            ;;
        *)
            echo "${VLLM_SHARD_WORKERS}"
            ;;
    esac
}

_run_infer() {
    python "$@"
}

_merge_shards() {
    local final_file="$1"
    shift
    : > "${final_file}"
    for shard_file in "$@"; do
        if [ -f "${shard_file}" ]; then
            cat "${shard_file}" >> "${final_file}"
            rm -f "${shard_file}"
        fi
    done
}

_run_vllm_sharded() {
    local result_file="$1"
    shift
    local -a base_args=("$@")
    if [ "${BACKEND}" != "vllm" ]; then
        python "${INFER_SCRIPT}" "${base_args[@]}"
        return
    fi
    mapfile -t GPU_LIST < <(_gpu_list)
    local gpu_count="${#GPU_LIST[@]}"
    local shard_workers
    shard_workers="$(_resolve_shard_workers "${gpu_count}")"

    if [ "${shard_workers}" -le 1 ]; then
        _run_infer "${INFER_SCRIPT}" "${base_args[@]}" --output_file "${result_file}" ${VLLM_EXTRA_ARGS}
        return
    fi

    if [ "${shard_workers}" -gt "${gpu_count}" ]; then
        shard_workers="${gpu_count}"
    fi

    echo "  [Sharding] ${shard_workers} workers across ${gpu_count} visible GPUs"
    local -a pids=()
    local -a shard_files=()
    local worker gpu shard_file
    for ((worker=0; worker<shard_workers; worker++)); do
        gpu="${GPU_LIST[worker]}"
        shard_file="${result_file%.jsonl}.shard${worker}.jsonl"
        shard_files+=("${shard_file}")
        CUDA_VISIBLE_DEVICES="${gpu}" python "${INFER_SCRIPT}" \
            "${base_args[@]}" \
            --output_file "${shard_file}" \
            --num_shards "${shard_workers}" \
            --shard_rank "${worker}" \
            --tp_size 1 \
            ${VLLM_EXTRA_ARGS} &
        pids+=("$!")
    done

    local pid
    for pid in "${pids[@]}"; do
        wait "${pid}"
    done

    _merge_shards "${result_file}" "${shard_files[@]}"
}

# ============================================================
#  FutureOmni
# ============================================================
run_futureomni() {
    echo ""
    echo "── [FutureOmni] ──────────────────────────────────────"
    if [ ! -f "${FUTUREOMNI_DATA_FILE}" ]; then
        echo "  [ERROR] Data file not found: ${FUTUREOMNI_DATA_FILE}"
        echo "  Update FUTUREOMNI_DATA_FILE in config.sh."
        return 1
    fi

    AUDIO_ARG=""
    if [ -n "${FUTUREOMNI_AUDIO_ROOT}" ]; then
        AUDIO_ARG="--audio_root ${FUTUREOMNI_AUDIO_ROOT}"
    fi

    DATA_NAME="$(basename "${FUTUREOMNI_DATA_FILE}" .json)"
    RESULT="${OUTPUT_DIR}/futureomni_${DATA_NAME}_${MODEL_NAME}_${FUTUREOMNI_NFRAMES}f_${RESULT_SUFFIX}.jsonl"

    _run_vllm_sharded "${RESULT}" \
        --dataset futureomni \
        --data_file "${FUTUREOMNI_DATA_FILE}" \
        --video_root "${FUTUREOMNI_VIDEO_ROOT}" \
        --ckpt "${MODEL_CKPT}" \
        --output_dir "${OUTPUT_DIR}" \
        --nframes "${FUTUREOMNI_NFRAMES}" \
        ${AUDIO_ARG}

    _register "${RESULT}"
    echo "  [FutureOmni] Done"
}

# ============================================================
#  seeAoT
# ============================================================
run_seeaot() {
    echo ""
    echo "── [seeAoT] ──────────────────────────────────────────"
    VIDEO_ROOT="${SEEAOT_ROOT}/videos"
    DATA_FILES_DIR="${SEEAOT_ROOT}/data_files"

    for SUBSET in "${SEEAOT_SUBSETS[@]}"; do
        DATA_FILE="${DATA_FILES_DIR}/${SUBSET}.json"
        if [ ! -f "${DATA_FILE}" ]; then
            echo "  [Skip] ${SUBSET}: not found at ${DATA_FILE}"
            continue
        fi

        RESULT="${OUTPUT_DIR}/seeaot_${SUBSET}_${MODEL_NAME}_${SEEAOT_NFRAMES}f_${RESULT_SUFFIX}.jsonl"
        echo "  [${SUBSET}] Running..."

        _run_vllm_sharded "${RESULT}" \
            --dataset seeaot \
            --data_file "${DATA_FILE}" \
            --video_root "${VIDEO_ROOT}" \
            --ckpt "${MODEL_CKPT}" \
            --output_dir "${OUTPUT_DIR}" \
            --nframes "${SEEAOT_NFRAMES}"

        _register "${RESULT}"
    done
    echo "  [seeAoT] Done"
}

# ============================================================
#  MVBench
# ============================================================
run_mvbench() {
    echo ""
    echo "── [MVBench] ─────────────────────────────────────────"
    if [ -n "${MVBENCH_DATA_ROOT}" ] && [ ! -d "${MVBENCH_DATA_ROOT}" ]; then
        echo "  [Info] MVBench root not found locally: ${MVBENCH_DATA_ROOT}"
        echo "  [Info] Loader will try Hugging Face cache / auto-download."
    fi

    SAMPLE_TAG="${MVBENCH_NFRAMES}f"
    FPS_ARG=""
    if awk "BEGIN {exit !(${MVBENCH_FPS:-0} > 0)}"; then
        SAMPLE_TAG="${MVBENCH_FPS}fps"
        FPS_ARG="--fps ${MVBENCH_FPS}"
    fi
    RESULT="${OUTPUT_DIR}/mvbench_${MODEL_NAME}_${SAMPLE_TAG}_${RESULT_SUFFIX}.jsonl"

    _run_vllm_sharded "${RESULT}" \
        --dataset mvbench \
        --data_root "${MVBENCH_DATA_ROOT}" \
        --ckpt "${MODEL_CKPT}" \
        --output_dir "${OUTPUT_DIR}" \
        --nframes "${MVBENCH_NFRAMES}" \
        ${FPS_ARG}

    _register "${RESULT}"
    echo "  [MVBench] Done"
}

# ============================================================
#  LongVideoBench
# ============================================================
run_longvideobench() {
    echo ""
    echo "── [LongVideoBench] ──────────────────────────────────"
    if [ -n "${LONGVIDEOBENCH_DATA_ROOT}" ] && [ ! -d "${LONGVIDEOBENCH_DATA_ROOT}" ]; then
        echo "  [Info] LongVideoBench root not found locally: ${LONGVIDEOBENCH_DATA_ROOT}"
        echo "  [Info] Loader will try Hugging Face cache / auto-download."
    fi

    NO_SUB_ARG=""
    if [ "${LONGVIDEOBENCH_NO_SUBTITLES:-false}" = "true" ]; then
        NO_SUB_ARG="--no_subtitles"
    fi

    RESULT="${OUTPUT_DIR}/longvideobench_${MODEL_NAME}_${LONGVIDEOBENCH_NFRAMES}f_${RESULT_SUFFIX}.jsonl"

    _run_vllm_sharded "${RESULT}" \
        --dataset longvideobench \
        --data_root "${LONGVIDEOBENCH_DATA_ROOT}" \
        --ckpt "${MODEL_CKPT}" \
        --output_dir "${OUTPUT_DIR}" \
        --nframes "${LONGVIDEOBENCH_NFRAMES}" \
        ${NO_SUB_ARG}

    _register "${RESULT}"
    echo "  [LongVideoBench] Done"
}

# ============================================================
#  Dispatch
# ============================================================
case "${BENCHMARK}" in
    futureomni)
        run_futureomni ;;
    seeaot)
        run_seeaot ;;
    mvbench)
        run_mvbench ;;
    longvideobench)
        run_longvideobench ;;
    all)
        run_futureomni
        run_seeaot
        run_mvbench
        run_longvideobench
        ;;
    metrics)
        # Collect all existing JSONL files in OUTPUT_DIR
        while IFS= read -r -d '' f; do
            RESULT_FILES+=("$f")
        done < <(find "${OUTPUT_DIR}" -name "*.jsonl" -print0 2>/dev/null)
        ;;
    *)
        echo "Usage: $0 [futureomni|seeaot|mvbench|longvideobench|all|metrics]"
        exit 1
        ;;
esac

# ============================================================
#  Final metrics report
# ============================================================
if [ ${#RESULT_FILES[@]} -gt 0 ]; then
    echo ""
    echo "── [Metrics] ─────────────────────────────────────────"
    EXISTING=()
    for f in "${RESULT_FILES[@]}"; do
        [ -f "$f" ] && EXISTING+=("$f")
    done
    if [ ${#EXISTING[@]} -gt 0 ]; then
        python "${EVAL_METRICS}" "${EXISTING[@]}"
    else
        echo "  No result files found."
    fi
fi

echo ""
echo "All done! Results saved to: ${OUTPUT_DIR}"
