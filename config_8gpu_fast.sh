# ============================================================
#  Qwen3 Unified Eval — 8-GPU Fastest Full Run Preset
# ============================================================
#
# Optimized for small / medium dense VL models such as Qwen3-VL-4B/7B.
# Uses dataset sharding across 8 single-GPU workers for max throughput.

BACKEND="vllm"

# ------------------------------------------------------------
#  Model
# ------------------------------------------------------------
MODEL_CKPT="/home/xuboshen/models/Qwen3-VL-4B-Instruct"
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# ------------------------------------------------------------
#  Output / cache
# ------------------------------------------------------------
OUTPUT_DIR="./results"
HF_HUB_CACHE_ROOT="/m2v_intern/xuboshen/zgw/hf_cache_temp"

# ------------------------------------------------------------
#  vLLM
# ------------------------------------------------------------
VLLM_BATCH_SIZE=8
VLLM_MAX_BATCHES=0
VLLM_GPU_MEM_UTIL=0.8
VLLM_TP_SIZE=""
VLLM_MAX_NUM_SEQS=8
VLLM_SHARD_WORKERS="auto"

# ------------------------------------------------------------
#  FutureOmni
# ------------------------------------------------------------
FUTUREOMNI_DATA_FILE="/m2v_intern/xuboshen/zgw/data/FutureOmni/futureomni_test.json"
FUTUREOMNI_VIDEO_ROOT="/m2v_intern/xuboshen/zgw/data/FutureOmni/videos"
FUTUREOMNI_AUDIO_ROOT=""
FUTUREOMNI_NFRAMES=32

# ------------------------------------------------------------
#  seeAoT
# ------------------------------------------------------------
SEEAOT_ROOT="/m2v_intern/xuboshen/zgw/data/AoTBench"
SEEAOT_SUBSETS=(
    "ReverseFilm"
    "UCF101"
    "Rtime_t2v"
    "Rtime_v2t"
    "AoTBench_QA"
)
SEEAOT_NFRAMES=16

# ------------------------------------------------------------
#  MVBench
# ------------------------------------------------------------
MVBENCH_DATA_ROOT="/m2v_intern/xuboshen/zgw/hf_cache_temp/datasets--OpenGVLab--MVBench"
MVBENCH_NFRAMES=16
MVBENCH_FPS=1.0

# ------------------------------------------------------------
#  LongVideoBench
# ------------------------------------------------------------
LONGVIDEOBENCH_DATA_ROOT="/m2v_intern/xuboshen/zgw/hf_cache_temp/datasets--longvideobench--LongVideoBench"
LONGVIDEOBENCH_NFRAMES=32
LONGVIDEOBENCH_NO_SUBTITLES="false"
