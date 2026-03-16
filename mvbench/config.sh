# ============================================================
#  MVBench — Config
#  使用 VLMEvalKit 原生 run.py (torchrun 多卡)
# ============================================================

# ---- 继承全局配置 ----
_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_SELF_DIR}/../config.sh"

# ---- 以下参数已从全局 config.sh 继承, 按需覆盖 ----
# VLMEVALKIT_REPO="/home/xuboshen/VLMEvalKit"
# MODEL_PATH → 此处用 VLMEvalKit 注册名

MODEL_NAME="Qwen3-VL-4B-Instruct"

# ---- GPU ----
CUDA_VISIBLE_DEVICES="4,5"
NPROC=2                     # torchrun 进程数
MASTER_PORT=29501

# ---- Dataset (VLMEvalKit 注册名, 见 video_dataset_config.py) ----
# 可选: MVBench_MP4_1fps / MVBench_MP4_8frame / MVBench_8frame 等
DATASET="MVBench_MP4_1fps"

# ---- Output ----
WORK_DIR="${RESULTS_ROOT}/mvbench"
