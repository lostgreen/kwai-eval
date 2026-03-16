# ============================================================
#  LongVideoBench — Config
#  使用 VLMEvalKit 原生 run.py (torchrun 多卡)
# ============================================================

# ---- 继承全局配置 ----
_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_SELF_DIR}/../config.sh"

# ---- 以下参数已从全局 config.sh 继承, 按需覆盖 ----
# VLMEVALKIT_REPO="/home/xuboshen/VLMEvalKit"

MODEL_NAME="Qwen3-VL-4B-Instruct"

# ---- GPU ----
CUDA_VISIBLE_DEVICES="6,7"
NPROC=2                     # torchrun 进程数
MASTER_PORT=29502

# ---- Dataset (VLMEvalKit 注册名, 见 video_dataset_config.py) ----
# 可选: LongVideoBench_8frame_subs / LongVideoBench_0.5fps_subs / LongVideoBench_64frame 等
DATASET="LongVideoBench_8frame_subs"

# ---- Output ----
WORK_DIR="${RESULTS_ROOT}/longvideobench"
