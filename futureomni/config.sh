# ============================================================
#  FutureOmni — Config
#  使用 FutureOmni 仓库原生 infer_ddp.py (torchrun DDP)
# ============================================================

# ---- 继承全局配置 ----
_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_SELF_DIR}/../config.sh"

# ---- 以下参数已从全局 config.sh 继承, 按需覆盖 ----
# FUTUREOMNI_REPO="/home/xuboshen/FutureOmni"
# MODEL_PATH="/home/xuboshen/models/Qwen3-VL-4B-Instruct"

MODEL_TYPE="qwen3_vl"       # qwen3_vl | qwen2_5omni | qwen3omni | qwen2_5_vl

# ---- GPU ----
CUDA_VISIBLE_DEVICES="0,1"
NPROC_PER_NODE=1            # DDP 进程数 (1=单卡, 2=双卡 DDP)
MASTER_PORT=29500

# ---- Dataset ----
DATA_FILE="${FUTUREOMNI_DATA_FILE}"
VIDEO_ROOT="${FUTUREOMNI_VIDEO_ROOT}"

# ---- 视频帧/Token 配置 ----
MAX_FRAMES=256              # RL 训练使用 256 帧
TOKENS_PER_FRAME=48         # 每帧限制 48 tokens
BATCH_SIZE=1

# ---- Output ----
OUTPUT_DIR="${RESULTS_ROOT}/futureomni"
