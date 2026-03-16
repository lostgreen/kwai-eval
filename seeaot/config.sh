# ============================================================
#  seeAoT — Config
#  使用 seeAoT 仓库原生 eval/run_qwen3.py (单卡 transformers)
# ============================================================

# ---- 继承全局配置 ----
_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_SELF_DIR}/../config.sh"

# ---- 以下参数已从全局 config.sh 继承, 按需覆盖 ----
# SEEAOT_REPO="/home/xuboshen/seeAoT"
# MODEL_PATH="/home/xuboshen/models/Qwen3-VL-4B-Instruct"

CKPT="${MODEL_PATH}"

# ---- GPU (单卡就够了) ----
CUDA_VISIBLE_DEVICES="2"

# ---- Dataset ----
DATA_ROOT="${SEEAOT_DATA_ROOT}"
SUBSETS=(
    "ReverseFilm"
    "UCF101"
    "Rtime_t2v"
    "Rtime_v2t"
    "AoTBench_QA"
)

# ---- Inference ----
NFRAMES=16

# ---- Output ----
OUTPUT_DIR="${RESULTS_ROOT}/seeaot"
