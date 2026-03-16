# ============================================================
#  Qwen3-Eval 全局配置
#
#  在服务器上先改好这里的路径，各子文件夹会自动继承
#  如需覆盖某个评测的特定参数，直接改对应子文件夹的 config.sh
# ============================================================

# ---- 仓库路径 ----
FUTUREOMNI_REPO="/home/xuboshen/FutureOmni"
SEEAOT_REPO="/home/xuboshen/seeAoT"
VLMEVALKIT_REPO="/home/xuboshen/VLMEvalKit"

# ---- 模型 ----
MODEL_PATH="/home/xuboshen/models/Qwen3-VL-4B-Instruct"

# ---- 数据 ----
FUTUREOMNI_DATA_FILE="/m2v_intern/xuboshen/zgw/data/FutureOmni/futureomni_test.json"
FUTUREOMNI_VIDEO_ROOT="/m2v_intern/xuboshen/zgw/data/FutureOmni/videos"
SEEAOT_DATA_ROOT="/m2v_intern/xuboshen/zgw/data/AoTBench"

# ---- 结果输出 (所有评测结果统一存到这个目录下) ----
GLOBAL_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_ROOT="${GLOBAL_CONFIG_DIR}/results"
