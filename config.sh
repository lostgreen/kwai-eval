# ============================================================
#  Qwen3 Unified Eval — Configuration
#
#  Edit this file to configure paths, model, and per-benchmark
#  frame counts. run_eval.sh sources this automatically.
# ============================================================

# ------------------------------------------------------------
#  Inference backend
#
#  "transformers" — single-GPU, uses eval/infer.py
#  "vllm"         — multi-GPU tensor parallelism, uses eval/infer_vllm.py
#                   Requires: pip install vllm
#                   Required env: export VLLM_WORKER_MULTIPROC_METHOD=spawn
# ------------------------------------------------------------
BACKEND="transformers"

# ------------------------------------------------------------
#  Model
# ------------------------------------------------------------
# HuggingFace model ID or local checkpoint path.
# Supported: Qwen3-VL-4B/7B/30B/72B, Qwen2.5-VL-7B, Qwen3-Omni, etc.
MODEL_CKPT="Qwen/Qwen3-VL-7B-Instruct"

# GPU device(s) to use (comma-separated).
# For vLLM tensor parallelism, list all GPUs you want to use:
#   e.g. "0,1,2,3" for 4-GPU inference
CUDA_VISIBLE_DEVICES="0"

# ------------------------------------------------------------
#  Output
# ------------------------------------------------------------
OUTPUT_DIR="./results"

# ------------------------------------------------------------
#  vLLM-specific options (only used when BACKEND="vllm")
# ------------------------------------------------------------
# Samples sent to llm.generate() per call. Higher = more throughput
# but more GPU memory. Recommended: 4-16 depending on video length.
VLLM_BATCH_SIZE=8

# GPU memory fraction allocated to vLLM (0.0–1.0).
VLLM_GPU_MEM_UTIL=0.9

# Tensor parallel size. Leave empty to auto-detect (= GPU count).
# Set explicitly to override, e.g. VLLM_TP_SIZE=4
VLLM_TP_SIZE=""

# ------------------------------------------------------------
#  FutureOmni
#  Benchmark: omni-modal future forecasting (video + audio QA)
#  Dataset  : 919 videos, 1034 multiple-choice QA pairs
#  Options  : 4–6 choices (A–F) per question
# ------------------------------------------------------------
FUTUREOMNI_DATA_FILE="/path/to/FutureOmni/eval/data/test.json"
FUTUREOMNI_VIDEO_ROOT="/path/to/FutureOmni/videos"

# (Optional) Directory containing audio files named {qid}.wav.
# Set to "" to disable audio input.
FUTUREOMNI_AUDIO_ROOT=""

# Frames to sample per video.
# Default: 32  (matches FutureOmni infer_ddp.py / infer_vllm.py defaults)
# Note: RL-trained models (e.g. ArrowRL) typically use 256 frames.
FUTUREOMNI_NFRAMES=32

# ------------------------------------------------------------
#  seeAoT
#  Benchmark: temporal direction sensitivity (AoTBench)
#  Dataset  : 5 subsets, video-only multiple-choice QA
#  Options  : 3–4 choices (A–C/D) per question
# ------------------------------------------------------------

# Root containing data_files/ and videos/ subdirectories.
SEEAOT_ROOT="/path/to/seeAoT"

# Subsets to evaluate. Comment out any you want to skip.
SEEAOT_SUBSETS=(
    "ReverseFilm"   # reversed film clips — temporal order understanding
    "UCF101"        # action recognition with temporal sensitivity
    "Rtime_t2v"     # text-to-video temporal matching
    "Rtime_v2t"     # video-to-text temporal matching
    "AoTBench_QA"   # general arrow-of-time QA
)

# Frames to sample per video.
# Default: 16  (matches seeAoT run_qwen25.py / run_qwen3.py defaults)
SEEAOT_NFRAMES=16

# ------------------------------------------------------------
#  MVBench
#  Benchmark: multi-task action/object/scene understanding
#  Dataset  : ~4000 QA pairs across 20 task types
#  Options  : 4 choices (A–D) per question
#  Tasks    : Action Sequence, Action Prediction, Action Antonym,
#             Fine-grained Action, Unexpected Action, Object Existence,
#             Object Interaction, Object Shuffle, Moving Direction,
#             Action Localization, Scene Transition, Action Count,
#             Moving Count, Moving Attribute, State Change,
#             Fine-grained Pose, Character Order, Egocentric Navigation,
#             Episodic Reasoning, Counterfactual Inference
# ------------------------------------------------------------

# Root directory with json/ and video/ subdirectories.
MVBENCH_DATA_ROOT="/path/to/MVBench"

# Frames to sample per video.
# Default: 16  (VLMEvalKit uses 8; 16 gives better accuracy for Qwen3)
MVBENCH_NFRAMES=16

# ------------------------------------------------------------
#  LongVideoBench
#  Benchmark: long-form video QA (15s – 1h videos) with subtitles
#  Dataset  : validation split (lvb_val.json)
#  Options  : 4 choices (A–D) per question
#  Categories: 17 task types (S2E, S2O, E2O, T2E, ...) × 4 duration groups
# ------------------------------------------------------------

# Root directory containing lvb_val.json, videos/, subtitles/.
LONGVIDEOBENCH_DATA_ROOT="/path/to/LongVideoBench"

# Frames to sample per video.
# Default: 32  (long videos benefit from more frames; VLMEvalKit uses 64 max)
# Increase to 64 for better accuracy on 10-min+ videos (needs more VRAM).
LONGVIDEOBENCH_NFRAMES=32

# Set to "true" to disable subtitle loading (faster, lower accuracy).
LONGVIDEOBENCH_NO_SUBTITLES="false"
