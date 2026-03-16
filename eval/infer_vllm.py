#!/usr/bin/env python3
"""
vLLM-based inference for all four benchmarks with automatic multi-GPU acceleration.

Key advantages over infer.py (transformers):
  - Tensor parallelism: auto-detects GPU count, distributes model across all GPUs
  - Continuous batching: processes BATCH_SIZE requests in parallel via vLLM engine
  - Higher throughput: ~3-5× faster than single-GPU transformers inference

Supports: FutureOmni, seeAoT, MVBench, LongVideoBench

Usage:
  # Single dataset (auto-detects all GPUs)
  python eval/infer_vllm.py \
      --dataset seeaot \
      --data_file /path/to/AoTBench_QA.json \
      --video_root /path/to/videos \
      --ckpt Qwen/Qwen3-VL-7B-Instruct \
      --output_dir ./results \
      --nframes 16 \
      --batch_size 8

  # MVBench (uses data_root instead of data_file)
  python eval/infer_vllm.py \
      --dataset mvbench \
      --data_root /path/to/MVBench \
      --ckpt Qwen/Qwen3-VL-7B-Instruct \
      --nframes 16

  # LongVideoBench
  python eval/infer_vllm.py \
      --dataset longvideobench \
      --data_root /path/to/LongVideoBench \
      --ckpt Qwen/Qwen3-VL-7B-Instruct \
      --nframes 32

Environment:
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  export VLLM_WORKER_MULTIPROC_METHOD=spawn  # required for multi-GPU
"""
import argparse
import json
import math
import os
import sys

import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from typing import Optional
sys.path.insert(0, os.path.dirname(__file__))
from dataset import MVBenchDataset, QASample, load_dataset


# ---------------------------------------------------------------------------
# Video reading utilities (cv2-based, avoids qwen_vl_utils nframes issues)
# ---------------------------------------------------------------------------

_IMAGE_FACTOR = 28
_VIDEO_MIN_PIXELS = 128 * 28 * 28
_VIDEO_MAX_PIXELS = 768 * 28 * 28


def _smart_resize(height: int, width: int, factor: int = _IMAGE_FACTOR,
                  min_pixels: int = _VIDEO_MIN_PIXELS,
                  max_pixels: int = _VIDEO_MAX_PIXELS):
    """Resize keeping aspect ratio, divisible by factor, within pixel budget."""
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _read_video_cv2(video_path: str, nframes: int) -> torch.Tensor:
    """Read *nframes* evenly-spaced frames from *video_path* using OpenCV.

    Returns a float32 tensor of shape ``(nframes, 3, H', W')`` where H'/W'
    are the smart-resized dimensions matching Qwen VL requirements.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0 or fps <= 0:
        cap.release()
        raise IOError(f"Invalid video metadata: {video_path}")

    duration = total_frames / fps
    timestamps = np.linspace(0, duration, nframes, endpoint=False)

    frames = []
    for t in timestamps:
        frame_id = int(round(t * fps))
        frame_id = min(frame_id, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            continue
        # BGR→RGB, HWC→CHW
        frames.append(np.transpose(frame[:, :, ::-1], (2, 0, 1)))

    cap.release()

    if len(frames) < 2:
        raise IOError(f"Could only read {len(frames)} frames from {video_path}")

    video_tensor = torch.tensor(np.array(frames))  # (N, 3, H, W)
    _, _, h, w = video_tensor.shape
    rh, rw = _smart_resize(h, w)
    video_tensor = transforms.functional.resize(
        video_tensor, [rh, rw],
        interpolation=InterpolationMode.BICUBIC, antialias=True,
    ).float()
    return video_tensor

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_FO_PROMPT = (
    "Based on the audio and video content, select the most likely future event.\n"
    "{question}\n{options}\n"
    "Answer with only the letter of the correct option ({letters})."
)

_AOT_PROMPT = "{question}\n{options}\nAnswer with only the option letter ({letters})."

# VLMEvalKit-aligned: mirrors mvbench.py system + qa_template
_MVB_SYS = (
    "Carefully watch the video and pay attention to the cause and sequence of "
    "events, the detail and movement of objects, and the action and pose of "
    "persons. Based on your observations, select the best option that accurately "
    "addresses the question."
)
_MVB_PROMPT = "Question: {question}\nOptions:\n{options}\nOnly give the best option.\nBest option:("

# VLMEvalKit-aligned: mirrors longvideobench.py prompt
_LVB_PROMPT = (
    "{question}\n{options}\n"
    "Answer with the option's letter from the given choices directly."
)


def _letters(n: int) -> str:
    return ", ".join("ABCDEF"[:n])


def build_prompt(sample: QASample, dataset_type: str) -> str:
    opts_plain = "\n".join(sample.options)  # "A. opt\nB. opt\n..."
    n = len(sample.options)
    if dataset_type == "futureomni":
        return _FO_PROMPT.format(
            question=sample.question, options=opts_plain, letters=_letters(n)
        )
    elif dataset_type == "seeaot":
        return _AOT_PROMPT.format(
            question=sample.question, options=opts_plain, letters=_letters(n)
        )
    elif dataset_type == "mvbench":
        # MVBench uses (A) / (B) style internally in prompt; options already formatted
        opts_mvb = "\n".join(
            f"({chr(ord('A') + i)}) {sample.options[i][3:]}"  # strip "A. " prefix
            if len(sample.options[i]) > 2 and sample.options[i][1] == "."
            else sample.options[i]
            for i in range(n)
        )
        return _MVB_PROMPT.format(question=sample.question, options=opts_mvb)
    elif dataset_type == "longvideobench":
        return _LVB_PROMPT.format(question=sample.question, options=opts_plain)
    else:
        return f"{sample.question}\n{opts_plain}\nAnswer with only the option letter."


def build_system_prompt(dataset_type: str) -> Optional[str]:
    if dataset_type == "mvbench":
        return _MVB_SYS
    return None


# ---------------------------------------------------------------------------
# vLLM model loading
# ---------------------------------------------------------------------------

def auto_select_tp_size(ckpt: str, n_gpus: int) -> int:
    """
    Pick a conservative TP size for dense VL models.

    Using all visible GPUs for small models (e.g. 4B) often hurts throughput
    because communication cost dominates. This heuristic keeps TP modest for
    smaller checkpoints while still scaling large/MoE models across all GPUs.
    """
    if n_gpus <= 1:
        return 1

    ckpt_lower = ckpt.lower()
    if any(tag in ckpt for tag in ["-A3B", "-A22B", "-A47B", "MoE"]):
        return n_gpus

    if any(tag in ckpt_lower for tag in ["72b", "70b", "32b", "30b", "27b", "34b"]):
        return n_gpus
    if any(tag in ckpt_lower for tag in ["14b", "13b", "9b", "8b", "7b"]):
        return min(4, n_gpus)
    if any(tag in ckpt_lower for tag in ["4b", "3b", "2b"]):
        return min(2, n_gpus)
    return min(4, n_gpus)


def load_vllm_model(
    ckpt: str,
    tp_size: Optional[int],
    gpu_mem_util: float,
    max_num_seqs: int,
):
    """
    Initialize vLLM engine with automatic tensor parallelism.

    Args:
        ckpt:         Model path or HuggingFace ID.
        tp_size:      Tensor parallel size. None = auto (all visible GPUs).
        gpu_mem_util: GPU memory utilization fraction (0.0–1.0).
    """
    from vllm import LLM

    n_gpus = torch.cuda.device_count()
    if tp_size is None:
        tp_size = auto_select_tp_size(ckpt, n_gpus)

    print(f"[vLLM] Initializing: model={ckpt}, tp={tp_size}/{n_gpus} GPUs, "
          f"mem_util={gpu_mem_util}")

    # MoE models (e.g. Qwen3-A22B) need expert parallelism
    is_moe = any(tag in ckpt for tag in ["-A3B", "-A22B", "-A47B", "MoE"])

    # Omni models need vLLM v0 engine
    if "omni" in ckpt.lower():
        os.environ["VLLM_USE_V1"] = "0"
        limit_mm = {"image": 3, "video": 3, "audio": 3}
    else:
        limit_mm = {"image": 24, "video": 4}

    llm = LLM(
        model=ckpt,
        tensor_parallel_size=tp_size,
        enable_expert_parallel=is_moe,
        max_num_seqs=max_num_seqs,
        limit_mm_per_prompt=limit_mm,
        gpu_memory_utilization=gpu_mem_util,
        trust_remote_code=True,
        seed=0,
    )

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)

    return llm, processor


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------

def build_request(
    processor,
    sample: QASample,
    dataset_type: str,
    nframes: int,
    fps: float,
    sys_prompt: Optional[str],
) -> dict:
    """
    Build a single vLLM request dict for a QASample.

    Reads video frames with OpenCV (bypasses qwen_vl_utils decord issues)
    and passes pre-loaded tensors to vLLM.
    """
    prompt_text = build_prompt(sample, dataset_type)

    # Build content list (for chat template only — actual pixels loaded separately)
    if sample.frame_paths:
        content = [{"type": "image", "image": fp} for fp in sample.frame_paths]
    else:
        video_item = {"type": "video", "video": sample.video_path}
        if fps > 0:
            video_item["fps"] = fps
        else:
            video_item["nframes"] = nframes
        video_item["min_pixels"] = 100352
        video_item["max_pixels"] = 602112
        video_item["total_pixels"] = 38535168 if nframes >= 64 else 19267584
        content = [video_item]

    if sys_prompt:
        content.append({"type": "text", "text": prompt_text})
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": content},
        ]
    else:
        content.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    if sample.frame_paths:
        from qwen_vl_utils import process_vision_info
        image_data, _ = process_vision_info(messages)
        return {
            "prompt": text,
            "multi_modal_data": {"image": image_data},
        }

    # Read video with OpenCV (matches FutureOmni's extract.py approach)
    video_tensor = _read_video_cv2(sample.video_path, nframes)

    return {
        "prompt": text,
        "multi_modal_data": {"video": [video_tensor]},
    }


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(response: str) -> str:
    s = response.strip()
    # MVBench prompt ends with "Best option:(", so response often starts with the letter
    if s and s[0].upper() in "ABCDEF":
        return s[0].upper()
    for ch in s.upper():
        if ch in "ABCDEF":
            return ch
    return s[:1].upper() if s else "N/A"


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def load_done_ids(output_file: str) -> set:
    done = set()
    if not os.path.exists(output_file):
        return done
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["idx"])
                except Exception:
                    pass
    return done


def select_indices(indices: list[int], shard_rank: int, num_shards: int) -> list[int]:
    if num_shards <= 1:
        return indices
    return indices[shard_rank::num_shards]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Fix Optional import at top of file
from typing import Optional


def main():
    parser = argparse.ArgumentParser(
        description="vLLM multi-GPU inference for FutureOmni / seeAoT / MVBench / LongVideoBench"
    )
    parser.add_argument(
        "--dataset", required=True,
        choices=["futureomni", "seeaot", "mvbench", "longvideobench"],
    )
    # FutureOmni / seeAoT
    parser.add_argument("--data_file", default="", help="Path to input JSON (FutureOmni/seeAoT)")
    parser.add_argument("--video_root", default="", help="Video directory (FutureOmni/seeAoT)")
    parser.add_argument("--audio_root", default="", help="Audio directory (FutureOmni only)")
    # MVBench / LongVideoBench
    parser.add_argument("--data_root", default="", help="Data root dir (MVBench/LongVideoBench)")
    # Common
    parser.add_argument("--ckpt", required=True, help="Model path or HuggingFace model ID")
    parser.add_argument("--output_dir", default="./results", help="Output directory")
    parser.add_argument("--output_file", default="", help="Output JSONL path override")
    parser.add_argument("--nframes", type=int, default=16, help="Frames to sample per video")
    parser.add_argument("--fps", type=float, default=-1.0, help="Sample video at fixed FPS (>0 overrides nframes)")
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Number of samples to batch per vLLM generate() call",
    )
    parser.add_argument(
        "--tp_size", type=int, default=None,
        help="Tensor parallel size (default: auto = all visible GPUs)",
    )
    parser.add_argument(
        "--gpu_mem_util", type=float, default=0.9,
        help="vLLM GPU memory utilization fraction (default: 0.9)",
    )
    parser.add_argument(
        "--max_num_seqs", type=int, default=8,
        help="Max concurrent sequences inside vLLM engine (default: 8)",
    )
    parser.add_argument(
        "--no_subtitles", action="store_true",
        help="(LongVideoBench) Disable subtitle loading",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None,
        help="(MVBench) Specific task names to evaluate (default: all 20)",
    )
    parser.add_argument("--num_shards", type=int, default=1, help="Total dataset shards")
    parser.add_argument("--shard_rank", type=int, default=0, help="Current shard rank")
    parser.add_argument("--max_batches", type=int, default=0, help="Stop after this many batches (0 = all)")
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_rank < 0 or args.shard_rank >= args.num_shards:
        raise ValueError("--shard_rank must be in [0, num_shards)")

    # Require VLLM_WORKER_MULTIPROC_METHOD=spawn for multi-GPU
    if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") != "spawn":
        print("[WARN] Set 'export VLLM_WORKER_MULTIPROC_METHOD=spawn' for multi-GPU stability.")

    # Build output filename
    if args.dataset in ("mvbench", "longvideobench"):
        data_root = args.data_root or args.data_file
        tag = args.dataset
    else:
        data_root = ""
        tag = f"{args.dataset}_{os.path.splitext(os.path.basename(args.data_file))[0]}"

    model_name = os.path.basename(args.ckpt.rstrip("/"))
    os.makedirs(args.output_dir, exist_ok=True)
    sample_tag = f"{args.fps:g}fps" if args.fps > 0 else f"{args.nframes}f"
    output_file = args.output_file or os.path.join(
        args.output_dir, f"{tag}_{model_name}_{sample_tag}_vllm.jsonl"
    )

    print(
        f"[Config] dataset={args.dataset} | nframes={args.nframes} | "
        f"fps={args.fps} | batch={args.batch_size} | "
        f"max_num_seqs={args.max_num_seqs} | "
        f"shard={args.shard_rank + 1}/{args.num_shards}"
    )
    print(f"[Config] model={args.ckpt}")
    print(f"[Output] {output_file}")

    # Load dataset
    if args.dataset == "futureomni":
        kw = {"audio_root": args.audio_root} if args.audio_root else {}
        dataset = load_dataset("futureomni", args.data_file, args.video_root, **kw)
    elif args.dataset == "seeaot":
        dataset = load_dataset("seeaot", args.data_file, args.video_root)
    elif args.dataset == "mvbench":
        kw = {"tasks": args.tasks} if args.tasks else {}
        dataset = load_dataset("mvbench", data_root=data_root, **kw)
    elif args.dataset == "longvideobench":
        dataset = load_dataset(
            "longvideobench", data_root=data_root,
            use_subtitles=not args.no_subtitles,
        )

    print(f"[Dataset] {len(dataset)} samples loaded")

    # Resume
    done_ids = load_done_ids(output_file)
    indices = [i for i in range(len(dataset)) if dataset[i].idx not in done_ids]
    indices = select_indices(indices, args.shard_rank, args.num_shards)
    print(f"[Resume] {len(done_ids)} done, {len(indices)} remaining in shard")
    if not indices:
        print("[Skip] All samples already processed.")
        return

    # Load vLLM model
    llm, processor = load_vllm_model(
        args.ckpt, args.tp_size, args.gpu_mem_util, args.max_num_seqs
    )

    from vllm import SamplingParams
    sampling_params = SamplingParams(
        temperature=0.01,
        max_tokens=16,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.0,
    )

    sys_prompt = build_system_prompt(args.dataset)

    # Batch inference
    errors = 0
    with open(output_file, "a", encoding="utf-8") as fout:
        batch_iter = range(0, len(indices), args.batch_size)
        if args.max_batches > 0:
            batch_iter = list(batch_iter)[:args.max_batches]
        for batch_start in tqdm(
            batch_iter,
            desc=f"Inference (batch={args.batch_size})",
        ):
            batch_indices = indices[batch_start: batch_start + args.batch_size]
            samples = [dataset[i] for i in batch_indices]

            # Build requests, skipping missing videos
            requests = []
            valid_samples = []
            for sample in samples:
                if sample.frame_paths is None and not os.path.exists(sample.video_path):
                    print(f"\n[Skip] Missing: {sample.video_path}")
                    continue
                try:
                    req = build_request(
                        processor, sample, args.dataset, args.nframes, args.fps, sys_prompt
                    )
                    requests.append(req)
                    valid_samples.append(sample)
                except Exception as e:
                    print(f"\n[Error] Build request for idx={sample.idx}: {e}")
                    errors += 1

            if not requests:
                continue

            # Generate
            try:
                outputs = llm.generate(requests, sampling_params)
            except Exception as e:
                print(f"\n[Error] llm.generate batch starting at {batch_start}: {e}")
                errors += len(requests)
                continue

            # Save results
            for sample, output in zip(valid_samples, outputs):
                response = output.outputs[0].text.strip()
                pred = extract_answer(response)
                result = {
                    "idx": sample.idx,
                    "pred": pred,
                    "response": response,
                    "answer": sample.answer,
                    "correct": pred == sample.answer,
                    "video_path": sample.video_path,
                    "metadata": sample.metadata,
                }
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"\n[Done] Results → {output_file}  (errors: {errors})")


if __name__ == "__main__":
    main()
