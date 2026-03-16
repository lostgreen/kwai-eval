#!/usr/bin/env python3
"""
Unified Qwen3-VL inference script for FutureOmni and seeAoT benchmarks.

Supports:
  - Qwen3-VL (and Qwen2.5-VL) via transformers
  - Resume: skips already-processed samples in the output JSONL
  - Both datasets via --dataset flag

Usage:
  python eval/infer.py \
      --dataset seeaot \
      --data_file /path/to/AoTBench_QA.json \
      --video_root /path/to/videos \
      --ckpt Qwen/Qwen3-VL-4B-Instruct \
      --output_dir ./results \
      --nframes 16
"""
import argparse
import json
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from dataset import QASample, load_dataset

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
# FutureOmni: model must reason over audio-visual future context
_FO_PROMPT_BASE = (
    "Based on the audio and video content, select the most likely future event.\n"
    "{question}\n{options}\n"
    "Answer with only the letter of the correct option ({letters})."
)

# seeAoT: straightforward multiple-choice over temporal video content
_AOT_PROMPT = "{question}\n{options}\nAnswer with only the option letter ({letters})."


def _letters(n: int) -> str:
    return ", ".join("ABCDEF"[:n])


def build_prompt(sample: QASample, dataset_type: str) -> str:
    opts = "\n".join(sample.options)
    n = len(sample.options)
    if dataset_type == "futureomni":
        return _FO_PROMPT_BASE.format(
            question=sample.question, options=opts, letters=_letters(n)
        )
    else:
        return _AOT_PROMPT.format(
            question=sample.question, options=opts, letters=_letters(n)
        )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(ckpt: str):
    """Load Qwen3-VL (or Qwen2.5-VL) model and processor."""
    from transformers import AutoProcessor

    # Try Qwen3VLForConditionalGeneration (transformers >= 4.52)
    try:
        from transformers import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            ckpt, dtype=torch.bfloat16, device_map="auto"
        )
        print(f"[Model] Loaded as Qwen3VLForConditionalGeneration")
    except (ImportError, Exception):
        # Fallback: Qwen2.5-VL or AutoModelForCausalLM
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                ckpt, torch_dtype=torch.bfloat16, device_map="auto"
            )
            print(f"[Model] Loaded as Qwen2_5_VLForConditionalGeneration")
        except (ImportError, Exception):
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                ckpt, torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True
            )
            print(f"[Model] Loaded as AutoModelForCausalLM (fallback)")

    processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    model.eval()
    return model, processor


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def infer_one(model, processor, sample: QASample, dataset_type: str,
              nframes: int) -> str:
    """Run inference on a single sample. Returns raw model response string."""
    if not os.path.exists(sample.video_path):
        return f"[FILE_NOT_FOUND: {sample.video_path}]"

    prompt = build_prompt(sample, dataset_type)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": sample.video_path,
                    "nframes": nframes,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Process vision inputs via qwen_vl_utils (installed with Qwen models)
    try:
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            **video_kwargs,
        )
    except ImportError:
        # Minimal fallback: text-only (no vision) – warns user
        print("[WARN] qwen_vl_utils not found; falling back to text-only input.")
        inputs = processor(text=[text], return_tensors="pt")

    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=16)

    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
    return response.strip()


def extract_answer(response: str) -> str:
    """Extract the single-letter answer from model output."""
    s = response.strip()
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
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["idx"])
                    except Exception:
                        pass
    return done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Unified Qwen3-VL evaluation for FutureOmni and seeAoT"
    )
    parser.add_argument(
        "--dataset", required=True, choices=["futureomni", "seeaot"],
        help="Dataset type"
    )
    parser.add_argument("--data_file", required=True, help="Path to input JSON file")
    parser.add_argument("--video_root", required=True, help="Directory containing videos")
    parser.add_argument(
        "--ckpt", required=True, help="Model checkpoint path or HuggingFace model ID"
    )
    parser.add_argument("--output_dir", default="./results", help="Output directory")
    parser.add_argument("--nframes", type=int, default=16, help="Frames to sample per video")
    parser.add_argument(
        "--audio_root", default=None,
        help="(FutureOmni only) Directory containing .wav audio files"
    )
    args = parser.parse_args()

    # Build output file path
    data_name = os.path.splitext(os.path.basename(args.data_file))[0]
    model_name = os.path.basename(args.ckpt.rstrip("/"))
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        f"{args.dataset}_{data_name}_{model_name}_{args.nframes}f.jsonl",
    )

    print(f"[Config] dataset={args.dataset} | data={args.data_file}")
    print(f"[Config] model={args.ckpt} | nframes={args.nframes}")
    print(f"[Output] {output_file}")

    # Load dataset
    kwargs = {}
    if args.dataset == "futureomni" and args.audio_root:
        kwargs["audio_root"] = args.audio_root
    dataset = load_dataset(args.dataset, args.data_file, args.video_root, **kwargs)
    print(f"[Dataset] {len(dataset)} samples loaded")

    # Resume
    done_ids = load_done_ids(output_file)
    remaining = len(dataset) - len(done_ids)
    print(f"[Resume] {len(done_ids)} done, {remaining} remaining")
    if remaining == 0:
        print("[Skip] All samples already processed.")
        return output_file

    # Load model
    print(f"[Model] Loading {args.ckpt} ...")
    model, processor = load_model(args.ckpt)
    print("[Model] Ready")

    # Run inference
    errors = 0
    with open(output_file, "a", encoding="utf-8") as fout:
        for i in tqdm(range(len(dataset)), desc="Inference"):
            sample = dataset[i]
            if sample.idx in done_ids:
                continue

            try:
                response = infer_one(model, processor, sample, args.dataset, args.nframes)
                pred = extract_answer(response)
            except Exception as e:
                print(f"\n[Error] idx={sample.idx}: {e}")
                response = f"ERROR: {e}"
                pred = "N/A"
                errors += 1

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
    return output_file


if __name__ == "__main__":
    main()
