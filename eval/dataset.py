#!/usr/bin/env python3
"""
Unified QA dataset loader for all four benchmarks:
  - FutureOmni:     video + optional audio, future forecasting
  - seeAoT:         video-only, temporal direction sensitivity (5 subsets)
  - MVBench:        video + gif + frame, 20 action/object/scene task types
  - LongVideoBench: long video QA with optional subtitle context

Data format details
-------------------
FutureOmni (test.json):
  {"qid": int, "question": str, "options": [str,...], "answer": str,
   "video_domain": str, "audio_type": str, ...}
  Videos named: {qid}.mp4

seeAoT (*.json):
  {"qa_idx": int, "video_name": str,
   "question": "text\nA. opt\nB. opt\n...", "ans": str}

MVBench (per-task JSON files under <root>/json/):
  [{"video": str, "question": str, "answer": str (text), "candidates": [str,...]}]
  Videos under <root>/video/<task_prefix>/<video>

LongVideoBench (lvb_val.json):
  {"video_path": str, "question": str, "candidates": [str,...],
   "correct_choice": int (0-3), "question_category": str,
   "duration_group": int, "subtitle_path": str, ...}
  Videos at <root>/<video_path>
"""
import glob
import json
import os
import shutil
import tarfile
import zipfile
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class QASample:
    idx: int
    video_path: str
    question: str
    options: List[str]         # ["A. Option 1", "B. Option 2", ...]
    answer: str                # single letter "A"/"B"/...
    audio_path: Optional[str] = None
    # For frame-based tasks (MVBench Episodic Reasoning): list of image paths
    frame_paths: Optional[List[str]] = None
    metadata: dict = field(default_factory=dict)

    def full_question(self) -> str:
        """Question + options as a single string."""
        return self.question + "\n" + "\n".join(self.options)


def _looks_like_hf_cache_dir(path: str) -> bool:
    base = os.path.basename(os.path.abspath(path))
    return base.startswith("datasets--")


def _latest_snapshot_dir(cache_root: str) -> Optional[str]:
    snapshots_dir = os.path.join(cache_root, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return None
    candidates = [
        os.path.join(snapshots_dir, name)
        for name in os.listdir(snapshots_dir)
        if os.path.isdir(os.path.join(snapshots_dir, name))
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _hf_cache_roots() -> List[str]:
    roots: List[str] = []
    for env_name in ("HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE"):
        value = os.environ.get(env_name)
        if value:
            roots.append(value)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(os.path.join(hf_home, "hub"))
    roots.append(os.path.expanduser("~/.cache/huggingface/hub"))
    # Keep order but drop duplicates.
    out = []
    seen = set()
    for root in roots:
        root = os.path.abspath(root)
        if root not in seen:
            out.append(root)
            seen.add(root)
    return out


def _default_cache_dirname(repo_id: str) -> str:
    namespace, name = repo_id.split("/", 1)
    return f"datasets--{namespace.replace('/', '--')}--{name.replace('/', '--')}"


def _find_dataset_root(
    data_root: str,
    repo_id: str,
    required_relpaths: List[str],
) -> Optional[str]:
    candidates: List[str] = []
    if data_root:
        candidates.append(os.path.abspath(data_root))
        if _looks_like_hf_cache_dir(data_root):
            snap = _latest_snapshot_dir(data_root)
            if snap:
                candidates.append(snap)

    cache_dirname = _default_cache_dirname(repo_id)
    for hub_root in _hf_cache_roots():
        cache_root = os.path.join(hub_root, cache_dirname)
        candidates.append(cache_root)
        snap = _latest_snapshot_dir(cache_root)
        if snap:
            candidates.append(snap)

    seen = set()
    for root in candidates:
        if not root or root in seen:
            continue
        seen.add(root)
        if all(os.path.exists(os.path.join(root, rel)) for rel in required_relpaths):
            return root
    return None


def _snapshot_download_dataset(repo_id: str) -> str:
    from huggingface_hub import login, snapshot_download

    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if token:
        try:
            login(token, add_to_git_credential=False)
        except Exception:
            pass
    return snapshot_download(repo_id=repo_id, repo_type="dataset")


def _prepare_mvbench_root(root: str) -> None:
    video_root = os.path.join(root, "video")
    if os.path.isdir(video_root):
        for filename in os.listdir(video_root):
            if not filename.endswith(".zip"):
                continue
            zip_path = os.path.join(video_root, filename)
            marker = zip_path + ".extracted"
            if os.path.exists(marker):
                continue
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(video_root)
            open(marker, "w").close()

    src_folder = os.path.join(video_root, "data0613")
    if os.path.isdir(src_folder):
        for subdir in os.listdir(src_folder):
            subdir_path = os.path.join(src_folder, subdir)
            if not os.path.isdir(subdir_path):
                continue
            for subsubdir in os.listdir(subdir_path):
                subsubdir_path = os.path.join(subdir_path, subsubdir)
                if not os.path.isdir(subsubdir_path):
                    continue
                target_folder = os.path.join(video_root, subdir, subsubdir)
                os.makedirs(target_folder, exist_ok=True)
                for item in os.listdir(subsubdir_path):
                    src_path = os.path.join(subsubdir_path, item)
                    dst_path = os.path.join(target_folder, item)
                    if not os.path.exists(dst_path):
                        shutil.move(src_path, dst_path)


def _prepare_longvideobench_root(root: str) -> None:
    videos_dir = os.path.join(root, "videos")
    if os.path.isdir(videos_dir):
        return

    tar_files = glob.glob(os.path.join(root, "**", "*.tar*"), recursive=True)
    if not tar_files:
        return

    grouped = {}
    for tar_file in tar_files:
        base_name = tar_file.split(".tar")[0]
        grouped.setdefault(base_name, []).append(tar_file)

    for base_name, parts in grouped.items():
        output_tar = base_name + ".tar"
        if len(parts) == 1 and parts[0].endswith(".tar"):
            output_tar = parts[0]
        elif not os.path.exists(output_tar):
            with open(output_tar, "wb") as out_tar:
                for part in sorted(parts):
                    with open(part, "rb") as part_file:
                        shutil.copyfileobj(part_file, out_tar)

        extract_dir = os.path.join(root, os.path.basename(base_name))
        if os.path.exists(extract_dir):
            continue
        with tarfile.open(output_tar, "r") as tar_ref:
            tar_ref.extractall(root)


def _longvideobench_missing_videos(root: str, max_report: int = 8) -> List[str]:
    ann_path = os.path.join(root, "lvb_val.json")
    if not os.path.exists(ann_path):
        return ["lvb_val.json"]

    try:
        with open(ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return ["lvb_val.json (unreadable)"]

    missing: List[str] = []
    for item in data:
        rel_path = str(item.get("video_path", "")).lstrip("./")
        if not rel_path:
            missing.append("[empty video_path]")
        elif not os.path.exists(os.path.join(root, rel_path)):
            missing.append(rel_path)
        if len(missing) >= max_report:
            break
    return missing


def _longvideobench_is_complete(root: str) -> bool:
    return len(_longvideobench_missing_videos(root, max_report=1)) == 0


def _resolve_mvbench_root(data_root: str) -> str:
    repo_id = "OpenGVLab/MVBench"
    required = ["json", "video"]
    root = _find_dataset_root(data_root, repo_id, required)
    if root is None:
        root = _snapshot_download_dataset(repo_id)
    _prepare_mvbench_root(root)
    root = _find_dataset_root(root, repo_id, required) or root
    if not all(os.path.exists(os.path.join(root, rel)) for rel in required):
        raise FileNotFoundError(
            f"MVBench dataset root is invalid: {root}. Expected json/ and video/."
        )
    return root


def _resolve_longvideobench_root(data_root: str) -> str:
    repo_id = "longvideobench/LongVideoBench"
    required = ["lvb_val.json"]
    root = _find_dataset_root(data_root, repo_id, required)
    if root is not None:
        _prepare_longvideobench_root(root)
        if _longvideobench_is_complete(root):
            return root
        missing = _longvideobench_missing_videos(root)
        print(
            "[LongVideoBench] Incomplete local cache detected, "
            f"will resume download. Missing examples: {missing}"
        )

    root = _snapshot_download_dataset(repo_id)
    _prepare_longvideobench_root(root)
    root = _find_dataset_root(root, repo_id, required) or root
    if not os.path.exists(os.path.join(root, "lvb_val.json")):
        raise FileNotFoundError(
            f"LongVideoBench dataset root is invalid: {root}. Expected lvb_val.json."
        )
    if not _longvideobench_is_complete(root):
        missing = _longvideobench_missing_videos(root)
        print(
            "[LongVideoBench] WARNING: download/extraction is still incomplete at "
            f"{root}. Missing examples: {missing}\n"
            "  These samples will be skipped during inference."
        )
    return root


# ---------------------------------------------------------------------------
# FutureOmni
# ---------------------------------------------------------------------------

class FutureOmniDataset:
    """Video + optional audio QA — future forecasting, 4-6 choices (A-F)."""

    DATASET_NAME = "futureomni"

    def __init__(self, data_file: str, video_root: str, audio_root: Optional[str] = None):
        self.video_root = video_root
        self.audio_root = audio_root
        with open(data_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> QASample:
        item = self.data[idx]
        qid = item["qid"]

        audio_path = None
        if self.audio_root:
            ap = os.path.join(self.audio_root, f"{qid}.wav")
            if os.path.exists(ap):
                audio_path = ap

        return QASample(
            idx=qid,
            video_path=os.path.join(self.video_root, f"{qid}.mp4"),
            question=item["question"],
            options=item.get("options", []),
            answer=item["answer"],
            audio_path=audio_path,
            metadata={
                "video_domain": item.get("video_domain"),
                "audio_type": item.get("audio_type"),
                "forecasting_pattern": item.get("forecasting_pattern"),
                "source": item.get("source"),
                "seconds": item.get("seconds"),
            },
        )


# ---------------------------------------------------------------------------
# seeAoT
# ---------------------------------------------------------------------------

class SeeAoTDataset:
    """Video-only temporal direction QA — question field embeds options."""

    DATASET_NAME = "seeaot"

    def __init__(self, data_file: str, video_root: str):
        self.video_root = video_root
        with open(data_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> QASample:
        item = self.data[idx]
        lines = item["question"].strip().split("\n")
        question = lines[0]
        options = [ln for ln in lines[1:] if ln.strip()]

        return QASample(
            idx=item["qa_idx"],
            video_path=os.path.join(self.video_root, item["video_name"]),
            question=question,
            options=options,
            answer=item["ans"],
            metadata={"subset": os.path.splitext(os.path.basename(self.video_root))[0]},
        )


# ---------------------------------------------------------------------------
# MVBench
# ---------------------------------------------------------------------------

# (json_file, video_prefix_under_video/, data_type, has_time_bound)
_MVBENCH_TASK_MAP = {
    "Action Sequence":         ("action_sequence.json",        "star/Charades_v1_480/",         "video", True),
    "Action Prediction":       ("action_prediction.json",      "star/Charades_v1_480/",         "video", True),
    "Action Antonym":          ("action_antonym.json",         "ssv2_video/",                   "video", False),
    "Fine-grained Action":     ("fine_grained_action.json",    "Moments_in_Time_Raw/videos/",   "video", False),
    "Unexpected Action":       ("unexpected_action.json",      "FunQA_test/test/",              "video", False),
    "Object Existence":        ("object_existence.json",       "clevrer/video_validation/",     "video", False),
    "Object Interaction":      ("object_interaction.json",     "star/Charades_v1_480/",         "video", True),
    "Object Shuffle":          ("object_shuffle.json",         "perception/videos/",            "video", False),
    "Moving Direction":        ("moving_direction.json",       "clevrer/video_validation/",     "video", False),
    "Action Localization":     ("action_localization.json",    "sta/sta_video/",                "video", True),
    "Scene Transition":        ("scene_transition.json",       "scene_qa/video/",               "video", False),
    "Action Count":            ("action_count.json",           "perception/videos/",            "video", False),
    "Moving Count":            ("moving_count.json",           "clevrer/video_validation/",     "video", False),
    "Moving Attribute":        ("moving_attribute.json",       "clevrer/video_validation/",     "video", False),
    "State Change":            ("state_change.json",           "perception/videos/",            "video", False),
    "Fine-grained Pose":       ("fine_grained_pose.json",      "nturgbd/",                      "video", False),
    "Character Order":         ("character_order.json",        "perception/videos/",            "video", False),
    "Egocentric Navigation":   ("egocentric_navigation.json",  "vlnqa/",                        "video", False),
    "Episodic Reasoning":      ("episodic_reasoning.json",     "tvqa/frames_fps3_hq/",          "frame", True),
    "Counterfactual Inference":("counterfactual_inference.json","clevrer/video_validation/",    "video", False),
}


class MVBenchDataset:
    """
    MVBench: 20 task types, ~4000 QA pairs.

    Expected directory layout:
      <data_root>/
        json/          ← per-task JSON files
        video/         ← video/gif/frame subdirectories

    answer in raw JSON is the text of the correct candidate;
    this loader converts it to the option letter (A/B/C/D).
    """

    DATASET_NAME = "mvbench"

    SYS = (
        "Carefully watch the video and pay attention to the cause and sequence of "
        "events, the detail and movement of objects, and the action and pose of "
        "persons. Based on your observations, select the best option that "
        "accurately addresses the question."
    )

    def __init__(self, data_root: str, tasks: Optional[List[str]] = None):
        """
        Args:
            data_root: Path to MVBench root directory.
            tasks:     Subset of task names to load (None = all 20 tasks).
        """
        self.data_root = _resolve_mvbench_root(data_root)
        self.video_root = os.path.join(self.data_root, "video")
        self.json_root = os.path.join(self.data_root, "json")
        self._samples: List[QASample] = []
        self._load(tasks)

    def _load(self, tasks: Optional[List[str]]):
        task_names = tasks if tasks else list(_MVBENCH_TASK_MAP.keys())
        global_idx = 0
        for task_name in task_names:
            if task_name not in _MVBENCH_TASK_MAP:
                print(f"[MVBench] Unknown task: {task_name!r}, skipping.")
                continue
            json_file, prefix, data_type, _ = _MVBENCH_TASK_MAP[task_name]
            json_path = os.path.join(self.json_root, json_file)
            if not os.path.exists(json_path):
                print(f"[MVBench] JSON not found: {json_path}, skipping task '{task_name}'.")
                continue

            with open(json_path, "r", encoding="utf-8") as f:
                items = json.load(f)

            for item in items:
                candidates = item["candidates"]
                answer_text = item["answer"]
                try:
                    answer_idx = candidates.index(answer_text)
                except ValueError:
                    # Fuzzy match: find candidate that starts with answer text
                    answer_idx = next(
                        (i for i, c in enumerate(candidates) if answer_text in c), 0
                    )
                answer_letter = chr(ord("A") + answer_idx)
                options = [f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(candidates)]

                if data_type == "frame":
                    # Episodic Reasoning: directory of frames
                    frame_dir = os.path.join(self.video_root, prefix, item["video"])
                    frame_paths = _list_frames(frame_dir)
                    sample = QASample(
                        idx=global_idx,
                        video_path=frame_dir,  # directory path
                        question=item["question"],
                        options=options,
                        answer=answer_letter,
                        frame_paths=frame_paths,
                        metadata={"task_type": task_name, "data_type": data_type},
                    )
                else:
                    video_path = os.path.join(self.video_root, prefix, item["video"])
                    sample = QASample(
                        idx=global_idx,
                        video_path=video_path,
                        question=item["question"],
                        options=options,
                        answer=answer_letter,
                        metadata={"task_type": task_name, "data_type": data_type},
                    )
                self._samples.append(sample)
                global_idx += 1

        print(f"[MVBench] Loaded {len(self._samples)} samples across {len(task_names)} tasks.")

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx: int) -> QASample:
        return self._samples[idx]


def _list_frames(frame_dir: str) -> List[str]:
    """Return sorted list of image paths inside a frame directory."""
    if not os.path.isdir(frame_dir):
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [
        os.path.join(frame_dir, f)
        for f in sorted(os.listdir(frame_dir))
        if os.path.splitext(f)[1].lower() in exts
    ]
    return files


# ---------------------------------------------------------------------------
# LongVideoBench
# ---------------------------------------------------------------------------

_LVB_DURATION_GROUPS = [15, 60, 600, 3600]
_LVB_TASK_CATEGORIES = [
    "S2E", "S2O", "S2A", "E2O", "O2E", "T2E",
    "T2O", "T2A", "E3E", "O3O", "SSS", "SOS",
    "SAA", "T3E", "T3O", "TOS", "TAA",
]


class LongVideoBenchDataset:
    """
    LongVideoBench (validation split): long-form video QA with subtitle context.

    Expected directory layout:
      <data_root>/
        lvb_val.json       ← main annotation file
        videos/            ← video files (video_path field in JSON)
        subtitles/         ← optional subtitle JSON files

    correct_choice is an integer index (0-3); this loader converts to letter.
    Subtitles are loaded and appended to the question as plain text when available.
    """

    DATASET_NAME = "longvideobench"

    def __init__(self, data_root: str, use_subtitles: bool = True):
        """
        Args:
            data_root:     Path to LongVideoBench root directory.
            use_subtitles: Append subtitle text to question when available.
        """
        self.data_root = _resolve_longvideobench_root(data_root)
        self.use_subtitles = use_subtitles
        ann_path = os.path.join(self.data_root, "lvb_val.json")
        with open(ann_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> QASample:
        item = self.data[idx]

        # video_path in JSON is like "./videos/xxx.mp4"
        rel_path = item["video_path"].lstrip("./")
        video_path = os.path.join(self.data_root, rel_path)

        candidates = item["candidates"]
        answer_letter = chr(ord("A") + item["correct_choice"])
        options = [f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(candidates)]

        question = item["question"]
        if self.use_subtitles:
            subtitle_text = self._load_subtitle(item)
            if subtitle_text:
                question = f"[Subtitles]\n{subtitle_text}\n\n[Question]\n{question}"

        return QASample(
            idx=idx,
            video_path=video_path,
            question=question,
            options=options,
            answer=answer_letter,
            metadata={
                "question_category": item.get("question_category"),
                "duration_group": item.get("duration_group"),
                "duration": item.get("duration"),
                "video_id": item.get("video_id"),
            },
        )

    def _load_subtitle(self, item: dict) -> str:
        """Load subtitle file and return plain text, or empty string."""
        sub_path = item.get("subtitle_path", "")
        if not sub_path:
            return ""
        full_path = os.path.join(self.data_root, sub_path.lstrip("./"))
        if not os.path.exists(full_path):
            return ""
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                subs = json.load(f)
            start_ts = item.get("starting_timestamp_for_subtitles", 0) or 0
            lines = []
            for s in subs:
                # Two possible subtitle formats used by LongVideoBench
                if "timestamp" in s:
                    ts_start = s["timestamp"][0] if isinstance(s["timestamp"], list) else 0
                    if ts_start >= start_ts:
                        lines.append(s.get("text", "").strip())
                elif "start" in s:
                    lines.append(s.get("line", s.get("text", "")).strip())
            return "\n".join(ln for ln in lines if ln)
        except Exception:
            return ""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def load_dataset(dataset_type: str, data_file: str = "", video_root: str = "", **kwargs):
    """
    Factory to load a dataset by type.

    dataset_type options:
      "futureomni"     – data_file + video_root required
      "seeaot"         – data_file + video_root required
      "mvbench"        – pass data_root=<path> in kwargs (data_file/video_root unused)
      "longvideobench" – pass data_root=<path> in kwargs
    """
    if dataset_type == "futureomni":
        return FutureOmniDataset(data_file, video_root, **kwargs)
    elif dataset_type == "seeaot":
        return SeeAoTDataset(data_file, video_root)
    elif dataset_type == "mvbench":
        data_root = kwargs.pop("data_root", data_file)
        return MVBenchDataset(data_root, **kwargs)
    elif dataset_type == "longvideobench":
        data_root = kwargs.pop("data_root", data_file)
        return LongVideoBenchDataset(data_root, **kwargs)
    else:
        raise ValueError(
            f"Unknown dataset_type {dataset_type!r}. "
            "Choose: 'futureomni', 'seeaot', 'mvbench', 'longvideobench'."
        )
