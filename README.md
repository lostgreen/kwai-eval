# Qwen3-Eval 使用指南

## 目录结构

```
Qwen3-Eval/
├── config.sh                   ← 第一步: 只改这里
├── run_all_8gpu_parallel.sh    ← 一键8卡并行
├── futureomni/
│   ├── config.sh               ← 覆盖局部参数(可选)
│   └── run.sh                  ← 单独跑 FutureOmni
├── seeaot/
│   ├── config.sh
│   └── run.sh
├── mvbench/
│   ├── config.sh
│   ├── eval_config.json        ← VLMEvalKit 模型注册 (见第2步)
│   └── run.sh
├── longvideobench/
│   ├── config.sh
│   ├── eval_config.json
│   └── run.sh
├── results/                    ← 所有评测结果统一输出到这里
│   ├── futureomni/
│   ├── seeaot/
│   ├── mvbench/
│   └── longvideobench/
└── logs/                       ← 并行跑时每个评测的日志
```

---

## 第一步: 修改全局配置

只需编辑根目录的 `config.sh`，填入服务器上的实际路径：

```bash
# 仓库路径
FUTUREOMNI_REPO="/path/to/FutureOmni"
SEEAOT_REPO="/path/to/seeAoT"
VLMEVALKIT_REPO="/path/to/VLMEvalKit"

# 模型路径 (本地权重目录)
MODEL_PATH="/path/to/models/Qwen3-VL-4B-Instruct"

# 数据路径
FUTUREOMNI_DATA_FILE="/path/to/FutureOmni/futureomni_test.json"
FUTUREOMNI_VIDEO_ROOT="/path/to/FutureOmni/videos"
SEEAOT_DATA_ROOT="/path/to/AoTBench"

# HuggingFace 缓存根目录 (VLMEvalKit 用这个路径找 MVBench / LongVideoBench)
# 指向直接包含 datasets--OpenGVLab--MVBench / datasets--longvideobench--LongVideoBench 的那一级目录
# run.sh 会自动在该目录下创建 hub/ 符号链接 (VLMEvalKit 内部会追加 /hub)
HF_HUB_CACHE_ROOT="/path/to/hf_cache"
```

> 各子文件夹的 config.sh 会自动 source 这个全局配置，无需重复填写。
> 如需覆盖某个评测的特定参数（GPU分配、帧数等），直接改对应子文件夹的 config.sh。

---

## 第二步: VLMEvalKit 模型注册 (MVBench / LongVideoBench)

VLMEvalKit 默认用 HuggingFace Hub ID 下载模型。服务器离线环境下需要用 JSON config
指定本地路径，**无需修改 VLMEvalKit 源码**。

在 `mvbench/` 和 `longvideobench/` 下各创建一个 `eval_config.json`：

**mvbench/eval_config.json**
```json
{
    "model": {
        "Qwen3-VL-4B-Instruct": {
            "class": "Qwen3VLChat",
            "model_path": "/path/to/models/Qwen3-VL-4B-Instruct",
            "use_custom_prompt": false,
            "use_vllm": true,
            "temperature": 0.7,
            "max_new_tokens": 16384,
            "top_p": 0.8,
            "top_k": 20
        }
    },
    "data": {
        "MVBench_MP4_1fps": {
            "class": "MVBench_MP4",
            "dataset": "MVBench_MP4",
            "fps": 1.0
        }
    }
}
```

**longvideobench/eval_config.json**
```json
{
    "model": {
        "Qwen3-VL-4B-Instruct": {
            "class": "Qwen3VLChat",
            "model_path": "/path/to/models/Qwen3-VL-4B-Instruct",
            "use_custom_prompt": false,
            "use_vllm": true,
            "temperature": 0.7,
            "max_new_tokens": 16384,
            "top_p": 0.8,
            "top_k": 20
        }
    },
    "data": {
        "LongVideoBench_8frame_subs": {
            "class": "LongVideoBench",
            "dataset": "LongVideoBench",
            "nframe": 8,
            "use_subtitle": true
        }
    }
}
```

### VLMEvalKit 可选数据集配置

**MVBench 可选项：**

| 配置名 | 说明 |
|---|---|
| `MVBench_8frame` | GIF/帧模式，抽8帧 |
| `MVBench_64frame` | GIF/帧模式，抽64帧 |
| `MVBench_MP4_8frame` | MP4模式，抽8帧 **（推荐）** |
| `MVBench_MP4_1fps` | MP4模式，1fps采样 **（推荐）** |

**LongVideoBench 可选项：**

| 配置名 | 说明 |
|---|---|
| `LongVideoBench_8frame` | 8帧，无字幕 |
| `LongVideoBench_8frame_subs` | 8帧 + 字幕 **（推荐）** |
| `LongVideoBench_64frame` | 64帧，无字幕（更准但更慢） |
| `LongVideoBench_1fps` | 1fps采样 |
| `LongVideoBench_0.5fps` | 0.5fps采样 |
| `LongVideoBench_0.5fps_subs` | 0.5fps + 字幕 |

修改 `eval_config.json` 中 `"data"` 字段下的 `class` / `dataset` / `nframe` / `fps` / `use_subtitle` 即可切换。

### VLMEvalKit 可选模型参数

```json
{
    "class": "Qwen3VLChat",      // 固定，不用改
    "model_path": "/path/to/model",
    "use_custom_prompt": false,   // 用模型自带prompt格式
    "use_vllm": true,             // true=vLLM加速, false=transformers
    "temperature": 0.7,           // 生成温度
    "max_new_tokens": 16384,      // 最大输出tokens
    "top_p": 0.8,
    "top_k": 20,
    "repetition_penalty": 1.0
}
```

然后在 `mvbench/config.sh` 和 `longvideobench/config.sh` 里把 `eval_config.json` 路径告诉 run.sh（见第三步）。

---

## 第三步: 运行评测

### 单独跑某个评测

```bash
# FutureOmni (torchrun DDP)
bash futureomni/run.sh

# seeAoT (单卡 transformers)
bash seeaot/run.sh

# MVBench (VLMEvalKit torchrun)
bash mvbench/run.sh

# LongVideoBench (VLMEvalKit torchrun)
bash longvideobench/run.sh
```

覆盖 GPU：
```bash
CUDA_VISIBLE_DEVICES=2,3 bash mvbench/run.sh
```

### 8卡一键并行（四个评测同时跑）

```bash
bash run_all_8gpu_parallel.sh
```

GPU 分配：
```
GPU 0     → FutureOmni   (torchrun, port 29500)
GPU 1     → seeAoT       (单卡 python)
GPU 2,3   → MVBench      (torchrun, port 29501)
GPU 4,5   → LongVideoBench (torchrun, port 29502)
GPU 6,7   → 空闲
```

日志实时写入 `logs/<benchmark>_<timestamp>.log`，可 `tail -f` 查看。

---

## 第四步: 查看结果

所有结果统一在 `results/` 下：

```
results/
├── futureomni/         ← infer_ddp.py 每个样本一个 JSON
├── seeaot/             ← 每个 subset 一个 .jsonl (从 AoTBench/data_files/output/ 拷贝)
├── mvbench/            ← VLMEvalKit 输出 (含 CSV 精度报告)
└── longvideobench/     ← VLMEvalKit 输出 (含 CSV 精度报告)
```

seeAoT accuracy 在 run.sh 最后会自动打印；
MVBench / LongVideoBench accuracy 由 VLMEvalKit 自动计算并写入 CSV。

---

## 常见问题

**Q: 换一个模型怎么办？**
只改 `config.sh` 里的 `MODEL_PATH`，同时改 `mvbench/eval_config.json` 和 `longvideobench/eval_config.json` 里的 `model_path` 和外层 key 名。

**Q: FutureOmni 支持哪些模型类型？**
在 `futureomni/config.sh` 里设置 `MODEL_TYPE`：
- `qwen3_vl` — Qwen3-VL 系列
- `qwen2_5_vl` — Qwen2.5-VL 系列
- `qwen2_5omni` — Qwen2.5-Omni（需要音频）
- `qwen3omni` — Qwen3-Omni（需要音频）

**Q: 端口冲突怎么办？**
各评测 torchrun 端口已错开（29500/29501/29502）。单独跑时也可在子文件夹 config.sh 里修改 `MASTER_PORT`。
