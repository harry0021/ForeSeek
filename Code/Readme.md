This directory contains the core code for the methodology, organized into the following pipelines:

1. **SFT trajectory construction** — Teacher-driven (Gemini/GPT) trajectory generation
2. **GRPO RL training** — Group Relative Policy Optimization fine-tuning
3. **Evaluation** — Multi-dataset VLM evaluation harness
4. **Pass@K filtering** — Top-K trajectory selection for iterative RL

---

## File Index

```
Code/
├── SFT trajectory construction
│   └── cgbench_v9k2_sft_builder-Final.py   # SFT trajectory builder (teacher: Gemini/GPT)
├── Prompts
│   ├── prompts.py                          # Base prompt templates
│   ├── prompts_v9k2.py                     # Prompt templates for v9k2 agent
│   └── prompts_v9k2_no_pv.py               # Prompt templates without Phase Verify
├── Agents
│   ├── foresight_agent.py                  # Base agent logic
│   ├── foresight_agent_api.py              # API-only agent mode
│   ├── foresight_agent_v9k2.py             # v9k2 agent (evaluation inference)
│   └── foresight_agent_v9k2_no_pv.py       # v9k2 agent without Phase Verify
├── Evaluation
│   ├── eval_foresight.py                   # Evaluation entry point
│   ├── frame_utils.py                      # Video frame extraction utilities
│   ├── config.py                           # Shared configuration
│   ├── run_gpu.sh                          # Evaluation launch script
│   └── run_passk_eval.sh                   # Pass@K evaluation script
└── GRPO RL training
    ├── run_v9k2_dapo_8gpu_1node.sh         # DAPO training launcher (8 GPU)
    ├── run_v9k2_gspo_8gpu_1node.sh         # GSPO training launcher (8 GPU)
    ├── run_v9k2_gspo_8gpu_1node-24f.sh     # GSPO 24-frame ablation
    ├── run_v9k2_gspo_8gpu_1node-64f.sh     # GSPO 64-frame ablation
    ├── run_v9k2_gspo_8gpu_1node-originWithoutPV.sh
    │                                          # GSPO origin-without-PV ablation
    ├── run_v9k2_gspo_8gpu_1node-removePV.sh # GSPO remove-PV ablation
    ├── v9k2_agent_loop.py                  # GRPO agent loop (training rollout)
    ├── v9k2_agent_config.yaml              # Agent loop Hydra config
    ├── v9k2_reward.py                      # GRPO reward function
    ├── grpo_val_reward.py                  # GRPO validation reward
    ├── prep_grpo_data.py                   # Single-dataset GRPO data preparation
    └── prep_grpo_data_multi.py             # Multi-dataset GRPO data preparation
```

---

## 1. SFT Trajectory Construction

Builds teacher-guided trajectories for supervised fine-tuning. Uses Gemini/GPT models as the
teacher to generate multi-round reasoning traces.

### Running

```bash
MODEL=gemini-3-flash python3 cgbench_v9k2_sft_builder-Final.py \
  --input-json /path/to/samples.json \
  --video-root /path/to/videos \
  --output-dir /path/to/output \
  --api-keys "KEY1,KEY2" \
  --api-key-strategy random \
  --sample 0 \
  --workers 50 \
  --qpm 50 \
  --resume
```

### Key Arguments

| Argument               | Default    | Description                                 |
| ---------------------- | ---------- | ------------------------------------------- |
| `--input-json`       | (required) | Input JSON/JSONL of CG samples              |
| `--video-root`       | (required) | Root directory for video files              |
| `--output-dir`       | (required) | Output directory for results                |
| `--model`            | `gpt-5`  | Teacher model (gemini-3-flash, gpt-5, etc.) |
| `--api-keys`         | (env)      | Comma/semicolon-separated API keys          |
| `--api-key-strategy` | `random` | Key selection:`random` or `round_robin` |
| `--sample`           | `5`      | Number of samples (`0` = all)             |
| `--workers`          | `5`      | Parallel worker threads                     |
| `--qpm`              | `50`     | API queries-per-minute cap                  |
| `--max-rounds`       | `5`      | Max trajectory rounds per sample            |
| `--dense-fps`        | `0.5`    | Dense frame sampling rate (fps)             |
| `--max-dense-frames` | `96`     | Max dense frames                            |
| `--resume`           | off        | Skip already-completed qids                 |

### Dependencies

- Python 3.10+
- `openai` (pip install openai)
- `ffmpeg` / `ffprobe` (system install)
- `prompts_v9k2.py` (included)

### Pipeline

1. **Phase 1**: Sparse frame sampling (deployment-faithful, no clue/GT)
2. **Oracle**: Dense frames with clue + ground truth for teacher guidance
3. Output: `v9k2_result.jsonl`, `v9k2_sft.jsonl`, frames/, `build_trace.jsonl`

---

## 2. GRPO RL Training

Group Relative Policy Optimization fine-tuning of Qwen3-VL-8B using the verl framework.
The released launchers cover both DAPO (token-level PPO with Clip-Higher and Dual-Clip)
and GSPO (sequence-level importance-ratio clipping), together with frame-budget and
Phase-Verify ablations.

### Running

```bash
# DAPO
bash run_v9k2_dapo_8gpu_1node.sh

# GSPO
bash run_v9k2_gspo_8gpu_1node.sh

# GSPO ablations
bash run_v9k2_gspo_8gpu_1node-24f.sh
bash run_v9k2_gspo_8gpu_1node-64f.sh
bash run_v9k2_gspo_8gpu_1node-originWithoutPV.sh
bash run_v9k2_gspo_8gpu_1node-removePV.sh
```

All launchers use anonymous placeholder defaults. Override them through environment variables:

```bash
GRPO_DIR=/path/to/foresight/grpo \
VERL_DIR=/path/to/verl \
MODEL_PATH=/path/to/sft/checkpoint \
GRPO_DATA=/path/to/datasets/CG-Bench/grpo_parquet_multi \
VIDEO_ROOT=/path/to/datasets/CG-Bench/videos \
OUTPUT_DIR=/path/to/output \
bash run_v9k2_gspo_8gpu_1node.sh
```

### Prerequisites

1. **Ray cluster** must be started before training (see comments in script)
2. **Data**: Train/test parquet files prepared via `prep_grpo_data_multi.py`
3. **Model**: SFT checkpoint from the trajectory construction pipeline
4. **Packages**: `verl`, `torch`, `vllm`, `transformers`, `ray`, `hydra-core`, `cv2`, `PIL`

### Key Configuration (set inside script)

| Variable            | Description                    |
| ------------------- | ------------------------------ |
| `GRPO_DIR`        | Directory containing the agent loop and reward code |
| `VERL_DIR`        | Local verl repository path      |
| `MODEL_PATH`      | Path to the SFT model checkpoint |
| `GRPO_DATA`       | Directory containing train/test parquet files |
| `VIDEO_ROOT`      | Root directory for video files |
| `OUTPUT_DIR`      | Training output directory      |
| `TRAIN_BATCH_SIZE`| Global training batch size      |
| `ROLLOUT_N`       | GRPO rollout group size         |
| `MAX_ROUNDS`      | Maximum multi-turn rollout rounds |

### Core Files

| File                       | Role                                                       |
| -------------------------- | ---------------------------------------------------------- |
| `v9k2_agent_loop.py`     | Agent rollout loop (imports prompts, reward)               |
| `v9k2_reward.py`         | Multi-component reward (IoU, consistency, narrowing bonus) |
| `grpo_val_reward.py`     | Letter-match accuracy for validation                       |
| `v9k2_agent_config.yaml` | Hydra config registering the agent loop                    |

The ablation launchers reference their corresponding agent-loop configurations
(`v9k2_agent_config_unlock24f.yaml`, `v9k2_agent_config_unlock64f.yaml`, or
`v9k2_agent_config_no_pv.yaml`). Place those files under `GRPO_DIR`, or override
the `AGENT_LOOP_CONFIG` assignment in the launcher for a custom layout.

---

## 3. Evaluation

Multi-dataset VLM evaluation supporting both cloud API (OpenAI-compatible) and local vLLM modes.

### Running

```bash
# Cloud mode (default)
bash run_gpu.sh

# Specific dataset with multiple GPUs
DATASET=mlvu NUM_GPUS=80 bash run_gpu.sh

# Local vLLM mode
LOCAL=1 DATASET=mlvu bash run_gpu.sh

# Custom agent and prompt modules
PROMPTS_MODULE=prompts_v9k2 AGENT_MODULE=foresight_agent_v9k2 bash run_gpu.sh
```

### Key Environment Variables

| Variable                      | Default                      | Description                                 |
| ----------------------------- | ---------------------------- | ------------------------------------------- |
| `DATASET`                   | `videomme_test`            | Dataset selection                           |
| `NUM_GPUS`                  | `1`                        | Number of parallel GPU workers              |
| `LOCAL`                     | `0`                        | Use local vLLM (`1`) or cloud API (`0`) |
| `LOCAL_URL`                 | `http://localhost:8000/v1` | Local vLLM endpoint                         |
| `OPENAI_API_BASE`           | (env)                        | API base URL                                |
| `OPENAI_API_KEY`            | (env)                        | API key                                     |
| `PROMPTS_MODULE`            | `""`                       | Custom prompts module name                  |
| `AGENT_MODULE`              | `foresight_agent`          | Agent module name                           |
| `OUTPUT_DIR`                | `$(pwd)/output_gpu`        | Results output directory                    |
| `SHARD_ID` / `NUM_SHARDS` | `0` / `1`                | Multi-machine sharding                      |
| `SAMPLE`                    | `""`                       | Sample subset size                          |

### Agent Modes

- **`foresight_agent.py`**: Unified VLM mode (all visual processing via API)
- **`foresight_agent_v9k2.py`**: v9k2 agent with multi-round Phase Verify
- **`foresight_agent_v9k2_no_pv.py`**: v9k2 agent without Phase Verify

---

## 4. Pass@K Filtering

Evaluates the model K times per sample and filters for best trajectories for iterative RL.

### Running

```bash
cd /path/to/Foresight-Uni-0601
SHARD_ID=0 NUM_SHARDS=3 K=8 NUM_GPUS=40 TEMPERATURE=0.7 \
  OUTPUT_BASE=./output_passk \
  bash run_passk_eval.sh
```

### Key Environment Variables

| Variable                      | Default       | Description                            |
| ----------------------------- | ------------- | -------------------------------------- |
| `K`                         | (required)    | Number of evaluation passes per sample |
| `TEMPERATURE`               | `0`         | Sampling temperature                   |
| `NUM_GPUS`                  | `40`        | GPUs per shard                         |
| `SHARD_ID` / `NUM_SHARDS` | `0` / `1` | Multi-machine sharding                 |
| `OUTPUT_BASE`               | (required)    | Base output directory                  |

### Post-Processing

After running K evaluations, use `filter_by_passk.py` to select the best trajectory per sample.
The agent internally handles temperature settings — when `TEMPERATURE=0.7`, sampling is enabled
for diverse K rollouts.

---

## Configuration (config.py)

Shared configuration for both evaluation and training:

- **`BASE_URL`**: Default API base URL — set via `OPENAI_API_BASE` env var
- **`API_KEY`**: Default API key — set via `OPENAI_API_KEY` env var
- **`TEMPERATURE`**: Sampling temperature (default `0.0`, only sent to API when > 0)
- **Model paths**: Auto-detected via `_find_model()` or set via model-specific env vars

---

## Environment Setup

### Python Dependencies

```
torch >= 2.5.0
vllm >= 0.7.0
openai
transformers
opencv-python (cv2)
Pillow (PIL)
numpy
```

### System Dependencies

- `ffmpeg` / `ffprobe` (for video frame extraction)
- CUDA toolkit (for GPU acceleration)
- NVIDIA driver >= 535 (for vLLM inference)

### For GRPO Training additionally

```
verl (custom RL training framework)
ray (distributed computing)
hydra-core (configuration management)
```

---

## Data Format

### Input data

JSON or JSONL with fields:

- `qid`: question ID
- `question`: question text / QA pair
- `video_path`: relative video path
- `answer`: ground truth answer
- `clue`: hint/region for teacher (optional)

### Output data

- `v9k2_result.jsonl`: per-sample trajectory results
- `v9k2_sft.jsonl`: training-ready SFT data
- `build_trace.jsonl`: detailed build trace for debugging
- `frames/`: extracted video frames

---

## Citation

If you use this code, please cite the corresponding paper.
