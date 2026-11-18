# ForeSeek: Foresight-Driven Evidence Seeking for Long-Video Reasoning

ForeSeek is a video agent that searches a long video by predicting what unseen intervals should contain, then checking those predictions. A failed expectation is kept as evidence: the implausible interval is set aside and the search moves to another region of the timeline, instead of continuing to sample wherever the last frame looked salient.

<p align="center">
  <img src="ForeSeek_main.png" alt="ForeSeek overview. Phase 1 builds a belief map from a 10–20 frame sparse scan. Phase 2 checks the prediction on the chosen interval and narrows or redirects the search." width="100%">
</p>

## Method

Long-video agents often retrieve the next clip from local similarity. ForeSeek keeps an explicit predictive state and uses each new observation to test it.

**Predictive state.** At step \(r\) the state is a global interval belief, a visually grounded expectation for the interval under inspection, and the reasoning trace so far. Each belief interval stores a time span, a short hint, and a relevance label in {High, Medium, Low, Elim}. Medium is the default. High marks positive evidence. Low marks a weak prior or a search that did not find the expected content. Elim is a hard exclusion declared after verification. The scheduler tries High before Medium, and uses Low only when no higher interval remains.

**Phase 1.** A duration-adaptive sparse scan, between 10 and 20 frames, covers the whole video. From those frames and the question, the model writes the initial belief and the first expectation.

**Phase 2.** Each later step reads a small set of frames inside the current interval and returns one of three verdicts:

| Verdict | What happens next |
|---|---|
| Confirmed | The expectation is supported, and the agent may answer. |
| Partially confirmed | The expectation is refined and the next step inspects a narrower sub-interval. |
| Not found | The current interval is deprioritized and the search moves to the highest remaining region. |

The loop stops when an answer is produced, the round limit is reached (5), or frames cannot be read.

**Training.** Supervision covers the expectation, the verdict, and the belief update, not only the final letter. Questions are drawn from CG-Bench, LongVideoBench, VideoHolmes, and the academic and YouTube subsets of LLaVA-Video, with evaluation overlap removed by question id, normalized question text, and source video. LongVideoBench answers are filled by a dual-answerer-plus-judge protocol; the other sources keep their original labels. The pool is split evenly into 10,021 pre-SFT and 10,021 pre-RL questions. A teacher writes each SFT trajectory in two calls: a Phase-1 call that sees only the sparse frames, and an oracle call that also sees the gold answer and must recover from a wrong region through `not_found`, a redirect, and a confirmation. Failed syntheses are dropped, leaving 8,165 SFT pairs. Reinforcement learning then keeps the 4,663 pre-RL questions (46.5%) that the SFT policy answers correctly on 1 to 6 of 8 rollouts at temperature 0.7.

Stage 1 is token-level supervised fine-tuning on those trajectories. Stage 2 is Group Sequence Policy Optimization. The trajectory reward is the product of answer correctness and a round penalty: full credit for a correct answer within 3 rounds, then 0.9 at 4 rounds and 0.8 at 5. An incorrect answer scores 0 at any length. The sequence-level clip is \([1-10^{-3},\, 1+1.5\times10^{-3}]\), with a loss-level KL penalty of 0.02 toward the SFT policy.

On five long-video benchmarks, ForeSeek matches or exceeds the accuracy of the strongest search-based agent while reading \(2.5\)–\(4.4\times\) fewer frames.

## Released checkpoints

Weights are hosted at [huggingface.co/Harry0021/ForeSeek](https://huggingface.co/Harry0021/ForeSeek).

Download a checkpoint and point evaluation or the training launcher at that directory. This repository is the agent, the trajectory builder, and the GSPO training launchers. `README.md` is the file-by-file index of the same code.

## Evaluation

`run_gpu.sh` calls `eval_foresight.py`. Questions are read from JSONL files under `DATA_DIR` (default `./data`). One worker is one GPU process.

```bash
# Local vLLM. The ForeSeek agent and prompts are the v9k2 modules.
PROMPTS_MODULE=prompts_v9k2 AGENT_MODULE=foresight_agent_v9k2 \
LOCAL=1 DATASET=videomme VIDEO_ROOT=/path/to/videos \
bash run_gpu.sh

# OpenAI-compatible API.
PROMPTS_MODULE=prompts_v9k2 AGENT_MODULE=foresight_agent_v9k2 \
OPENAI_API_BASE=http://127.0.0.1:8000/v1 OPENAI_API_KEY=EMPTY \
DATASET=longvideobench VIDEO_ROOT_LVB=/path/to/videos \
bash run_gpu.sh
```

| `DATASET` | Question file under `DATA_DIR` | Video root |
|---|---|---|
| `videomme` | `videomme_test_wosubtitles_raw_list_full.jsonl` | `VIDEO_ROOT` |
| `longvideobench` | `LongVideoBench_nosub_raw_full.jsonl` | `VIDEO_ROOT_LVB` |
| `lvbench` | `LVBench_raw_full_root.jsonl` | `VIDEO_ROOT_LVBENCH` |
| `videoholmes` | `videoholmes_raw_full.jsonl` | `VIDEO_ROOT_HOLMES` |
| `mlvu` | `MLVU_MCQ_raw_full.jsonl` | `VIDEO_ROOT_MLVU` |
| `videommu` | `VideoMMMU_raw_full.jsonl` | `VIDEO_ROOT_MMU` |

`NUM_GPUS` sets the worker count. `SAMPLE` evaluates the first N items. `SHARD_ID` and `NUM_SHARDS` split a run across machines. Results go to `OUTPUT_DIR` (default `./output_gpu`). `foresight_agent_v9k2_no_pv.py` with `prompts_v9k2_no_pv.py` is the variant without the verification phase.

Frame extraction needs `ffmpeg` and `ffprobe`.

## Trajectory construction

`cgbench_v9k2_sft_builder-Final.py` turns a QA json into multi-round SFT conversations. The paper's teacher is Gemini-3-Flash; `--model` selects the endpoint.

```bash
python3 cgbench_v9k2_sft_builder-Final.py \
  --input-json /path/to/samples.json \
  --video-root /path/to/videos \
  --output-dir /path/to/output \
  --model gemini-3-flash \
  --api-keys "KEY1,KEY2" \
  --sample 0 --workers 50 --qpm 50 --resume
```

`--sample 0` runs every item. The builder writes `v9k2_sft.jsonl`, `v9k2_result.jsonl`, and `build_trace.jsonl`. Phase 1 uses the same 10–20 frame sparse scan as inference. The oracle call may read a denser strip, at 0.5 fps and at most 96 frames.

## GSPO

Training uses [verl](https://github.com/volcengine/verl) and a Ray cluster. The launchers are one node and 8 GPUs. The rollout group size is 8, the round limit is 5, the learning rate is \(1\times10^{-6}\), and the KL coefficient is 0.02. `v9k2_agent_loop.py` rolls out the agent. `v9k2_reward.py` applies the correctness-times-length reward. `grpo_val_reward.py` scores validation by the option letter.

```bash
ray start --head --num-gpus=8 --num-cpus=64 --include-dashboard=false

GRPO_DIR=/path/to/this/directory \
VERL_DIR=/path/to/verl \
MODEL_PATH=/path/to/sft/checkpoint \
GRPO_DATA=/path/to/parquet \
VIDEO_ROOT=/path/to/videos \
OUTPUT_DIR=/path/to/output \
bash run_v9k2_gspo_8gpu_1node.sh
```

`prep_grpo_data_multi.py` builds the parquet directory. It must contain `train_select.parquet` and `test.parquet`. `run_passk_eval.sh` repeats the SFT policy 8 times at temperature 0.7 so the 1–6 correct filter can be applied; edit the paths at the top of that script before running it.

The default launcher looks for `v9k2_agent_config_unlock64f.yaml` inside `GRPO_DIR`. This directory ships `v9k2_agent_config.yaml`. Point `AGENT_LOOP_CONFIG` at the yaml you intend to use, or place the expected file next to the launcher. The other launchers are the 24-frame and 64-frame settings (`run_v9k2_gspo_8gpu_1node-24f.sh`, `run_v9k2_gspo_8gpu_1node-64f.sh`) and the two runs without the verification phase (`run_v9k2_gspo_8gpu_1node-removePV.sh`, `run_v9k2_gspo_8gpu_1node-originWithoutPV.sh`).

## Setup

```bash
pip install torch openai transformers vllm opencv-python pillow numpy
```

GSPO additionally needs `verl`, `ray`, and `hydra-core`. Keep a CUDA driver that matches the installed `torch` and `vllm`.
