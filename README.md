# ForeSeek: Foresight-Driven Evidence Seeking for Long-Video Reasoning

ForeSeek is a video agent that searches a long video by predicting what unseen intervals should contain, then checking those predictions. A failed expectation is kept as evidence: the implausible interval is set aside and the search moves to another region of the timeline.

<p align="center">
  <img src="ForeSeek_main.png" alt="ForeSeek overview. Phase 1 builds a belief map from a sparse scan. Phase 2 checks the prediction and narrows or redirects the search." width="100%">
</p>

## Method

ForeSeek keeps a predictive state: a global interval belief, an expectation for the interval under inspection, and the reasoning trace so far. Each interval carries a relevance label in {High, Medium, Low, Elim}.

**Phase 1.** A short sparse scan covers the whole video. From those frames and the question, the model writes the initial belief and the first expectation.

**Phase 2.** Each later step reads frames inside the current interval and returns one verdict:

| Verdict | What happens next |
|---|---|
| Confirmed | The expectation is supported, and the agent may answer. |
| Partially confirmed | The expectation is refined on a narrower sub-interval. |
| Not found | The current interval is deprioritized and the search moves elsewhere. |

Training has two stages. Supervised fine-tuning learns full foresight trajectories, including the expectation, the verdict, and the belief update. Group Sequence Policy Optimization then rewards a correct answer and favors trajectories that stop once the evidence is sufficient. Questions come from CG-Bench, LongVideoBench, VideoHolmes, and LLaVA-Video, held out from the evaluation sets.

On five long-video benchmarks, ForeSeek matches or exceeds the accuracy of the strongest search-based agent while reading 2.5–4.4× fewer frames.

## Checkpoints

Weights: [huggingface.co/Harry0021/ForeSeek](https://huggingface.co/Harry0021/ForeSeek).

This repository contains the agent, the trajectory builder, and the training launchers. Script-level options are in `README.md`.

## Run

Evaluation uses a local vLLM server or any OpenAI-compatible endpoint. Set `DATA_DIR` to the question files and the matching `VIDEO_ROOT*` to the videos.

```bash
PROMPTS_MODULE=prompts_v9k2 AGENT_MODULE=foresight_agent_v9k2 \
LOCAL=1 DATASET=videomme VIDEO_ROOT=/path/to/videos \
bash run_gpu.sh
```

`DATASET` can be `videomme`, `longvideobench`, `lvbench`, `videoholmes`, or `mlvu`.

Trajectories for supervised fine-tuning are built with `cgbench_v9k2_sft_builder-Final.py`. Training is launched with `run_v9k2_gspo_8gpu_1node.sh` on a Ray cluster; pass the checkpoint, data, and video paths through the environment variables at the top of that script. Frame extraction needs `ffmpeg`.
