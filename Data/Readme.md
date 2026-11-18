## Data File Provenance

### SFT samples

The three SFT sample files share the **same 24 unique (qid, video_uid) pairs** but
differ in their data-generation pipelines:

| File | Relation |
| ---- | -------- |
| `sft_origin_sample.jsonl` | Anchor — randomly sampled |
| `sft_removePV_sample.jsonl` | Row-by-row filtered version of origin; matched by `(qid, video_uid, video, stage)` |
| `sft_withoutPV_sample.jsonl` | **Independently regenerated** — matched by `(qid, video_uid)` taking **all available stages** for each pair |

**Key difference**: `removePV` is a deterministic filtering pass over the original
SFT data, so it has the exact same stage distribution. `withoutPV` was generated from
scratch and therefore the number of rollout rounds (`unified_round_1`–`unified_round_5`)
may differ per sample.

**Stage distribution**:

| Stage               | Origin | removePV | withoutPV |
| ------------------- | ------ | -------- | --------- |
| `phase1`          | 24     | 24       | 24        |
| `unified_round_1` | 24     | 24       | 24        |
| `unified_round_2` | 24     | 24       | 24        |
| `unified_round_3` | 16     | 16       | 17        |
| `unified_round_4` | 13     | 13       | 14        |
| `unified_round_5` | —     | —       | 1         |

**JSONL record schema** (all SFT files):

```json
{
  "qid": 96614,
  "video_uid": "6ibZezGV0Kc",
  "video": "6ibZezGV0Kc.mp4",
  "stage": "phase1",
  "gt": "C",
  "conversations": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
  "images": ["path/to/frame1.jpg", ...],
  "_source": "...",
  "_ablation": "no_pv"   // only in removePV
}
```

### RL sample

| File | Contents |
| ---- | -------- |
| `rl_sample.jsonl` | First 100 rows from GRPO training set |

**Record schema**:

```json
{
  "data_source": "grpo_multi_cgbench_grpo",
  "prompt": [{"role": "system", "content": "..."}, {"role": "user", "content": [...]}],
  "reward_model": {"style": "rule", "ground_truth": "D"},
  "extra_info": {"index": 2, "qid": 10076, "video_path": "...", "video_uid": "...", "source": "...", "question": "...", "ground_truth": "...", "clue_intervals": [...], "sub_category": "...", "domain": "...", "duration": 123.45}
}
```
