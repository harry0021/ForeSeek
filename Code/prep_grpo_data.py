import argparse
import json
import os
import sys

import pandas as pd

# ── system prompt (must match prompts_v9k2.UNIFIED_SYSTEM) ──────────────────
UNIFIED_SYSTEM = (
    "You are a video analysis expert who uses a principled Predict→Verify→Act "
    "loop to answer multiple-choice questions about long videos.\n"
    "Each round you receive sampled frames from a specific time interval.\n"
    "Your goal: form testable hypotheses, verify them with the frames you see, "
    "and progressively narrow down to the correct answer.\n"
    "Always output well-formed XML tags as instructed."
)

# ── phase-1 user template (must match prompts_v9k2.PHASE1_USER_TEMPLATE) ────
PHASE1_USER_TEMPLATE = (
    "These are {n_frames} uniformly sampled frames from the full video "
    "(timestamps shown on each frame).\n\n"
    "Question: {question}\n\n"
    "Based on these sparse frames, output:\n"
    "1. <intervals> — label every segment HIGH/MEDIUM/LOW with a 3-5 word "
    "scene_hint and 1-3 reason_token words from the question.\n"
    "2. <prediction> — one testable hypothesis: WHERE to look next, WHAT you "
    "expect to see, and WHY (must include a time range).\n"
    "3. <query_type> — one of: temporal_localization | object_state | "
    "action_recognition | causal_reasoning | counting\n\n"
    "Constraints:\n"
    "- At least one HIGH interval; no all-MEDIUM output.\n"
    "- <prediction> must reference a concrete time range."
)


def build_row(item: dict, idx: int, video_root: str) -> dict:
    """Convert one cgbench item to a verl-compatible parquet row."""
    video_uid = item.get("video_uid", "")
    video_file = video_uid if video_uid.endswith(".mp4") else video_uid + ".mp4"
    video_path = os.path.join(video_root, video_file)

    question = item.get("question_with_choices") or item.get("question", "")

    # Phase1 user message — no images yet (AgentLoop fills them in at rollout)
    phase1_user_text = PHASE1_USER_TEMPLATE.format(
        n_frames="{n_frames}",   # placeholder, filled by AgentLoop
        question=question,
    )

    prompt = [
        {"role": "system", "content": UNIFIED_SYSTEM},
        {"role": "user",   "content": phase1_user_text},
    ]

    return {
        "data_source": "cgbench_v9k2",
        "prompt": prompt,
        "reward_model": {
            "style":         "rule",
            "ground_truth":  item.get("right_answer", ""),
        },
        "extra_info": {
            "index":          idx,
            "qid":            item.get("qid", idx),
            "video_path":     video_path,
            "video_uid":      video_uid,
            "question":       question,
            "ground_truth":   item.get("right_answer", ""),
            "clue_intervals": item.get("clue_intervals", []),
            "sub_category":   item.get("sub_category", ""),
            "domain":         item.get("domain", ""),
            "duration":       item.get("duration", 0),
        },
    }


def convert(src_path: str, video_root: str) -> pd.DataFrame:
    with open(src_path, encoding="utf-8") as f:
        # Support both JSON array and JSONL
        text = f.read().strip()
        if text.startswith("["):
            items = json.loads(text)
        else:
            items = [json.loads(line) for line in text.splitlines() if line.strip()]

    rows = [build_row(item, idx, video_root) for idx, item in enumerate(items)]
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train",      required=True,  help="Train JSON/JSONL (cgbench_train_2k)")
    ap.add_argument("--test",       required=True,  help="Test JSON/JSONL  (cgbench_test_1k)")
    ap.add_argument("--out",        required=True,  help="Output directory for parquet files")
    ap.add_argument("--video-root", default="",     help="Root dir for video files")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Processing train: {args.train}")
    train_df = convert(args.train, args.video_root)
    train_path = os.path.join(args.out, "train.parquet")
    train_df.to_parquet(train_path, index=False)
    print(f"  → {len(train_df)} rows saved to {train_path}")

    print(f"Processing test: {args.test}")
    test_df = convert(args.test, args.video_root)
    test_path = os.path.join(args.out, "test.parquet")
    test_df.to_parquet(test_path, index=False)
    print(f"  → {len(test_df)} rows saved to {test_path}")

    print("\nSample row keys:", list(train_df.columns))
    sample_extra = json.loads(train_df.iloc[0]["extra_info"]) if isinstance(
        train_df.iloc[0]["extra_info"], str
    ) else train_df.iloc[0]["extra_info"]
    print("Sample extra_info:", {k: v for k, v in sample_extra.items() if k != "question"})


if __name__ == "__main__":
    main()
