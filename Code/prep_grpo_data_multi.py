import argparse
import json
import os
import random

import pandas as pd

# ── Canonical video root paths (override via CLI args if needed) ─────────────
DEFAULT_LLAVA_VIDEO_ROOT = (
    "/path/to/datasets/llava-video-178k/videos"
)
DEFAULT_CGBENCH_VIDEO_ROOT = (
    "/path/to/datasets/CG-Bench/videos"
)
DEFAULT_VH_VIDEO_ROOT = (
    "/path/to/datasets/Video-Holmes/videos"
)

# ── Default JSON source paths ────────────────────────────────────────────────
_GRPO_SRC = (
    "/path/to/datasets/CG-Bench/makeNewData/outputNewAdded/GRPO"
)
DEFAULT_CGBENCH_JSON  = f"{_GRPO_SRC}/cgbench_grpo_annotated.json"
DEFAULT_ACADEMIC_JSON = f"{_GRPO_SRC}/academic_grpo_annotated.json"
DEFAULT_YOUTUBE_JSON  = f"{_GRPO_SRC}/youtube_grpo_annotated.json"
DEFAULT_VH_JSON       = f"{_GRPO_SRC}/vh_grpo_annotated.json"

# ── System / Phase-1 prompt (must match v9k2_agent_loop.py prompts) ──────────
UNIFIED_SYSTEM = (
    "You are a video analysis expert who uses a principled Predict→Verify→Act "
    "loop to answer multiple-choice questions about long videos.\n"
    "Each round you receive sampled frames from a specific time interval.\n"
    "Your goal: form testable hypotheses, verify them with the frames you see, "
    "and progressively narrow down to the correct answer.\n"
    "Always output well-formed XML tags as instructed."
)

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


# ── Source → video root resolver ─────────────────────────────────────────────

def make_source_roots(llava_root: str, cgbench_root: str, vh_root: str) -> dict:
    return {
        "academic_grpo": os.path.join(llava_root, "academic_source"),
        "youtube_grpo":  os.path.join(llava_root, "youtube"),
        "cgbench_grpo":  cgbench_root,
        "videoholmes":   vh_root,
    }


def resolve_video_path(video_uid: str, source: str, source_roots: dict) -> str:
    """Return absolute video path given uid and _source value."""
    root = source_roots.get(source, "")
    if not root:
        return video_uid  # fallback: return uid as-is (will fail at rollout)
    for ext in ("", ".mp4", ".mkv", ".webm"):
        p = os.path.join(root, video_uid + ext)
        if os.path.isfile(p):
            return p
    # Return best-guess path even if file not found (for dry-run / remote data)
    return os.path.join(root, video_uid + ".mp4")


# ── Row builder ──────────────────────────────────────────────────────────────

def build_row(item: dict, idx: int, source_roots: dict) -> dict:
    """Convert one annotated item to a verl-compatible parquet row."""
    video_uid = item.get("video_uid", "")
    source    = item.get("_source", "")
    video_path = resolve_video_path(video_uid, source, source_roots)

    question = item.get("question_with_choices") or item.get("question", "")

    phase1_user_text = PHASE1_USER_TEMPLATE.format(
        n_frames="{n_frames}",   # placeholder; AgentLoop fills at rollout time
        question=question,
    )

    prompt = [
        {"role": "system", "content": UNIFIED_SYSTEM},
        {"role": "user",   "content": phase1_user_text},
    ]

    return {
        "data_source": f"grpo_multi_{source}",
        "prompt": prompt,
        "reward_model": {
            "style":        "rule",
            "ground_truth": item.get("right_answer", ""),
        },
        "extra_info": {
            "index":          idx,
            "qid":            item.get("qid", idx),
            "video_path":     video_path,
            "video_uid":      video_uid,
            "source":         source,
            "question":       question,
            "ground_truth":   item.get("right_answer", ""),
            "clue_intervals": item.get("clue_intervals", []),
            "sub_category":   item.get("sub_category", ""),
            "domain":         item.get("domain", ""),
            "duration":       item.get("duration", 0),
        },
    }


# ── Loader ───────────────────────────────────────────────────────────────────

def load_json(path: str) -> list[dict]:
    if not path or not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        text = f.read().strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# ── Proportional sampling ─────────────────────────────────────────────────────

def proportional_sample(
    groups: dict[str, list],
    total: int,
    rng: random.Random,
) -> list:
    """
    Sample `total` items from multiple groups proportionally by group size.
    Each group contributes floor(group_size / all_size * total) items.
    Remainder items are filled greedily from the largest groups.
    """
    all_size = sum(len(v) for v in groups.values())
    if total >= all_size:
        merged = []
        for items in groups.values():
            merged.extend(items)
        rng.shuffle(merged)
        return merged

    counts: dict[str, int] = {}
    for src, items in groups.items():
        counts[src] = int(len(items) / all_size * total)

    # Distribute remainder to largest groups first
    remainder = total - sum(counts.values())
    for src in sorted(groups, key=lambda s: len(groups[s]), reverse=True):
        if remainder <= 0:
            break
        counts[src] += 1
        remainder -= 1

    sampled = []
    for src, items in groups.items():
        n = counts[src]
        chosen = rng.sample(items, min(n, len(items)))
        sampled.extend(chosen)
        print(f"  {src}: {len(items)} → sampled {len(chosen)}")

    rng.shuffle(sampled)
    return sampled


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Merge multi-source GRPO JSONs into verl parquet files."
    )

    # Source JSON paths
    ap.add_argument("--cgbench-json",  default=DEFAULT_CGBENCH_JSON)
    ap.add_argument("--academic-json", default=DEFAULT_ACADEMIC_JSON)
    ap.add_argument("--youtube-json",  default=DEFAULT_YOUTUBE_JSON)
    ap.add_argument("--vh-json",       default=DEFAULT_VH_JSON)

    # Video roots (rarely need to change)
    ap.add_argument("--llava-video-root", default=DEFAULT_LLAVA_VIDEO_ROOT,
                    help="Root containing academic_source/ and youtube/ subdirs")
    ap.add_argument("--cgbench-video-root", default=DEFAULT_CGBENCH_VIDEO_ROOT)
    ap.add_argument("--vh-video-root",      default=DEFAULT_VH_VIDEO_ROOT)

    # Output
    ap.add_argument("--out", required=True, help="Output directory for parquet files")

    # Sampling
    ap.add_argument("--sample", type=int, default=0,
                    help="Target total items (0 = use all). Proportional across sources.")
    ap.add_argument("--train-ratio", type=float, default=0.9,
                    help="Fraction of sampled items for train split (rest → test)")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)

    source_roots = make_source_roots(
        args.llava_video_root,
        args.cgbench_video_root,
        args.vh_video_root,
    )

    # Load all sources
    print("Loading source JSONs …")
    groups: dict[str, list] = {}
    source_files = {
        "cgbench_grpo":  args.cgbench_json,
        "academic_grpo": args.academic_json,
        "youtube_grpo":  args.youtube_json,
        "videoholmes":   args.vh_json,
    }
    for src_key, fpath in source_files.items():
        items = load_json(fpath)
        if not items:
            print(f"  SKIP (not found): {fpath}")
            continue
        groups[src_key] = items
        print(f"  {src_key}: {len(items)} items from {os.path.basename(fpath)}")

    total_available = sum(len(v) for v in groups.values())
    print(f"Total available: {total_available}")

    target = args.sample if args.sample > 0 else total_available
    print(f"\nSampling {target} items (proportional across sources):")
    sampled = proportional_sample(groups, target, rng)
    print(f"Total sampled: {len(sampled)}")

    # Video-level train/test split (prevents the same video appearing in both splits).
    # Questions about the same video are always assigned to the same split.
    video_to_items: dict[str, list] = {}
    for item in sampled:
        vid = item.get("video_uid", f"__unknown_{id(item)}")
        video_to_items.setdefault(vid, []).append(item)

    all_vids = list(video_to_items.keys())
    rng.shuffle(all_vids)
    n_test_vids = max(1, int(len(all_vids) * (1 - args.train_ratio)))
    test_vids  = set(all_vids[:n_test_vids])
    train_vids = set(all_vids[n_test_vids:])

    train_items = [item for vid in train_vids for item in video_to_items[vid]]
    test_items  = [item for vid in test_vids  for item in video_to_items[vid]]
    rng.shuffle(train_items)
    rng.shuffle(test_items)

    print(f"\nVideo-level split:")
    print(f"  Unique videos: {len(all_vids)}  →  train: {len(train_vids)}, test: {len(test_vids)}")
    print(f"  Items         →  train: {len(train_items)}, test: {len(test_items)}")
    assert not (train_vids & test_vids), "BUG: video overlap between splits"

    # Build parquet rows
    def to_df(items: list, offset: int = 0) -> pd.DataFrame:
        rows = [build_row(item, offset + i, source_roots) for i, item in enumerate(items)]
        return pd.DataFrame(rows)

    print("\nBuilding train parquet …")
    train_df = to_df(train_items, offset=0)
    train_path = os.path.join(args.out, "train.parquet")
    train_df.to_parquet(train_path, index=False)
    print(f"  → {len(train_df)} rows → {train_path}")

    print("Building test parquet …")
    test_df = to_df(test_items, offset=len(train_items))
    test_path = os.path.join(args.out, "test.parquet")
    test_df.to_parquet(test_path, index=False)
    print(f"  → {len(test_df)} rows → {test_path}")

    # Verify a sample path
    sample_extra = test_df.iloc[0]["extra_info"]
    if isinstance(sample_extra, str):
        sample_extra = json.loads(sample_extra)
    vp = sample_extra.get("video_path", "")
    exists = os.path.isfile(vp)
    print(f"\nSample video_path: {vp}")
    print(f"  File exists: {exists}")

    # Source distribution summary
    print("\nSource distribution in train:")
    src_counts: dict[str, int] = {}
    for row in train_df.itertuples():
        ei = row.extra_info
        if isinstance(ei, str):
            ei = json.loads(ei)
        s = ei.get("source", "?")
        src_counts[s] = src_counts.get(s, 0) + 1
    for s, c in sorted(src_counts.items()):
        print(f"  {s}: {c}")

    print(f"\nDone. Data saved to {args.out}/")
    print(
        f"To use: set GRPO_DATA={args.out} in run_v9k2_grpo_4gpu_2nodes.sh "
        f"(or export GRPO_DATA={args.out} before launching)"
    )


if __name__ == "__main__":
    main()
