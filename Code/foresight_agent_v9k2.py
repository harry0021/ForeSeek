import os
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

import config
import importlib as _imp
_prompts_mod = os.environ.get("PROMPTS_MODULE", "prompts_v9k2")
prompts = _imp.import_module(_prompts_mod)

from frame_utils import (
    uniform_timestamps,
    timestamps_in_range,
    format_timestamp,
    estimate_visual_tokens,
)
from foresight_agent import (
    _call_api,
    _build_image_content,
    _encode_image_base64,
    get_video_meta,
    extract_frames,
    _extract_answer,
    _resize_factor,
)


# ═══════════════════════════════════════════════════════════════
# Interval data structure (flat list, no tree)
# ═══════════════════════════════════════════════════════════════

def _parse_intervals_block(text: str, duration: float) -> List[Dict[str, Any]]:
    """Parse <intervals> block into a flat list of interval dicts.
    Supports both seconds [126-158s] and MM:SS [02:06-02:38] formats.
    """
    m = re.search(r"<intervals>(.*?)</intervals>", text, re.DOTALL)
    if not m:
        return []
    intervals = []
    for line in m.group(1).strip().splitlines():
        line = line.strip()
        if not line:
            continue

        # Try seconds format: [start-end] or [start_s-end_s]
        range_m = re.search(r"\[(\d+)\s*s?\s*[-–—>]+\s*(\d+)\s*s?\]", line)
        if range_m:
            start = float(range_m.group(1))
            end = float(range_m.group(2))
        else:
            # Try MM:SS format: [02:06-02:38]
            mmss_m = re.search(r"\[(\d+):(\d+)\s*[-–—>]+\s*(\d+):(\d+)\]", line)
            if mmss_m:
                start = int(mmss_m.group(1)) * 60 + int(mmss_m.group(2))
                end = int(mmss_m.group(3)) * 60 + int(mmss_m.group(4))
            else:
                continue

        start = max(0.0, min(duration, start))
        end = max(0.0, min(duration, end))
        if end <= start:
            continue

        if "ELIMINATED" in line.upper():
            intervals.append({
                "start": start, "end": end,
                "relevance": "ELIMINATED", "scene_hint": "", "reason_token": "",
            })
            continue

        rel_m = re.search(r"relevance\s*=\s*(HIGH|MEDIUM|LOW)", line, re.IGNORECASE)
        hint_m = re.search(r"scene_hint\s*=\s*([^\n\r]+?)(?:\s+reason_token\s*=|$)", line)
        rt_m = re.search(r"reason_token\s*=\s*([^\n\r]+)", line)

        intervals.append({
            "start": start,
            "end": end,
            "relevance": rel_m.group(1).upper() if rel_m else "MEDIUM",
            "scene_hint": hint_m.group(1).strip()[:60] if hint_m else "",
            "reason_token": rt_m.group(1).strip()[:40] if rt_m else "",
        })
    return intervals


def _format_intervals_block(intervals: List[Dict[str, Any]]) -> str:
    """Format interval list for prompt injection."""
    if not intervals:
        return "(no intervals)"
    lines = []
    for iv in intervals:
        if iv["relevance"] == "ELIMINATED":
            lines.append(f"  [{int(iv['start'])}-{int(iv['end'])}s] ELIMINATED")
        else:
            hint_str = f" scene_hint={iv['scene_hint']}" if iv.get("scene_hint") else ""
            lines.append(
                f"  [{int(iv['start'])}-{int(iv['end'])}s] "
                f"relevance={iv['relevance']}{hint_str}"
            )
    return "\n".join(lines)


def _pick_best_interval(intervals: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Pick the highest-priority active interval."""
    active = [iv for iv in intervals if iv["relevance"] not in ("ELIMINATED", "LOW")]
    if not active:
        # Fallback: pick widest non-eliminated
        active = [iv for iv in intervals if iv["relevance"] != "ELIMINATED"]
    if not active:
        return None
    # Sort: HIGH > MEDIUM > LOW, then by width (wider first)
    rank = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}
    active.sort(key=lambda iv: (rank.get(iv["relevance"], 0), iv["end"] - iv["start"]), reverse=True)
    return active[0]


def _parse_next_action(text: str) -> Dict[str, Any]:
    """Parse <search> tag to determine next region to explore.
    Returns {"action": "ANSWER"} if model output <answer> instead (meaning it's done).
    Supports both seconds format [126-158s] and MM:SS format [02:06-02:38].
    """
    search_m = re.search(r"<search>(.*?)</search>", text, re.DOTALL)

    if search_m:
        content = search_m.group(1).strip()
        # Try seconds format: [126-158s] or [126s-158s]
        sec_m = re.search(r"\[(\d+)\s*s?\s*[-–—>]+\s*(\d+)\s*s?\]", content)
        if sec_m:
            start = float(sec_m.group(1))
            end = float(sec_m.group(2))
        else:
            # Try MM:SS format: [02:06-02:38]
            mmss_m = re.search(r"\[(\d+):(\d+)\s*[-–—>]+\s*(\d+):(\d+)\]", content)
            if mmss_m:
                start = int(mmss_m.group(1)) * 60 + int(mmss_m.group(2))
                end = int(mmss_m.group(3)) * 60 + int(mmss_m.group(4))
            else:
                # Try plain numbers: 126-158 or 126 158
                plain_m = re.search(r"(\d+)\s*[-–—>\s]+\s*(\d+)", content)
                if plain_m:
                    start = float(plain_m.group(1))
                    end = float(plain_m.group(2))
                else:
                    return {"action": "ANSWER"}

        # Parse frames=N
        frames_m = re.search(r"frames\s*=\s*(\d+)", content)
        frames = int(frames_m.group(1)) if frames_m else 12
        # Cap frames to unlock range (aligned with 64f SFT)
        frames = max(4, min(frames, 64))

        return {
            "action": "SEARCH",
            "start": float(start),
            "end": float(end),
            "frames": frames,
        }

    # No <search> tag — model is answering
    return {"action": "ANSWER"}


def _extract_observation(text: str) -> str:
    m = re.search(r"<observation>(.*?)</observation>", text, re.DOTALL)
    if m:
        return m.group(1).strip()[:300]
    return ""


def _extract_conclusion(text: str) -> str:
    """Extract <conclusion> tag (v9j compat fallback)."""
    m = re.search(r"<conclusion>(.*?)</conclusion>", text, re.DOTALL)
    if m:
        return m.group(1).strip()[:300]
    return ""


def _extract_verification(text: str) -> str:
    """Extract <verification> tag — what the model found vs what was predicted."""
    m = re.search(r"<verification>(.*?)</verification>", text, re.DOTALL)
    if m:
        return m.group(1).strip()[:300]
    return ""


def _extract_prediction(text: str) -> str:
    """Extract <prediction> tag — the model's hypothesis for the next region."""
    m = re.search(r"<prediction>(.*?)</prediction>", text, re.DOTALL)
    if m:
        return m.group(1).strip()[:300]
    return ""


# ═══════════════════════════════════════════════════════════════
# Phase 1: Initial assessment (reuses existing VLM call logic)
# ═══════════════════════════════════════════════════════════════

def _phase1_assess(
    client: OpenAI,
    frame_paths: List[str],
    timestamps: List[float],
    duration: float,
    duration_fmt: str,
    resolution: str,
    raw_query: str,
) -> Tuple[List[Dict[str, Any]], str, str, str, int]:
    """Phase 1: assess intervals from sparse frames.

    Returns (intervals, query_type, initial_prediction, raw_output, tokens).
    """
    content: List[Dict[str, Any]] = []
    for i, (ts, fp) in enumerate(zip(timestamps, frame_paths)):
        content.append({"type": "text", "text": f"Frame {i + 1} at [{format_timestamp(ts)}]:"})
        content.append(_build_image_content(fp))

    boundary_ts = [0.0] + list(timestamps) + [duration]
    content.append({
        "type": "text",
        "text": prompts.PHASE1_USER_TEMPLATE.format(
            num_frames=len(frame_paths),
            num_intervals=len(boundary_ts) - 1,
            boundary_timestamps=", ".join(f"{int(ts)}s" for ts in boundary_ts),
            duration=int(duration),
            duration_fmt=duration_fmt,
            resolution=resolution,
            query=raw_query,
        ),
    })

    response, tok = _call_api(
        client, config.VERIFY_MODEL,
        prompts.PHASE1_SYSTEM,
        content,
        max_tokens=config.MAX_TOKENS_PHASE1,
    )

    # Parse intervals
    intervals = _parse_intervals_block(response, duration)

    # Fallback: if parsing failed, create uniform intervals
    if not intervals:
        for i in range(len(boundary_ts) - 1):
            intervals.append({
                "start": boundary_ts[i],
                "end": boundary_ts[i + 1],
                "relevance": "MEDIUM",
                "scene_hint": "",
                "reason_token": "",
            })

    # Parse query_type
    qt_m = re.search(r"<query_type>(.*?)</query_type>", response, re.DOTALL)
    query_type = qt_m.group(1).strip().split()[0] if qt_m else "global_understanding"

    # Parse natural language prediction (v9k2: model's own first hypothesis)
    pred_m = re.search(r"<prediction>(.*?)</prediction>", response, re.DOTALL)
    initial_prediction = pred_m.group(1).strip()[:300] if pred_m else ""

    return intervals, query_type, initial_prediction, response, tok


# ═══════════════════════════════════════════════════════════════
# Unified Loop
# ═══════════════════════════════════════════════════════════════

def foresight_single(
    index: int,
    item: Dict[str, Any],
    client: OpenAI,
    video_root: str,
    output_dir: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run ForesightAgent v9j on a single video QA sample.

    Phase 1: Sparse frame assessment → interval list
    Unified Loop: iterative verify + update intervals
    """
    video_file = item["videos"][0]
    video_path = os.path.join(video_root, video_file)
    raw_query = ""
    for turn in item.get("prompt", []):
        if turn.get("role") == "user":
            raw_query = str(turn.get("content", "")).strip()
            break

    gt = item.get("reward_model", {}).get("ground_truth", "")
    video_id = os.path.splitext(os.path.basename(video_file))[0]
    frame_dir = os.path.join(output_dir, config.FRAME_SAVE_ROOT, f"{video_id}_{index}_{os.getpid()}")
    total_tokens = 0

    # ── Tracking ──
    phase1_frames: int = 0
    phase1_visual_tokens: int = 0
    loop_frames_per_round: List[int] = []
    loop_visual_tokens_per_round: List[int] = []
    loop_regions_per_round: List[List[float]] = []
    loop_responses_per_round: List[str] = []

    # ─── Phase 1 ───
    duration, resolution = get_video_meta(video_path)
    width, height = (int(v) for v in resolution.split("x"))
    duration_fmt = format_timestamp(duration)

    # Adaptive initial frame count: scale with video length
    _min_probe = int(os.environ.get("MIN_PROBE_FRAMES", "10"))
    _max_probe = int(os.environ.get("MAX_PROBE_FRAMES_ADAPTIVE", "20"))
    # _min_probe = int(os.environ.get("MIN_PROBE_FRAMES", "8"))
    # _max_probe = int(os.environ.get("MAX_PROBE_FRAMES_ADAPTIVE", "13"))
    _adaptive_n = min(_max_probe, max(_min_probe, _min_probe + int(duration / 300)))
    probe_timestamps = uniform_timestamps(duration, _adaptive_n)
    # Phase 1 resize: default 0.8 (matches grpo/v9k2_agent_loop.py training setting).
    # Ablation override via env PHASE1_RESIZE; default preserves original behavior.
    _PHASE1_RESIZE = float(os.environ.get("PHASE1_RESIZE", "0.8"))
    probe_frame_paths = extract_frames(
        video_path, probe_timestamps,
        os.path.join(frame_dir, "phase1"),
        resize=_PHASE1_RESIZE,
    )

    intervals, query_type, initial_prediction, phase1_raw, tok = _phase1_assess(
        client, probe_frame_paths, probe_timestamps,
        duration, duration_fmt,
        f"{min(width, height)}p",
        raw_query,
    )
    total_tokens += tok
    phase1_frames = len(probe_frame_paths)
    phase1_visual_tokens = estimate_visual_tokens(width, height, phase1_frames, _PHASE1_RESIZE)

    if verbose:
        print(f"\n{'='*60}")
        print(f"[Phase 1] Video: {video_file} ({duration:.0f}s)")
        print(f"[Phase 1] Initial prediction: {initial_prediction[:100]}")
        print(f"[Phase 1] Query: {raw_query[:100]}")
        print(f"[Phase 1] Intervals ({len(intervals)}):")
        for iv in intervals:
            print(f"  [{int(iv['start'])}-{int(iv['end'])}s] {iv['relevance']} hint=\"{iv.get('scene_hint','')}\"")
        print(f"[Phase 1] Query type: {query_type}")

    # ─── Unified Loop ───
    max_rounds = int(os.environ.get("MAX_UNIFIED_ROUNDS", str(config.MAX_VERIFY_ROUNDS)))
    answer = ""
    confidence = 1
    stop_reason = "max_rounds_exceeded"
    search_history: List[Tuple[float, float, str, str]] = []  # (start, end, verification, prediction)

    # Determine initial target from Phase 1 prediction
    best_iv = _pick_best_interval(intervals)
    if best_iv is None:
        best_iv = {"start": 0, "end": duration, "relevance": "MEDIUM", "scene_hint": ""}

    next_start = best_iv["start"]
    next_end = best_iv["end"]
    next_frames = 12

    # Use the model's own Phase 1 prediction as Round 1's "To verify"
    if initial_prediction:
        current_prediction = initial_prediction
        # Try to extract region from prediction to override best_iv
        region_m = re.search(r'\[(\d+)[s]?-(\d+)[s]?\]', initial_prediction)
        mm_m = re.search(r'\[(\d+):(\d+)-(\d+):(\d+)\]', initial_prediction)
        if mm_m:
            next_start = int(mm_m.group(1)) * 60 + int(mm_m.group(2))
            next_end = int(mm_m.group(3)) * 60 + int(mm_m.group(4))
        elif region_m:
            next_start = int(region_m.group(1))
            next_end = int(region_m.group(2))
    else:
        # Fallback: construct from best interval
        current_prediction = (
            f"I predict answer evidence at [{int(best_iv['start'])}-{int(best_iv['end'])}s] "
            f"because scene_hint suggests: {best_iv.get('scene_hint', 'relevant content')}"
        )

    for round_num in range(max_rounds):
        is_final = (round_num == max_rounds - 1)
        start_t = max(0.0, min(duration, next_start))
        end_t = max(0.0, min(duration, next_end))
        if end_t <= start_t:
            end_t = min(duration, start_t + 30)

        n_frames = min(next_frames, 64)
        # Unified rounds resize: default max_side=576 cap then factor=0.8
        # (matches grpo/v9k2_agent_loop.py: _DEFAULT_MAX_SIDE_LOOP=576, resize_factor=0.8).
        # Ablation override via env LOOP_MAX_SIDE / LOOP_FACTOR; defaults preserve original behavior.
        _LOOP_MAX_SIDE = int(os.environ.get("LOOP_MAX_SIDE", "576"))
        _LOOP_FACTOR = float(os.environ.get("LOOP_FACTOR", "0.8"))
        _max_dim = max(width, height)
        resize = (min(_LOOP_MAX_SIDE, _max_dim) / _max_dim) * _LOOP_FACTOR

        # Token budget check
        est_tokens = estimate_visual_tokens(width, height, n_frames, resize)
        if est_tokens > config.MAX_VISUAL_TOKENS:
            n_frames = max(4, int(n_frames * config.MAX_VISUAL_TOKENS / est_tokens))

        # Extract frames
        target_ts = timestamps_in_range(start_t, end_t, n_frames)
        target_paths = extract_frames(
            video_path, target_ts,
            os.path.join(frame_dir, f"loop_r{round_num}"),
            resize=resize,
        )

        if not target_paths:
            stop_reason = "frame_extraction_failed"
            break

        round_vis_tok = estimate_visual_tokens(width, height, len(target_paths), resize)
        loop_frames_per_round.append(len(target_paths))
        loop_visual_tokens_per_round.append(round_vis_tok)
        loop_regions_per_round.append([round(float(start_t), 3), round(float(end_t), 3)])

        # Build VLM message
        verify_content: List[Dict[str, Any]] = []
        for i, fp in enumerate(target_paths):
            ts = target_ts[i] if i < len(target_ts) else 0
            verify_content.append({"type": "text", "text": f"Frame {i} at [{format_timestamp(ts)}]:"})
            verify_content.append(_build_image_content(fp))

        # Format search history as predict→verify chain
        if search_history:
            history_lines = []
            for i, (s, e, verif, pred) in enumerate(search_history):
                line = f"  Round {i+1}: viewed [{int(s)}-{int(e)}s]"
                if verif:
                    line += f"\n    Verified: {verif}"
                if pred:
                    line += f"\n    Predicted: {pred}"
                history_lines.append(line)
            history_str = "\n".join(history_lines)
        else:
            history_str = "  (first round — verifying Phase 1 prediction)"

        # Select template — pass current_prediction for verification
        if is_final:
            user_text = prompts.UNIFIED_USER_TEMPLATE_FINAL.format(
                duration=int(duration),
                duration_fmt=duration_fmt,
                start_fmt=format_timestamp(start_t),
                end_fmt=format_timestamp(end_t),
                current_prediction=current_prediction,
                search_history=history_str,
                query=raw_query,
            )
        else:
            user_text = prompts.UNIFIED_USER_TEMPLATE.format(
                duration=int(duration),
                duration_fmt=duration_fmt,
                start_fmt=format_timestamp(start_t),
                end_fmt=format_timestamp(end_t),
                intervals_block=_format_intervals_block(intervals),
                search_history=history_str,
                current_prediction=current_prediction,
                query=raw_query,
            )

        verify_content.append({"type": "text", "text": user_text})

        # Call VLM
        vlm_response, tok = _call_api(
            client, config.VERIFY_MODEL,
            prompts.UNIFIED_SYSTEM,
            verify_content,
            max_tokens=config.MAX_TOKENS_VERIFY,
        )
        total_tokens += tok
        loop_responses_per_round.append(vlm_response[:1500])

        # Parse response — VERIFY then ACT
        answer = _extract_answer(vlm_response)
        verification = _extract_verification(vlm_response)
        prediction = _extract_prediction(vlm_response)
        next_action = _parse_next_action(vlm_response)

        # Record search history as predict→verify chain
        search_history.append((start_t, end_t, verification[:200], prediction[:200] if prediction else ""))

        # Update intervals from model output (if <intervals> present)
        new_intervals = _parse_intervals_block(vlm_response, duration)
        if new_intervals:
            updated_ranges = set()
            for niv in new_intervals:
                updated_ranges.add((int(niv["start"]), int(niv["end"])))
            merged = []
            for old_iv in intervals:
                old_key = (int(old_iv["start"]), int(old_iv["end"]))
                if old_key not in updated_ranges:
                    merged.append(old_iv)
            merged.extend(new_intervals)
            merged.sort(key=lambda iv: iv["start"])
            intervals = merged

        if verbose:
            print(f"\n[Round {round_num}] Region=[{start_t:.0f}-{end_t:.0f}s], {len(target_paths)} frames")
            print(f"  Decision: {next_action['action']}, Answer={answer}")
            print(f"  Verification: {verification[:80]}")
            print(f"  Prediction: {prediction[:80]}")

        # Decision logic: model outputs ANSWER → stop; SEARCH → continue
        if next_action["action"] == "ANSWER" and answer:
            confidence = 3  # answering = confident
            stop_reason = "answer_found"
            break

        if is_final:
            # Last round — must have an answer
            confidence = 1
            break

        # Model wants to continue searching — update prediction for next round
        if next_action["action"] == "SEARCH":
            next_start = next_action.get("start", start_t)
            next_end = next_action.get("end", end_t)
            next_frames = next_action.get("frames", 12)
            # Carry prediction forward — this is what the next round will verify
            if prediction:
                current_prediction = prediction
            else:
                current_prediction = f"Searching [{int(next_start)}-{int(next_end)}s] for answer evidence"
        else:
            # Fallback: model didn't output valid <search> and no <answer>
            # Mark current region as explored, pick next best
            for iv in intervals:
                if iv["start"] <= start_t + 5 and iv["end"] >= end_t - 5:
                    if iv["relevance"] != "ELIMINATED":
                        iv["relevance"] = "LOW"
            best_iv = _pick_best_interval(intervals)
            if best_iv:
                next_start = best_iv["start"]
                next_end = best_iv["end"]
                next_frames = 12
                current_prediction = f"Fallback: trying [{int(next_start)}-{int(next_end)}s] (next best interval)"
            else:
                next_start = 0
                next_end = duration
                next_frames = 12
                current_prediction = "No intervals left, searching full video"

    if not answer:
        stop_reason = "no_answer"
    elif stop_reason != "answer_found":
        stop_reason = "low_confidence_answer"

    # Cleanup
    shutil.rmtree(frame_dir, ignore_errors=True)

    loop_frames_total = sum(loop_frames_per_round)
    loop_visual_tokens_total = sum(loop_visual_tokens_per_round)
    all_visual_tokens = phase1_visual_tokens + loop_visual_tokens_total

    return {
        "index": index,
        "videos": item.get("videos", []),
        "gt": gt,
        "answer": answer,
        "stop_reason": stop_reason,
        "num_rounds": len(loop_frames_per_round),
        "total_tokens": total_tokens,
        "frame_stats": {
            "phase1_frames": phase1_frames,
            "phase1_visual_tokens": phase1_visual_tokens,
            "phase25_frames": 0,  # no longer separate
            "phase25_visual_tokens": 0,
            "phase3_frames_per_round": loop_frames_per_round,
            "phase3_visual_tokens_per_round": loop_visual_tokens_per_round,
            "phase3_frames_total": loop_frames_total,
            "phase3_visual_tokens_total": loop_visual_tokens_total,
            "total_frames": phase1_frames + loop_frames_total,
            "total_visual_tokens": all_visual_tokens,
        },
        "estimated_visual_tokens": loop_visual_tokens_total,
        "phase1_captions": [""] * phase1_frames,
        "phase1_boundaries": "None detected",
        "phase1_belief_raw_output": phase1_raw,
        "phase2_structure": "",
        "phase2_raw_output": phase1_raw,
        "final_navigation_raw_output": "",
        "phase2_prediction": {
            "primary": [next_start, next_end],
            "secondary": [0, 0],
            "confidence": "medium",
            "query_type": query_type,
            "uncertain_regions": [],
        },
        "phase2_belief_nodes": intervals,  # flat list instead of tree
        "trajectory": {
            "phase1_probe_timestamps": [round(float(t), 3) for t in probe_timestamps],
            "phase25_probe_targets_per_round": [],
            "phase25_probe_nodes_per_round": [],
            "phase25_update_raw_outputs_per_round": [],
            "phase3_regions_per_round": loop_regions_per_round,
            "phase3_timestamps_per_round": [[round(float(t), 3) for t in target_ts] for target_ts in [timestamps_in_range(r[0], r[1], f) for r, f in zip(loop_regions_per_round, loop_frames_per_round)] if target_ts],
            "phase3_responses_per_round": loop_responses_per_round,
            "correction_raw_outputs": [],
            "correction_regions": [],
            "frontier_restores": [],
        },
        "query": raw_query,
    }
