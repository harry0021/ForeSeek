import os
import re
import subprocess
import json
import base64
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

import config
import prompts
from frame_utils import (
    uniform_timestamps,
    timestamps_in_range,
    format_timestamp,
    estimate_visual_tokens,
)


# ─── Helpers (copied from foresight_agent.py, no local_models import) ───

@dataclass
class StructurePrediction:
    structure_text: str = ""
    uncertain_regions: List[Tuple[float, float]] = field(default_factory=list)
    reasoning: str = ""
    primary_region: Tuple[float, float] = (0, 0)
    secondary_region: Tuple[float, float] = (0, 0)
    primary_confidence: str = "low"
    num_frames: int = 8
    resolution: str = "medium"
    query_type: str = "global_understanding"
    raw_output: str = ""


def _parse_time_range(text: str) -> Optional[Tuple[float, float]]:
    m = re.search(r"\[(\d+):(\d+)\s*-\s*(\d+):(\d+)\]", text)
    if m:
        start = int(m.group(1)) * 60 + int(m.group(2))
        end = int(m.group(3)) * 60 + int(m.group(4))
        return float(start), float(end)
    m = re.search(r"\[(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\]", text)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


def parse_structure_output(text: str, duration: float) -> StructurePrediction:
    pred = StructurePrediction(raw_output=text)
    m = re.search(r"<structure>(.*?)</structure>", text, re.DOTALL)
    if m:
        pred.structure_text = m.group(1).strip()
    m = re.search(r"<uncertain(?:_regions)?>(.*?)</uncertain(?:_regions)?>", text, re.DOTALL)
    if m:
        region_text = m.group(1).strip()
        if region_text.lower() != "none":
            for line in region_text.split("\n"):
                r = _parse_time_range(line)
                if r:
                    pred.uncertain_regions.append(r)
    m = re.search(r"<prediction>(.*?)</prediction>", text, re.DOTALL)
    if m:
        for line in m.group(1).strip().split("\n"):
            low = line.lower()
            if "primary" in low:
                r = _parse_time_range(line)
                if r:
                    pred.primary_region = (max(0, r[0]), min(duration, r[1]))
                if "high" in low:
                    pred.primary_confidence = "high"
                elif "medium" in low:
                    pred.primary_confidence = "medium"
            elif "secondary" in low:
                r = _parse_time_range(line)
                if r:
                    pred.secondary_region = (max(0, r[0]), min(duration, r[1]))
            elif "frames" in low:
                fm = re.search(r"(\d+)", line)
                if fm:
                    pred.num_frames = min(int(fm.group(1)), 16)
            elif "resolution" in low:
                if "high" in low:
                    pred.resolution = "high"
                elif "low" in low:
                    pred.resolution = "low"
    m = re.search(r"<query_type>(.*?)</query_type>", text, re.DOTALL)
    if m:
        pred.query_type = m.group(1).strip().split()[0]
    if pred.primary_region == (0, 0) or pred.primary_region[0] >= pred.primary_region[1]:
        pred.primary_region = (0, duration)
        pred.primary_confidence = "low"
    return pred


def _extract_answer(text: str) -> str:
    # supports A-H (CG-Bench / LVBench have up to 7-8 options)
    m = re.search(r"<answer>\s*([A-Ha-h])\s*</answer>", text)
    if m:
        return m.group(1).strip().upper()
    m = re.search(r"[Aa]nswer[\.:\s]\s*([A-Ha-h])", text)
    if m:
        return m.group(1).strip().upper()
    return ""


def _extract_confidence(text: str) -> int:
    m = re.search(r"<confidence>\s*(\d)\s*</confidence>", text)
    if m:
        return int(m.group(1))
    return 1


def _format_captions_block(captions: List[Tuple[float, str]]) -> str:
    return "\n".join(
        f"[{format_timestamp(ts)}] {cap}"
        for ts, cap in sorted(captions, key=lambda x: x[0])
    )


def _resize_factor(resolution: str) -> float:
    return {"high": 1.0, "medium": 0.5, "low": 0.25}.get(resolution, 0.5)


def _encode_image_base64(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _build_image_content(file_path: str) -> Dict[str, Any]:
    b64 = _encode_image_base64(file_path)
    ext = os.path.splitext(file_path)[1].lstrip(".").lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "image/png")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def _call_api(
    client: OpenAI,
    model: str,
    system: str,
    user_content: Any,
    max_tokens: int = 2048,
) -> Tuple[str, int]:
    """Call the cloud API (sync, non-streaming). Returns (text, total_tokens)."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    try:
        response = client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens,
        )
        text = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0
        return text, tokens
    except Exception as e:
        print(f"  API error ({type(e).__name__}): {e}")
        return "", 0


def get_video_meta(video_path: str) -> Tuple[float, str]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=duration,width,height",
         "-show_entries", "format=duration",
         "-of", "json", video_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()}")
    data = json.loads(result.stdout or "{}")
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video streams in {video_path}")
    stream = streams[0]
    duration = stream.get("duration") or data.get("format", {}).get("duration")
    if duration is None:
        raise RuntimeError(f"Cannot determine duration for {video_path}")
    return float(duration), f"{int(stream.get('width',0))}x{int(stream.get('height',0))}"


def extract_frames(
    video_path: str, timestamps: List[float], save_dir: str, resize: float = 1.0,
) -> List[str]:
    os.makedirs(save_dir, exist_ok=True)
    paths = []
    for i, ts in enumerate(timestamps):
        out_path = os.path.join(save_dir, f"frame_{i:04d}.png")
        cmd = ["ffmpeg", "-y", "-ss", f"{ts:.3f}", "-i", video_path,
               "-frames:v", "1", "-q:v", "2"]
        if resize < 1.0:
            cmd.extend(["-vf", f"scale='trunc(iw*{resize}/2)*2:trunc(ih*{resize}/2)*2'"])
        cmd.append(out_path)
        subprocess.run(cmd, capture_output=True)
        if os.path.exists(out_path):
            paths.append(out_path)
    return paths


# ─── Main agent function (API-only, no torch) ───

def foresight_single_api(
    index: int,
    item: Dict[str, Any],
    client: OpenAI,
    video_root: str,
    output_dir: str,
) -> Dict[str, Any]:
    """ForesightAgent with all local models replaced by API calls.

    BLIP-2 → Gemini captioning via API
    CLIP   → skipped (boundaries_block = "None detected")
    """
    video_file = item["videos"][0]
    video_path = os.path.join(video_root, video_file)
    raw_query = ""
    for turn in item.get("prompt", []):
        if turn.get("role") == "user":
            raw_query = str(turn.get("content", "")).strip()
            break

    gt = item.get("reward_model", {}).get("ground_truth", "")
    video_id = os.path.splitext(video_file)[0]
    frame_dir = os.path.join(output_dir, config.FRAME_SAVE_ROOT, video_id)
    total_tokens = 0

    # ─── Phase 1: Sparse Captioning via API ───
    duration, resolution = get_video_meta(video_path)
    width, height = (int(v) for v in resolution.split("x"))

    probe_timestamps = uniform_timestamps(duration, config.NUM_PROBE_FRAMES)
    probe_frame_paths = extract_frames(
        video_path, probe_timestamps,
        os.path.join(frame_dir, "phase1"),
        resize=config.PHASE1_RESIZE,
    )

    # Use Gemini API for captioning (no local BLIP-2)
    captions: List[Tuple[float, str]] = []
    for ts, fp in zip(probe_timestamps, probe_frame_paths):
        content = [
            {"type": "text", "text": prompts.CAPTION_PROMPT},
            _build_image_content(fp),
        ]
        caption, tok = _call_api(
            client, config.VERIFY_MODEL,  # use VLM for captioning
            "You are a helpful video frame captioner.",
            content,
            max_tokens=150,
        )
        total_tokens += tok
        captions.append((ts, caption.strip()))

    # CLIP boundary detection → skipped (no torch)
    boundaries_block = "None detected (API-only mode, CLIP unavailable)"

    # ─── Phase 2: LLM Structure Prediction ───
    duration_fmt = format_timestamp(duration)
    captions_block = _format_captions_block(captions)

    structure_input = prompts.STRUCTURE_USER_TEMPLATE.format(
        duration=int(duration),
        duration_fmt=duration_fmt,
        resolution=f"{min(width,height)}p",
        num_frames=len(captions),
        captions_block=captions_block,
        boundaries_block=boundaries_block,
        query=raw_query,
    )
    structure_output, tok = _call_api(
        client, config.STRUCTURE_MODEL,
        prompts.STRUCTURE_SYSTEM,
        structure_input,
        max_tokens=config.MAX_TOKENS_STRUCTURE,
    )
    total_tokens += tok
    prediction = parse_structure_output(structure_output, duration)

    # ─── Phase 2.5: Adaptive Probing via API ───
    for probe_round in range(config.MAX_PROBE_ROUNDS):
        if not prediction.uncertain_regions or prediction.primary_confidence == "high":
            break

        probe_targets = []
        remaining = config.MAX_EXTRA_FRAMES
        for s, e in prediction.uncertain_regions:
            if remaining <= 0:
                break
            seg_len = max(0.0, e - s)
            n = max(1, min(remaining, int(seg_len / 30)))
            probe_targets.extend(timestamps_in_range(s, e, n))
            remaining -= n

        if not probe_targets:
            break

        new_frame_paths = extract_frames(
            video_path, probe_targets,
            os.path.join(frame_dir, f"phase2_5_r{probe_round}"),
            resize=config.PHASE1_RESIZE,
        )

        new_captions_list = []
        for ts, fp in zip(probe_targets, new_frame_paths):
            content = [
                {"type": "text", "text": prompts.CAPTION_PROMPT},
                _build_image_content(fp),
            ]
            cap, tok = _call_api(
                client, config.VERIFY_MODEL,
                "You are a helpful video frame captioner.",
                content,
                max_tokens=150,
            )
            total_tokens += tok
            new_captions_list.append((ts, cap.strip()))
            captions.append((ts, cap.strip()))

        all_captions_block = _format_captions_block(captions)
        new_captions_block = _format_captions_block(new_captions_list)

        probe_input = prompts.ADAPTIVE_PROBE_TEMPLATE.format(
            previous_structure=prediction.raw_output,
            new_captions_block=new_captions_block,
            all_captions_block=all_captions_block,
            boundaries_block=boundaries_block,
            query=raw_query,
        )
        probe_output, tok = _call_api(
            client, config.STRUCTURE_MODEL,
            prompts.STRUCTURE_SYSTEM,
            probe_input,
            max_tokens=config.MAX_TOKENS_STRUCTURE,
        )
        total_tokens += tok
        prediction = parse_structure_output(probe_output, duration)

    # ─── Phase 3: VLM Verification ───
    answer = ""
    confidence = 1
    stop_reason = "max_rounds_exceeded"
    num_vlm_calls = 0
    total_visual_tokens = 0

    for round_num in range(config.MAX_VERIFY_ROUNDS):
        region = prediction.primary_region
        start_t, end_t = region
        n_frames = prediction.num_frames if round_num == 0 else max(4, prediction.num_frames // 2)
        resize = _resize_factor(prediction.resolution)

        est_tokens = estimate_visual_tokens(width, height, n_frames, resize)
        if est_tokens > config.MAX_VISUAL_TOKENS:
            r = (config.MAX_VISUAL_TOKENS / est_tokens) ** (1 / 3)
            n_frames = max(1, int(n_frames * r))
            resize = resize * r

        target_ts = timestamps_in_range(start_t, end_t, n_frames)
        target_paths = extract_frames(
            video_path, target_ts,
            os.path.join(frame_dir, f"phase3_r{round_num}"),
            resize=resize,
        )

        if not target_paths:
            stop_reason = "frame_extraction_failed"
            break

        total_visual_tokens += estimate_visual_tokens(width, height, len(target_paths), resize)
        num_vlm_calls += 1

        verify_content: List[Dict[str, Any]] = []
        for i, fp in enumerate(target_paths):
            ts = target_ts[i] if i < len(target_ts) else 0
            verify_content.append({"type": "text", "text": f"Frame {i} at [{format_timestamp(ts)}]:"})
            verify_content.append(_build_image_content(fp))

        verify_content.append({
            "type": "text",
            "text": prompts.VERIFICATION_USER_TEMPLATE.format(
                start_fmt=format_timestamp(start_t),
                end_fmt=format_timestamp(end_t),
                duration_fmt=duration_fmt,
                query=raw_query,
            ),
        })

        vlm_response, tok = _call_api(
            client, config.VERIFY_MODEL,
            prompts.VERIFICATION_SYSTEM,
            verify_content,
            max_tokens=config.MAX_TOKENS_VERIFY,
        )
        total_tokens += tok

        answer = _extract_answer(vlm_response)
        confidence = _extract_confidence(vlm_response)

        if answer and confidence >= config.CONFIDENCE_THRESHOLD:
            stop_reason = "answer_found"
            break

        if round_num < config.MAX_VERIFY_ROUNDS - 1:
            found_status = "found with low confidence" if answer else "NOT found"
            instruction = prompts.CORRECTION_FOUND if answer else prompts.CORRECTION_NOT_FOUND
            correction_input = prompts.CORRECTION_USER_TEMPLATE.format(
                previous_structure=prediction.raw_output,
                examined_start_fmt=format_timestamp(start_t),
                examined_end_fmt=format_timestamp(end_t),
                vlm_observation=vlm_response[:500],
                found_status=found_status,
                all_captions_block=_format_captions_block(captions),
                query=raw_query,
                correction_instruction=instruction,
            )
            correction_output, tok = _call_api(
                client, config.STRUCTURE_MODEL,
                prompts.CORRECTION_SYSTEM,
                correction_input,
                max_tokens=config.MAX_TOKENS_STRUCTURE,
            )
            total_tokens += tok
            prediction = parse_structure_output(correction_output, duration)

    if not answer:
        stop_reason = "no_answer"
    elif stop_reason != "answer_found":
        stop_reason = "low_confidence_answer"

    shutil.rmtree(frame_dir, ignore_errors=True)

    return {
        "index": index,
        "videos": item.get("videos", []),
        "gt": gt,
        "answer": answer,
        "stop_reason": stop_reason,
        "num_rounds": num_vlm_calls,
        "total_tokens": total_tokens,
        "estimated_visual_tokens": total_visual_tokens,
        "phase2_structure": prediction.structure_text,
        "phase2_prediction": {
            "primary": list(prediction.primary_region),
            "confidence": prediction.primary_confidence,
            "query_type": prediction.query_type,
        },
    }
