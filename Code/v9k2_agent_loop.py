from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any
from uuid import uuid4

import cv2
from PIL import Image

# v9k2 prompt set (shared with foresight_agent_v9k2.py)
_prompts_mod = os.environ.get("PROMPTS_MODULE", "prompts_v9k2")
import importlib as _imp
prompts = _imp.import_module(_prompts_mod)

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (unchanged from previous version)
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_ROUND_FRAMES   = 12
_DEFAULT_MAX_SIDE_LOOP  = 576
_DEFAULT_MAX_ROUNDS     = 5


def _format_timestamp(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 60:02d}:{s % 60:02d}"


def _resize_pil(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def _resize_pil_factor(img: Image.Image, factor: float) -> Image.Image:
    """Resize by relative factor (e.g. 0.8 = shrink to 80% per dimension).
    Used for Phase1 to reduce visual token count without changing aspect ratio.
    """
    if factor >= 1.0:
        return img
    w, h = img.size
    return img.resize((int(w * factor), int(h * factor)), Image.LANCZOS)


def _uniform_ts_internal(start: float, end: float, n: int) -> list[float]:
    if n <= 0:
        return []
    span = end - start
    return [start + span * (i + 1) / (n + 1) for i in range(n)]


def _extract_frames_at_ts(
    video_path: str,
    timestamps: list[float],
    max_side: int | None = None,
    resize_factor: float | None = None,
) -> tuple[list[Image.Image], list[float]]:
    """Extract frames at the given timestamps.

    Returns ``(frames, kept_timestamps)`` where ``kept_timestamps[i]`` is the
    timestamp of ``frames[i]``. Frames that fail to decode are skipped from
    BOTH lists, so the two stay index-aligned. This lets callers emit a correct
    per-frame ``Frame i at [mm:ss]:`` label (matching inference) without any
    risk of label/frame desync when an occasional frame fails to read.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        return [], []
    frames: list[Image.Image] = []
    kept_ts: list[float] = []
    for ts in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            continue
        if frame.shape[0] == 0 or frame.shape[1] == 0:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        if pil.width == 0 or pil.height == 0:
            continue
        if max_side is not None:
            pil = _resize_pil(pil, max_side)
        if resize_factor is not None and resize_factor < 1.0:
            pil = _resize_pil_factor(pil, resize_factor)
        if pil.width == 0 or pil.height == 0:
            continue
        frames.append(pil)
        kept_ts.append(ts)
    cap.release()
    return frames, kept_ts


def _get_video_meta(video_path: str) -> tuple[float, str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0, "720p"
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    nf  = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = nf / fps
    return duration, f"{min(w, h)}p"


def _phase1_n_frames(duration: float) -> int:
    _min = int(os.environ.get("MIN_PROBE_FRAMES", "10"))
    _max = int(os.environ.get("MAX_PROBE_FRAMES_ADAPTIVE", "20"))
    return max(_min, min(_max, _min + int(duration / 300)))


# XML parsers ────────────────────────────────────────────────────────────────

def _parse_prediction(text: str) -> str:
    m = re.search(r"<prediction>(.*?)</prediction>", text, re.DOTALL)
    return m.group(1).strip()[:300] if m else ""


def _parse_answer(text: str) -> str:
    m = re.search(r"<answer>\s*([A-Ha-h])\s*</answer>", text)
    return m.group(1).upper() if m else ""


def _parse_verification(text: str) -> str:
    m = re.search(r"<verification>(.*?)</verification>", text, re.DOTALL)
    return m.group(1).strip()[:200] if m else ""


def _parse_next_action(text: str) -> dict:
    search_m = re.search(r"<search>(.*?)</search>", text, re.DOTALL)
    if search_m:
        content = search_m.group(1).strip()
        sec_m = re.search(r"\[(\d+)\s*s?\s*[-–—>]+\s*(\d+)\s*s?\]", content)
        if sec_m:
            start, end = float(sec_m.group(1)), float(sec_m.group(2))
        else:
            mm_m = re.search(r"\[(\d+):(\d+)\s*[-–—>]+\s*(\d+):(\d+)\]", content)
            if mm_m:
                start = int(mm_m.group(1)) * 60 + int(mm_m.group(2))
                end   = int(mm_m.group(3)) * 60 + int(mm_m.group(4))
            else:
                plain_m = re.search(r"(\d+)\s*[-–—>\s]+\s*(\d+)", content)
                if plain_m:
                    start, end = float(plain_m.group(1)), float(plain_m.group(2))
                else:
                    return {"action": "ANSWER"}
        frames_m = re.search(r"frames\s*=\s*(\d+)", content)
        frames = int(frames_m.group(1)) if frames_m else 12
        frames = max(4, min(frames, 16))
        return {"action": "SEARCH", "start": float(start), "end": float(end), "frames": frames}
    return {"action": "ANSWER"}


def _parse_intervals_block(text: str, duration: float) -> list[dict]:
    m = re.search(r"<intervals>(.*?)</intervals>", text, re.DOTALL)
    if not m:
        return []
    intervals = []
    for line in m.group(1).strip().splitlines():
        line = line.strip()
        if not line:
            continue
        range_m = re.search(r"\[(\d+)\s*s?\s*[-–—>]+\s*(\d+)\s*s?\]", line)
        if range_m:
            start, end = float(range_m.group(1)), float(range_m.group(2))
        else:
            mmss_m = re.search(r"\[(\d+):(\d+)\s*[-–—>]+\s*(\d+):(\d+)\]", line)
            if mmss_m:
                start = int(mmss_m.group(1)) * 60 + int(mmss_m.group(2))
                end   = int(mmss_m.group(3)) * 60 + int(mmss_m.group(4))
            else:
                continue
        start = max(0.0, min(duration, start))
        end   = max(0.0, min(duration, end))
        if end <= start:
            continue
        if "ELIMINATED" in line.upper():
            intervals.append({"start": start, "end": end,
                              "relevance": "ELIMINATED", "scene_hint": "", "reason_token": ""})
            continue
        rel_m  = re.search(r"relevance\s*=\s*(HIGH|MEDIUM|LOW)", line, re.IGNORECASE)
        hint_m = re.search(r"scene_hint\s*=\s*([^\n\r]+?)(?:\s+reason_token\s*=|$)", line)
        rt_m   = re.search(r"reason_token\s*=\s*([^\n\r]+)", line)
        intervals.append({
            "start":        start,
            "end":          end,
            "relevance":    rel_m.group(1).upper() if rel_m else "MEDIUM",
            "scene_hint":   hint_m.group(1).strip()[:60] if hint_m else "",
            "reason_token": rt_m.group(1).strip()[:40]   if rt_m   else "",
        })
    return intervals


def _format_intervals_block(intervals: list[dict]) -> str:
    if not intervals:
        return "(no intervals)"
    lines = []
    for iv in intervals:
        if iv["relevance"] == "ELIMINATED":
            lines.append(f"  [{int(iv['start'])}-{int(iv['end'])}s] ELIMINATED")
        else:
            hint = f" scene_hint={iv['scene_hint']}" if iv.get("scene_hint") else ""
            lines.append(f"  [{int(iv['start'])}-{int(iv['end'])}s] relevance={iv['relevance']}{hint}")
    return "\n".join(lines)


def _pick_best_interval(intervals: list[dict]) -> dict | None:
    active = [iv for iv in intervals if iv["relevance"] not in ("ELIMINATED", "LOW")]
    if not active:
        active = [iv for iv in intervals if iv["relevance"] != "ELIMINATED"]
    if not active:
        return None
    rank = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}
    active.sort(key=lambda iv: (rank.get(iv["relevance"], 0), iv["end"] - iv["start"]), reverse=True)
    return active[0]


def _merge_intervals(current: list[dict], updates: list[dict]) -> list[dict]:
    updated_keys = {(int(iv["start"]), int(iv["end"])) for iv in updates}
    merged = [iv for iv in current
              if (int(iv["start"]), int(iv["end"])) not in updated_keys]
    merged.extend(updates)
    merged.sort(key=lambda iv: iv["start"])
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Empty single-output (used when frame extraction or generation fails)
# ─────────────────────────────────────────────────────────────────────────────

def _empty_single_output() -> AgentLoopOutput:
    # reward_score=None → verl skips rm_scores assignment (avoids IndexError on empty response)
    return AgentLoopOutput(
        prompt_ids=[0],      # minimal 1-token dummy to avoid zero-dim tensor crashes
        response_ids=[0],
        response_mask=[0],
        reward_score=None,
        num_turns=0,
        metrics=AgentLoopMetrics(),
        extra_fields={"reward_extra_info": {
            "correct": 0.0, "clue_iou": 0.0,
            "consistency": 0.0, "verify_q": 0.0, "n_rounds": 0,
        }},
    )


# ─────────────────────────────────────────────────────────────────────────────
# AgentLoop
# ─────────────────────────────────────────────────────────────────────────────

@register("v9k2_agent")
class V9K2AgentLoop(AgentLoopBase):
    """
    Each VLM call → one separate AgentLoopOutput.

    run() returns list[AgentLoopOutput]:
        outputs[0]: Phase 1 (sparse frames, intervals + initial prediction)
        outputs[1..N]: Round i (dense frames, verify + answer-or-search)
    Only outputs[-1].reward_score is set; verl will broadcast that to all
    earlier outputs (see main_ppo_sync.py:402-404).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_assistant_turns = getattr(
            self.rollout_config.multi_turn, "max_assistant_turns", None
        ) or _DEFAULT_MAX_ROUNDS
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    async def run(
        self, sampling_params: dict[str, Any], **kwargs
    ) -> list[AgentLoopOutput]:
        # ── trajectory metadata injected by patched verl _run_agent_loop ────
        # (verl/experimental/agent_loop/agent_loop.py passes these). All have
        # safe defaults so this still works if verl is reverted.
        self._grpo_step          = int(kwargs.pop("_grpo_step", -1))
        self._grpo_sample_index  = int(kwargs.pop("_grpo_sample_index", -1))
        self._grpo_rollout_n     = int(kwargs.pop("_grpo_rollout_n", -1))
        self._grpo_validate      = bool(kwargs.pop("_grpo_validate", False))

        # ── extract dataset fields ──────────────────────────────────────────
        extra: dict        = kwargs.get("extra_info") or {}
        video_path: str    = extra.get("video_path", "")
        gt_answer: str     = extra.get("ground_truth", "")
        clue_intervals     = extra.get("clue_intervals", []) or []
        question: str      = extra.get("question", "")
        duration: float    = float(extra.get("duration", 0.0) or 0.0)
        max_rounds = int(self.max_assistant_turns)

        # ── resolve duration / resolution from video metadata ───────────────
        duration_meta, resolution = await asyncio.to_thread(_get_video_meta, video_path)
        if duration <= 0:
            duration = duration_meta
        duration_fmt = _format_timestamp(duration)

        # ── Phase 1: extract sparse frames ──────────────────────────────────
        n_phase1 = _phase1_n_frames(duration)
        probe_ts = _uniform_ts_internal(0.0, duration, n_phase1)
        # Phase1 frames: resize_factor=0.8 (originally 1.0/none), reduces visual
        # token count by ~36% (0.8² = 0.64) without changing aspect ratio.
        # Required to prevent backward OOM when actor params/grads are on GPU
        # (offload_policy=False). Inference still uses full resolution; this is
        # a training-only memory optimization.
        phase1_frames, phase1_ts = await asyncio.to_thread(
            _extract_frames_at_ts, video_path, probe_ts, None, 0.8
        )
        if not phase1_frames or duration <= 0.0:
            return [_empty_single_output()]
        boundary_ts = [0.0] + probe_ts + [duration]

        outputs: list[AgentLoopOutput] = []

        # ── Trajectory bookkeeping (carried across calls for reward) ────────
        intervals: list[dict] = []
        search_history: list[tuple[float, float, str, str]] = []
        trajectory_regions: list[list[float]] = []
        predicted_next_regions: list = []  # per-round <search> declared region or None
        verifications: list[str] = []
        final_answer = ""
        n_rounds_completed = 0  # how many "rounds" (excluding Phase1) finished

        # ── Phase 1 call (independent system+user) ──────────────────────────
        phase1_user_text = prompts.PHASE1_USER_TEMPLATE.format(
            num_frames=len(phase1_frames),
            num_intervals=len(boundary_ts) - 1,
            boundary_timestamps=", ".join(f"{int(ts)}s" for ts in boundary_ts),
            duration=int(duration),
            duration_fmt=duration_fmt,
            resolution=resolution,
            query=question,
        )
        # Interleave per-frame timestamp labels with images (matches inference
        # foresight_agent_v9k2: "Frame i at [mm:ss]:" before each frame).
        # Phase 1 uses 1-indexed frame numbers (Frame 1..N) like inference.
        phase1_user_content: list[dict] = []
        for i, ts in enumerate(phase1_ts):
            phase1_user_content.append(
                {"type": "text", "text": f"Frame {i + 1} at [{_format_timestamp(ts)}]:"}
            )
            phase1_user_content.append({"type": "image"})
        phase1_user_content.append({"type": "text", "text": phase1_user_text})
        phase1_messages = [
            {"role": "system",  "content": prompts.PHASE1_SYSTEM},
            {"role": "user",    "content": phase1_user_content},
        ]
        p1_out = await self._single_call_output(
            messages=phase1_messages,
            images=phase1_frames,
            sampling_params=sampling_params,
        )
        if p1_out is None:
            return [_empty_single_output()]
        outputs.append(p1_out)

        # Parse Phase 1 response
        phase1_text = self.tokenizer.decode(p1_out.response_ids, skip_special_tokens=True)
        intervals          = _parse_intervals_block(phase1_text, duration) or intervals
        initial_prediction = _parse_prediction(phase1_text)
        early_answer       = _parse_answer(phase1_text)

        if early_answer:
            # Phase 1 already produced an answer — terminate immediately.
            final_answer = early_answer
            return self._finalize(
                outputs=outputs,
                final_answer=final_answer,
                gt_answer=gt_answer,
                trajectory_regions=trajectory_regions,
                predicted_next_regions=predicted_next_regions,
                verifications=verifications,
                clue_intervals=clue_intervals,
                n_rounds=n_rounds_completed,
                max_rounds=max_rounds,
                duration=duration,
            )

        # Determine first search region
        best_iv = _pick_best_interval(intervals)
        if best_iv is None:
            best_iv = {"start": 0.0, "end": duration, "relevance": "MEDIUM", "scene_hint": ""}
        next_start = best_iv["start"]
        next_end   = best_iv["end"]
        next_frames = _DEFAULT_ROUND_FRAMES

        if initial_prediction:
            current_prediction = initial_prediction
            region_m = re.search(r'\[(\d+)[s]?-(\d+)[s]?\]', initial_prediction)
            mm_m     = re.search(r'\[(\d+):(\d+)-(\d+):(\d+)\]', initial_prediction)
            if mm_m:
                next_start = int(mm_m.group(1)) * 60 + int(mm_m.group(2))
                next_end   = int(mm_m.group(3)) * 60 + int(mm_m.group(4))
            elif region_m:
                next_start = int(region_m.group(1))
                next_end   = int(region_m.group(2))
        else:
            current_prediction = (
                f"I predict answer evidence at [{int(best_iv['start'])}-{int(best_iv['end'])}s] "
                f"because scene_hint suggests: {best_iv.get('scene_hint', 'relevant content')}"
            )

        # ── Unified loop: each round is a SEPARATE call ─────────────────────
        for round_idx in range(max_rounds):
            n_rounds_completed = round_idx + 1
            is_final = (round_idx == max_rounds - 1)

            start_t = max(0.0, min(duration, next_start))
            end_t   = max(0.0, min(duration, next_end))
            if end_t <= start_t:
                end_t = min(duration, start_t + 30.0)

            n_frames = min(next_frames, 16)
            round_ts = _uniform_ts_internal(start_t, end_t, n_frames)
            # Round frames: max_side=576 cap + resize_factor=0.8
            # → final max side ≈ 460 px (576 × 0.8). Reduces visual tokens
            # by ~36% per frame. With 12 frames per round and up to 5 rounds,
            # this is the dominant memory saving for multi-turn trajectories.
            round_frames, round_kept_ts = await asyncio.to_thread(
                _extract_frames_at_ts, video_path, round_ts, _DEFAULT_MAX_SIDE_LOOP, 0.8
            )
            if not round_frames:
                break
            trajectory_regions.append([start_t, end_t])

            # Build search history block
            if search_history:
                hlines = []
                for i, (s, e, verif, pred) in enumerate(search_history):
                    line = f"  Round {i+1}: viewed [{int(s)}-{int(e)}s]"
                    if verif:
                        line += f"\n    Verified: {verif}"
                    if pred:
                        line += f"\n    Predicted: {pred}"
                    hlines.append(line)
                history_str = "\n".join(hlines)
            else:
                history_str = "  (first round — verifying Phase 1 prediction)"

            # Build round user template
            if is_final:
                user_text = prompts.UNIFIED_USER_TEMPLATE_FINAL.format(
                    duration=int(duration),
                    duration_fmt=duration_fmt,
                    start_fmt=_format_timestamp(start_t),
                    end_fmt=_format_timestamp(end_t),
                    current_prediction=current_prediction,
                    search_history=history_str,
                    query=question,
                )
            else:
                user_text = prompts.UNIFIED_USER_TEMPLATE.format(
                    duration=int(duration),
                    duration_fmt=duration_fmt,
                    start_fmt=_format_timestamp(start_t),
                    end_fmt=_format_timestamp(end_t),
                    intervals_block=_format_intervals_block(intervals),
                    search_history=history_str,
                    current_prediction=current_prediction,
                    query=question,
                )

            # Interleave per-frame timestamp labels with images (matches inference
            # foresight_agent_v9k2: "Frame i at [mm:ss]:" before each frame).
            # Rounds use 0-indexed frame numbers (Frame 0..N-1) like inference.
            round_user_content = []
            for i, ts in enumerate(round_kept_ts):
                round_user_content.append(
                    {"type": "text", "text": f"Frame {i} at [{_format_timestamp(ts)}]:"}
                )
                round_user_content.append({"type": "image"})
            round_user_content.append({"type": "text", "text": user_text})
            round_messages = [
                {"role": "system",  "content": prompts.UNIFIED_SYSTEM},
                {"role": "user",    "content": round_user_content},
            ]

            r_out = await self._single_call_output(
                messages=round_messages,
                images=round_frames,
                sampling_params=sampling_params,
            )
            if r_out is None:
                break
            outputs.append(r_out)

            round_text = self.tokenizer.decode(r_out.response_ids, skip_special_tokens=True)

            # Parse round response
            answer       = _parse_answer(round_text)
            verification = _parse_verification(round_text)
            prediction   = _parse_prediction(round_text)
            next_action  = _parse_next_action(round_text)

            verifications.append(verification or "")
            if next_action["action"] == "SEARCH":
                predicted_next_regions.append(
                    [float(next_action["start"]), float(next_action["end"])]
                )
            else:
                predicted_next_regions.append(None)

            search_history.append((
                start_t, end_t,
                verification[:200],
                prediction[:200] if prediction else "",
            ))
            new_ivs = _parse_intervals_block(round_text, duration)
            if new_ivs:
                intervals = _merge_intervals(intervals, new_ivs)

            if next_action["action"] == "ANSWER" and answer:
                final_answer = answer
                break
            if is_final:
                if answer:
                    final_answer = answer
                break

            # Continue searching: update next region/prediction
            if next_action["action"] == "SEARCH":
                next_start  = next_action["start"]
                next_end    = next_action["end"]
                next_frames = next_action["frames"]
                current_prediction = prediction or (
                    f"Searching [{int(next_start)}-{int(next_end)}s] for answer evidence"
                )
            else:
                # Fallback: degrade current region, pick next-best
                for iv in intervals:
                    if iv["start"] <= start_t + 5 and iv["end"] >= end_t - 5:
                        if iv["relevance"] != "ELIMINATED":
                            iv["relevance"] = "LOW"
                best_iv = _pick_best_interval(intervals)
                if best_iv:
                    next_start  = best_iv["start"]
                    next_end    = best_iv["end"]
                    next_frames = _DEFAULT_ROUND_FRAMES
                    current_prediction = (
                        f"Fallback: trying [{int(next_start)}-{int(next_end)}s] (next best interval)"
                    )
                else:
                    next_start, next_end, next_frames = 0.0, duration, _DEFAULT_ROUND_FRAMES
                    current_prediction = "No intervals left, searching full video"

        return self._finalize(
            outputs=outputs,
            final_answer=final_answer,
            gt_answer=gt_answer,
            trajectory_regions=trajectory_regions,
            predicted_next_regions=predicted_next_regions,
            verifications=verifications,
            clue_intervals=clue_intervals,
            n_rounds=n_rounds_completed,
            max_rounds=max_rounds,
            duration=duration,
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    async def _single_call_output(
        self,
        *,
        messages: list[dict],
        images: list[Image.Image] | None,
        sampling_params: dict[str, Any],
    ) -> AgentLoopOutput | None:
        """
        One isolated VLM call → one AgentLoopOutput.

        - Tokenises [system, user] into prompt_ids via apply_chat_template.
        - Calls vLLM once with a fresh request_id.
        - Wraps the response into a standalone AgentLoopOutput
          (response_mask is all-1; the entire assistant message is generated).

        Returns None if generate() failed.
        """
        request_id = uuid4().hex
        # Guard: filter out any zero-size images before sending to vLLM
        if images:
            images = [img for img in images if img.width > 0 and img.height > 0]
        if images is not None and len(images) == 0:
            logger.warning("_single_call_output: all images are zero-size, skipping")
            return None

        try:
            prompt_ids = await self.apply_chat_template(messages, images=images)
        except Exception as exc:
            logger.warning("apply_chat_template failed: %s", exc, exc_info=True)
            return None
        if not prompt_ids:
            logger.warning("_single_call_output: empty prompt_ids after chat template, skipping")
            return None

        try:
            output = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images if images else None,
            )
        except Exception as exc:
            logger.warning("generate failed: %s", exc, exc_info=True)
            return None
        if output is None:
            return None

        response_ids = list(output.token_ids)
        # Cap to response_length (matches verl convention)
        rl = self.response_length
        response_ids  = response_ids[:rl]
        response_mask = [1] * len(response_ids)

        multi_modal_data = {"images": list(images)} if images else None

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            multi_modal_data=multi_modal_data,
            reward_score=None,   # filled in _finalize on the LAST output only
            num_turns=2,
            metrics=AgentLoopMetrics(),
            extra_fields={},     # filled in _finalize on the LAST output only
        )

    def _finalize(
        self,
        *,
        outputs: list[AgentLoopOutput],
        final_answer: str,
        gt_answer: str,
        trajectory_regions: list[list[float]],
        predicted_next_regions: list,
        verifications: list[str],
        clue_intervals,
        n_rounds: int,
        max_rounds: int,
        duration: float = 0.0,
    ) -> list[AgentLoopOutput]:
        """
        Compute trajectory-level reward and attach it ONLY to the last output.

        verl's _agent_loop_postprocess (main_ppo_sync.py:402-404) will copy
        reward_score and extra_fields["reward_extra_info"] from outputs[-1]
        to every earlier output, so the whole trajectory shares the reward.
        """
        if not outputs:
            return [_empty_single_output()]

        from v9k2_reward import (  # noqa: PLC0415
            compute_trajectory_reward,
            _final_clue_iou,
            _consistency_score,
            _verify_quality_score,
            _narrowing_bonus,
        )
        reward = compute_trajectory_reward(
            final_answer=final_answer,
            gt_answer=gt_answer,
            trajectory_regions=trajectory_regions,
            predicted_next_regions=predicted_next_regions,
            verifications=verifications,
            clue_intervals=clue_intervals,
            num_rounds=max(1, n_rounds),
            max_rounds=max_rounds,
            duration=duration,
        )

        clue_iou_val   = _final_clue_iou(trajectory_regions, clue_intervals or [])
        consistency_val = _consistency_score(
            trajectory_regions, predicted_next_regions or []
        )
        verify_q_val   = _verify_quality_score(verifications or [])
        correct_val    = (
            1.0 if final_answer and final_answer.upper() == gt_answer.upper() else 0.0
        )
        # v4: narrowing is always computed (no longer short-circuited by correct).
        # Background: in v3 the "all-or-nothing" scheme zeroed every process score
        # when the answer was wrong; that short-circuit was the matching implementation.
        # v4 removed the "all-or-nothing" rule, so narrowing should faithfully reflect
        # whether the model narrowed the interval along the trajectory, independent
        # of answer correctness — otherwise reward_trace analysis is distorted.
        narrowing_val = _narrowing_bonus(
            trajectory_regions, float(duration), max(1, int(n_rounds))
        )
        # WARNING: schema must match grpo_val_reward.py exactly. Otherwise verl's
        # val aggregation (process_validation_metrics) will fill None between the
        # two paths -> np.mean(None / int) raises TypeError.
        # naive.py:90 also requires the dict to contain a "score" key.
        reward_extra_info = {
            "score":        float(reward),
            "correct":      correct_val,
            "clue_iou":     clue_iou_val,
            "narrowing":    narrowing_val,
            "consistency":  consistency_val,
            "verify_q":     verify_q_val,
            "n_rounds":     int(n_rounds),
        }

        # ──────────────────────────────────────────────────────────────────
        # Side-channel per-rollout reward log.
        # verl's main_ppo_sync.py does NOT aggregate reward_extra_info into the
        # training-step metrics (only into val). To make the distributions of
        # correct/clue_iou/narrowing/consistency/verify_q visible during training
        # too, we append a single JSON line to REWARD_TRACE_PATH per rollout.
        # Default path: grpo/logs/reward_trace.jsonl; override via env var.
        # Disable: export REWARD_TRACE_PATH=""
        # ──────────────────────────────────────────────────────────────────
        _trace_path = os.environ.get(
            "REWARD_TRACE_PATH",
            "/path/to/foresight/grpo/logs/reward_trace.jsonl",
        )
        if _trace_path:
            try:
                import json as _json
                import time as _time
                _entry = {
                    "ts": _time.time(),
                    "step": int(getattr(self, "_grpo_step", -1)),
                    "is_val": bool(getattr(self, "_grpo_validate", False)),
                    "sample_idx": int(getattr(self, "_grpo_sample_index", -1)),
                    "rollout_n": int(getattr(self, "_grpo_rollout_n", -1)),
                    "reward": float(reward),
                    "correct": float(correct_val),
                    "clue_iou": float(clue_iou_val),
                    "narrowing": float(narrowing_val),
                    "consistency": float(consistency_val),
                    "verify_q": float(verify_q_val),
                    "n_rounds": int(n_rounds),
                }
                # A single append + flush is enough: concurrent <4KB writes are atomic on POSIX
                with open(_trace_path, "a") as _f:
                    _f.write(_json.dumps(_entry) + "\n")
            except Exception as _e:
                # Log-write failures must never affect training
                logger.warning("REWARD_TRACE write failed: %s", _e)

        # Set reward and extra_fields ONLY on the last output.
        # verl will copy these to outputs[:-1] automatically.
        outputs[-1].reward_score = float(reward)
        outputs[-1].extra_fields = {"reward_extra_info": reward_extra_info}

        # Earlier outputs need extra_fields["reward_extra_info"] to exist as
        # a dict so that the broadcast assignment in main_ppo_sync.py:404
        # (output.extra_fields["reward_extra_info"] = ...) does NOT fail.
        for o in outputs[:-1]:
            if not isinstance(o.extra_fields, dict):
                o.extra_fields = {}

        # Replace the hardcoded num_turns=2 with the actual trajectory length.
        # IMPORTANT: AgentLoopOutput.as_dict() uses pydantic model_dump(exclude_unset=True),
        # which only includes fields set at construction time. Direct attribute
        # assignment (o.num_turns = X) does NOT mark the field as "set", so it
        # would be excluded from as_dict() output. We must use model_copy(update={...})
        # to produce a new AgentLoopOutput with num_turns properly registered.
        # verl aggregates this into training/num_turns/{mean,max,min}.
        # Each sample in the same trajectory carries the same value so that
        # the per-trajectory length is correctly weighted by sample count.
        outputs = [o.model_copy(update={"num_turns": int(n_rounds)}) for o in outputs]

        return outputs
