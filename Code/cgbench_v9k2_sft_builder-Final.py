from __future__ import annotations

import argparse
import base64
import json
import os
import re
import random
import subprocess
import sys
import time
import traceback
import concurrent.futures
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

# ===== Local hardcoded defaults (edit here directly) =====
DEFAULT_INPUT_JSON = "/path/to/datasets/CG-Bench/makeNewData/cgbench_train_2k.json"
DEFAULT_VIDEO_ROOT = "/path/to/datasets/CG-Bench/videos"
DEFAULT_OUTPUT_DIR = "/path/to/datasets/CG-Bench/makeNewData/output/out_sft-Final"
DEFAULT_SAMPLE = 5

DEFAULT_MODEL = "gpt-5"
DEFAULT_API_BASE = "YOUR_API_BASE_URL"
DEFAULT_API_KEY = "YOUR_API_KEY"  # Fill your key here if you don't want env/CLI
# DEFAULT_API_KEY = "YOUR_API_KEY_BACKUP"
DEFAULT_MAX_ROUNDS = 5  # Maximum trajectory rounds per sample
DEFAULT_MIN_PROBE_FRAMES = 10   # Phase1 sparse frame lower bound (matches v9k2 MIN_PROBE_FRAMES)
DEFAULT_MAX_PROBE_FRAMES = 20  # Phase1 sparse frame upper bound (matches v9k2 MAX_PROBE_FRAMES_ADAPTIVE)
DEFAULT_DENSE_FPS = 0.5  # Oracle dense-stage sampling rate (frames per second)
DEFAULT_MAX_DENSE_FRAMES = 96   # Oracle dense-stage max frames
DEFAULT_DENSE_MAX_SIDE = 576    # Oracle dense-frame max resolution (0.8x720, reduces tokens by 36%)
DEFAULT_CLUE_IOU_THR = 0.2  # IoU threshold with clue in the Oracle dense stage
DEFAULT_RESOLUTION = "720p"
DEFAULT_ORACLE_RETRIES = 3
DEFAULT_WORKERS = 5
DEFAULT_QPM = 50  # API queries-per-minute limit (gemini-3-flash)


class RateLimiter:
    """Thread-safe QPM limiter: hands out evenly spaced time slots across threads."""

    def __init__(self, qpm: float):
        self.min_interval = 60.0 / qpm if qpm and qpm > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.min_interval
            wait = start - now
        if wait > 0:
            time.sleep(wait)


# Set by main(); call_real_phase1_assess / call_oracle_plan throttle through it.
_RATE_LIMITER: Optional["RateLimiter"] = None


def parse_api_keys(single_key: str, multi_keys: str) -> List[str]:
    """Parse API keys from --api-key and --api-keys (comma/newline/semicolon separated)."""
    keys: List[str] = []

    if multi_keys:
        for part in re.split(r"[,;\n]", multi_keys):
            k = part.strip()
            if k:
                keys.append(k)

    if single_key and single_key.strip():
        keys.append(single_key.strip())

    # de-duplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for k in keys:
        if k not in seen:
            uniq.append(k)
            seen.add(k)
    return uniq


def mask_key(key: str) -> str:
    if not key:
        return "<empty>"
    if len(key) <= 8:
        return key[:2] + "***"
    return f"{key[:4]}***{key[-4:]}"


# Foresight-Uni-0601 root (auto fallback to fixed absolute path)
_ROOT_CANDIDATES = [
    Path(__file__).resolve().parent.parent,
    Path("/path/to/foresight"),
]
for _root in _ROOT_CANDIDATES:
    if (_root / "prompts_v9k2.py").exists():
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        break

import prompts_v9k2 as prompts  # noqa: E402


ORACLE_SYSTEM = """You are an expert video-QA teacher designing a search trajectory for
anticipation-guided video QA. You see dense frames, the question, clue intervals, ground
truth, and the student's real Phase 1 prediction.

You must decide EVERYTHING in the plan yourself:
- How many unified rounds (at least min_steps, at most max_rounds)
- Which time window each round views (including wrong tries and corrections)
- verification_result per round (must match what dense frames actually show)
- next_prediction text chaining round to round

═══ CORE PRINCIPLE — OBSERVATION-DRIVEN NARROWING ═══
Although you know the clue_intervals and ground truth, you MUST pretend you are
discovering the answer through genuine visual exploration. Every narrowing of the search
window MUST be justified by what was SEEN in the current step's frames — not by your
privileged knowledge of where the answer is.

The correct reasoning chain is:
  Step N: search [A–B]s → observe [concrete visual content] → verification_result: partially confirmed
  → next_prediction: "The key evidence (X) was partially visible near [timestamp], suggesting
    the answer is in the sub-region [C–D]s. I will zoom in there next."
  Step N+1: search [C–D]s ⊂ [A–B]s → observe [more specific content] → verified

NOT:
  Step N: search [A–B]s → next_prediction: "I'll check [0–5]s" ← fabricated shortcut
  Step N+1: search [0–5]s → confirmed                          ← "magic" jump using clue knowledge

The why_this_region and next_prediction fields must read as if a student is reasoning
from the frames they just saw, not as if they already know the answer location.

OBSERVATION RULES (critical — violations produce unusable training data):
- expected_observation MUST describe ONLY concrete visual content visible in the dense
  frames: objects, people, colors, text, actions, spatial relationships, UI elements.
- NEVER write intentions, plans, expectations, or reasoning in expected_observation.
  ✗ BAD: "I expect to find a transition slide", "I will look for red text",
         "The scoreline should show MIA vs CLT", "I predict the cat is on the left"
  ✓ GOOD: "A transition slide listing upcoming matches including Charlotte FC",
          "Scoreboard showing FC Dallas 2 - Inter Miami 1 in white text on black background",
          "A lady in dark blue holding a girl in pink; a tabby cat sits on the sofa to her left"
- expected_observation and verification_result MUST be logically consistent:
  * If observation describes the EXACT target scene → confirmed
  * If observation describes RELATED content but missing key detail → partially confirmed
  * If observation describes UNRELATED content or target is absent → not found
- NEVER mark "not found" when the observation clearly describes the target scene.
- NEVER mark "confirmed" when the observation only describes related but insufficient content.

Return strict JSON only with a non-empty "steps" array."""

ORACLE_USER_TEMPLATE = """Question:
{question}

Ground-truth option letter: {gt}
Answer text (for your reasoning only): {answer_text}

Clue intervals (seconds, teacher-only — student never sees these):
{clue_intervals}

--- Real Phase 1 (sparse frames, NO clue/GT) ---
Phase 1 interval assessment (model did NOT see clue/GT when producing this):
{phase1_intervals_block}
<prediction>{real_phase1_prediction}</prediction>
<query_type>{real_query_type}</query_type>
{phase1_miss_notice}
Video duration: {duration:.1f}s
You MUST use at least {min_steps} and at most {max_rounds} steps.
Even when Phase 1 already points near the clue, never collapse the search into fewer than {min_steps} rounds —
the student needs to see broad → narrowing → confirmation; that progression IS the learning signal.

Return JSON:
{{
  "query_type": "{real_query_type}",
  "steps": [
    {{
      "round": 1,
      "target_start": 0.0,
      "target_end": 120.0,
      "why_this_region": "Why search here given the Phase 1 prediction above.",
      "expected_observation": "Concrete visuals in dense frames for this window.",
      "verification_result": "not found",
      "next_prediction": "I predict [content] at [start-end]s because [reason from THIS round's observation].",
      "search_frames": 12
    }}
  ]
}}

Guidelines:
- steps.length: YOU choose between {min_steps} and {max_rounds}. MINIMUM {min_steps} steps are
  REQUIRED. The student only sees sparse Phase 1 frames; even when you already know the clue
  location, the student cannot jump straight there — it must navigate: broad Phase 1 region →
  sub-region narrowing → final confirmation. Do NOT collapse multiple narrowing hops into one
  step just because you know the answer location. Use more steps for long videos or when Phase 1
  missed the clue. Never pad with invented failed searches, never stop early when uncertain.
- Step 1 [target_start,target_end] MUST overlap the Phase 1 prediction time range, and Step 1
  should cover a Phase 1-sized segment (not a pre-narrowed small window).
- PROGRESSIVE NARROWING (no shortcuts): A search window NARROWER than the Phase 1 segment that
  contains it may ONLY appear AFTER an earlier step has already searched that broader Phase 1
  segment. The student has no reason to jump straight to a small precise window it has never
  looked at. So: search a Phase 1-sized region first; only once it is (partially) verified may
  a later step shrink to a sub-window. This holds even when Phase 1's prediction is already
  near the clue — verify the broad region first, THEN narrow. The only exception is when the
  Phase 1 segment itself is already small (≤ ~30s), in which case one step may suffice.
- verification_result must match dense frames (honest mistakes allowed on non-final steps):
  * confirmed = decisive evidence for option {gt}
  * partially confirmed = related but insufficient
  * not found = prediction not supported in this window
- Only the LAST step may be "confirmed" (or a 1-step plan with Step 1 confirmed).
- Non-last steps: verification_result not "confirmed"; next_prediction must cite a NEW time range
  and reason from THIS step's observation (wrong turns and corrections are OK).
- LOGICAL CONSISTENCY between verification_result and the next step:
  * not found: the entire current window is eliminated. The next step MUST search a region that
    does NOT overlap the just-eliminated window. Do not "narrow within" a not-found window —
    that contradicts the not-found result. Importantly, when jumping to a new unexplored
    region, ALIGN with the Phase 1 interval boundaries shown above — the student at inference
    navigates by those intervals. Prefer a next window that corresponds to one of the existing
    Phase 1 segments (or a union of adjacent ones) rather than an arbitrary narrow window that
    would only be reachable with privileged knowledge. For example, if Phase 1 has a [0-130s]
    segment and the evidence is expected somewhere in there, set the next target to cover
    [0-130s] (or a broad sub-range), NOT directly to [0-20s].
  * partially confirmed: the current window contains relevant but insufficient detail. The next
    step SHOULD either (a) narrow to a sub-region within the current window, OR (b) jump to a
    clearly different region only if the observation text explicitly identifies evidence there
    (e.g., "the sign is partially visible at the very start of the video"). Random jumps without
    observation-based justification are not allowed.
  * Both cases: the next_prediction text must explain WHY that specific region is the next
    logical target based on what the current step actually observed.
- OBSERVATION-DRIVEN NARROWING — each step must earn its sub-window:
  The reason a later step searches a narrower window MUST come from what the current step
  actually observed in its frames. You may use clue_intervals to CHOOSE which sub-region
  to zoom into, but the next_prediction and why_this_region must be written as visual
  discoveries ("I saw partial text near 45s → zooming into 40–55s"), NOT as clue reads
  ("the clue is at 45–50s so I'll check there"). A student following this trajectory
  must be able to understand WHY each window was chosen from the preceding observation.
- Final step time range should overlap clue intervals and contain answer evidence.
- FINAL STEP window size — HARD LIMITS (strictly enforced):
  * detail_recognition (color, text, object detail, person appearance): final window ≤ 30s.
    The student cannot identify a specific visual detail from sparse frames over a wide window.
    Always add an extra narrowing step rather than confirming from a wide window directly.
  * temporal_localization, causal_reasoning, counting, comparison: final window ≤ 60s.
  * global_understanding (requires seeing multiple parts of the video): final window ≤ 120s.
  If the Phase 1 prediction covers a window wider than these limits, you MUST plan intermediate
  steps to narrow down before the confirming final step — do NOT skip straight to "confirmed"
  from a wide window even when you already know the clue location.
  A student who samples 12 frames uniformly from a 200s window has only ~1 frame per 17 seconds;
  it is statistically unlikely any frame falls exactly on a 2-5 second clue interval.
- expected_observation: concrete visuals; search_frames in [4, 16]; times in [0, {duration:.1f}].
- Do NOT include phase1_coarse_prediction in JSON.
{retry_feedback}
"""


def format_timestamp(seconds: float) -> str:
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes:02d}:{secs:02d}"


def format_duration_fmt(seconds: float) -> str:
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def ffprobe_duration(video_path: str) -> float:
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        text=True,
    ).strip()
    return float(out)


def uniform_timestamps(start: float, end: float, n: int) -> List[float]:
    if n <= 1:
        return [max(start, min(end, (start + end) / 2.0))]
    step = (end - start) / (n - 1)
    return [start + i * step for i in range(n)]


def adaptive_phase1_n(duration: float, min_f: int, max_f: int) -> int:
    return min(max_f, max(min_f, min_f + int(duration / 300)))


def _extract_frame_jpeg_bytes(video_path: str, ts: float, max_side: int = 720) -> bytes:
    """Extract one frame as JPEG bytes with small backward retries for hard timestamps."""
    vf = f"scale='if(gt(iw,ih),{max_side},-2)':'if(gt(iw,ih),-2,{max_side})',format=yuvj420p"
    errors: List[str] = []

    # Some videos fail exactly at boundary timestamps; retry slightly earlier.
    for delta in (0.0, -0.2, -0.5, -1.0, -2.0):
        t = max(0.0, ts + delta)
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", f"{t:.3f}",
            "-i", video_path,
            "-vf", vf,
            "-frames:v", "1",
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-strict", "unofficial",
            "-q:v", "2",
            "pipe:1",
        ]
        try:
            img = subprocess.check_output(cmd, stderr=subprocess.PIPE)
            if img:
                return img
            errors.append(f"empty output at ts={t:.3f}")
        except subprocess.CalledProcessError as e:
            err = (e.stderr or b"").decode("utf-8", errors="ignore").strip().replace("\n", " | ")
            errors.append(f"ts={t:.3f} rc={e.returncode}: {err[:240]}")

    raise RuntimeError("frame extraction failed: " + " ; ".join(errors[:3]))


def extract_frame_file(video_path: str, ts: float, out_path: str, max_side: int = 720) -> str:
    out_path = str(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img = _extract_frame_jpeg_bytes(video_path, ts, max_side=max_side)
    with open(out_path, "wb") as f:
        f.write(img)
    return out_path


def extract_frame_b64(video_path: str, ts: float, max_side: int = 720) -> str:
    img = _extract_frame_jpeg_bytes(video_path, ts, max_side=max_side)
    return base64.b64encode(img).decode("utf-8")


def resolve_video_path(video_root: str, video_uid: str) -> str:
    for name in (video_uid, f"{video_uid}.mp4", f"{video_uid}.mkv", f"{video_uid}.webm"):
        p = os.path.join(video_root, name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"video not found under {video_root}: {video_uid}")


def parse_json_from_text(text: str) -> Dict[str, Any]:
    """Parse model JSON; never raise (bad JSON used to crash the whole batch)."""
    text = (text or "").strip()
    if not text:
        return {}

    def _loads(s: str) -> Optional[Dict[str, Any]]:
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            return None
        return obj if isinstance(obj, dict) else None

    for pat in (r"```json\s*([\s\S]*?)```", r"```\s*([\s\S]*?)```"):
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            got = _loads(m.group(1).strip())
            if got is not None:
                return got

    got = _loads(text)
    if got is not None:
        return got

    # Last resort: first {...} block (may be invalid — do not raise)
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        got = _loads(m.group(0))
        if got is not None:
            return got
    return {}


class BuildTraceLog:
    """Append-only JSONL trace (one file per run) for debugging builder steps."""

    def __init__(self, path: Optional[str]):
        self.path = path
        self._fh = None
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(path, "a", encoding="utf-8")

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def event(self, event: str, **fields: Any) -> None:
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        line = json.dumps(rec, ensure_ascii=False, default=str)
        if self._fh:
            self._fh.write(line + "\n")
            self._fh.flush()
        brief = " ".join(
            f"{k}={str(v)[:120]}" for k, v in fields.items() if k not in ("oracle_raw", "tb")
        )
        # print(f"[trace] {event}" + (f" {brief}" if brief else ""))


def response_text(resp: Any) -> str:
    content = resp.choices[0].message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                parts.append(str(c.get("text", "")))
        return "\n".join(parts).strip()
    return ""


def image_file_to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def phase1_search_window(
    phase1_prediction: str,
    intervals: List[Dict[str, Any]],
    duration: float,
) -> List[float]:
    """Primary time range implied by real Phase1 (prediction text, else best interval)."""
    ranges = extract_time_ranges_from_text(phase1_prediction, duration)
    if ranges:
        return ranges[0]
    high = [iv for iv in intervals if iv.get("relevance") == "HIGH"]
    if high:
        return [float(high[0]["start"]), float(high[0]["end"])]
    active = [
        iv for iv in intervals
        if iv.get("relevance") not in ("ELIMINATED", "LOW")
    ]
    if active:
        iv = max(active, key=lambda x: float(x["end"]) - float(x["start"]))
        return [float(iv["start"]), float(iv["end"])]
    return [0.0, min(30.0, duration)]


def segments_intersect(a: List[float], b: List[float]) -> bool:
    """True if two time intervals share any overlap (use instead of IoU for wide Phase1)."""
    return float(a[0]) < float(b[1]) and float(a[1]) > float(b[0])


def segment_contains(outer: List[float], inner: List[float]) -> bool:
    """True if inner lies entirely inside outer (e.g. clue inside wide Phase1 window)."""
    return float(outer[0]) <= float(inner[0]) and float(outer[1]) >= float(inner[1])


def step_aligns_phase1(
    step: Dict[str, Any],
    phase1_prediction: str,
    intervals: List[Dict[str, Any]],
    duration: float,
) -> bool:
    """Step1 search window must overlap Phase1 prediction range (intersection, not IoU)."""
    seg = [float(step["target_start"]), float(step["target_end"])]
    p1 = phase1_search_window(phase1_prediction, intervals, duration)
    return segments_intersect(seg, p1)


def inject_phase1_into_plan(
    plan: Dict[str, Any],
    phase1_prediction: str,
) -> Dict[str, Any]:
    """Attach real Phase1 text to plan metadata only — never rewrite step time ranges."""
    plan["phase1_coarse_prediction"] = phase1_prediction
    plan["phase1_prediction"] = phase1_prediction
    return plan


def normalize_verification_result(value: str) -> str:
    v = (value or "").strip().lower()
    if "partial" in v:
        return "partially confirmed"
    if "not" in v and "found" in v:
        return "not found"
    if "confirm" in v:
        return "confirmed"
    return v or "not found"


def _append_time_range(
    ranges: List[List[float]],
    s: float,
    e: float,
    duration: float,
) -> None:
    s = max(0.0, min(duration, float(s)))
    e = max(s + 1e-3, min(duration, float(e)))
    ranges.append([s, e])


def extract_time_ranges_from_text(text: str, duration: float) -> List[List[float]]:
    """Parse numeric time spans from prediction / next_prediction text."""
    ranges: List[List[float]] = []
    if not text:
        return ranges

    for m in re.finditer(
        r"\[?\s*(\d+(?:\.\d+)?)\s*s?\s*[-–—]\s*(\d+(?:\.\d+)?)\s*s?\]?",
        text,
        re.IGNORECASE,
    ):
        _append_time_range(ranges, float(m.group(1)), float(m.group(2)), duration)

    for m in re.finditer(
        r"\[?(\d+):(\d+)\s*[-–—]\s*(\d+):(\d+)\]?",
        text,
    ):
        s = int(m.group(1)) * 60 + int(m.group(2))
        e = int(m.group(3)) * 60 + int(m.group(4))
        _append_time_range(ranges, float(s), float(e), duration)

    for m in re.finditer(
        r"(?:between|from)\s+(\d+(?:\.\d+)?)\s*s?\s+(?:and|to)\s+(\d+(?:\.\d+)?)\s*s?",
        text,
        re.IGNORECASE,
    ):
        _append_time_range(ranges, float(m.group(1)), float(m.group(2)), duration)

    for m in re.finditer(
        r"(\d+(?:\.\d+)?)\s*(?:s|seconds)?\s+to\s+(\d+(?:\.\d+)?)\s*(?:s|seconds)?",
        text,
        re.IGNORECASE,
    ):
        _append_time_range(ranges, float(m.group(1)), float(m.group(2)), duration)

    for m in re.finditer(
        r"around\s+\[?\s*(\d+(?:\.\d+)?)\s*s?\s*[-–—]\s*(\d+(?:\.\d+)?)\s*s?\]?",
        text,
        re.IGNORECASE,
    ):
        _append_time_range(ranges, float(m.group(1)), float(m.group(2)), duration)

    return ranges


def segment_overlaps(a: List[float], b: List[float], min_ratio: float = 0.25) -> bool:
    """True if intervals overlap with meaningful intersection vs the shorter span."""
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    inter = max(0.0, e - s)
    if inter <= 0:
        return False
    short = min(a[1] - a[0], b[1] - b[0])
    return inter / max(short, 1e-3) >= min_ratio


def interval_iou(a: List[float], b: List[float]) -> float:
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    inter = max(0.0, e - s)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def make_step_prediction(
    start: float,
    end: float,
    why: str = "",
    template: str = "check",
) -> str:
    st, ed = int(start), int(end)
    reason = (why or "this interval may contain relevant evidence.").strip()[:200]
    if template == "clue":
        return (
            f"I predict decisive answer evidence at [{st}s-{ed}s] because {reason} "
            f"Check this first."
        )
    return (
        f"I predict relevant content at [{st}s-{ed}s] because {reason} "
        f"Check this first."
    )


def _short_scene_hint(text: str, max_words: int = 5) -> str:
    words = re.findall(r"[\u4e00-\u9fff]{1,8}|[A-Za-z]{2,}", (text or "").strip())
    return " ".join(words[:max_words]) if words else "evidence region"


def prediction_from_observation(
    next_start: float,
    next_end: float,
    expected_observation: str,
    why_this_region: str = "",
) -> str:
    """Build next_prediction from this round's Actual, not a generic template."""
    hint = _short_scene_hint(expected_observation)
    reason = hint if hint != "evidence region" else (
        why_this_region or "prior view was inconclusive"
    )
    return make_step_prediction(next_start, next_end, reason, template="check")


def align_phase1_prediction_to_round1(
    phase1_prediction: str,
    phase1_raw: str,
    step1: Dict[str, Any],
    plan: Dict[str, Any],
    duration: float,
) -> Tuple[str, str]:
    """
    Round1 frames come from step1 [start,end]. If Phase1 <prediction> points elsewhere,
    rewrite ONLY the <prediction> tag so To verify matches the viewed window.
    Intervals block from the real Phase1 model is left unchanged.
    """
    st = float(step1["target_start"])
    ed = float(step1["target_end"])
    step_seg = [st, ed]
    why = str(step1.get("why_this_region", "")).strip()

    pred_ranges = extract_time_ranges_from_text(phase1_prediction, duration)
    if pred_ranges and segments_intersect(pred_ranges[0], step_seg):
        return phase1_prediction, phase1_raw

    coarse = str(
        plan.get("phase1_coarse_prediction") or plan.get("phase1_prediction") or ""
    ).strip()
    if coarse:
        coarse_ranges = extract_time_ranges_from_text(coarse, duration)
        if coarse_ranges and segments_intersect(coarse_ranges[0], step_seg):
            aligned = coarse
        else:
            aligned = make_step_prediction(st, ed, why)
    else:
        aligned = make_step_prediction(st, ed, why)

    if phase1_raw and "<prediction>" in phase1_raw:
        new_raw = re.sub(
            r"<prediction>.*?</prediction>",
            f"<prediction>{aligned}</prediction>",
            phase1_raw,
            count=1,
            flags=re.DOTALL,
        )
    else:
        new_raw = phase1_raw
    return aligned, new_raw


def text_overlaps_clue(text: str, clues: List[List[float]], duration: float) -> bool:
    for seg in extract_time_ranges_from_text(text, duration):
        if segment_intersects_clue(seg, clues):
            return True
    return False


def phase1_boundary_timestamps(probe_ts: List[float], duration: float) -> List[float]:
    """Avoid duplicate 0s from [0]+probe_ts where probe_ts already starts at 0."""
    pts = sorted({0.0, float(duration)} | {float(t) for t in probe_ts if float(t) > 0})
    return pts


def parse_intervals_block_v9k2(text: str, duration: float) -> List[Dict[str, Any]]:
    m = re.search(r"<intervals>(.*?)</intervals>", text, re.DOTALL)
    if not m:
        return []

    intervals: List[Dict[str, Any]] = []
    for line in m.group(1).strip().splitlines():
        line = line.strip()
        if not line:
            continue

        range_m = re.search(r"\[(\d+)\s*s?\s*[-–—>]+\s*(\d+)\s*s?\]", line)
        if range_m:
            start = float(range_m.group(1))
            end = float(range_m.group(2))
        else:
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
                "start": start,
                "end": end,
                "relevance": "ELIMINATED",
                "scene_hint": "",
                "reason_token": "",
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


def call_real_phase1_assess(
    client: OpenAI,
    model: str,
    phase1_paths: List[str],
    phase1_timestamps: List[float],
    duration: float,
    resolution: str,
    question: str,
) -> Tuple[List[Dict[str, Any]], str, str, str]:
    """Deployment-faithful Phase1: prompts_v9k2, sparse frames, NO clue/GT."""
    content: List[Dict[str, Any]] = []
    for i, (ts, fp) in enumerate(zip(phase1_timestamps, phase1_paths)):
        content.append({"type": "text", "text": f"Frame {i + 1} at [{format_timestamp(ts)}]:"})
        content.append({"type": "image_url", "image_url": {"url": image_file_to_data_url(fp)}})

    boundary_ts = phase1_boundary_timestamps(phase1_timestamps, duration)
    phase1_user = prompts.PHASE1_USER_TEMPLATE.format(
        num_frames=len(phase1_paths),
        boundary_timestamps=", ".join(f"{int(ts)}s" for ts in boundary_ts),
        duration=int(duration),
        duration_fmt=format_duration_fmt(duration),
        resolution=resolution,
        query=question,
    )
    content.append({"type": "text", "text": phase1_user})

    effort = os.getenv("REASONING_EFFORT", "low")
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompts.PHASE1_SYSTEM},
            {"role": "user", "content": content},
        ],
        "max_completion_tokens": 4096,
    }
    if effort:
        kwargs["reasoning_effort"] = effort
    if _RATE_LIMITER:
        _RATE_LIMITER.acquire()
    api_client = client() if callable(client) else client
    resp = api_client.chat.completions.create(**kwargs)
    raw = response_text(resp)

    intervals = parse_intervals_block_v9k2(raw, duration)
    qt_m = re.search(r"<query_type>(.*?)</query_type>", raw, re.DOTALL)
    pred_m = re.search(r"<prediction>(.*?)</prediction>", raw, re.DOTALL)

    query_type = qt_m.group(1).strip().split()[0] if qt_m else "detail_recognition"
    prediction = pred_m.group(1).strip()[:500] if pred_m else ""

    # Ensure the raw stored as SFT gpt value has the required tags.
    # The model sometimes omits <query_type>; append the fallback so training
    # data always has the full expected Phase1 format.
    normalized = raw
    if not qt_m:
        normalized = normalized.rstrip() + f"\n\n<query_type>{query_type}</query_type>"
    if not pred_m and prediction:
        normalized = normalized.rstrip() + f"\n\n<prediction>{prediction}</prediction>"

    return intervals, query_type, prediction, normalized


def validate_oracle_plan_autonomous(
    plan: Dict[str, Any],
    clue_intervals: List[List[float]],
    duration: float,
    max_rounds: int,
    phase1_prediction: str = "",
    phase1_intervals: Optional[List[Dict[str, Any]]] = None,
    sub_category: str = "",
    min_steps: int = 2,
) -> Tuple[bool, str]:
    """Protocol checks only — verification text comes from Oracle plan + compile_v9k2."""
    steps = plan.get("steps") or []
    if len(steps) < min_steps:
        return False, f"too_few_steps:{len(steps)}<min:{min_steps}"
    if len(steps) > max_rounds:
        return False, f"too_many_steps:{len(steps)}>{max_rounds}"

    if phase1_prediction and not step_aligns_phase1(
        steps[0], phase1_prediction, phase1_intervals or [], duration,
    ):
        return False, "step1_misaligned_with_phase1"

    if not any(
        search_hits_clue(
            [float(s["target_start"]), float(s["target_end"])],
            clue_intervals,
        )
        for s in steps
    ):
        return False, "no_step_overlaps_clue"

    # ── Final step window hard limit ───────────────────────────────
    # Priority: sub_category (from source data) > query_type (from Oracle output)
    # Thresholds derived from p90(clue_length) + 20s on CG-Bench 12k.
    _SUBCAT_LIMITS: Dict[str, float] = {
        "2D Spatial Perception":  40.0,
        "Text Perception":        50.0,
        "Scene Perception":       50.0,
        "Entity Perception":      60.0,
        "Entity Cognition":       60.0,
        "Event Perception":       60.0,
        "Event Cognition":        60.0,
        "Time Perception":        60.0,
        "Hallucination":          60.0,
        "Text Cognition":         60.0,
        "Scene Cognition":        80.0,
        "Time Cognition":        170.0,
    }

    if sub_category and sub_category in _SUBCAT_LIMITS:
        final_window_limit = _SUBCAT_LIMITS[sub_category]
    else:
        # Fallback: derive from Oracle's query_type
        qt = str(plan.get("query_type", "detail_recognition")).strip().lower()
        if "detail" in qt or "text" in qt or "spatial" in qt:
            final_window_limit = 50.0
        elif "global" in qt or "scene" in qt or "hallucin" in qt:
            final_window_limit = 80.0
        elif "time" in qt or "temporal" in qt:
            final_window_limit = 80.0
        else:
            final_window_limit = 60.0

    # For multi-interval clues with large spread (scattered), relax to the
    # smallest individual interval + margin — hitting any one is sufficient.
    if clue_intervals and len(clue_intervals) > 1:
        min_s = min(float(iv[0]) for iv in clue_intervals)
        max_e = max(float(iv[1]) for iv in clue_intervals)
        merged_span = max_e - min_s
        if duration > 0 and merged_span > duration * 0.2:
            iv_lens = [float(iv[1]) - float(iv[0]) for iv in clue_intervals]
            final_window_limit = max(final_window_limit, min(iv_lens) + 20.0)

    last_step = steps[-1]
    last_width = float(last_step.get("target_end", 0)) - float(last_step.get("target_start", 0))
    if last_width > final_window_limit:
        return False, (
            f"final_step_window_too_wide:{last_width:.0f}s>{final_window_limit:.0f}s "
            f"(sub_category={sub_category!r})"
        )

    for i, step in enumerate(steps):
        vres = normalize_verification_result(step.get("verification_result", ""))
        is_last = i == len(steps) - 1
        if not is_last:
            if vres == "confirmed":
                return False, f"step{i+1}_confirmed_before_last"
            if not str(step.get("next_prediction", "")).strip():
                return False, f"step{i+1}_empty_next_prediction"
            nxt = steps[i + 1]
            np_ranges = extract_time_ranges_from_text(
                str(step.get("next_prediction", "")), duration,
            )
            nxt_seg = [float(nxt["target_start"]), float(nxt["target_end"])]
            if np_ranges and not segments_intersect(np_ranges[0], nxt_seg):
                return False, f"step{i+1}_next_prediction_mismatch_step{i+2}_window"

    return True, "ok"


def normalize_plan_step_light(
    step: Dict[str, Any],
    duration: float,
    is_last: bool,
    next_step: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Clamp times and normalize strings only — preserve Oracle verification labels."""
    st = float(step.get("target_start", 0))
    ed = float(step.get("target_end", min(duration, st + 10)))
    st = max(0.0, min(duration, st))
    ed = max(st + 1e-3, min(duration, ed))
    vres = normalize_verification_result(step.get("verification_result", ""))

    next_pred = str(step.get("next_prediction", "")).strip()
    if not is_last and next_step is not None:
        ns = float(next_step["target_start"])
        ne = float(next_step["target_end"])
        obs = str(step.get("expected_observation", "")).strip()
        why = str(step.get("why_this_region", "")).strip()
        if not extract_time_ranges_from_text(next_pred, duration):
            next_pred = prediction_from_observation(ns, ne, obs, why)
        elif obs and "only partially informative" in next_pred.lower():
            next_pred = prediction_from_observation(ns, ne, obs, why)

    return {
        "round": int(step.get("round", 1)),
        "target_start": st,
        "target_end": ed,
        "why_this_region": str(step.get("why_this_region", ""))[:400],
        "expected_observation": str(step.get("expected_observation", ""))[:800],
        "verification_result": vres,
        "next_prediction": next_pred if not is_last else "",
        "search_frames": normalize_search_frames(step.get("search_frames", 12)),
    }



def finalize_oracle_plan_light(
    plan: Dict[str, Any],
    duration: float,
    max_rounds: int,
) -> Dict[str, Any]:
    """Trim over-long plans and clamp fields — no clue/phase1 time overrides."""
    steps = trim_steps_if_too_long(plan.get("steps") or [], max_rounds, [])
    fixed: List[Dict[str, Any]] = []
    for i, step in enumerate(steps):
        nxt = steps[i + 1] if i + 1 < len(steps) else None
        fixed.append(normalize_plan_step_light(step, duration, i == len(steps) - 1, nxt))
    plan["steps"] = fixed[:max_rounds]
    return plan


def trim_steps_if_too_long(
    steps: List[Dict[str, Any]],
    max_n: int,
    clue_intervals: List[List[float]],
) -> List[Dict[str, Any]]:
    """Only shorten over-long Oracle plans; never pad to a step quota."""
    if len(steps) <= max_n:
        return steps
    if max_n <= 1:
        return [dict(steps[-1])]
    if max_n == 2:
        return [dict(steps[0]), dict(steps[-1])]
    mid = dict(steps[len(steps) // 2])
    return [dict(steps[0]), mid, dict(steps[-1])][:max_n]



def _phase1_hits_clue(
    phase1_intervals: List[Dict[str, Any]],
    clue_intervals: List[List[float]],
) -> bool:
    """Return True if any Phase1 interval has a time intersection with any clue interval."""
    for iv in phase1_intervals:
        seg = [iv.get("start", 0.0), iv.get("end", 0.0)]
        for c in clue_intervals:
            if max(seg[0], c[0]) < min(seg[1], c[1]):  # any intersection
                return True
    return False


def _compute_min_steps(
    duration: float,
    sub_category: str,
    phase1_intervals: List[Dict[str, Any]],
    clue_intervals: Optional[List[List[float]]] = None,
) -> int:
    """Minimum Oracle steps required — ensures training data shows genuine narrowing.

    Base rule: broad-verify → sub-narrow → confirm = 3 steps for most videos.
    Phase1-miss rule: if Phase1 completely missed the clue region the student would
    need to (a) explore the wrong region, (b) get not-found, (c) jump to a new region,
    (d) narrow, (e) confirm — that naturally needs 4+ steps and we enforce it so the
    training data includes realistic correction trajectories.
    """
    min_s = 2
    if duration > 300:
        min_s = 4
    # Tight/detail sub_categories need more narrowing before confirmation
    tight_cats = {
        "Scene Perception", "Text Perception", "2D Spatial Perception",
        "Entity Perception", "Entity Cognition",
    }
    if sub_category in tight_cats:
        min_s = max(min_s, 4)
    # If Phase 1 produced a very wide window, more hops needed to narrow down
    if phase1_intervals:
        widest = max(
            (iv.get("end", 0.0) - iv.get("start", 0.0)) for iv in phase1_intervals
        )
        if widest > 150:
            min_s = max(min_s, 4)
    # Phase1-miss: Phase1 completely missed the clue → student must first explore
    # wrong region (not_found), then redirect → inherently needs ≥5 steps.
    if (
        clue_intervals
        and phase1_intervals
        and not _phase1_hits_clue(phase1_intervals, clue_intervals)
    ):
        min_s = max(min_s, 5)
    return min_s


def obtain_oracle_plan_autonomous(
    client: OpenAI,
    model: str,
    video_path: str,
    duration: float,
    question: str,
    gt: str,
    answer_text: str,
    clue_intervals: List[List[float]],
    dense_fps: float,
    max_dense_frames: int,
    max_rounds: int,
    phase1_prediction: str,
    phase1_intervals: List[Dict[str, Any]],
    query_type: str,
    oracle_retries: int,
    qid: Any = None,
    trace: Optional[BuildTraceLog] = None,
    sub_category: str = "",
    dense_max_side: int = DEFAULT_DENSE_MAX_SIDE,
) -> Tuple[Optional[Dict[str, Any]], str, str]:
    """
    Call Oracle up to N times with validation feedback. No scaffold / no repair overrides.
    """
    min_steps = _compute_min_steps(duration, sub_category, phase1_intervals, clue_intervals)
    # Determine if Phase1 missed the clue (drives the "wrong-region correction" narrative)
    phase1_missed_clue = (
        bool(clue_intervals)
        and bool(phase1_intervals)
        and not _phase1_hits_clue(phase1_intervals, clue_intervals)
    )
    last_reason = ""
    raws: List[str] = []
    for attempt in range(max(1, oracle_retries)):
        feedback = ""
        if last_reason:
            if last_reason.startswith("empty_api_response"):
                feedback = (
                    f"\n\nRETRY {attempt + 1}: Your previous response was completely empty "
                    f"(the API returned no content). Please return a valid JSON object with "
                    f"a non-empty steps[] array containing at least {min_steps} steps."
                )
            elif last_reason.startswith("too_few_steps:") and "<min:" in last_reason:
                if phase1_missed_clue:
                    feedback = (
                        f"\n\nRETRY {attempt + 1}: Rejected — {last_reason}. "
                        f"IMPORTANT: Phase 1 predicted a region that does NOT overlap the clue "
                        f"interval. This means the student would first explore the wrong region "
                        f"(getting 'not found'), then redirect to the correct region, then narrow "
                        f"and confirm. This correction arc requires at least {min_steps} steps. "
                        f"Plan the failed exploration explicitly as Step 1 (not_found), then the "
                        f"corrective search steps after."
                    )
                else:
                    feedback = (
                        f"\n\nRETRY {attempt + 1}: Your previous JSON was rejected because: "
                        f"{last_reason}. You MUST include at least {min_steps} steps. "
                        f"Even when Phase 1 is near the clue, the student needs to see the full "
                    f"broad→narrow→confirm progression. Add the missing narrowing step(s)."
                )
            else:
                feedback = (
                    f"\n\nRETRY {attempt + 1}: Your previous JSON was rejected because: "
                    f"{last_reason}. Fix the plan; keep Step 1 overlapping Phase 1 prediction. "
                    f"Return non-empty steps[] with at least {min_steps} steps."
                )
        try:
            plan, raw = call_oracle_plan(
                client, model, video_path, duration, question, gt,
                answer_text, clue_intervals, dense_fps, max_dense_frames, max_rounds,
                phase1_prediction, query_type,
                phase1_intervals=phase1_intervals,
                retry_feedback=feedback,
                min_steps=min_steps,
                dense_max_side=dense_max_side,
                phase1_missed_clue=phase1_missed_clue,
            )
        except Exception as e:
            last_reason = f"oracle_api_error:{type(e).__name__}:{e}"
            raws.append(f"<api_error>{e}</api_error>")
            if trace:
                trace.event(
                    "oracle_api_error",
                    qid=qid,
                    attempt=attempt + 1,
                    error=str(e),
                    tb=traceback.format_exc()[-4000:],
                )
            continue

        # Detect empty API response (content filter / proxy silent failure)
        if plan.get("_empty_response"):
            finish_reason = plan.get("_finish_reason", "unknown")
            last_reason = f"empty_api_response:{finish_reason}"
            raws.append(f"<empty_response finish_reason={finish_reason}/>")
            if trace:
                trace.event(
                    "oracle_empty_response",
                    qid=qid,
                    attempt=attempt + 1,
                    finish_reason=finish_reason,
                    hint="API returned empty content; likely safety filter or proxy issue",
                )
            print(f"[qid={qid}] Oracle attempt {attempt + 1}/{oracle_retries} "
                  f"empty API response (finish_reason={finish_reason}); retrying")
            continue

        raws.append(raw)
        n_steps_raw = len(plan.get("steps") or [])
        if trace:
            trace.event(
                "oracle_attempt",
                qid=qid,
                attempt=attempt + 1,
                n_steps_raw=n_steps_raw,
                raw_preview=(raw or "")[:2000],
                raw_len=len(raw or ""),
                min_steps=min_steps,
            )
        if n_steps_raw == 0 and (raw or "").strip() and trace:
            trace.event(
                "oracle_parse_empty_steps",
                qid=qid,
                attempt=attempt + 1,
                hint="model replied but JSON had no steps[] — check raw_preview",
            )
        plan = finalize_oracle_plan_light(plan, duration, max_rounds)
        ok, reason = validate_oracle_plan_autonomous(
            plan, clue_intervals, duration, max_rounds,
            phase1_prediction, phase1_intervals,
            sub_category=sub_category,
            min_steps=min_steps,
        )
        if ok:
            if trace:
                trace.event(
                    "oracle_ok",
                    qid=qid,
                    attempt=attempt + 1,
                    n_steps=len(plan.get("steps") or []),
                    min_steps=min_steps,
                )
            return plan, "\n---\n".join(raws), reason
        last_reason = reason
        print(f"[qid={qid}] Oracle attempt {attempt + 1}/{oracle_retries} invalid: {reason}")
        if trace:
            trace.event(
                "oracle_invalid",
                qid=qid,
                attempt=attempt + 1,
                reason=reason,
                n_steps_after_finalize=len(plan.get("steps") or []),
                min_steps=min_steps,
            )
    return None, "\n---\n".join(raws), last_reason



def normalize_intervals(raw: Any, duration: float) -> List[List[float]]:
    out: List[List[float]] = []
    if not isinstance(raw, list):
        return out
    for x in raw:
        if isinstance(x, (list, tuple)) and len(x) >= 2:
            s, e = float(x[0]), float(x[1])
        elif isinstance(x, dict):
            s = float(x.get("start", x.get("start_time", 0)))
            e = float(x.get("end", x.get("end_time", s)))
        else:
            continue
        s = max(0.0, min(duration, s))
        e = max(s + 1e-3, min(duration, e))
        out.append([s, e])
    out.sort(key=lambda z: z[0])
    return out


def iou_1d(a: List[float], b: List[float]) -> float:
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    inter = max(0.0, e - s)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def overlap_clue(seg: List[float], clues: List[List[float]], thr: float = 0.2) -> bool:
    """IoU-based clue overlap (strict for narrow windows). Prefer search_hits_clue for plans."""
    return any(iou_1d(seg, c) >= thr for c in clues)


def segment_intersects_clue(seg: List[float], clues: List[List[float]]) -> bool:
    """Any time intersection between search window and a clue interval."""
    return any(segments_intersect(seg, c) for c in clues)


def search_hits_clue(seg: List[float], clues: List[List[float]]) -> bool:
    """
    Plan step overlaps or contains clue (intersection / containment).
    Use instead of overlap_clue when the step window may be much wider than clue.
    """
    return any(
        segments_intersect(seg, c) or segment_contains(seg, c) for c in clues
    )


def probe_segments(probe_ts: List[float], duration: float) -> List[List[float]]:
    boundary = phase1_boundary_timestamps(probe_ts, duration)
    segs: List[List[float]] = []
    for i in range(len(boundary) - 1):
        s, e = boundary[i], boundary[i + 1]
        if e > s:
            segs.append([s, e])
    return segs


def format_boundary_lines(segments: List[List[float]]) -> str:
    return "\n".join(
        f"Segment {i + 1}: [{int(s)}s - {int(e)}s]"
        for i, (s, e) in enumerate(segments)
    )


def normalize_relevance(value: str) -> str:
    v = (value or "MEDIUM").strip().upper()
    if v in ("HIGH", "MEDIUM", "LOW", "ELIMINATED"):
        return v
    return "MEDIUM"


def guess_reason_tokens(question: str, max_words: int = 3) -> str:
    words = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}", question)
    stop = {
        "which", "what", "when", "where", "video", "following", "correct",
        "option", "choose", "answer", "question",
    }
    picked = [w for w in words if w.lower() not in stop][:max_words]
    return " ".join(picked) if picked else "evidence"


def normalize_phase1_intervals_from_teacher(
    raw_intervals: Any,
    segments: List[List[float]],
    clue_intervals: List[List[float]],
) -> List[Dict[str, Any]]:
    if not isinstance(raw_intervals, list) or len(raw_intervals) != len(segments):
        return []
    nodes: List[Dict[str, Any]] = []
    for seg, item in zip(segments, raw_intervals):
        if not isinstance(item, dict):
            return []
        rel = normalize_relevance(str(item.get("relevance", "MEDIUM")))
        hint = str(item.get("scene_hint", "")).strip()[:60]
        rt = str(item.get("reason_token", "")).strip()[:40]
        if rel != "ELIMINATED" and (not hint or not rt):
            return []
        nodes.append({
            "start": float(seg[0]),
            "end": float(seg[1]),
            "relevance": rel,
            "scene_hint": hint,
            "reason_token": rt,
        })
    if clue_intervals and not any(
        n["relevance"] == "HIGH" and segment_intersects_clue([n["start"], n["end"]], clue_intervals)
        for n in nodes
    ):
        for n in nodes:
            if segment_intersects_clue([n["start"], n["end"]], clue_intervals):
                n["relevance"] = "HIGH"
                break
    return nodes


def build_intervals_fallback(
    segments: List[List[float]],
    clue_intervals: List[List[float]],
    plan: Dict[str, Any],
    question: str,
) -> List[Dict[str, Any]]:
    """Rule fallback when Phase1 teacher JSON is invalid."""
    steps = plan.get("steps") or []
    clue_hint = ""
    if steps:
        clue_hint = str(steps[0].get("expected_observation", "")).strip()[:50]
    default_rt = guess_reason_tokens(question)
    nodes: List[Dict[str, Any]] = []
    for seg in segments:
        if segment_intersects_clue(seg, clue_intervals):
            rel = "HIGH"
            hint = clue_hint or "evidence region activity"
        elif clue_intervals and seg[1] <= clue_intervals[0][0] + 1:
            rel = "LOW"
            hint = "before evidence region"
        else:
            rel = "MEDIUM"
            hint = "sparse boundary unclear"
        nodes.append({
            "start": float(seg[0]),
            "end": float(seg[1]),
            "relevance": rel,
            "scene_hint": hint,
            "reason_token": default_rt,
        })
    return nodes


def normalize_search_frames(value: Any, default_frames: int = 12) -> int:
    """Match v9k2 _parse_next_action frame policy: default 12, clamp to [4, 16]."""
    try:
        n = int(value)
    except Exception:
        n = int(default_frames)
    return max(4, min(16, n))


def build_phase1_belief_text(
    query_type: str,
    prediction: str,
    intervals: List[Dict[str, Any]],
) -> str:
    lines = ["<intervals>"]
    for iv in intervals:
        lines.append(
            f"[{int(iv['start'])}s-{int(iv['end'])}s] relevance={iv['relevance']} "
            f"scene_hint={iv.get('scene_hint', '')} reason_token={iv.get('reason_token', '')}"
        )
    lines.append("</intervals>")
    lines.append("")
    lines.append(f"<prediction>{prediction}</prediction>")
    lines.append("")
    lines.append(f"<query_type>{query_type}</query_type>")
    return "\n".join(lines)


def compile_v9k2_responses(
    plan: Dict[str, Any],
    phase1_prediction: str,
    gt: str,
    duration: float,
    phase1_intervals: List[Dict[str, Any]],
) -> Tuple[
    List[str],
    List[List[float]],
    List[List[float]],
    List[int],
    List[List[Dict[str, Any]]],
    List[Dict[str, Any]],
]:
    """
    Build v9k2 XML per unified round (matches prompts_v9k2 + foresight_agent_v9k2).

    - Non-last: <verification> + <prediction> + <search> + <intervals>
    - Last: <verification> + <answer>
    - Belief list merged each round; returned for per-round SFT user prompts.
    """
    steps = plan.get("steps") or []
    if not steps:
        clue = plan.get("_clue_fallback", [[0, min(30.0, duration)]])
        steps = [{
            "target_start": clue[0][0],
            "target_end": clue[0][1],
            "expected_observation": "Evidence visible in clue region.",
            "verification_result": "confirmed",
            "next_prediction": "",
            "search_frames": 12,
        }]

    responses: List[str] = []
    regions: List[List[float]] = []
    ts_per_round: List[List[float]] = []
    frames_per_round: List[int] = []
    beliefs_before_round: List[List[Dict[str, Any]]] = []

    belief = [dict(iv) for iv in phase1_intervals]
    current_prediction = phase1_prediction
    for i, step in enumerate(steps):
        beliefs_before_round.append([dict(iv) for iv in belief])

        st = float(step["target_start"])
        ed = float(step["target_end"])
        st = max(0.0, min(duration, st))
        ed = max(st + 1e-3, min(duration, ed))
        n_frames = normalize_search_frames(step.get("search_frames", 12))
        cur_ts = uniform_timestamps(st, ed, n_frames)

        actual = str(
            step.get("actual_observation") or step.get("expected_observation", ""),
        ).strip()
        vres = normalize_verification_result(step.get("verification_result", ""))
        is_last = i == len(steps) - 1

        verif = (
            f"Expected: {current_prediction}\n"
            f"Actual: {actual}\n"
            f"Result: {vres}"
        )

        if is_last:
            resp = f"<verification>{verif}</verification>\n<answer>{gt}</answer>"
        else:
            next_step = steps[i + 1]
            next_pred = str(step.get("next_prediction") or "").strip()
            ns = float(next_step["target_start"])
            ne = float(next_step["target_end"])
            if not next_pred or "only partially informative" in next_pred.lower():
                next_pred = prediction_from_observation(
                    ns,
                    ne,
                    actual,
                    str(step.get("why_this_region", "")),
                )
            search_st = max(0.0, min(duration, ns))
            search_ed = max(search_st + 1e-3, min(duration, ne))
            n_search = normalize_search_frames(next_step.get("search_frames", 12))

            hint_src = (
                str(next_step.get("why_this_region", ""))
                or str(next_step.get("expected_observation", ""))
            )
            interval_updates = build_teacher_interval_updates(
                st, ed, vres, search_st, search_ed, hint_src,
            )
            intervals_xml = format_intervals_xml(interval_updates)

            resp = (
                f"<verification>{verif}</verification>\n"
                f"<prediction>{next_pred}</prediction>\n"
                f"<search>[{int(search_st)}s-{int(search_ed)}s] frames={n_search}</search>\n"
                f"{intervals_xml}"
            )
            current_prediction = next_pred

            belief = apply_viewed_to_belief(belief, [st, ed], vres)
            belief = merge_intervals_belief(belief, interval_updates)

        responses.append(resp)
        regions.append([st, ed])
        ts_per_round.append(cur_ts)
        frames_per_round.append(n_frames)

    return responses, regions, ts_per_round, frames_per_round, beliefs_before_round, belief


def build_search_history(
    regions: List[List[float]],
    responses: List[str],
    upto: int,
) -> str:
    if upto <= 0:
        return "  (first round — verifying Phase 1 prediction)"
    lines = []
    for i in range(upto):
        st, ed = regions[i]
        verif = ""
        pred = ""
        m = re.search(r"<verification>(.*?)</verification>", responses[i], re.DOTALL)
        if m:
            verif = m.group(1).strip()[:200]
        m2 = re.search(r"<prediction>(.*?)</prediction>", responses[i], re.DOTALL)
        if m2:
            pred = m2.group(1).strip()[:200]
        line = f"  Round {i+1}: viewed [{int(st)}-{int(ed)}s]"
        if verif:
            line += f"\n    Verified: {verif}"
        if pred:
            line += f"\n    Predicted: {pred}"
        lines.append(line)
    return "\n".join(lines)


def merge_intervals_belief(
    intervals: List[Dict[str, Any]],
    new_intervals: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Same merge policy as foresight_agent_v9k2 unified loop."""
    if not new_intervals:
        return [dict(iv) for iv in intervals]
    updated_ranges = {(int(n["start"]), int(n["end"])) for n in new_intervals}
    merged: List[Dict[str, Any]] = []
    for old_iv in intervals:
        old_key = (int(old_iv["start"]), int(old_iv["end"]))
        if old_key not in updated_ranges:
            merged.append(dict(old_iv))
    for niv in new_intervals:
        merged.append(dict(niv))
    merged.sort(key=lambda iv: iv["start"])
    return merged


def apply_viewed_to_belief(
    belief: List[Dict[str, Any]],
    viewed: List[float],
    vres: str,
) -> List[Dict[str, Any]]:
    """Downgrade segments that heavily overlap the region just verified."""
    out: List[Dict[str, Any]] = []
    for iv in belief:
        node = dict(iv)
        if segment_overlaps([node["start"], node["end"]], viewed, min_ratio=0.35):
            if vres == "not found":
                node["relevance"] = "ELIMINATED"
                node["scene_hint"] = node.get("scene_hint") or "explored not found"
            elif vres == "partially confirmed" and node.get("relevance") == "HIGH":
                node["relevance"] = "MEDIUM"
        out.append(node)
    return out


def build_teacher_interval_updates(
    viewed_start: float,
    viewed_end: float,
    vres: str,
    next_start: float,
    next_end: float,
    hint_source: str,
) -> List[Dict[str, Any]]:
    """Intervals emitted in this round's <intervals> tag (then merged into belief).

    Only reflects what can be inferred from the CURRENT window's result —
    does NOT inject Oracle's private next-step coordinates as HIGH.
    Navigation guidance is carried by <prediction> + <search> tags instead.
    """
    updates: List[Dict[str, Any]] = []
    vs, ve = viewed_start, viewed_end

    if vres == "not found":
        updates.append({
            "start": vs,
            "end": ve,
            "relevance": "ELIMINATED",
            "scene_hint": "searched not found",
            "reason_token": "",
        })
    elif vres == "partially confirmed":
        updates.append({
            "start": vs,
            "end": ve,
            "relevance": "MEDIUM",
            "scene_hint": _short_scene_hint(hint_source),
            "reason_token": "",
        })
        # Navigation to the next region is carried by <prediction> + <search> tags.
        # Do NOT inject Oracle's next-step coordinates here — the model cannot derive
        # a precise sub-region from frames without privileged clue/GT knowledge.
    else:
        updates.append({
            "start": vs,
            "end": ve,
            "relevance": "LOW",
            "scene_hint": "explored",
            "reason_token": "",
        })

    return updates


def format_intervals_xml(intervals: List[Dict[str, Any]]) -> str:
    """v9k2 unified-loop <intervals> block (assistant turn)."""
    lines = ["<intervals>"]
    for iv in intervals:
        rel = iv.get("relevance", "MEDIUM")
        hint = (iv.get("scene_hint") or "").strip()
        if rel == "ELIMINATED":
            line = f"[{int(iv['start'])}-{int(iv['end'])}] relevance=ELIMINATED"
            if hint:
                line += f" scene_hint={hint}"
        else:
            line = f"[{int(iv['start'])}-{int(iv['end'])}] relevance={rel}"
            if hint:
                line += f" scene_hint={hint}"
        lines.append(line)
    lines.append("</intervals>")
    return "\n".join(lines)


def format_intervals_block(intervals: List[Dict[str, Any]]) -> str:
    if not intervals:
        return "(no intervals)"
    lines = []
    for iv in intervals:
        if iv.get("relevance") == "ELIMINATED":
            lines.append(f"  [{int(iv['start'])}-{int(iv['end'])}s] ELIMINATED")
        else:
            hint = f" scene_hint={iv['scene_hint']}" if iv.get("scene_hint") else ""
            lines.append(
                f"  [{int(iv['start'])}-{int(iv['end'])}s] relevance={iv['relevance']}{hint}"
            )
    return "\n".join(lines)


def call_oracle_plan(
    client: OpenAI,
    model: str,
    video_path: str,
    duration: float,
    question: str,
    gt: str,
    answer_text: str,
    clue_intervals: List[List[float]],
    dense_fps: float,
    max_dense_frames: int,
    max_rounds: int,
    real_phase1_prediction: str,
    real_query_type: str,
    phase1_intervals: Optional[List[Dict[str, Any]]] = None,
    retry_feedback: str = "",
    min_steps: int = 2,
    dense_max_side: int = DEFAULT_DENSE_MAX_SIDE,
    phase1_missed_clue: bool = False,
) -> Tuple[Dict[str, Any], str]:
    n = min(max_dense_frames, max(8, int(duration * dense_fps)))
    ts = uniform_timestamps(0.0, max(0.0, duration - 1e-3), n)

    content: List[Dict[str, Any]] = [{"type": "text", "text": "Dense global keyframes (teacher)."}]
    for t in ts:
        b64 = extract_frame_b64(video_path, t, max_side=dense_max_side)
        content.append({"type": "text", "text": f"[t={t:.2f}s]"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    p1_ivs = phase1_intervals or []
    if p1_ivs:
        iv_lines = [
            f"[{int(iv['start'])}s-{int(iv['end'])}s] relevance={iv.get('relevance','MEDIUM')}"
            + (f" scene_hint={iv['scene_hint']}" if iv.get("scene_hint") else "")
            for iv in p1_ivs
        ]
        phase1_intervals_block = "\n".join(iv_lines)
    else:
        phase1_intervals_block = "(no interval data)"

    # Build optional Phase1-miss notice injected into the prompt
    if phase1_missed_clue:
        phase1_miss_notice = (
            "\n⚠️  PHASE 1 MISSED THE CLUE REGION: The Phase 1 intervals above do NOT overlap "
            "the clue intervals. This means the student's initial prediction was wrong. "
            f"You MUST model the correction arc explicitly — at minimum {min_steps} steps: "
            "Step 1 searches the Phase1 region and gets 'not found' or 'partially confirmed' "
            "without hitting the clue; later steps redirect to the actual clue region and "
            "narrow down. Do NOT skip the failed exploration — it is the learning signal.\n"
        )
    else:
        phase1_miss_notice = ""

    user_text = ORACLE_USER_TEMPLATE.format(
        question=question,
        gt=gt,
        answer_text=answer_text or gt,
        clue_intervals=json.dumps(clue_intervals),
        duration=duration,
        max_rounds=max_rounds,
        min_steps=min_steps,
        phase1_intervals_block=phase1_intervals_block,
        real_phase1_prediction=real_phase1_prediction[:1200],
        real_query_type=real_query_type or "detail_recognition",
        retry_feedback=retry_feedback or "",
        phase1_miss_notice=phase1_miss_notice,
    )
    content.append({"type": "text", "text": user_text})

    if _RATE_LIMITER:
        _RATE_LIMITER.acquire()
    api_client = client() if callable(client) else client
    resp = api_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ORACLE_SYSTEM},
            {"role": "user", "content": content},
        ],
        max_completion_tokens=16384,
    )
    text = response_text(resp)
    # Capture finish_reason to diagnose empty-content responses
    try:
        finish_reason = str(resp.choices[0].finish_reason or "unknown")
    except Exception:
        finish_reason = "unknown"
    if not text.strip():
        plan: Dict[str, Any] = {
            "steps": [],
            "_finish_reason": finish_reason,
            "_empty_response": True,
        }
        plan.setdefault("query_type", real_query_type or "detail_recognition")
        plan.setdefault("final_answer", gt)
        return plan, ""
    plan = parse_json_from_text(text)
    plan.setdefault("query_type", real_query_type or "detail_recognition")
    plan["phase1_coarse_prediction"] = real_phase1_prediction
    plan["phase1_prediction"] = real_phase1_prediction
    plan.setdefault("steps", [])
    plan.setdefault("final_answer", gt)

    fixed = []
    for i, s in enumerate(plan["steps"][:max_rounds], start=1):
        st = float(s.get("target_start", 0))
        ed = float(s.get("target_end", min(duration, st + 10)))
        st = max(0.0, min(duration, st))
        ed = max(st + 1e-3, min(duration, ed))
        fixed.append({
            "round": i,
            "target_start": st,
            "target_end": ed,
            "why_this_region": str(s.get("why_this_region", ""))[:400],
            "expected_observation": str(s.get("expected_observation", ""))[:800],
            "verification_result": normalize_verification_result(
                s.get("verification_result", "not found"),
            ),
            "next_prediction": str(s.get("next_prediction", ""))[:400],
            "search_frames": normalize_search_frames(s.get("search_frames", 12)),
        })
    plan["steps"] = fixed
    return plan, text


def save_frames_for_turn(
    video_path: str,
    frame_dir: str,
    prefix: str,
    timestamps: List[float],
) -> List[str]:
    paths = []
    for i, ts in enumerate(timestamps):
        p = os.path.join(frame_dir, f"{prefix}_{i:02d}_{ts:.2f}s.jpg")
        extract_frame_file(video_path, ts, p)
        paths.append(p)
    return paths


def build_llamafactory_user(images: List[str], text: str) -> str:
    """Qwen-VL / LLaMA-Factory style: <image>path per line + instruction."""
    parts = [f"<image>{p}" for p in images]
    parts.append(text)
    return "\n".join(parts)


def build_sft_turns(
    question: str,
    duration: float,
    resolution: str,
    phase1_paths: List[str],
    phase1_ts: List[float],
    phase1_assistant: str,
    intervals: List[Dict[str, Any]],
    phase1_prediction: str,
    responses: List[str],
    regions: List[List[float]],
    ts_per_round: List[List[float]],
    round_image_paths: List[List[str]],
    beliefs_before_round: List[List[Dict[str, Any]]],
    max_rounds: int = 5,
) -> List[Dict[str, Any]]:
    turns: List[Dict[str, Any]] = []
    duration_fmt = format_duration_fmt(duration)
    boundary_ts = phase1_boundary_timestamps(phase1_ts, duration)

    phase1_user = prompts.PHASE1_USER_TEMPLATE.format(
        num_frames=len(phase1_paths),
        boundary_timestamps=", ".join(f"{int(ts)}s" for ts in boundary_ts),
        duration=int(duration),
        duration_fmt=duration_fmt,
        resolution=resolution,
        query=question,
    )
    turns.append({
        "stage": "phase1",
        "system": prompts.PHASE1_SYSTEM,
        "user": build_llamafactory_user(phase1_paths, phase1_user),
        "assistant": phase1_assistant,
        "images": phase1_paths,
    })

    for r, (resp, region, r_ts, r_paths) in enumerate(
        zip(responses, regions, ts_per_round, round_image_paths)
    ):
        st, ed = region
        # Match agent: FINAL template only at round max_rounds (0-based: max_rounds-1)
        # so the model learns to answer when confirmed from a regular prompt (not just FINAL)
        is_final = (r == max_rounds - 1)
        history = build_search_history(regions, responses, r)
        verify_pred = phase1_prediction if r == 0 else ""
        if r > 0:
            m = re.search(r"<prediction>(.*?)</prediction>", responses[r - 1], re.DOTALL)
            if m:
                verify_pred = m.group(1).strip()

        if is_final:
            user_text = prompts.UNIFIED_USER_TEMPLATE_FINAL.format(
                duration=int(duration),
                duration_fmt=duration_fmt,
                start_fmt=format_timestamp(st),
                end_fmt=format_timestamp(ed),
                current_prediction=verify_pred,
                search_history=history,
                query=question,
            )
        else:
            belief_for_user = (
                beliefs_before_round[r]
                if r < len(beliefs_before_round)
                else intervals
            )
            intervals_block = format_intervals_block(belief_for_user)
            user_text = prompts.UNIFIED_USER_TEMPLATE.format(
                duration=int(duration),
                duration_fmt=duration_fmt,
                start_fmt=format_timestamp(st),
                end_fmt=format_timestamp(ed),
                intervals_block=intervals_block,
                current_prediction=verify_pred,
                search_history=history,
                query=question,
            )

        turns.append({
            "stage": f"unified_round_{r+1}",
            "system": prompts.UNIFIED_SYSTEM,
            "user": build_llamafactory_user(r_paths, user_text),
            "assistant": resp,
            "images": r_paths,
        })

    return turns


def load_input(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    return [data]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="CG-Bench -> v9k2 SFT (Final: Phase1 + Oracle, 2 API calls/sample)",
    )
    ap.add_argument("--input-json", default=DEFAULT_INPUT_JSON, help="JSON array or JSONL of CG samples")
    ap.add_argument("--video-root", default=DEFAULT_VIDEO_ROOT)
    ap.add_argument(
        "--output-dir",
        default=(os.getenv("OUTPUT_DIR") or DEFAULT_OUTPUT_DIR),
        help="Writes v9k2_sft.jsonl + v9k2_result.jsonl (env OUTPUT_DIR overrides default)",
    )
    ap.add_argument("--sample", type=int, default=DEFAULT_SAMPLE, help="0 = all")
    ap.add_argument("--seed", type=int, default=42)

    # Prefer runtime env override; fallback to code default.
    ap.add_argument("--model", default=(os.getenv("MODEL") or os.getenv("GEMINI_MODEL") or DEFAULT_MODEL))
    ap.add_argument("--api-base", default=DEFAULT_API_BASE)
    ap.add_argument("--api-key", default=(DEFAULT_API_KEY or os.getenv("OPENAI_API_KEY", "")))
    ap.add_argument(
        "--api-keys",
        default=os.getenv("OPENAI_API_KEYS", ""),
        help="Multiple API keys separated by comma/semicolon/newline; used for cost/load split.",
    )
    ap.add_argument(
        "--api-key-strategy",
        choices=["random", "round_robin"],
        default=os.getenv("API_KEY_STRATEGY", "random"),
        help="How to select key per API call when multiple keys are provided.",
    )

    ap.add_argument("--dense-fps", type=float, default=DEFAULT_DENSE_FPS)
    ap.add_argument("--max-dense-frames", type=int, default=DEFAULT_MAX_DENSE_FRAMES)
    ap.add_argument("--dense-max-side", type=int, default=DEFAULT_DENSE_MAX_SIDE,
                    help="Max pixel side for Oracle dense frames (default 576 = 0.8×720, "
                         "saves ~36%% tokens vs 720p with negligible quality loss)")
    ap.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    ap.add_argument("--min-probe-frames", type=int, default=DEFAULT_MIN_PROBE_FRAMES)
    ap.add_argument("--max-probe-frames", type=int, default=DEFAULT_MAX_PROBE_FRAMES)
    ap.add_argument("--clue-iou-thr", type=float, default=DEFAULT_CLUE_IOU_THR)
    ap.add_argument("--resolution", default=DEFAULT_RESOLUTION)
    ap.add_argument(
        "--oracle-retries",
        type=int,
        default=int(os.getenv("ORACLE_RETRIES", DEFAULT_ORACLE_RETRIES)),
        help="Oracle JSON retries before skip (no scaffold)",
    )
    ap.add_argument(
        "--log-file",
        default="",
        help="JSONL trace log (default: <output-dir>/build_trace.jsonl)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("WORKERS", DEFAULT_WORKERS)),
        help="Parallel workers (thread pool; each worker makes 2 API calls)",
    )
    ap.add_argument(
        "--qpm",
        type=float,
        default=float(os.getenv("QPM", DEFAULT_QPM)),
        help="API queries-per-minute cap (0 = unlimited). Each sample = 2 queries.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        default=os.getenv("RESUME", "").lower() in ("1", "true", "yes"),
        help="Skip qids already completed (ok) in existing v9k2_result.jsonl; append output.",
    )
    ap.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Force overwrite (default unless --resume).",
    )

    args = ap.parse_args()
    random.seed(args.seed)

    api_keys = parse_api_keys(args.api_key, args.api_keys)
    if not api_keys:
        raise SystemExit("OPENAI_API_KEY is empty (provide --api-key or --api-keys)")

    global _RATE_LIMITER
    # qpm means TOTAL cap across all keys to avoid accidental overrun.
    _RATE_LIMITER = RateLimiter(args.qpm)

    items = load_input(args.input_json)
    if args.sample > 0:
        items = items[: args.sample]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sft_path = out_dir / "v9k2_sft.jsonl"
    result_path = out_dir / "v9k2_result.jsonl"
    frames_root = out_dir / "frames"
    log_path = args.log_file.strip() or str(out_dir / "build_trace.jsonl")

    # ── Resume: collect already-completed qids and keep their existing output ──
    # qid fallback uses the stable post-sample index, so indexing is fixed across runs.
    indexed_items = list(enumerate(items))
    done_qids: set = set()
    file_mode = "w"
    if args.resume and result_path.exists():
        for line in result_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Only treat fully-built samples as done; skip/fail records are retried.
            if "skip_reason" not in rec and "error" not in rec:
                done_qids.add(rec.get("qid"))
        before = len(indexed_items)
        indexed_items = [
            (i, it) for (i, it) in indexed_items
            if it.get("qid", i) not in done_qids
        ]
        file_mode = "a"
        print(f"[resume] {len(done_qids)} qids already done; "
              f"{before - len(indexed_items)} skipped, {len(indexed_items)} to build.")

    trace = BuildTraceLog(log_path)
    trace_lock = threading.Lock()

    # Multi-key client scheduler (thread-safe)
    key_lock = threading.Lock()
    key_idx = 0
    key_usage = {k: 0 for k in api_keys}

    def get_client() -> OpenAI:
        nonlocal key_idx
        if len(api_keys) == 1:
            k = api_keys[0]
        else:
            with key_lock:
                if args.api_key_strategy == "round_robin":
                    k = api_keys[key_idx % len(api_keys)]
                    key_idx += 1
                else:  # random
                    k = random.choice(api_keys)
                key_usage[k] += 1
        return OpenAI(api_key=k, base_url=args.api_base)

    trace.event(
        "run_start",
        model=args.model,
        sample=args.sample,
        max_rounds=args.max_rounds,
        oracle_retries=args.oracle_retries,
        n_items=len(indexed_items),
        n_done_resumed=len(done_qids),
        api_calls_per_sample=2,
        workers=args.workers,
        qpm=args.qpm,
        resume=args.resume,
        log_file=log_path,
        n_api_keys=len(api_keys),
        api_key_strategy=args.api_key_strategy,
        api_keys_masked=[mask_key(k) for k in api_keys],
    )

    # ── Per-item worker ──────────────────────────────────────────────────────
    def _work(idx_item: tuple) -> Dict[str, Any]:
        """Run Phase1 + Oracle for one sample. Returns a result dict."""
        i, item = idx_item
        qid = item.get("qid", i)
        try:
            with trace_lock:
                trace.event("sample_start", qid=qid, index=i)
            uid = str(item.get("video_uid", "")).strip()
            question = str(
                item.get("question_with_choices") or item.get("question") or ""
            ).strip()
            gt = str(item.get("right_answer", "")).strip().upper()
            answer_text = str(item.get("answer", "")).strip()

            video_path = resolve_video_path(args.video_root, uid)
            video_file = os.path.basename(video_path)
            duration = float(item.get("duration") or ffprobe_duration(video_path))
            clue_intervals = normalize_intervals(item.get("clue_intervals", []), duration)

            if not question or not gt or not clue_intervals:
                return {"status": "skip", "qid": qid, "uid": uid,
                        "skip_reason": "missing_fields", "sft_turns": [], "record": None}

            p1_n = adaptive_phase1_n(duration, args.min_probe_frames, args.max_probe_frames)
            phase1_ts = uniform_timestamps(0.0, max(0.0, duration - 1e-3), p1_n)
            frame_dir = str(frames_root / f"q{qid}_{uid}")
            phase1_paths = save_frames_for_turn(video_path, frame_dir, "phase1", phase1_ts)

            intervals, query_type, phase1_prediction, phase1_raw = call_real_phase1_assess(
                client=get_client,
                model=args.model,
                phase1_paths=phase1_paths,
                phase1_timestamps=phase1_ts,
                duration=duration,
                resolution=args.resolution,
                question=question,
            )
            if not intervals or not phase1_prediction:
                with trace_lock:
                    trace.event("skip", qid=qid, reason="phase1_parse_empty")
                return {"status": "skip", "qid": qid, "uid": uid,
                        "skip_reason": "phase1_parse_empty", "sft_turns": [],
                        "record": {"qid": qid, "skip_reason": "phase1_parse_empty"}}
            with trace_lock:
                trace.event("phase1_ok", qid=qid, n_intervals=len(intervals),
                            query_type=query_type, prediction_preview=phase1_prediction[:200])

            plan, raw_oracle, validate_reason = obtain_oracle_plan_autonomous(
                get_client, args.model, video_path, duration, question, gt,
                answer_text, clue_intervals,
                args.dense_fps, args.max_dense_frames, args.max_rounds,
                phase1_prediction, intervals, query_type, args.oracle_retries,
                qid=qid, trace=trace,
                sub_category=str(item.get("sub_category", "")),
                dense_max_side=args.dense_max_side,
            )
            if plan is None:
                print(f"[qid={qid}] Oracle failed: {validate_reason}")
                with trace_lock:
                    trace.event("skip", qid=qid,
                                reason=f"oracle_failed:{validate_reason}",
                                oracle_raw_preview=(raw_oracle or "")[:1500])
                return {"status": "skip", "qid": qid, "uid": uid,
                        "skip_reason": f"oracle_failed:{validate_reason}", "sft_turns": [],
                        "record": {"qid": qid,
                                   "skip_reason": f"oracle_failed:{validate_reason}",
                                   "teacher_metadata": {"oracle_raw": raw_oracle}}}

            plan["_clue_fallback"] = clue_intervals
            plan = inject_phase1_into_plan(plan, phase1_prediction)
            plan_source = "oracle_unified"

            steps = plan.get("steps") or []
            if not steps:
                return {"status": "skip", "qid": qid, "uid": uid,
                        "skip_reason": "empty_plan_steps", "sft_turns": [],
                        "record": {"qid": qid, "skip_reason": "empty_plan_steps"}}

            round_paths: List[List[str]] = []
            ts_per_round: List[List[float]] = []
            for r, step in enumerate(steps):
                st = float(step["target_start"])
                ed = float(step["target_end"])
                st = max(0.0, min(duration, st))
                ed = max(st + 1e-3, min(duration, ed))
                n_frames = normalize_search_frames(step.get("search_frames", 12))
                r_ts = uniform_timestamps(st, ed, n_frames)
                ts_per_round.append(r_ts)
                round_paths.append(
                    save_frames_for_turn(video_path, frame_dir, f"round{r+1}", r_ts)
                )

            responses, regions, ts_per_round, frames_per_round, beliefs_before, final_belief = (
                compile_v9k2_responses(plan, phase1_prediction, gt, duration, intervals)
            )

            sft_turns_raw = build_sft_turns(
                question, duration, args.resolution,
                phase1_paths, phase1_ts, phase1_raw, intervals,
                phase1_prediction, responses, regions, ts_per_round, round_paths,
                beliefs_before,
                max_rounds=args.max_rounds,
            )

            total_frames = len(phase1_paths) + sum(len(x) for x in round_paths)
            v9k2_record = {
                "qid": qid, "index": qid, "videos": [video_file],
                "query": question, "gt": gt, "answer": gt,
                "stop_reason": "answer_found",
                "num_rounds": len(responses),
                "phase1_belief_raw_output": phase1_raw,
                "phase2_belief_nodes": final_belief,
                "trajectory": {
                    "phase1_probe_timestamps": [round(float(t), 3) for t in phase1_ts],
                    "phase3_regions_per_round": regions,
                    "phase3_timestamps_per_round": [
                        [round(float(t), 3) for t in ts] for ts in ts_per_round
                    ],
                    "phase3_responses_per_round": responses,
                },
                "frame_stats": {
                    "phase1_frames": len(phase1_paths),
                    "phase3_frames_per_round": [len(x) for x in round_paths],
                    "total_frames": total_frames,
                },
                "teacher_metadata": {
                    "clue_intervals": clue_intervals,
                    "oracle_plan": plan,
                    "oracle_raw_output": raw_oracle,
                    "plan_source": plan_source,
                    "plan_validation": validate_reason,
                    "num_plan_steps": len(steps),
                    "num_unified_rounds": len(responses),
                    "phase1_source": "real_v9k2_prompt_no_clue",
                    "builder_version": "Final-unified",
                    "api_calls": 2,
                    "oracle_retries": args.oracle_retries,
                    "phase1_rewrite_for_round1": False,
                    "sft_v9k2_interval_updates": True,
                },
            }

            sft_lines = []
            for turn in sft_turns_raw:
                sft_lines.append({
                    "qid": qid, "video_uid": uid, "video": video_file,
                    "stage": turn["stage"], "gt": gt,
                    "conversations": [
                        {"from": "system", "value": turn["system"]},
                        {"from": "human", "value": turn["user"]},
                        {"from": "gpt", "value": turn["assistant"]},
                    ],
                    "images": turn["images"],
                })

            with trace_lock:
                trace.event("sample_ok", qid=qid, plan_source=plan_source,
                            num_rounds=len(responses))
            return {"status": "ok", "qid": qid, "uid": uid,
                    "sft_turns": sft_lines, "record": v9k2_record}

        except Exception as e:
            tb = traceback.format_exc()
            with trace_lock:
                trace.event("exception", qid=qid,
                            error=f"{type(e).__name__}: {e}", tb=tb[-6000:])
            print(f"[qid={qid}] EXCEPTION: {e}")
            return {"status": "fail", "qid": qid, "uid": "",
                    "sft_turns": [], "record": {
                        "qid": qid, "error": f"{type(e).__name__}: {e}",
                        "traceback": tb[-8000:],
                    }}

    def process_one(idx_item: tuple) -> Dict[str, Any]:
        t0 = time.monotonic()
        res = _work(idx_item)
        res["elapsed_s"] = time.monotonic() - t0
        return res

    # ── Submit all, collect futures IN ORDER, write in order ─────────────────
    ok, skip, fail = 0, 0, 0
    elapsed_list: List[float] = []
    n_total = len(indexed_items)
    wall_start = time.monotonic()
    if n_total == 0:
        print("[resume] nothing to build — all samples already done.")
    try:
        with sft_path.open(file_mode, encoding="utf-8") as fsft, result_path.open(
            file_mode, encoding="utf-8",
        ) as fres:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.workers
            ) as executor:
                # submit preserves order → futures list matches input order
                futures = [
                    executor.submit(process_one, (i, item))
                    for i, item in indexed_items
                ]
                for n, fut in enumerate(futures):
                    result = fut.result()  # blocks until this item is done
                    elapsed_list.append(result.get("elapsed_s", 0.0))
                    status = result["status"]
                    if status == "ok":
                        ok += 1
                        fres.write(json.dumps(result["record"], ensure_ascii=False) + "\n")
                        for sft_line in result["sft_turns"]:
                            fsft.write(json.dumps(sft_line, ensure_ascii=False) + "\n")
                    elif status == "skip":
                        skip += 1
                        if result.get("record"):
                            fres.write(json.dumps(result["record"], ensure_ascii=False) + "\n")
                    else:  # fail
                        fail += 1
                        if result.get("record"):
                            fres.write(json.dumps(result["record"], ensure_ascii=False) + "\n")
                    fres.flush()
                    fsft.flush()

                    if (n + 1) % 10 == 0 or (n + 1) == n_total:
                        done = n + 1
                        wall = time.monotonic() - wall_start
                        avg = sum(elapsed_list) / len(elapsed_list) if elapsed_list else 0.0
                        thru = done / (wall / 60.0) if wall > 0 else 0.0
                        cur_qpm = (done * 2) / (wall / 60.0) if wall > 0 else 0.0
                        print(
                            f"[{done}/{n_total}] ok={ok} skip={skip} fail={fail} | "
                            f"avg/sample={avg:.1f}s wall={wall:.0f}s "
                            f"throughput={thru:.1f}/min apiQPM≈{cur_qpm:.1f}"
                        )
    finally:
        wall = time.monotonic() - wall_start
        avg = sum(elapsed_list) / len(elapsed_list) if elapsed_list else 0.0
        cur_qpm = ((ok + skip + fail) * 2) / (wall / 60.0) if wall > 0 else 0.0
        trace.event(
            "run_end", ok=ok, skip=skip, fail=fail,
            wall_s=round(wall, 1), avg_sample_s=round(avg, 2),
            api_qpm=round(cur_qpm, 1),
            api_key_usage={mask_key(k): v for k, v in key_usage.items()},
        )
        trace.close()

    wall = time.monotonic() - wall_start
    avg = sum(elapsed_list) / len(elapsed_list) if elapsed_list else 0.0
    cur_qpm = ((ok + skip + fail) * 2) / (wall / 60.0) if wall > 0 else 0.0
    print(f"Done ok={ok} skip={skip} fail={fail}")
    print(f"Timing: avg/sample={avg:.1f}s | wall={wall:.0f}s | "
          f"throughput={(ok+skip+fail)/(wall/60.0) if wall>0 else 0:.1f} samples/min | "
          f"observed API QPM≈{cur_qpm:.1f} (cap={args.qpm})")
    if api_keys:
        usage_str = ", ".join(f"{mask_key(k)}={v}" for k, v in key_usage.items())
        print(f"API key usage ({args.api_key_strategy}): {usage_str}")
    print(f"SFT turns: {sft_path}")
    print(f"v9k2 records: {result_path}")
    print(f"Frames under: {frames_root}")
    print(f"Trace log: {log_path}")


if __name__ == "__main__":
    main()
