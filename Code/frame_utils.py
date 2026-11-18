"""Frame extraction utilities using ffmpeg/ffprobe subprocess calls."""

import asyncio
import json
import math
import os
from typing import List, Optional, Tuple


async def detect_scene_boundaries(
    video_path: str,
    timestamps: List[float],
    frame_paths: List[str],
    threshold: float = 0.3,
) -> List[Tuple[float, float]]:

    if len(frame_paths) < 2:
        return []
    try:
        import torch
        import clip as openai_clip
        from PIL import Image
    except ImportError:
        # CLIP not installed — skip boundary detection silently
        return []

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = openai_clip.load("ViT-B/32", device=device)
        model.eval()

        features = []
        for fp in frame_paths:
            if not os.path.exists(fp):
                features.append(None)
                continue
            img = preprocess(Image.open(fp).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.encode_image(img)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            features.append(feat)

        boundaries: List[Tuple[float, float]] = []
        for i in range(1, len(features)):
            if features[i - 1] is None or features[i] is None:
                continue
            sim = (features[i - 1] * features[i]).sum().item()
            dist = 1.0 - sim
            if dist > threshold:
                boundaries.append((timestamps[i - 1], timestamps[i]))
        return boundaries
    except Exception:
        # Any CLIP error → degrade gracefully
        return []


async def get_video_meta(video_path: str) -> Tuple[float, str]:
    """Get video duration and resolution via ffprobe.

    Returns (duration_seconds, "WIDTHxHEIGHT").
    """
    proc = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=duration,width,height",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {video_path}: {stderr.decode().strip()}")

    data = json.loads(stdout.decode() or "{}")
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError(f"ffprobe returned no video streams for {video_path}")

    stream = streams[0]
    # Try stream-level duration first, fall back to format-level
    duration = stream.get("duration")
    if duration is None:
        duration = data.get("format", {}).get("duration")
    if duration is None:
        raise RuntimeError(f"Cannot determine duration for {video_path}")
    duration = float(duration)

    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))
    return duration, f"{width}x{height}"


def uniform_timestamps(duration: float, n: int) -> List[float]:
    """Generate n uniformly spaced timestamps across [0, duration].

    Places frames at evenly spaced points, avoiding the very start (0.0)
    and very end to get representative content.
    """
    if n <= 0:
        return []
    if n == 1:
        return [duration / 2.0]
    step = duration / (n + 1)
    return [step * (i + 1) for i in range(n)]


def timestamps_in_range(start: float, end: float, n: int) -> List[float]:
    """Generate n uniformly spaced timestamps within [start, end]."""
    if n <= 0:
        return []
    if n == 1:
        return [(start + end) / 2.0]
    step = (end - start) / (n + 1)
    return [start + step * (i + 1) for i in range(n)]


async def extract_frames(
    video_path: str,
    timestamps: List[float],
    save_dir: str,
    resize: float = 1.0,
    factor: int = 28,
) -> List[str]:
    """Extract frames at specific timestamps using ffmpeg.

    Returns list of saved frame file paths.
    """
    os.makedirs(save_dir, exist_ok=True)
    paths = []

    for i, ts in enumerate(timestamps):
        out_path = os.path.join(save_dir, f"frame_{i:04d}.png")

        # Build ffmpeg command: seek to timestamp, extract one frame
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{ts:.3f}",
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "2",
        ]

        # Apply resize if needed (round to even numbers for codec compatibility)
        if resize < 1.0:
            cmd.extend(["-vf", f"scale='trunc(iw*{resize}/2)*2:trunc(ih*{resize}/2)*2'"])

        cmd.append(out_path)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

        if os.path.exists(out_path):
            paths.append(out_path)
        else:
            print(f"Warning: failed to extract frame at t={ts:.1f}s from {video_path}")

    return paths


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS string."""
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes:02d}:{secs:02d}"


def estimate_visual_tokens(
    width: int,
    height: int,
    nframes: int,
    resize: float = 1.0,
    patch_size: int = 16,
    merge_size: int = 2,
    min_pixels: int = 65536,
    max_pixels: int = 16777216,
) -> int:
    """Count Qwen3-VL image tokens after ffmpeg and processor resizing.

    Defaults match Qwen3-VL-8B-Instruct's local model configuration:
    patch_size=16, spatial_merge_size=2, min_pixels=65536, and
    max_pixels=16777216.
    """
    if nframes <= 0:
        return 0

    # Match extract_frames(): trunc(iw*resize/2)*2 and likewise for height.
    if resize < 1.0:
        scaled_width = max(2, int(width * resize / 2) * 2)
        scaled_height = max(2, int(height * resize / 2) * 2)
    else:
        scaled_width = width
        scaled_height = height

    # Match Transformers Qwen2/3-VL smart_resize().
    factor = patch_size * merge_size
    if max(scaled_height, scaled_width) / min(scaled_height, scaled_width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got "
            f"{scaled_width}x{scaled_height}"
        )
    resized_height = round(scaled_height / factor) * factor
    resized_width = round(scaled_width / factor) * factor
    if resized_height * resized_width > max_pixels:
        beta = math.sqrt((scaled_height * scaled_width) / max_pixels)
        resized_height = max(
            factor, math.floor(scaled_height / beta / factor) * factor
        )
        resized_width = max(
            factor, math.floor(scaled_width / beta / factor) * factor
        )
    elif resized_height * resized_width < min_pixels:
        beta = math.sqrt(min_pixels / (scaled_height * scaled_width))
        resized_height = math.ceil(scaled_height * beta / factor) * factor
        resized_width = math.ceil(scaled_width * beta / factor) * factor

    grid_h = resized_height // patch_size
    grid_w = resized_width // patch_size
    tokens_per_image = grid_h * grid_w // (merge_size**2)
    return nframes * tokens_per_image
