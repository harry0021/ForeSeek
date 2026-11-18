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
# PROMPTS_MODULE env var lets you swap prompt variants without editing code:
#   PROMPTS_MODULE=prompts_v9b  → Phase 1 fix + lightweight Round 1 observation
#   (default)                   → Phase 1 fix only (prompts.py)
import importlib as _imp
_prompts_mod = os.environ.get("PROMPTS_MODULE", "prompts")
prompts = _imp.import_module(_prompts_mod)
from frame_utils import (
    uniform_timestamps,
    timestamps_in_range,
    format_timestamp,
    estimate_visual_tokens,
)


@dataclass
class StructurePrediction:
    """Parsed output from Phase 2 LLM structural reasoning."""
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
    # World-model fields (v9e+): only populated when the prompt emits them.
    # gap_inferences: ordered list of (gap_start, gap_end, predicted_content_with_reasoning)
    gap_inferences: List[Tuple[float, float, str]] = field(default_factory=list)
    # query_reasoning: free-form chain linking the query to the temporal model.
    query_reasoning: str = ""
    # world_state (v9g+): query-AGNOSTIC global memory of the video — scene type,
    # main entities, setting, and a coarse scene timeline. Acts as the durable
    # textual replacement for the Phase 1 frames downstream phases would
    # otherwise have to re-watch. Empty string when the prompt doesn't emit it.
    world_state: str = ""
    # v9i: dynamic interval belief tree. Phase 1+2 creates depth-0 nodes;
    # Phase 2.5 probes split active nodes into child intervals.
    belief_nodes: List[Dict[str, Any]] = field(default_factory=list)
    # truncated: True when the VLM response appears cut off (missing </prediction> etc).
    truncated: bool = False


def _parse_time_range(text: str) -> Optional[Tuple[float, float]]:
    """Parse '[start-end]' from text. Supports MM:SS and plain seconds."""
    m = re.search(r"\[(\d+):(\d+)\s*(?:-|->|→|to)\s*(\d+):(\d+)\]", text)
    if m:
        start = int(m.group(1)) * 60 + int(m.group(2))
        end = int(m.group(3)) * 60 + int(m.group(4))
        return float(start), float(end)
    m = re.search(r"\[(\d+\.?\d*)\s*s?\s*(?:-|->|→|to)\s*(\d+\.?\d*)\s*s?\]", text)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


def parse_structure_output(text: str, duration: float) -> StructurePrediction:
    """Parse the LLM's structured output into a StructurePrediction."""
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
        pred_text = m.group(1).strip()
        for line in pred_text.split("\n"):
            low = line.lower()
            if "primary" in low:
                r = _parse_time_range(line)
                if r:
                    pred.primary_region = (max(0, r[0]), min(duration, r[1]))
                # 4-level confidence: certain / high / medium / low.
                # Order matters — check "uncertain" before "certain" (substring),
                # and "certain" before "high"/"medium"/"low" so the strongest tier wins.
                conf_m = re.search(r"confidence\s*:?\s*(certain|high|medium|low|uncertain)", low)
                if conf_m:
                    raw = conf_m.group(1)
                    pred.primary_confidence = "low" if raw == "uncertain" else raw
                elif "certain" in low and "uncertain" not in low:
                    pred.primary_confidence = "certain"
                elif "uncertain" in low:
                    pred.primary_confidence = "low"
            elif "secondary" in low:
                r = _parse_time_range(line)
                if r:
                    pred.secondary_region = (max(0, r[0]), min(duration, r[1]))
            elif "frames" in low:
                fm = re.search(r"(\d+)", line)
                if fm:
                    # Clamp to a safe [4, 16] range — VLMs occasionally emit
                    # "frames: 0" when they think the query is unanswerable,
                    # which silently kills Phase 3 frame extraction.
                    pred.num_frames = max(4, min(int(fm.group(1)), 16))
            elif "resolution" in low:
                if "high" in low:
                    pred.resolution = "high"
                elif "low" in low:
                    pred.resolution = "low"

    m = re.search(r"<query_type>(.*?)</query_type>", text, re.DOTALL)
    if m:
        pred.query_type = m.group(1).strip().split()[0]

    # ── World-model extensions (v9e/v9f) ──
    gi_m = re.search(r"<gap_inference>(.*?)</gap_inference>", text, re.DOTALL)
    if gi_m:
        for line in gi_m.group(1).strip().splitlines():
            line = line.strip()
            if not line:
                continue
            # Accept "Gap [T1s → T2s]: prediction" or "Gap [T1-T2]: prediction"
            mm = re.search(
                r"\[?\s*(\d+)\s*s?\s*(?:->|→|-|to)\s*(\d+)\s*s?\s*\]?\s*:?\s*(.*)",
                line,
            )
            if mm:
                t1, t2 = float(mm.group(1)), float(mm.group(2))
                content = mm.group(3).strip()
                if t2 > t1 and content:
                    pred.gap_inferences.append((t1, t2, content[:300]))

    qr_m = re.search(r"<query_reasoning>(.*?)</query_reasoning>", text, re.DOTALL)
    if qr_m:
        pred.query_reasoning = qr_m.group(1).strip()[:600]

    # v9g: parse the query-agnostic world-state memory block.
    ws_m = re.search(r"<world_state>(.*?)</world_state>", text, re.DOTALL)
    if ws_m:
        # Keep up to 800 chars — enough room for scene_type + main_entities + setting
        # + scene_timeline without bloating later prompts.
        pred.world_state = ws_m.group(1).strip()[:800]

    pred.belief_nodes = _parse_belief_tree(text, duration)
    if pred.belief_nodes:
        _derive_prediction_from_belief(pred)

    # Truncation detection: the prompts always end with a closing tag; if none is present
    # the response was cut off and downstream parsing will silently degrade.
    if not re.search(r"</(prediction|query_type|belief_tree)>", text):
        pred.truncated = True

    if pred.primary_region == (0, 0) or pred.primary_region[0] >= pred.primary_region[1]:
        pred.primary_region = (0, duration)
        pred.primary_confidence = "low"

    return pred


def _gap_inference_for_region(
    gap_inferences: List[Tuple[float, float, str]],
    start: float,
    end: float,
    max_chars: int = 500,
) -> str:
    """Pick the gap-inference snippets that overlap [start, end] and concatenate them.

    Used to inject world-model context into the Phase 3 verification prompt so the
    VLM can compare predicted state against the actual frames it is about to see.
    """
    if not gap_inferences:
        return ""
    overlapping = [
        (s, e, c) for s, e, c in gap_inferences
        if not (e <= start or s >= end)
    ]
    if not overlapping:
        return ""
    lines = [f"[{int(s)}-{int(e)}s] {c}" for s, e, c in overlapping]
    block = "\n".join(lines)
    return block if len(block) <= max_chars else block[: max_chars - 3] + "..."


def _parse_single_timestamp(text: str) -> Optional[float]:
    """Parse a single [MM:SS] or [SS] timestamp from a line of text."""
    m = re.search(r"\[(\d+):(\d+)\]", text)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    m = re.search(r"\[(\d+(?:\.\d+)?)\]", text)
    if m:
        return float(m.group(1))
    return None


def _phase1_unified_vlm(
    client: OpenAI,
    frame_paths: List[str],
    timestamps: List[float],
    duration: float,
    duration_fmt: str,
    resolution: str,
    raw_query: str,
) -> Tuple[List[Tuple[float, str]], List[Tuple[float, float]], int]:
    """Phase 1: single multi-image VLM call for captioning + boundary detection.

    Replaces separate BLIP-2 captioning and CLIP boundary detection.
    Returns (captions, boundaries, total_tokens).
    """
    content: List[Dict[str, Any]] = []
    for i, (ts, fp) in enumerate(zip(timestamps, frame_paths)):
        content.append({"type": "text", "text": f"Frame {i + 1} at [{format_timestamp(ts)}]:"})
        content.append(_build_image_content(fp))

    content.append({
        "type": "text",
        "text": prompts.PHASE1_USER_TEMPLATE.format(
            num_frames=len(frame_paths),
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

    # ── Parse captions ──
    captions: List[Tuple[float, str]] = []
    cap_m = re.search(r"<captions>(.*?)</captions>", response, re.DOTALL)
    if cap_m:
        for line in cap_m.group(1).strip().splitlines():
            line = line.strip()
            if not line:
                continue
            ts_val = _parse_single_timestamp(line)
            if ts_val is not None:
                desc = re.sub(r"^\[[\d:\.]+\]\s*", "", line).strip()
                if desc:
                    captions.append((ts_val, desc))

    # Fallback: if parsing failed, pair timestamps with empty captions
    if len(captions) < len(timestamps) // 2:
        captions = list(zip(timestamps, [""] * len(timestamps)))

    # ── Parse boundaries ──
    boundaries: List[Tuple[float, float]] = []
    bnd_m = re.search(r"<boundaries>(.*?)</boundaries>", response, re.DOTALL)
    if bnd_m:
        bnd_text = bnd_m.group(1).strip()
        if "none" not in bnd_text.lower():
            for line in bnd_text.splitlines():
                tss = re.findall(r"\[(\d+):(\d+)\]", line)
                if len(tss) >= 2:
                    t1 = int(tss[0][0]) * 60 + int(tss[0][1])
                    t2 = int(tss[1][0]) * 60 + int(tss[1][1])
                    boundaries.append((float(t1), float(t2)))

    return captions, boundaries, tok


def _phase1_and_structure_vlm(
    client: OpenAI,
    frame_paths: List[str],
    timestamps: List[float],
    duration: float,
    duration_fmt: str,
    resolution: str,
    raw_query: str,
) -> Tuple[List[Tuple[float, str]], List[Tuple[float, float]], "StructurePrediction", int]:
    """Merged Phase 1 + Phase 2: single multi-image VLM call that produces
    captions, scene boundaries, AND temporal structure prediction together.

    Eliminates the caption-text bottleneck of the two-step approach:
    the VLM reasons from visual evidence directly rather than through
    intermediate caption text that a separate LLM has to re-interpret.

    Returns (captions, boundaries, prediction, total_tokens).
    """
    content: List[Dict[str, Any]] = []
    for i, (ts, fp) in enumerate(zip(timestamps, frame_paths)):
        content.append({"type": "text", "text": f"Frame {i + 1} at [{format_timestamp(ts)}]:"})
        content.append(_build_image_content(fp))

    belief_boundaries = [0.0] + list(timestamps) + [duration]
    content.append({
        "type": "text",
        "text": prompts.PHASE1_STRUCTURE_USER_TEMPLATE.format(
            num_frames=len(frame_paths),
            num_intervals=max(0, len(belief_boundaries) - 1),
            boundary_timestamps=", ".join(f"{int(ts)}s" for ts in belief_boundaries),
            duration=int(duration),
            duration_fmt=duration_fmt,
            resolution=resolution,
            query=raw_query,
        ),
    })

    response, tok = _call_api(
        client, config.VERIFY_MODEL,
        prompts.PHASE1_STRUCTURE_SYSTEM,
        content,
        max_tokens=config.MAX_TOKENS_PHASE1,
    )

    # ── Parse captions ──
    # v9e format: [OBSERVED at T] description inside <structure>
    # v9a format: [MM:SS] description inside <captions>
    captions: List[Tuple[float, str]] = []
    cap_m = re.search(r"<captions>(.*?)</captions>", response, re.DOTALL)
    if cap_m:
        for line in cap_m.group(1).strip().splitlines():
            line = line.strip()
            if not line:
                continue
            ts_val = _parse_single_timestamp(line)
            if ts_val is not None:
                desc = re.sub(r"^\[[\d:\.]+\]\s*", "", line).strip()
                if desc:
                    captions.append((ts_val, desc))
    else:
        # v9e fallback: extract [OBSERVED at T] lines from <structure>
        struct_m = re.search(r"<structure>(.*?)</structure>", response, re.DOTALL)
        if struct_m:
            for line in struct_m.group(1).strip().splitlines():
                obs_m = re.search(r"\[OBSERVED\s+at\s+(\d+)\](.+)", line, re.IGNORECASE)
                if obs_m:
                    ts_val = float(obs_m.group(1))
                    desc = obs_m.group(2).strip()
                    if desc:
                        captions.append((ts_val, desc))
    if len(captions) < len(timestamps) // 2:
        captions = list(zip(timestamps, [""] * len(timestamps)))

    # ── Parse boundaries ──
    boundaries: List[Tuple[float, float]] = []
    bnd_m = re.search(r"<boundaries>(.*?)</boundaries>", response, re.DOTALL)
    if bnd_m:
        bnd_text = bnd_m.group(1).strip()
        if "none" not in bnd_text.lower():
            for line in bnd_text.splitlines():
                tss = re.findall(r"\[(\d+):(\d+)\]", line)
                if len(tss) >= 2:
                    t1 = int(tss[0][0]) * 60 + int(tss[0][1])
                    t2 = int(tss[1][0]) * 60 + int(tss[1][1])
                    boundaries.append((float(t1), float(t2)))

    # ── Parse structure prediction (reuse existing parser) ──
    prediction = parse_structure_output(response, duration)

    return captions, boundaries, prediction, tok


def _probe_caption_vlm(
    client: OpenAI,
    frame_path: str,
    timestamp: float,
    duration_fmt: str,
    raw_query: str,
) -> Tuple[str, int]:
    """Caption a single probe frame using the unified VLM (Phase 2.5).

    Replaces BLIP-2 for individual adaptive probe frames.
    Returns (caption_text, total_tokens).
    """
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": f"Frame at [{format_timestamp(timestamp)}]:"},
        _build_image_content(frame_path),
        {
            "type": "text",
            "text": prompts.PROBE_CAPTION_TEMPLATE.format(
                timestamp_fmt=format_timestamp(timestamp),
                duration_fmt=duration_fmt,
                query=raw_query,
            ),
        },
    ]
    response, tok = _call_api(
        client, config.VERIFY_MODEL,
        prompts.PROBE_CAPTION_SYSTEM,
        content,
        max_tokens=150,
    )
    return response.strip(), tok


def _extract_answer(text: str) -> str:
    # Primary: <answer>X</answer> — supports A-H (CG-Bench / LVBench have up to 7-8 options)
    m = re.search(r"<answer>\s*([A-Ha-h])\s*</answer>", text)
    if m:
        return m.group(1).strip().upper()
    # "Answer: X" variants
    m = re.search(r"[Aa]nswer\s*[:\.\s]\s*\**\s*([A-Ha-h])\b", text)
    if m:
        return m.group(1).strip().upper()
    # "The answer is X"
    m = re.search(r"(?:correct\s+)?answer\s+is\s+\**\s*([A-Ha-h])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().upper()
    # "Option X" / "Choice X"
    m = re.search(r"(?:[Oo]ption|[Cc]hoice)\s+\**\s*([A-Ha-h])\b", text)
    if m:
        return m.group(1).strip().upper()
    # "(X)" — LVBench uses (A)/(B)/(C)/(D) style options
    m = re.search(r"\(\s*([A-Ha-h])\s*\)", text[-300:])
    if m:
        return m.group(1).strip().upper()
    # Last-resort: lone letter at the very end
    m = re.search(r"\b([A-Ha-h])\b\s*$", text.rstrip())
    if m:
        return m.group(1).strip().upper()
    return ""


def _extract_confidence(text: str) -> int:
    """Extract numeric confidence (1/2/3) from response. Accepts binary fallback."""
    m = re.search(r"<confidence>\s*(\d)\s*</confidence>", text)
    if m:
        return int(m.group(1))
    # Binary fallback
    m = re.search(r"<confidence>\s*(certain|uncertain)\s*</confidence>", text, re.IGNORECASE)
    if m:
        return 3 if m.group(1).lower() == "certain" else 1
    return 1


def _extract_observation(text: str) -> str:
    """Extract the VLM's structured observation from <observation> tag.
    Format: 'Content: ...; Text: ...; Next: ...'
    """
    m = re.search(r"<observation>(.*?)</observation>", text, re.DOTALL)
    if m:
        return m.group(1).strip()[:400]
    return ""


def _format_captions_block(captions: List[Tuple[float, str]]) -> str:
    lines = []
    for ts, cap in sorted(captions, key=lambda x: x[0]):
        lines.append(f"[{format_timestamp(ts)}] {cap}")
    return "\n".join(lines)


def _score_for_likelihood(answer_likely: str) -> float:
    # Also handles relevance=HIGH|MEDIUM|LOW from v9h2
    mapping = {"YES": 0.80, "MAYBE": 0.50, "NO": 0.10,
               "HIGH": 0.80, "MEDIUM": 0.50, "LOW": 0.10}
    return mapping.get(answer_likely.upper(), 0.50)


def _node_prev_likely(node: Dict[str, Any], prompts_module: Any) -> str:
    """Return the relevance string matching the current prompt's vocabulary.

    v9h2 uses relevance=HIGH|MEDIUM|LOW in BELIEF_UPDATE_TEMPLATE.
    v9h / v9i use answer_likely=YES|MAYBE|NO.
    We detect which is needed by inspecting the template string.
    """
    al = node.get("answer_likely", "MAYBE")
    tmpl = getattr(prompts_module, "BELIEF_UPDATE_TEMPLATE", "")
    if "relevance=" in tmpl:
        return {"YES": "HIGH", "MAYBE": "MEDIUM", "NO": "LOW"}.get(al, "MEDIUM")
    return al


def _uncertainty_rank(uncertainty: str) -> int:
    return {"high": 3, "medium": 2, "low": 1}.get(uncertainty.lower(), 2)


def _clean_reason_token(reason: str) -> str:
    reason = re.split(r"\s+(?:left|right)\s*=", reason.strip(), maxsplit=1)[0]
    return " ".join(reason.split()[:3])[:60]


def _parse_interval_range(text: str) -> Optional[Tuple[float, float]]:
    """Parse v9i interval ranges. Supports seconds and MM:SS with -> / → / -."""
    m = re.search(
        r"\[?\s*(?:(\d+):(\d+)|(\d+(?:\.\d+)?))\s*s?\s*"
        r"(?:->|→|-|to)\s*"
        r"(?:(\d+):(\d+)|(\d+(?:\.\d+)?))\s*s?\s*\]?",
        text,
    )
    if not m:
        return None
    if m.group(1) is not None:
        start = int(m.group(1)) * 60 + int(m.group(2))
    else:
        start = float(m.group(3))
    if m.group(4) is not None:
        end = int(m.group(4)) * 60 + int(m.group(5))
    else:
        end = float(m.group(6))
    return float(start), float(end)


def _parse_interval_node_line(line: str, duration: float) -> Optional[Dict[str, Any]]:
    interval = _parse_interval_range(line)
    if not interval:
        return None
    start, end = interval
    if end <= start:
        return None
    id_m = re.search(r"\b(?:Node|node)\s+([A-Za-z0-9_.-]+)", line)
    parent_m = re.search(r"\bparent\s*=\s*([A-Za-z0-9_.-]+|None)", line, re.IGNORECASE)
    depth_m = re.search(r"\bdepth\s*=\s*(\d+)", line, re.IGNORECASE)
    status_m = re.search(r"\bstatus\s*=\s*(active|split|searched|eliminated)", line, re.IGNORECASE)
    # v9h2: relevance=HIGH|MEDIUM|LOW (single merged signal)
    rel_m = re.search(r"\brelevance\s*=\s*(HIGH|MEDIUM|LOW)", line, re.IGNORECASE)
    # v9h legacy: answer_likely=YES|MAYBE|NO
    al_m = re.search(r"\banswer_likely\s*=\s*(YES|NO|MAYBE)", line, re.IGNORECASE)
    score_m = re.search(r"\bscore\s*=\s*(0(?:\.\d+)?|1(?:\.0+)?)", line, re.IGNORECASE)
    unc_m = re.search(r"\buncertainty\s*=\s*(high|medium|low)", line, re.IGNORECASE)
    sc_m = re.search(r"\bscene_cut\s*=\s*(YES|NO)", line, re.IGNORECASE)
    rt_m = re.search(r"\breason_token\s*=\s*([^\n\r]+?)(?:\s+scene_hint\s*=|$)", line)
    sh_m = re.search(r"\bscene_hint\s*=\s*([^\n\r]+)", line)
    # Normalise: relevance overrides answer_likely; map HIGH→YES, MEDIUM→MAYBE, LOW→NO
    if rel_m:
        raw_rel = rel_m.group(1).upper()
        answer_likely = {"HIGH": "YES", "MEDIUM": "MAYBE", "LOW": "NO"}.get(raw_rel, "MAYBE")
    else:
        answer_likely = al_m.group(1).upper() if al_m else "MAYBE"
    parent_id = parent_m.group(1) if parent_m and parent_m.group(1) != "None" else None
    scene_hint = sh_m.group(1).strip()[:60] if sh_m else ""
    return {
        "node_id": id_m.group(1) if id_m else f"N{int(start)}_{int(end)}",
        "parent_id": parent_id,
        "gap_start": max(0.0, min(duration, start)),
        "gap_end": max(0.0, min(duration, end)),
        "depth": int(depth_m.group(1)) if depth_m else 0,
        "status": status_m.group(1).lower() if status_m else "active",
        "answer_likely": answer_likely,
        "score": float(score_m.group(1)) if score_m else _score_for_likelihood(answer_likely),
        "uncertainty": unc_m.group(1).lower() if unc_m else "medium",
        "scene_cut": sc_m.group(1).upper() == "YES" if sc_m else False,
        "reason_token": _clean_reason_token(rt_m.group(1)) if rt_m else "",
        "scene_hint": scene_hint,
    }


def _parse_belief_tree(text: str, duration: float) -> List[Dict[str, Any]]:
    m = re.search(r"<belief_tree>(.*?)</belief_tree>", text, re.DOTALL)
    if not m:
        return []
    nodes = []
    for line in m.group(1).strip().splitlines():
        line = line.strip()
        if line.lower().startswith("node"):
            node = _parse_interval_node_line(line, duration)
            if node:
                nodes.append(node)
    return nodes


def _format_belief_tree_block(belief_nodes: List[Dict[str, Any]]) -> str:
    if not belief_nodes:
        return "(no belief-tree nodes available)"
    lines = []
    for node in belief_nodes:
        sc = "YES" if node.get("scene_cut") else "NO"
        hint = node.get("scene_hint", "")
        hint_str = f" scene_hint={hint}" if hint else ""
        lines.append(
            f"  Node {node.get('node_id')} [{int(node['gap_start'])}s→{int(node['gap_end'])}s] "
            f"depth={node.get('depth', 0)} status={node.get('status', 'active')} "
            f"answer_likely={node.get('answer_likely', 'MAYBE')} "
            f"score={float(node.get('score', 0.5)):.2f} "
            f"uncertainty={node.get('uncertainty', 'medium')} "
            f"scene_cut={sc} reason_token={node.get('reason_token', '')}{hint_str}"
        )
    return "\n".join(lines)


def _active_belief_nodes(belief_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        node for node in belief_nodes
        if node.get("status", "active") == "active"
        and node.get("answer_likely", "MAYBE") in ("YES", "MAYBE")
        and node["gap_end"] > node["gap_start"]
    ]


def _restore_frontier_if_all_no(belief_nodes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Avoid a dead frontier by restoring the strongest active NO interval to MAYBE."""
    if _active_belief_nodes(belief_nodes):
        return None
    candidates = [
        node for node in belief_nodes
        if node.get("status", "active") == "active"
        and node["gap_end"] > node["gap_start"]
    ]
    if not candidates:
        return None
    best = max(
        candidates,
        key=lambda n: (
            float(n.get("score", _score_for_likelihood(n.get("answer_likely", "NO")))),
            n["gap_end"] - n["gap_start"],
        ),
    )
    best["answer_likely"] = "MAYBE"
    best["score"] = max(float(best.get("score", 0.0)), _score_for_likelihood("MAYBE"))
    best["uncertainty"] = "high"
    return best


def _rank_belief_node(node: Dict[str, Any]) -> Tuple[int, float, int, float]:
    label_rank = {"YES": 2, "MAYBE": 1, "NO": 0}.get(node.get("answer_likely", "MAYBE"), 1)
    width = node["gap_end"] - node["gap_start"]
    return (
        label_rank,
        float(node.get("score", _score_for_likelihood(node.get("answer_likely", "MAYBE")))),
        _uncertainty_rank(node.get("uncertainty", "medium")),
        width,
    )


def _top_belief_nodes(belief_nodes: List[Dict[str, Any]], limit: int = 2) -> List[Dict[str, Any]]:
    return sorted(_active_belief_nodes(belief_nodes), key=_rank_belief_node, reverse=True)[:limit]


def _derive_prediction_from_belief(prediction: StructurePrediction) -> bool:
    top = _top_belief_nodes(prediction.belief_nodes, limit=2)
    if not top:
        return False
    old = prediction.primary_region
    best = top[0]
    prediction.primary_region = (best["gap_start"], best["gap_end"])
    prediction.primary_confidence = (
        "high" if best.get("answer_likely") == "YES" and float(best.get("score", 0)) >= 0.65
        else "medium"
    )
    if len(top) > 1:
        prediction.secondary_region = (top[1]["gap_start"], top[1]["gap_end"])
    return prediction.primary_region != old


def _compute_unsearched_belief_hints(
    belief_nodes: List[Dict[str, Any]],
    searched_regions: List[Tuple[float, float, str]],
) -> str:
    hints = []
    for node in _top_belief_nodes(belief_nodes, limit=8):
        mid = (node["gap_start"] + node["gap_end"]) / 2.0
        if any((s - 5) <= mid <= (e + 5) for s, e, _ in searched_regions):
            continue
        hint = node.get("scene_hint", "")
        hint_str = f" scene_hint={hint}" if hint else ""
        hints.append(
            f"  Node {node.get('node_id')} [{int(node['gap_start'])}-{int(node['gap_end'])}s] "
            f"({node.get('answer_likely')} score={float(node.get('score', 0.5)):.2f} "
            f"uncertainty={node.get('uncertainty', 'medium')} reason_token={node.get('reason_token', '')}{hint_str})"
        )
    return "\n".join(hints) if hints else "  (all active belief nodes exhausted)"


def _parse_belief_child(line: str, key: str, parent: Dict[str, Any], duration: float) -> Optional[Dict[str, Any]]:
    m = re.search(key + r"\s*=\s*(.*)", line, re.IGNORECASE)
    if not m:
        return None
    child = _parse_interval_node_line(f"Node tmp {m.group(1)}", duration)
    if not child:
        return None
    suffix = "L" if key.lower() == "left" else "R"
    child["node_id"] = f"{parent.get('node_id', 'N')}.{suffix}"
    child["parent_id"] = parent.get("node_id")
    child["depth"] = int(parent.get("depth", 0)) + 1
    child["status"] = "active"
    return child


def _parse_belief_update(text: str, parent: Dict[str, Any], duration: float) -> Optional[Dict[str, Any]]:
    m = re.search(r"<belief_update>(.*?)</belief_update>", text, re.DOTALL)
    if not m:
        return None
    block = m.group(1).strip()
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    parent_line = next((ln for ln in lines if ln.lower().startswith("parent=")), lines[0] if lines else "")
    action_m = re.search(r"\baction\s*=\s*(SPLIT|KEEP|ELIMINATE)", parent_line, re.IGNORECASE)
    # v9h2: relevance field; v9h legacy: answer_likely
    rel_m2 = re.search(r"\brelevance\s*=\s*(HIGH|MEDIUM|LOW)", parent_line, re.IGNORECASE)
    al_m = re.search(r"\banswer_likely\s*=\s*(YES|NO|MAYBE)", parent_line, re.IGNORECASE)
    if rel_m2:
        raw_rel2 = rel_m2.group(1).upper()
        parsed_al = {"HIGH": "YES", "MEDIUM": "MAYBE", "LOW": "NO"}.get(raw_rel2, "MAYBE")
    else:
        parsed_al = al_m.group(1).upper() if al_m else parent.get("answer_likely", "MAYBE")
    score_m = re.search(r"\bscore\s*=\s*(0(?:\.\d+)?|1(?:\.0+)?)", parent_line, re.IGNORECASE)
    unc_m = re.search(r"\buncertainty\s*=\s*(high|medium|low)", parent_line, re.IGNORECASE)
    sc_m = re.search(r"\bscene_cut\s*=\s*(YES|NO)", parent_line, re.IGNORECASE)
    rt_m = re.search(r"\breason_token\s*=\s*([^\n\r]+?)(?:\s+scene_hint\s*=|$)", parent_line)
    sh_m = re.search(r"\bscene_hint\s*=\s*([^\n\r]+)", parent_line)
    children = []
    for line in lines:
        if line.lower().startswith("left"):
            child = _parse_belief_child(line, "left", parent, duration)
            if child:
                children.append(child)
        elif line.lower().startswith("right"):
            child = _parse_belief_child(line, "right", parent, duration)
            if child:
                children.append(child)
    return {
        "action": action_m.group(1).upper() if action_m else "KEEP",
        "answer_likely": parsed_al,
        "score": float(score_m.group(1)) if score_m else parent.get("score", 0.5),
        "uncertainty": unc_m.group(1).lower() if unc_m else parent.get("uncertainty", "medium"),
        "scene_cut": sc_m.group(1).upper() == "YES" if sc_m else parent.get("scene_cut", False),
        "reason_token": _clean_reason_token(rt_m.group(1)) if rt_m else parent.get("reason_token", ""),
        "scene_hint": sh_m.group(1).strip()[:60] if sh_m else parent.get("scene_hint", ""),
        "children": children,
    }


def _find_active_belief_node_at(belief_nodes: List[Dict[str, Any]], timestamp: float) -> Optional[Dict[str, Any]]:
    candidates = [
        node for node in _active_belief_nodes(belief_nodes)
        if node["gap_start"] <= timestamp <= node["gap_end"]
    ]
    return min(candidates, key=lambda n: n["gap_end"] - n["gap_start"]) if candidates else None


def _demote_belief_nodes_for_search(
    belief_nodes: List[Dict[str, Any]],
    start_t: float,
    end_t: float,
    found: bool,
) -> None:
    if found:
        return
    for node in _active_belief_nodes(belief_nodes):
        overlap = max(0.0, min(end_t, node["gap_end"]) - max(start_t, node["gap_start"]))
        width = max(1.0, node["gap_end"] - node["gap_start"])
        if overlap / width >= 0.50:
            node["score"] = min(float(node.get("score", 0.5)), 0.05)
            node["answer_likely"] = "NO"
            node["uncertainty"] = "low"
            node["status"] = "eliminated"


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
    """Call the cloud API with rate-limit retry. Returns (response_text, total_tokens)."""
    import time as _time
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    api_kwargs = dict(model=model, messages=messages, max_tokens=max_tokens)
    if config.TEMPERATURE > 0:
        api_kwargs["temperature"] = config.TEMPERATURE
    for attempt in range(5):
        try:
            response = client.chat.completions.create(**api_kwargs)
            text = response.choices[0].message.content or ""
            tokens = response.usage.total_tokens if response.usage else 0
            return text, tokens
        except Exception as e:
            if "429" in str(e) or "RateLimit" in type(e).__name__:
                wait = 3 * (attempt + 1)
                _time.sleep(wait)
                continue
            print(f"  API error ({type(e).__name__}): {e}")
            return "", 0
    print(f"  API failed after 5 retries (rate limited)")
    return "", 0


def _get_ffmpeg_exe() -> str:
    """Get ffmpeg executable path: system PATH first, then imageio-ffmpeg fallback."""
    import shutil
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def get_video_meta(video_path: str) -> Tuple[float, str]:
    """Get video duration and resolution using ffmpeg (no ffprobe needed)."""
    ffmpeg = _get_ffmpeg_exe()
    result = subprocess.run(
        [ffmpeg, "-i", video_path],
        capture_output=True, text=True,
    )
    # ffmpeg -i always returns non-zero, info is in stderr
    stderr = result.stderr
    dur_m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr)
    res_m = re.search(r",\s*(\d{2,5})x(\d{2,5})", stderr)
    if not dur_m or not res_m:
        raise RuntimeError(f"Cannot parse video meta from {video_path}: {stderr[:200]}")
    duration = int(dur_m.group(1)) * 3600 + int(dur_m.group(2)) * 60 + float(dur_m.group(3))
    width, height = int(res_m.group(1)), int(res_m.group(2))
    return duration, f"{width}x{height}"


def extract_frames(
    video_path: str, timestamps: List[float], save_dir: str, resize: float = 1.0,
) -> List[str]:
    """Extract frames at timestamps using ffmpeg."""
    ffmpeg = _get_ffmpeg_exe()
    os.makedirs(save_dir, exist_ok=True)
    paths = []
    for i, ts in enumerate(timestamps):
        out_path = os.path.join(save_dir, f"frame_{i:04d}.png")
        cmd = [ffmpeg, "-y", "-ss", f"{ts:.3f}", "-i", video_path,
               "-frames:v", "1", "-q:v", "2"]
        if resize < 1.0:
            cmd.extend(["-vf", f"scale='trunc(iw*{resize}/2)*2:trunc(ih*{resize}/2)*2'"])
        cmd.append(out_path)
        subprocess.run(cmd, capture_output=True)
        if os.path.exists(out_path):
            paths.append(out_path)
    return paths


# ═══════════════════════════════════════════════════════════════
# Main agent function
# ═══════════════════════════════════════════════════════════════

def _compute_remaining_hint(duration: float, searched: List[Tuple[float, float, str]]) -> str:
    """Compute unsearched time gaps given already-searched regions.

    Returns a human-readable string describing the largest unsearched region
    and all gaps, to guide the correction LLM toward unexplored areas.
    """
    if not searched:
        return f"Entire video [0-{int(duration)}] unsearched."

    # Merge overlapping searched regions
    intervals = sorted([(s, e) for s, e, _ in searched], key=lambda x: x[0])
    merged: List[Tuple[float, float]] = []
    for s, e in intervals:
        if merged and s <= merged[-1][1] + 5:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    # Find gaps
    gaps: List[Tuple[float, float]] = []
    prev_end = 0.0
    for s, e in merged:
        if s > prev_end + 5:
            gaps.append((prev_end, s))
        prev_end = max(prev_end, e)
    if prev_end < duration - 5:
        gaps.append((prev_end, duration))

    if not gaps:
        return "All regions have been searched. Use caption evidence to pick the most likely region."

    gaps.sort(key=lambda x: x[1] - x[0], reverse=True)
    largest = gaps[0]
    all_gaps = ", ".join(f"[{int(s)}-{int(e)}s]" for s, e in gaps)
    return (f"Unsearched regions: {all_gaps}. "
            f"Largest gap: [{int(largest[0])}-{int(largest[1])}s] — focus here.")


def foresight_single(
    index: int,
    item: Dict[str, Any],
    client: OpenAI,
    video_root: str,
    output_dir: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run ForesightAgent on a single video QA sample.

    Phase 1+2: Unified VLM — multi-image captioning + scene boundaries + structure prediction (single API call)
    Phase 2.5: Adaptive probing — single-frame VLM captions (per probe, API call)
    Phase 3:   VLM verification (targeted frames, API call, up to MAX_VERIFY_ROUNDS)
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
    # Workers often process multiple questions from the same video in parallel.
    # Include index/pid so one worker's cleanup cannot delete another's frames.
    frame_dir = os.path.join(output_dir, config.FRAME_SAVE_ROOT, f"{video_id}_{index}_{os.getpid()}")
    total_tokens = 0

    # ── Frame / visual-token tracking (all phases) ──────────────────────────
    phase1_frames: int = 0
    phase1_visual_tokens: int = 0
    phase25_frames: int = 0
    phase25_visual_tokens: int = 0
    phase3_frames_per_round: List[int] = []
    phase3_visual_tokens_per_round: List[int] = []
    phase25_probe_targets_per_round: List[List[float]] = []
    phase25_probe_nodes_per_round: List[List[Dict[str, Any]]] = []
    phase3_regions_per_round: List[List[float]] = []
    phase3_timestamps_per_round: List[List[float]] = []
    phase3_responses_per_round: List[str] = []
    correction_raw_outputs: List[str] = []
    correction_regions: List[List[float]] = []
    phase25_update_raw_outputs_per_round: List[List[str]] = []
    frontier_restores: List[Dict[str, Any]] = []
    frontier_restored_once: bool = False
    final_navigation_raw_output: str = ""

    # ─── Phase 1: Unified VLM Captioning + Scene Boundary Detection ───
    duration, resolution = get_video_meta(video_path)
    width, height = (int(v) for v in resolution.split("x"))
    duration_fmt = format_timestamp(duration)

    # Adaptive initial probe frame count: scale with video length.
    # Short videos (< 5 min): 5 frames baseline.
    # Each additional 5 minutes adds 1 frame, up to 10.
    #   duration_min  frames
    #         < 5       5
    #         10        6
    #         15        7
    #         20        8
    #         25        9
    #        >= 30      10
    _min_probe = int(os.environ.get("MIN_PROBE_FRAMES", "5"))
    _max_probe = int(os.environ.get("MAX_PROBE_FRAMES_ADAPTIVE", "10"))
    _adaptive_n = min(_max_probe, max(_min_probe, _min_probe + int(duration / 300)))
    probe_timestamps = uniform_timestamps(duration, _adaptive_n)
    probe_frame_paths = extract_frames(
        video_path, probe_timestamps,
        os.path.join(frame_dir, "phase1"),
        resize=config.PHASE1_RESIZE,
    )

    captions, boundaries, prediction, tok = _phase1_and_structure_vlm(
        client, probe_frame_paths, probe_timestamps,
        duration, duration_fmt,
        f"{min(width, height)}p",
        raw_query,
    )
    total_tokens += tok
    phase1_frames = len(probe_frame_paths)
    phase1_visual_tokens = estimate_visual_tokens(width, height, phase1_frames, config.PHASE1_RESIZE)
    phase1_belief_raw_output = prediction.raw_output

    if prediction.truncated:
        print(f"  [WARN] Phase 1+2 VLM response appears truncated for {video_file} "
              f"(no closing </prediction> tag). Consider raising MAX_TOKENS_PHASE1 "
              f"(current={config.MAX_TOKENS_PHASE1}).")

    if boundaries:
        boundaries_block = "\n".join(
            f"- Scene cut between [{format_timestamp(a)}] and [{format_timestamp(b)}]"
            for a, b in boundaries
        )
    else:
        boundaries_block = "None detected"

    if verbose:
        print(f"\n{'='*60}")
        print(f"[Phase 1+2] Video: {video_file} ({duration:.0f}s, {resolution})")
        print(f"[Phase 1+2] Query: {raw_query[:100]}")
        print(f"[Phase 1+2] Captions ({len(captions)}):")
        for ts, cap in sorted(captions, key=lambda x: x[0]):
            print(f"  [{format_timestamp(ts)}] {cap}")
        print(f"[Phase 1+2] Boundaries: {boundaries_block}")
        print(f"[Phase 1+2] Primary: [{prediction.primary_region[0]:.0f}-{prediction.primary_region[1]:.0f}s] "
              f"conf={prediction.primary_confidence}")
        print(f"[Phase 1+2] Query type: {prediction.query_type}")
        print(f"[Phase 1+2] Uncertain: {prediction.uncertain_regions}")

    # ─── Phase 2 is now merged into Phase 1 (see _phase1_and_structure_vlm above) ───
    # The prediction object was returned directly from the combined VLM call.

    # ── Effective confidence: override to "low" when region is too wide ──
    region_dur = prediction.primary_region[1] - prediction.primary_region[0]
    region_coverage = region_dur / duration if duration > 0 else 1.0
    if region_coverage > 0.80:
        _effective_confidence = "low"
    elif region_coverage < 0.40 and prediction.primary_confidence in ("high", "medium", "certain"):
        _effective_confidence = prediction.primary_confidence
    else:
        _effective_confidence = prediction.primary_confidence

    # ── Parse failure detection (placeholder text / invalid range) ──
    # Inject uniform probe candidates so Phase 2.5 still fires.
    _is_parse_failure = (
        prediction.primary_confidence in ("low", "uncertain")
        and prediction.primary_region == (0.0, duration)
        and not prediction.uncertain_regions
    )
    if _is_parse_failure:
        step = duration / (config.MAX_EXTRA_FRAMES + 1)
        prediction.uncertain_regions = [
            (round(i * step), round((i + 1) * step))
            for i in range(config.MAX_EXTRA_FRAMES)
        ]
        if verbose:
            print(f"[Phase 1+2] Parse failure — injecting {len(prediction.uncertain_regions)} uniform probe regions for Phase 2.5")

    # ─── Phase 2.5: Incremental Structure Refinement (hybrid trigger) ───
    for probe_round in range(config.MAX_PROBE_ROUNDS):
        # ── Early exit: query-type-aware confidence / width tradeoff ──
        if prediction.belief_nodes:
            top_nodes = _top_belief_nodes(prediction.belief_nodes, limit=1)
            best_node = top_nodes[0] if top_nodes else None
            qtype_for_probe = prediction.query_type.split("|")[0].strip()
            best_width = (
                best_node["gap_end"] - best_node["gap_start"]
                if best_node is not None
                else duration
            )
            if qtype_for_probe in ("detail_recognition", "counting"):
                stop_width = 30.0    # 10 frames × 3s/frame
            elif qtype_for_probe in ("temporal_localization", "causal_reasoning", "comparison"):
                stop_width = 50.0    # 10 frames × 5s/frame
            elif qtype_for_probe == "global_understanding":
                stop_width = duration
            else:
                stop_width = 40.0    # 10 frames × 4s/frame
            best_is_confident = (
                best_node is not None
                and best_node.get("answer_likely") == "YES"
                and float(best_node.get("score", 0.0)) >= 0.70
            )
            belief_width_ok = (
                best_node is not None
                and best_width <= stop_width
            )
            can_skip_probe = belief_width_ok and (
                best_is_confident or _effective_confidence in ("certain", "high")
            )
        else:
            can_skip_probe = _effective_confidence in ("certain", "high") and not prediction.uncertain_regions
        if can_skip_probe:
            if verbose:
                print(f"[Phase 2.5] High effective confidence, skipping.")
            break

        # Source 1: Time gap rule — large gaps between adjacent captions
        sorted_captions = sorted(captions, key=lambda x: x[0])
        gap_candidates = []
        for i in range(1, len(sorted_captions)):
            gap = sorted_captions[i][0] - sorted_captions[i - 1][0]
            if gap > config.CAPTION_GAP_THRESHOLD:
                gap_candidates.append((sorted_captions[i - 1][0] + sorted_captions[i][0]) / 2.0)

        # Source 2: VLM-detected scene boundaries (from Phase 1)
        boundary_candidates = [(a + b) / 2.0 for a, b in boundaries]

        # Source 3: LLM uncertain regions (from Phase 2)
        llm_candidates = [(s + e) / 2.0 for s, e in prediction.uncertain_regions]

        # Source 3b: v9i belief frontier
        belief_candidates: List[float] = []
        if hasattr(prompts, "BELIEF_UPDATE_TEMPLATE") and prediction.belief_nodes:
            for node in _top_belief_nodes(prediction.belief_nodes, limit=config.MAX_EXTRA_FRAMES):
                belief_candidates.append((node["gap_start"] + node["gap_end"]) / 2.0)

        # Source 4: Fallback — when low confidence and no other candidates
        fallback_candidates: List[float] = []
        if _effective_confidence in ("low", "uncertain") and not gap_candidates and not boundary_candidates and not llm_candidates and not belief_candidates:
            step = duration / (config.MAX_EXTRA_FRAMES + 1)
            fallback_candidates = [step * (i + 1) for i in range(config.MAX_EXTRA_FRAMES)]

        if belief_candidates:
            probe_targets = belief_candidates[:config.MAX_EXTRA_FRAMES]
        else:
            # Merge, deduplicate, exclude existing caption positions
            all_candidates = sorted(set(gap_candidates + boundary_candidates + llm_candidates + fallback_candidates))
            deduped = []
            for t in all_candidates:
                if not deduped or abs(t - deduped[-1]) > config.PROBE_DEDUP_DIST:
                    if all(abs(t - ct) > config.PROBE_DEDUP_DIST for ct, _ in captions):
                        deduped.append(t)

            probe_targets = deduped[:config.MAX_EXTRA_FRAMES]

        if verbose:
            print(f"\n[Phase 2.5] Round {probe_round}: gap={len(gap_candidates)}, "
                  f"boundary={len(boundary_candidates)}, llm={len(llm_candidates)}, "
                  f"belief={len(belief_candidates)}, "
                  f"fallback={len(fallback_candidates)} → {len(probe_targets)} targets")

        if not probe_targets:
            if verbose:
                print(f"[Phase 2.5] No probe targets, skipping.")
            break

        phase25_probe_targets_per_round.append([round(float(t), 3) for t in probe_targets])
        probe_nodes_this_round: List[Dict[str, Any]] = []
        update_outputs_this_round: List[str] = []
        # Extract probe frames
        new_frame_paths = extract_frames(
            video_path, probe_targets,
            os.path.join(frame_dir, f"phase2_5_r{probe_round}"),
            resize=config.PHASE1_RESIZE,
        )
        phase25_frames += len(new_frame_paths)
        phase25_visual_tokens += estimate_visual_tokens(width, height, len(new_frame_paths), config.PHASE1_RESIZE)

        if hasattr(prompts, "BELIEF_UPDATE_TEMPLATE") and prediction.belief_nodes:
            for fp, ts in zip(new_frame_paths, probe_targets):
                target_node = _find_active_belief_node_at(prediction.belief_nodes, ts)
                if target_node is None:
                    continue
                probe_nodes_this_round.append({
                    "node_id": target_node.get("node_id"),
                    "start": target_node.get("gap_start"),
                    "end": target_node.get("gap_end"),
                    "depth": target_node.get("depth"),
                    "answer_likely": target_node.get("answer_likely"),
                    "score": target_node.get("score"),
                    "uncertainty": target_node.get("uncertainty"),
                })
                belief_content: List[Dict[str, Any]] = [
                    {"type": "text", "text": f"Frame at [{format_timestamp(ts)}]:"},
                    _build_image_content(fp),
                    {
                        "type": "text",
                        "text": prompts.BELIEF_UPDATE_TEMPLATE.format(
                            node_id=target_node.get("node_id", ""),
                            gap_start=int(target_node["gap_start"]),
                            gap_end=int(target_node["gap_end"]),
                            depth=int(target_node.get("depth", 0)),
                            prev_likely=_node_prev_likely(target_node, prompts),
                            prev_score=float(target_node.get("score", 0.5)),
                            prev_uncertainty=target_node.get("uncertainty", "medium"),
                            prev_reason=target_node.get("reason_token", ""),
                            probe_time=int(ts),
                            timestamp_fmt=format_timestamp(ts),
                            duration_fmt=duration_fmt,
                            query=raw_query,
                        ),
                    },
                ]
                belief_resp, tok = _call_api(
                    client, config.VERIFY_MODEL,
                    prompts.BELIEF_UPDATE_SYSTEM,
                    belief_content,
                    max_tokens=160,
                )
                total_tokens += tok
                update_outputs_this_round.append(belief_resp)
                update = _parse_belief_update(belief_resp, target_node, duration)
                if update:
                    old_region = (target_node["gap_start"], target_node["gap_end"])
                    target_node["answer_likely"] = update["answer_likely"]
                    target_node["score"] = update["score"]
                    target_node["uncertainty"] = update["uncertainty"]
                    target_node["scene_cut"] = update["scene_cut"]
                    target_node["reason_token"] = update["reason_token"]
                    if update.get("scene_hint"):
                        target_node["scene_hint"] = update["scene_hint"]
                    if update["action"] == "SPLIT" and len(update["children"]) == 2:
                        target_node["status"] = "split"
                        prediction.belief_nodes.extend(update["children"])
                    elif update["action"] == "ELIMINATE" or update["answer_likely"] == "NO":
                        target_node["status"] = "eliminated"
                    if verbose:
                        print(
                            f"[Phase 2.5 v9i] Node {target_node.get('node_id')} "
                            f"[{old_region[0]:.0f}-{old_region[1]:.0f}s] "
                            f"action={update['action']} children={len(update['children'])} "
                            f"label={update['answer_likely']} score={update['score']:.2f}"
                        )
            old_primary = prediction.primary_region
            if _derive_prediction_from_belief(prediction):
                _effective_confidence = prediction.primary_confidence
                if verbose:
                    print(
                        f"[Phase 2.5 v9i] Re-derived primary: "
                        f"[{old_primary[0]:.0f}-{old_primary[1]:.0f}s] → "
                        f"[{prediction.primary_region[0]:.0f}-{prediction.primary_region[1]:.0f}s] "
                        f"conf={prediction.primary_confidence}"
                    )
            phase25_probe_nodes_per_round.append(probe_nodes_this_round)
            phase25_update_raw_outputs_per_round.append(update_outputs_this_round)
            continue

        # Legacy path: caption probe frame, then ask text model to update.
        phase25_probe_nodes_per_round.append(probe_nodes_this_round)
        phase25_update_raw_outputs_per_round.append(update_outputs_this_round)
        new_captions_list: List[Tuple[float, str]] = []
        for fp, ts in zip(new_frame_paths, probe_targets):
            caption_text, cap_tok = _probe_caption_vlm(
                client, fp, ts, duration_fmt, raw_query,
            )
            total_tokens += cap_tok
            new_captions_list.append((ts, caption_text))
        captions.extend(new_captions_list)

        if verbose:
            print(f"[Phase 2.5] New captions:")
            for ts, cap in new_captions_list:
                print(f"  [{format_timestamp(ts)}] {cap}")

        # Incremental refinement: ask LLM to LOCALLY update, not rewrite
        new_captions_block = _format_captions_block(new_captions_list)
        previous_prediction_str = (
            f"primary: [{prediction.primary_region[0]:.0f}-{prediction.primary_region[1]:.0f}] "
            f"confidence: {prediction.primary_confidence}\n"
            f"secondary: [{prediction.secondary_region[0]:.0f}-{prediction.secondary_region[1]:.0f}]"
        )

        probe_input = prompts.ADAPTIVE_PROBE_TEMPLATE.format(
            previous_structure=prediction.raw_output,
            previous_prediction=previous_prediction_str,
            new_captions_block=new_captions_block,
            query=raw_query,
        )
        probe_output, tok = _call_api(
            client, config.STRUCTURE_MODEL,
            prompts.ADAPTIVE_PROBE_SYSTEM,
            probe_input,
            max_tokens=config.MAX_TOKENS_STRUCTURE,
        )
        total_tokens += tok

        # Parse incremental update — only change prediction if LLM says "update"
        pred_update_m = re.search(r"<prediction_update>(.*?)</prediction_update>", probe_output, re.DOTALL)
        if pred_update_m:
            update_text = pred_update_m.group(1).strip()
            if update_text.lower().startswith("keep"):
                if verbose:
                    print(f"[Phase 2.5] LLM says KEEP prediction: [{prediction.primary_region[0]:.0f}-{prediction.primary_region[1]:.0f}s]")
            else:
                new_region = _parse_time_range(update_text)
                if new_region:
                    # Validate: reject 0-duration or inverted regions from the update
                    if new_region[0] >= new_region[1]:
                        if verbose:
                            print(f"[Phase 2.5] Rejected invalid update region {new_region}, keeping original.")
                    else:
                        old_primary = prediction.primary_region
                        prediction.primary_region = (max(0, new_region[0]), min(duration, new_region[1]))
                        # Re-compute effective confidence after update
                        new_region_dur = prediction.primary_region[1] - prediction.primary_region[0]
                        new_coverage = new_region_dur / duration if duration > 0 else 1.0
                        # Parse updated binary confidence from the tag if present
                        conf_m = re.search(r"confidence:\s*(certain|high|medium|low|uncertain)", update_text, re.IGNORECASE)
                        if conf_m:
                            raw_conf = conf_m.group(1).lower()
                            # 4-level scheme: keep certain/high/medium/low as-is.
                            # Legacy 'uncertain' (from v9e binary) maps to 'low'.
                            prediction.primary_confidence = "low" if raw_conf == "uncertain" else raw_conf
                        _effective_confidence = "low" if new_coverage > 0.80 else prediction.primary_confidence
                        if verbose:
                            print(f"[Phase 2.5] LLM says UPDATE: [{old_primary[0]:.0f}-{old_primary[1]:.0f}s] "
                                  f"→ [{prediction.primary_region[0]:.0f}-{prediction.primary_region[1]:.0f}s] "
                                  f"eff_conf={_effective_confidence}")
                            reason_m = re.search(r"reason:\s*(.*)", update_text, re.IGNORECASE)
                            if reason_m:
                                print(f"[Phase 2.5] Reason: {reason_m.group(1).strip()[:100]}")
                else:
                    if verbose:
                        print(f"[Phase 2.5] LLM said update but couldn't parse region, keeping original.")
        else:
            if verbose:
                print(f"[Phase 2.5] No <prediction_update> tag found, keeping original prediction.")

        # Parse local structure updates (for logging; structure_text stays mostly intact)
        local_m = re.search(r"<local_update>(.*?)</local_update>", probe_output, re.DOTALL)
        if local_m and verbose:
            print(f"[Phase 2.5] Local updates: {local_m.group(1).strip()[:200]}")

    # ─── Phase 3: VLM Verification (cloud API, with images) ───
    answer = ""
    confidence = 1
    stop_reason = "max_rounds_exceeded"
    num_vlm_calls = 0
    total_visual_tokens = 0  # Phase 3 only; full total computed at return

    # Exploration memory: tracks every region examined and what was observed there
    searched_regions: List[Tuple[float, float, str]] = []

    # Re-compute region coverage after Phase 2.5 updates
    region_dur = prediction.primary_region[1] - prediction.primary_region[0]
    region_coverage = region_dur / duration if duration > 0 else 1.0
    # Wide-region strategy only applies to long videos (>5 min).
    is_wide_region = (region_coverage > 0.65) and (duration > 300)

    # Query-type-aware frame selection strategy
    qtype = prediction.query_type.split("|")[0].strip()
    if is_wide_region or qtype == "global_understanding":
        default_frames = 12
        default_resize = "low"
        # global: cap the verify region at most 40% of duration to avoid dumping all into one call
        _verify_region_cap = duration
    elif qtype == "counting":
        default_frames = 6
        default_resize = "medium"
        # counting needs focused frames, cap verify region tightly
        _verify_region_cap = min(150.0, duration * 0.25)
    elif qtype == "detail_recognition":
        default_frames = 6
        default_resize = "high"
        # detail needs sharp close-up frames in a narrow window
        _verify_region_cap = min(120.0, duration * 0.20)
    else:
        default_frames = prediction.num_frames
        default_resize = prediction.resolution
        _verify_region_cap = duration

    for round_num in range(config.MAX_VERIFY_ROUNDS):
        is_final_round = (round_num == config.MAX_VERIFY_ROUNDS - 1)

        region = prediction.primary_region
        start_t, end_t = region

        # On the final round: expand search to merge primary + secondary for max coverage
        if is_final_round and prediction.secondary_region != (0, 0):
            sec_s, sec_e = prediction.secondary_region
            if sec_e - sec_s > 5 and (sec_s, sec_e) != (start_t, end_t):
                start_t = min(start_t, sec_s)
                end_t = max(end_t, sec_e)
                if verbose:
                    print(f"[Phase 3] Final round: expanded to merged region [{start_t:.0f}-{end_t:.0f}s]")

        # ── Query-type-aware region width cap ──────────────────────────────
        # detail/counting questions don't benefit from wide regions — more frames
        # in a narrow window beats sparse frames across a wide region.
        # The cap is only applied when the region is actually wider than the limit.
        if end_t - start_t > _verify_region_cap:
            mid = (start_t + end_t) / 2.0
            half = _verify_region_cap / 2.0
            start_t = max(0.0, mid - half)
            end_t = min(duration, start_t + _verify_region_cap)
            if verbose:
                print(f"[Phase 3] Region capped by qtype={qtype}: [{start_t:.0f}-{end_t:.0f}s]")

        # Frame count: use defaults on round 0, reduce on later rounds (except final)
        if round_num == 0:
            n_frames = default_frames
            resize = _resize_factor(default_resize)
        elif is_final_round:
            n_frames = max(default_frames, 10)
            resize = _resize_factor("low")
        else:
            n_frames = max(4, default_frames // 2)
            resize = _resize_factor(default_resize)

        est_tokens = estimate_visual_tokens(width, height, n_frames, resize)
        if est_tokens > config.MAX_VISUAL_TOKENS:
            r = (config.MAX_VISUAL_TOKENS / est_tokens) ** (1 / 3)
            n_frames = max(1, int(n_frames * r))
            resize = resize * r

        target_ts = timestamps_in_range(start_t, end_t, n_frames)
        phase3_regions_per_round.append([round(float(start_t), 3), round(float(end_t), 3)])
        phase3_timestamps_per_round.append([round(float(t), 3) for t in target_ts])
        target_paths = extract_frames(
            video_path, target_ts,
            os.path.join(frame_dir, f"phase3_r{round_num}"),
            resize=resize,
        )

        if not target_paths:
            stop_reason = "frame_extraction_failed"
            break

        round_vis_tok = estimate_visual_tokens(width, height, len(target_paths), resize)
        total_visual_tokens += round_vis_tok
        phase3_frames_per_round.append(len(target_paths))
        phase3_visual_tokens_per_round.append(round_vis_tok)
        num_vlm_calls += 1

        # Build multimodal message with base64 frames
        verify_content: List[Dict[str, Any]] = []
        for i, fp in enumerate(target_paths):
            ts = target_ts[i] if i < len(target_ts) else 0
            verify_content.append({"type": "text", "text": f"Frame {i} at [{format_timestamp(ts)}]:"})
            verify_content.append(_build_image_content(fp))

        user_tmpl = prompts.VERIFICATION_USER_TEMPLATE_FINAL if is_final_round else prompts.VERIFICATION_USER_TEMPLATE
        # Round 0 uses minimal system prompt (no observation) for speed.
        # Falls back gracefully if VERIFICATION_SYSTEM_R1 is not defined.
        if is_final_round:
            system_prompt = prompts.VERIFICATION_SYSTEM_FINAL
        elif round_num == 0 and hasattr(prompts, "VERIFICATION_SYSTEM_R1"):
            system_prompt = prompts.VERIFICATION_SYSTEM_R1
        else:
            system_prompt = prompts.VERIFICATION_SYSTEM

        # World-model context: gap predictions overlapping the verify region.
        # Falls back to empty string when the prompt set doesn't expose gap_inferences.
        wm_ctx = _gap_inference_for_region(prediction.gap_inferences, start_t, end_t)
        if not wm_ctx:
            wm_ctx = "(no inferred-state predictions available for this region)"

        verify_kwargs = {
            "structure_summary": (
                _format_belief_tree_block(prediction.belief_nodes)
                if prediction.belief_nodes
                else (prediction.structure_text[:300] if prediction.structure_text else "Not available")
            ),
            "start_fmt": format_timestamp(start_t),
            "end_fmt": format_timestamp(end_t),
            "duration_fmt": duration_fmt,
            "query": raw_query,
            "world_model_context": wm_ctx,
            "query_reasoning": prediction.query_reasoning or "(none)",
            # v9g: durable world-model memory and how many frames built it.
            # When the verify template doesn't reference these, they're silently
            # skipped — fully backward-compatible with v9a/d/e/f templates.
            "world_state": prediction.world_state or "(world_state not provided by Phase 1+2 prompt)",
            "num_frames_so_far": phase1_frames + phase25_frames,
        }
        # Only pass placeholders the template actually references → avoid KeyError
        # for old prompt files (prompts.py / v9d / v9e) that don't use them.
        try:
            tmpl_fields = {
                fname for _, fname, _, _ in __import__("string").Formatter().parse(user_tmpl)
                if fname
            }
        except Exception:
            tmpl_fields = set(verify_kwargs.keys())
        verify_text = user_tmpl.format(**{k: v for k, v in verify_kwargs.items() if k in tmpl_fields})
        verify_content.append({"type": "text", "text": verify_text})

        if verbose:
            print(f"\n[Phase 3] Round {round_num}{'(FINAL)' if is_final_round else ''}: "
                  f"region=[{start_t:.0f}-{end_t:.0f}s], {len(target_paths)} frames, "
                  f"resize={resize:.2f}, wide={is_wide_region}, qtype={qtype}")

        vlm_response, tok = _call_api(
            client, config.VERIFY_MODEL,
            system_prompt,
            verify_content,
            max_tokens=config.MAX_TOKENS_VERIFY,
        )
        total_tokens += tok
        phase3_responses_per_round.append(vlm_response[:1000])

        answer = _extract_answer(vlm_response)
        confidence = _extract_confidence(vlm_response)
        observation = _extract_observation(vlm_response)

        # Record this search in exploration memory
        obs_summary = observation or (f"answer={'found' if answer else 'not found'}")
        searched_regions.append((start_t, end_t, obs_summary))
        if prediction.belief_nodes:
            _demote_belief_nodes_for_search(
                prediction.belief_nodes,
                start_t,
                end_t,
                found=bool(answer and confidence >= config.CONFIDENCE_THRESHOLD),
            )
            if not (answer and confidence >= config.CONFIDENCE_THRESHOLD):
                restored_node = None
                if not frontier_restored_once:
                    restored_node = _restore_frontier_if_all_no(prediction.belief_nodes)
                if restored_node is not None:
                    frontier_restored_once = True
                    frontier_restores.append({
                        "phase": "phase3_after_search",
                        "node_id": restored_node.get("node_id"),
                        "start": restored_node.get("gap_start"),
                        "end": restored_node.get("gap_end"),
                        "round": round_num,
                    })
                _derive_prediction_from_belief(prediction)

        if verbose:
            print(f"[Phase 3] VLM response: {vlm_response[:200]}")
            print(f"[Phase 3] Answer={answer}, Confidence={confidence}, GT={gt}")
            print(f"[Phase 3] Observation: {obs_summary[:100]}")

        if answer and confidence >= config.CONFIDENCE_THRESHOLD:
            stop_reason = "answer_found"
            break

        if not is_final_round:
            found_status = "found but uncertain" if answer else "NOT found"
            instruction = prompts.CORRECTION_FOUND if answer else prompts.CORRECTION_NOT_FOUND

            # Build search history block (all rounds so far)
            history_lines = [
                f"  Round {i+1}: [{format_timestamp(s)}-{format_timestamp(e)}] — {obs}"
                for i, (s, e, obs) in enumerate(searched_regions)
            ]
            search_history_block = "\n".join(history_lines) if history_lines else "  (none yet)"

            # Compute remaining unsearched hint
            remaining_hint = _compute_remaining_hint(duration, searched_regions)

            correction_kwargs = {
                "all_captions_block": (
                    _format_belief_tree_block(prediction.belief_nodes)
                    if prediction.belief_nodes
                    else _format_captions_block(captions)
                ),
                "search_history_block": search_history_block,
                "remaining_hint": remaining_hint,
                "query": raw_query,
                "correction_instruction": instruction,
                "unsearched_gap_hints": (
                    _compute_unsearched_belief_hints(prediction.belief_nodes, searched_regions)
                    if prediction.belief_nodes
                    else "(belief frontier not available)"
                ),
            }
            try:
                corr_fields = {
                    fname for _, fname, _, _ in __import__("string").Formatter().parse(
                        prompts.CORRECTION_USER_TEMPLATE
                    )
                    if fname
                }
            except Exception:
                corr_fields = set(correction_kwargs.keys())
            correction_input = prompts.CORRECTION_USER_TEMPLATE.format(
                **{k: v for k, v in correction_kwargs.items() if k in corr_fields}
            )
            correction_output, tok = _call_api(
                client, config.STRUCTURE_MODEL,
                prompts.CORRECTION_SYSTEM,
                correction_input,
                max_tokens=config.MAX_TOKENS_STRUCTURE,
            )
            total_tokens += tok
            correction_raw_outputs.append(correction_output)
            final_navigation_raw_output = correction_output
            new_pred = parse_structure_output(correction_output, duration)
            prediction.primary_region = new_pred.primary_region
            prediction.secondary_region = new_pred.secondary_region
            correction_regions.append([
                round(float(new_pred.primary_region[0]), 3),
                round(float(new_pred.primary_region[1]), 3),
            ])
            prediction.primary_confidence = new_pred.primary_confidence
            prediction.num_frames = new_pred.num_frames
            prediction.resolution = new_pred.resolution
            if new_pred.query_type and new_pred.query_type != "global_understanding":
                prediction.query_type = new_pred.query_type

    if not answer:
        stop_reason = "no_answer"
    elif stop_reason != "answer_found":
        stop_reason = "low_confidence_answer"

    shutil.rmtree(frame_dir, ignore_errors=True)

    phase3_frames_total = sum(phase3_frames_per_round)
    phase3_visual_tokens_total = sum(phase3_visual_tokens_per_round)
    all_visual_tokens = phase1_visual_tokens + phase25_visual_tokens + phase3_visual_tokens_total

    return {
        "index": index,
        "videos": item.get("videos", []),
        "gt": gt,
        "answer": answer,
        "stop_reason": stop_reason,
        "num_rounds": num_vlm_calls,
        "total_tokens": total_tokens,
        # ── Per-phase frame / visual-token breakdown ──
        "frame_stats": {
            "phase1_frames": phase1_frames,
            "phase1_visual_tokens": phase1_visual_tokens,
            "phase25_frames": phase25_frames,
            "phase25_visual_tokens": phase25_visual_tokens,
            "phase3_frames_per_round": phase3_frames_per_round,
            "phase3_visual_tokens_per_round": phase3_visual_tokens_per_round,
            "phase3_frames_total": phase3_frames_total,
            "phase3_visual_tokens_total": phase3_visual_tokens_total,
            "total_frames": phase1_frames + phase25_frames + phase3_frames_total,
            "total_visual_tokens": all_visual_tokens,
        },
        # Legacy field — Phase 3 only (kept for backward-compatible eval scripts)
        "estimated_visual_tokens": phase3_visual_tokens_total,
        "phase1_captions": [c for _, c in sorted(captions, key=lambda x: x[0])],
        "phase1_boundaries": boundaries_block,
        "phase1_belief_raw_output": phase1_belief_raw_output,
        "phase2_structure": prediction.structure_text,
        "phase2_raw_output": prediction.raw_output,
        "final_navigation_raw_output": final_navigation_raw_output,
        "phase2_prediction": {
            "primary": list(prediction.primary_region),
            "secondary": list(prediction.secondary_region),
            "confidence": prediction.primary_confidence,
            "query_type": prediction.query_type,
            "uncertain_regions": prediction.uncertain_regions,
        },
        "phase2_belief_nodes": prediction.belief_nodes,
        "trajectory": {
            "phase1_probe_timestamps": [round(float(t), 3) for t in probe_timestamps],
            "phase25_probe_targets_per_round": phase25_probe_targets_per_round,
            "phase25_probe_nodes_per_round": phase25_probe_nodes_per_round,
            "phase25_update_raw_outputs_per_round": phase25_update_raw_outputs_per_round,
            "phase3_regions_per_round": phase3_regions_per_round,
            "phase3_timestamps_per_round": phase3_timestamps_per_round,
            "phase3_responses_per_round": phase3_responses_per_round,
            "correction_raw_outputs": correction_raw_outputs,
            "correction_regions": correction_regions,
            "frontier_restores": frontier_restores,
        },
        "query": raw_query,
    }
