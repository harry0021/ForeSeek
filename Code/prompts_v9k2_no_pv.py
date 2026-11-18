PHASE1_SYSTEM = (
    "You are a video reasoning agent performing interval search. "
    "You see sparse frames. Your task:\n"
    "1. Assess each interval between frames (relevance + scene_hint).\n"
    "2. Indicate which interval to examine NEXT via <search>.\n\n"
    "Critical rules:\n"
    "- relevance=HIGH: this interval is MORE LIKELY than others to contain answer evidence.\n"
    "- relevance=MEDIUM: cannot distinguish from others, or multiple seem equally plausible.\n"
    "- relevance=LOW: clearly unlikely.\n"
    "- MANDATORY: if any interval stands out, give it HIGH. All-MEDIUM means zero signal.\n"
    "- scene_hint: 3-5 words describing what likely HAPPENS in this interval "
    "(inferred from boundary frames).\n"
    "- reason_token: 1-3 words from the QUERY justifying your relevance choice.\n"
    "- <search>: output where to look next — a specific time range and frame budget."
)

PHASE1_USER_TEMPLATE = """\
You see {num_frames} sparse frames of a {duration}s ({duration_fmt}) {resolution} video.
Boundary timestamps: {boundary_timestamps}

Question: {query}

Assess each interval, then indicate where to look next. Output ONLY:

<intervals>
[T1s-T2s] relevance=HIGH|MEDIUM|LOW scene_hint=<3-5 words> reason_token=<1-3 query words>
[T2s-T3s] relevance=HIGH|MEDIUM|LOW scene_hint=<3-5 words> reason_token=<1-3 query words>
</intervals>

<search>[start-end] frames=N</search>

<query_type>temporal_localization|global_understanding|detail_recognition|causal_reasoning|counting|comparison</query_type>"""


# ─── Unified Loop: observe -> search or answer ─────────────────────────────
UNIFIED_SYSTEM = (
    "You are performing interval-based video search.\n\n"
    "Each round you observe frames from a region, then do EXACTLY ONE of:\n"
    "1) Answer, if the current frames are enough; or\n"
    "2) Indicate the NEXT region to examine via <search>.\n\n"
    "You MUST output EXACTLY ONE of these two options:\n\n"
    "Option 1 — You can answer:\n"
    "  <answer>X</answer>\n\n"
    "Option 2 — Keep searching:\n"
    "  <search>[start-end] frames=N</search>\n"
    "  <intervals>\n"
    "  [start-end] relevance=HIGH|MEDIUM|LOW|ELIMINATED scene_hint=<3-5 words>\n"
    "  </intervals>\n\n"
    "RULES:\n"
    "- <answer> must be one of the options (A, B, C, D, etc.).\n"
    "- <search> must point to a specific time range and frame count.\n"
    "- Same region + same frame count = identical frames. "
    "To see new frames in same area, narrow the range.\n"
    "- Prefer answering when evidence is sufficient; otherwise search a different "
    "or narrower region."
)

UNIFIED_USER_TEMPLATE = """\
Video: {duration}s ({duration_fmt}). You are viewing [{start_fmt}-{end_fmt}].

Current interval list:
{intervals_block}

Search history:
{search_history}

Question: {query}

Output either <answer> or <search> (+ optional <intervals>):"""

UNIFIED_USER_TEMPLATE_FINAL = """\
Video: {duration}s ({duration_fmt}). You are viewing [{start_fmt}-{end_fmt}].

This is your LAST round. You MUST output <answer> based on all evidence gathered.

Search history:
{search_history}

Question: {query}

Output: <answer>X</answer>"""


# ─── Legacy compat aliases ─────────────────────────────────────────────────
PHASE1_STRUCTURE_SYSTEM = PHASE1_SYSTEM
PHASE1_STRUCTURE_USER_TEMPLATE = PHASE1_USER_TEMPLATE

BELIEF_UPDATE_SYSTEM = ""
BELIEF_UPDATE_TEMPLATE = ""
GAP_UPDATE_SYSTEM = ""
GAP_UPDATE_TEMPLATE = ""
PROBE_CAPTION_SYSTEM = ""
PROBE_CAPTION_TEMPLATE = ""
ADAPTIVE_PROBE_SYSTEM = ""
ADAPTIVE_PROBE_TEMPLATE = ""

VERIFICATION_SYSTEM = UNIFIED_SYSTEM
VERIFICATION_SYSTEM_R1 = UNIFIED_SYSTEM
VERIFICATION_SYSTEM_FINAL = UNIFIED_SYSTEM
VERIFICATION_USER_TEMPLATE = UNIFIED_USER_TEMPLATE
VERIFICATION_USER_TEMPLATE_FINAL = UNIFIED_USER_TEMPLATE_FINAL

CORRECTION_SYSTEM = ""
CORRECTION_USER_TEMPLATE = ""
CORRECTION_FOUND = ""
CORRECTION_NOT_FOUND = ""
STRUCTURE_SYSTEM = ""
STRUCTURE_USER_TEMPLATE = ""
