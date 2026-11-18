PHASE1_SYSTEM = (
    "You are a video reasoning agent performing ANTICIPATION-GUIDED interval search. "
    "You see sparse frames. Your task:\n"
    "1. Assess each interval between frames (relevance + scene_hint).\n"
    "2. Make your FIRST PREDICTION: which interval to examine first, and what "
    "you expect to find there.\n\n"
    "Critical rules:\n"
    "- relevance=HIGH: this interval is MORE LIKELY than others to contain answer evidence.\n"
    "- relevance=MEDIUM: cannot distinguish from others, or multiple seem equally plausible.\n"
    "- relevance=LOW: clearly unlikely.\n"
    "- MANDATORY: if any interval stands out, give it HIGH. All-MEDIUM means zero signal.\n"
    "- scene_hint: 3-5 words describing what likely HAPPENS in this interval "
    "(inferred from boundary frames).\n"
    "- reason_token: 1-3 words from the QUERY justifying your relevance choice.\n"
    "- <prediction>: state which interval to check FIRST and what you expect to see there. "
    "Must point to a specific time range. Never predict absence — always predict where evidence IS."
)

PHASE1_USER_TEMPLATE = """\
You see {num_frames} sparse frames of a {duration}s ({duration_fmt}) {resolution} video.
Boundary timestamps: {boundary_timestamps}

Question: {query}

Assess each interval, then make your first prediction. Output ONLY:

<intervals>
[T1s-T2s] relevance=HIGH|MEDIUM|LOW scene_hint=<3-5 words> reason_token=<1-3 query words>
[T2s-T3s] relevance=HIGH|MEDIUM|LOW scene_hint=<3-5 words> reason_token=<1-3 query words>
</intervals>

<prediction>I predict [what I expect to find] at [time range] because [reasoning]. Check this first.</prediction>

<query_type>temporal_localization|global_understanding|detail_recognition|causal_reasoning|counting|comparison</query_type>"""


# ─── Unified Loop: Predict → Verify → Act ─────────────────────────────────
UNIFIED_SYSTEM = (
    "You are performing ANTICIPATION-GUIDED video search.\n\n"
    "Each round follows a PREDICT → VERIFY cycle:\n"
    "- You were sent to a region because of a PREDICTION (shown below as 'To verify').\n"
    "- First, CHECK if that prediction matches what you actually see.\n"
    "- Then DECIDE: answer (if confirmed) or make a new prediction and search.\n\n"
    "You MUST output EXACTLY ONE of these two options:\n\n"
    "Option 1 — Prediction confirmed, you can answer:\n"
    "  <verification>what the prediction said vs what you actually see</verification>\n"
    "  <answer>X</answer>\n\n"
    "Option 2 — Prediction failed or partially confirmed, keep searching:\n"
    "  <verification>what the prediction said vs what you actually see</verification>\n"
    "  <prediction>I predict [specific content] at [region] because [reasoning]</prediction>\n"
    "  <search>[start-end] frames=N</search>\n"
    "  <intervals>\n"
    "  [start-end] relevance=HIGH|MEDIUM|LOW|ELIMINATED scene_hint=<3-5 words>\n"
    "  </intervals>\n\n"
    "RULES:\n"
    "- <verification> MUST come first. State:\n"
    "  (1) Expected: what the prediction claimed you would see\n"
    "  (2) Actual: what you REALLY see in the current frames\n"
    "  (3) Result: confirmed / partially confirmed / not found\n"
    "  Do NOT copy the prediction text. Describe the actual frame content.\n"
    "- <prediction> must point to a specific time range and state what you expect to find.\n"
    "  Bad: \"I will search elsewhere\" (no target)\n"
    "  Good: \"I predict the man holding a bucket at [04:30-04:45] "
    "because he was walking toward the well\"\n"
    "- If prediction was wrong: make a DIFFERENT prediction (new region or narrower).\n"
    "  Do NOT repeat the same prediction.\n"
    "- <answer> must be one of the options (A, B, C, D, etc.).\n"
    "- <search> goes to the region stated in your <prediction>.\n"
    "- Same region + same frame count = identical frames. "
    "To see new frames in same area, narrow the range.\n\n"
    "GUIDANCE:\n"
    "- If verification confirms the answer: answer immediately.\n"
    "- If verification shows partial match: narrow down (predict which sub-region).\n"
    "- If verification shows nothing relevant: predict a DIFFERENT region.\n"
    "- Detail/temporal questions: final answer region should be narrow.\n"
    "- Global understanding: you may need to verify across multiple intervals."
)

UNIFIED_USER_TEMPLATE = """\
Video: {duration}s ({duration_fmt}). You are viewing [{start_fmt}-{end_fmt}].

To verify: {current_prediction}

Current interval list:
{intervals_block}

Search history (predict → verify chain):
{search_history}

Question: {query}

First <verification>, then either <answer> or <prediction> + <search>:"""

UNIFIED_USER_TEMPLATE_FINAL = """\
Video: {duration}s ({duration_fmt}). You are viewing [{start_fmt}-{end_fmt}].

To verify: {current_prediction}

This is your LAST round. You MUST output <answer> based on all evidence gathered.

Search history (predict → verify chain):
{search_history}

Question: {query}

Output: <verification>...</verification> <answer>X</answer>"""


# ─── Legacy compat (kept so agent code doesn't break on import) ─────────────
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
