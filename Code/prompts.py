"""Prompt templates for ForesightAgent."""

# ─── Phase 1+2 (Unified, Merged): Multi-image captioning + scene boundary + structure prediction ───
PHASE1_STRUCTURE_SYSTEM = (
    "You are a video analyst and temporal reasoning expert. "
    "You will see sparse frames from a video and a question. "
    "Your task is to: (1) describe each frame focusing on query-relevant details, "
    "(2) identify scene boundaries, "
    "(3) reason about the full video's temporal structure — including what happens BETWEEN frames, "
    "and (4) predict where in the video the answer is most likely located.\n\n"
    "Confidence calibration — be conservative, most questions should be 'low' or 'medium':\n"
    "- 'high': the answer is DIRECTLY VISIBLE in one of the sampled frames "
    "(e.g. you can read the answer off the screen right now).\n"
    "- 'medium': strong visual clues point to a specific region, but the answer "
    "may be just outside the sampled frames (requires viewing nearby content).\n"
    "- 'low': the answer requires content between or outside the observed frames — "
    "you are inferring based on context, not direct observation. "
    "Use this whenever you are not sure, or when the question asks about specific "
    "details, counts, causal events, or temporal order that the frames do not directly show."
)

PHASE1_STRUCTURE_USER_TEMPLATE = """\
The {num_frames} frames below are sampled uniformly from a {duration}s ({duration_fmt}) {resolution} video.
Each frame is labeled with its timestamp.

Question: {query}

Instructions:
1. Describe each frame (1-2 sentences):
   - What is happening / who is present / setting
   - Any on-screen text, subtitles, or numbers if directly relevant to the question
   - Query-relevant visual details
2. Identify major scene/content changes between consecutive frames.
3. Reason about what happens between the frames you see (mark inferred segments).
4. Predict the time range most likely to contain the answer.

Rules:
- All time values must be integers in seconds (0 to {duration}).
- Never write "None", "N", "SS", or placeholders — always use actual numbers.
- For <uncertain>: write a time range if a segment's content is unclear, else write None.
- Confidence: default to 'low' unless a sampled frame directly shows the answer.
  Most questions about specific details, counts, causes, or order should be 'low' or 'medium'.

Output ONLY these tags (no explanation, no other text):

<captions>
[MM:SS] your description of this frame
[MM:SS] your description of this frame
</captions>

<boundaries>
- Scene cut between [MM:SS] and [MM:SS]
OR
None detected
</boundaries>

<structure>
type: [actual video type, e.g. tutorial/news/animation/documentary]
- [start-end] what happens (observed/inferred)
- [start-end] what happens (observed/inferred)
</structure>

<uncertain>[start-end] or None</uncertain>

<prediction>
primary: [start-end] confidence: high/medium/low
secondary: [start-end] confidence: high/medium/low
frames: [integer 4-16]
resolution: high/medium/low
</prediction>

<query_type>temporal_localization|global_understanding|detail_recognition|causal_reasoning|counting|comparison</query_type>"""

# ─── Phase 1 (legacy, separate captioning only — kept for reference) ───
PHASE1_SYSTEM = (
    "You are a video analyst. You will see sparse frames from a video and a question. "
    "For each frame, describe what you observe, focusing on details relevant to the question. "
    "Also identify any major scene or content changes between consecutive frames."
)

PHASE1_USER_TEMPLATE = """\
The {num_frames} frames below are sampled uniformly from a {duration}s ({duration_fmt}) {resolution} video.
Each frame is labeled with its timestamp in [MM:SS] format.

Question to keep in mind while describing: {query}

For each frame, write 1-2 sentences describing:
- What is happening / who is present
- Objects, text, numbers, or visual elements relevant to the question
- The setting or scene

After describing all frames, identify scene boundaries (major changes in setting, subject, or scene between consecutive frames).

Output ONLY these tags (no other text):

<captions>
[MM:SS] your description of this frame
[MM:SS] your description of this frame
</captions>

<boundaries>
- Scene cut between [MM:SS] and [MM:SS]
OR
None detected
</boundaries>"""

# ─── Phase 2.5 Probe: Single-frame caption via unified VLM ───
PROBE_CAPTION_SYSTEM = (
    "Describe this video frame in 1-2 sentences. "
    "Focus on what is happening and any details relevant to the question provided."
)

PROBE_CAPTION_TEMPLATE = """\
This frame is from timestamp {timestamp_fmt} in a {duration_fmt} video.

Question: {query}

Describe what you see in 1-2 sentences: what is happening, query-relevant details, and any on-screen text/subtitles/numbers visible (or "none" if absent).
Output ONLY the description (no tags, no preamble)."""

# ─── Phase 2: Predictive Structure Modeling (text-only, used in Phase 2.5 update and Phase 3 correction) ───
STRUCTURE_SYSTEM = (
    "You are a video structure analyst. Given sparse captions and a question, "
    "predict the video's temporal structure and where the answer is located. "
    "Be concise. Output ONLY the required tags, nothing else.\n\n"
    "Confidence calibration:\n"
    "- 'high': a caption explicitly describes the answer location "
    "(e.g. 'at the end the final dish is shown' → high for 'what is the final dish').\n"
    "- 'medium': reasonable inference from captions, some uncertainty.\n"
    "- 'low': captions give little clue; answer could be anywhere."
)

STRUCTURE_USER_TEMPLATE = """\
Video: {duration}s ({duration_fmt}), {resolution}, {num_frames} sampled frames.

Captions (timestamps in seconds):
{captions_block}

Boundaries: {boundaries_block}

Question: {query}

Using ONLY the captions above, fill every field with real values derived from the caption timestamps.
Rules:
- Time ranges must be integers in seconds, e.g. "30-75" means from 30s to 75s.
- Cover the full video duration ({duration}s) across all structure segments.
- If a segment's exact boundaries are uncertain, interpolate between the nearest caption timestamps.
- Never write "None", "N", "SS", or any placeholder — always use actual numbers.
- For <uncertain>: write a time range if unsure about content there, or write None if everything is clear.
- For frames: integer between 4 and 16.
- Confidence is 3-level: 'high', 'medium', or 'low'. Default to 'low'.

Output ONLY these tags (no explanation, no other text):

<structure>
type: [actual video type, e.g. tutorial/news/animation/documentary]
- [start-end] what happens (observed/inferred)
- [start-end] what happens (observed/inferred)
</structure>

<uncertain>[start-end] or None</uncertain>

<prediction>
primary: [start-end] confidence: high/medium/low
secondary: [start-end] confidence: high/medium/low
frames: [integer 4-16]
resolution: high/medium/low
</prediction>

<query_type>temporal_localization|global_understanding|detail_recognition|causal_reasoning|counting|comparison</query_type>"""

# ─── Phase 2.5: Incremental Structure Refinement ───
ADAPTIVE_PROBE_SYSTEM = (
    "You are a video structure analyst performing an INCREMENTAL update. "
    "You have an existing structure prediction (the baseline). New frames have been sampled. "
    "Only update the parts affected by the new information. Keep everything else unchanged. "
    "Be concise. Output ONLY the required tags."
)

ADAPTIVE_PROBE_TEMPLATE = """\
## Baseline Structure (confirmed, do NOT change unless new captions contradict it)
{previous_structure}

## Baseline Answer Region
{previous_prediction}

## New Captions (just sampled at these timestamps)
{new_captions_block}

## Question
{query}

## Instructions
1. Review ONLY the time regions covered by the new captions.
2. Do NOT change structure segments that are far from the new captions.
3. Decide if the answer region needs adjustment based on the new information.
4. All time values must be integers in seconds — never write "None" or placeholder text.
5. Re-evaluate confidence (3-level): 'high' if new evidence clearly locates the answer, 'medium' if helpful but uncertain, 'low' if still unclear.

Output ONLY these tags:

<local_update>
- [start-end] no change / updated: [new description]
</local_update>

<prediction_update>keep / update: [start-end] confidence: high/medium/low reason: ...</prediction_update>

<query_type>temporal_localization|global_understanding|detail_recognition|causal_reasoning|counting|comparison</query_type>"""

# ─── Phase 3: Verification ───
VERIFICATION_SYSTEM = (
    "Answer the multiple-choice question based on the video frames. "
    "You are given the video's structure analysis as context to help you focus. "
    "Output ONLY (in this exact order):\n"
    "<answer>X</answer>\n"
    "<confidence>N</confidence>\n"
    "<observation>Content: [what frames show]; Text: [on-screen text/subtitles with timestamps, or 'none']; Next: [if answer not found, estimate where it is, e.g. 'try [30-45s]']</observation>\n"
    "where X is A/B/C/D/E/F and N is 1(insufficient)/2(partial)/3(sufficient)."
)

# Used on the FINAL verification round to guarantee an answer is produced.
VERIFICATION_SYSTEM_FINAL = (
    "Answer the multiple-choice question based on the video frames. "
    "This is your LAST attempt. You MUST choose one option — do not leave blank. "
    "Even if the frames are inconclusive, pick the most likely answer based on all available context. "
    "Output ONLY (in this exact order):\n"
    "<answer>X</answer>\n"
    "<confidence>N</confidence>\n"
    "<observation>Content: [what frames show]; Text: [on-screen text/subtitles, or 'none']; Next: [best remaining candidate region]</observation>\n"
    "where X is A/B/C/D/E/F and N is 1(insufficient)/2(partial)/3(sufficient)."
)

VERIFICATION_USER_TEMPLATE = """\
Video structure: {structure_summary}
You are viewing frames from [{start_fmt}-{end_fmt}] of a {duration_fmt} video.
This region was identified as most likely containing the answer.

Question: {query}

Output ONLY: <answer>X</answer> <confidence>N</confidence> <observation>Content: ...; Text: ...; Next: ...</observation>"""

VERIFICATION_USER_TEMPLATE_FINAL = """\
Video structure: {structure_summary}
You are viewing frames from [{start_fmt}-{end_fmt}] of a {duration_fmt} video.
This is the LAST attempt. You must select one answer even if uncertain.

Question: {query}

Output ONLY: <answer>X</answer> <confidence>N</confidence> <observation>Content: ...; Text: ...; Next: ...</observation>"""

# ─── Phase 3 Round 2+: Structure Correction with Exploration Memory ───
CORRECTION_SYSTEM = (
    "You are a video structure analyst performing SEARCH REFINEMENT. "
    "Follow the correction instruction carefully — behavior differs depending on "
    "whether the answer was found (but uncertain) or not found at all. "
    "Be concise. Output ONLY the required tags."
)

CORRECTION_USER_TEMPLATE = """\
## All Observations (captions accumulated so far)
{all_captions_block}

## Search History (regions already examined — ALL are ELIMINATED)
{search_history_block}

## Remaining Unsearched Regions
{remaining_hint}

## Question
{query}

{correction_instruction}
Output ONLY the same tags (<structure>, <uncertain>, <prediction>, <query_type>)."""

CORRECTION_FOUND = (
    "The answer WAS found but with low confidence. "
    "Stay near the current search region — do NOT jump to a completely different area. "
    "Refine: suggest a slightly narrower sub-region or adjacent segment. "
    "Ignore the 'Next' field in the observation for region selection."
)
CORRECTION_NOT_FOUND = (
    "The answer was NOT found in the searched region. "
    "Move to a NEW unsearched region. "
    "If the observation's 'Next' field contains a specific time estimate (e.g. 'try [30-45s]'), "
    "use it to set a NARROW primary region (10-25s) centered on that estimate. "
    "Otherwise, pick the most likely region from the remaining unsearched areas. "
    "Do NOT repeat any region listed in the Search History."
)
