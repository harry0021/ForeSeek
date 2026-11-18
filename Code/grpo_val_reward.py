from __future__ import annotations

import re
from typing import Any, Optional


_ANSWER_TAG_RE = re.compile(r"<answer>\s*([A-Za-z])", re.IGNORECASE)
_ANSWER_KV_RE = re.compile(r"\b(?:final\s+)?answer\s*[:：]\s*([A-Za-z])", re.IGNORECASE)
_LETTER_TAIL_RE = re.compile(r"\b([A-Da-d])\b")


def _extract_answer_letter(text: str) -> Optional[str]:
    if not text:
        return None
    # 1) Prefer <answer>X</answer>
    m = _ANSWER_TAG_RE.search(text)
    if m:
        return m.group(1).upper()
    # 2) "Answer: X" style
    m = _ANSWER_KV_RE.search(text)
    if m:
        return m.group(1).upper()
    # 3) Fallback: last standalone A-D letter in the tail
    tail = text[-800:]
    matches = _LETTER_TAIL_RE.findall(tail)
    if matches:
        return matches[-1].upper()
    return None


def _normalize_gt(gt: Any) -> str:
    if gt is None:
        return ""
    s = str(gt).strip()
    if not s:
        return ""
    # Sometimes ground_truth comes as "A", "A.", "A) text", "Answer: A"
    m = re.search(r"[A-Za-z]", s)
    return m.group(0).upper() if m else s.upper()[:1]


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    **kwargs,
):
    # For our custom data sources, compute a simple letter-match accuracy.
    #
    # ⚠️ CRITICAL: the returned dict MUST share the same keys as
    # v9k2_agent_loop._finalize's reward_extra_info, otherwise verl's
    # val aggregation (process_validation_metrics) fills missing keys with
    # None and np.mean crashes with "TypeError: unsupported operand for /:
    # 'NoneType' and 'int'".
    #
    # Schema must match v9k2_agent_loop._finalize exactly:
    #   {correct, clue_iou, narrowing, consistency, verify_q, n_rounds}
    # plus "score" (verl uses this as the primary reward signal).
    if isinstance(data_source, str) and data_source.startswith("grpo_multi_"):
        pred = _extract_answer_letter(solution_str or "")
        gt = _normalize_gt(ground_truth)
        correct = 1.0 if (pred is not None and gt and pred == gt) else 0.0
        return {
            "score":       correct,   # primary reward signal
            "correct":     correct,
            "clue_iou":    0.0,        # cannot recover trajectory info post-hoc
            "narrowing":   0.0,
            "consistency": 0.0,
            "verify_q":    0.0,
            "n_rounds":    0,
        }

    # Otherwise, fall back to verl's default registry
    from verl.utils.reward_score import default_compute_score

    return default_compute_score(
        data_source,
        solution_str,
        ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
