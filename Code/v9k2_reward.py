from __future__ import annotations

import argparse
import json
import os
from typing import Optional


W_CORRECT     = float(os.environ.get("W_CORRECT",     "1.0"))  # v5: 0.0 → 1.0 (sole term in total)
W_CLUE_IOU    = float(os.environ.get("W_CLUE_IOU",    "0.0"))  # v5: 1.0 → 0.0 (interface kept)
W_NARROWING   = float(os.environ.get("W_NARROWING",   "0.0"))  # interface kept
W_CONSISTENCY = float(os.environ.get("W_CONSISTENCY", "0.0"))  # interface kept
W_VERIFY      = float(os.environ.get("W_VERIFY",      "0.0"))  # interface kept

VERIFY_MIN_CHARS = 30  # minimum chars for a verification to count

# ── Time tax ──────────────────────────────────────────────────────────────────
# Gentle per-round discount applied to clue_iou and narrowing_bonus.
# Continuous gradient avoids the collapse modes seen with step-functions.
#   n=1 → 1.00,  n=2 → 0.98,  n=3 → 0.96,  n=4 → 0.94,  n=5 → 0.92
TIME_TAX_PER_ROUND = 0.02

# ── Late time tax (v5.1) ──────────────────────────────────────────────────────
# For n >= TIME_TAX_START_ROUND, discount `correct` by LATE_TIME_TAX_PER_ROUND.
# Defaults: START=4, RATE=0.10
#   n=1,2,3 → 1.00, n=4 → 0.90, n=5 → 0.80
# Override via env vars:
#   TIME_TAX_START_ROUND=3 LATE_TIME_TAX_PER_ROUND=0.15 bash run.sh
TIME_TAX_START_ROUND     = int(os.environ.get("TIME_TAX_START_ROUND",     "4"))
LATE_TIME_TAX_PER_ROUND  = float(os.environ.get("LATE_TIME_TAX_PER_ROUND", "0.10"))

# ── Anti-hacking thresholds ───────────────────────────────────────────────────
# Minimum effective final interval width for tightness (prevents sub-10s hacks).
MIN_FINAL_WIDTH_SEC = 10.0

# "Stuck" pair detection (used in narrowing and consistency).
# A consecutive pair is stuck when IoU > threshold AND widths are similar.
STUCK_IOU_THRESH  = 0.85
STUCK_WIDTH_RATIO = 0.85   # min(w1,w2)/max(w1,w2)

# Consistency promise-diversity discount weight (0.5 = halve when all stuck).
PROMISE_DIVERSITY_DISCOUNT = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _interval_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


def _time_discount(num_rounds: int) -> float:
    """Gentle per-round efficiency multiplier (2% per round).

    Applied to clue_iou and narrowing_bonus.  Shallow enough not to cause
    round collapse, but creates a continuous gradient favouring efficiency.
    """
    return max(0.0, 1.0 - TIME_TAX_PER_ROUND * (num_rounds - 1))


def _time_discount_after_start(num_rounds: int) -> float:
    """Late-onset time discount: free for n < START, then linear tax.

    n < TIME_TAX_START_ROUND          → 1.0
    n >= TIME_TAX_START_ROUND         → 1.0 - rate × (n - START + 1)

    Defaults START=4, rate=0.10 →
        n=1,2,3 → 1.00 ;  n=4 → 0.90 ;  n=5 → 0.80
    """
    if num_rounds < TIME_TAX_START_ROUND:
        return 1.0
    excess = num_rounds - TIME_TAX_START_ROUND + 1
    return max(0.0, 1.0 - LATE_TIME_TAX_PER_ROUND * excess)


def _stagnation_rate(trajectory_regions: list[list[float]]) -> float:
    """Fraction of consecutive visit-pairs that are 'stuck' (same place, same size)."""
    if len(trajectory_regions) < 2:
        return 0.0
    stuck = 0
    for i in range(len(trajectory_regions) - 1):
        s1, e1 = float(trajectory_regions[i][0]),     float(trajectory_regions[i][1])
        s2, e2 = float(trajectory_regions[i + 1][0]), float(trajectory_regions[i + 1][1])
        w1, w2 = e1 - s1, e2 - s2
        iou   = _interval_iou(s1, e1, s2, e2)
        ratio = min(w1, w2) / max(w1, w2, 1e-6)
        if iou > STUCK_IOU_THRESH and ratio > STUCK_WIDTH_RATIO:
            stuck += 1
    return stuck / (len(trajectory_regions) - 1)


def _final_clue_iou(
    trajectory_regions: list[list[float]],
    clue_intervals: list[list[float]],
) -> float:
    """IoU between the FINAL visited region and the best-matching clue interval.

    "Final commit" IoU: only the model's last committed region is scored.
    Using global-max previously let the model spray many regions and get credit
    for any accidental overlap, incentivising exhausting all 5 rounds.
    """
    if not trajectory_regions or not clue_intervals:
        return 0.0
    s, e = float(trajectory_regions[-1][0]), float(trajectory_regions[-1][1])
    best = 0.0
    for cs, ce in clue_intervals:
        iou = _interval_iou(s, e, float(cs), float(ce))
        if iou > best:
            best = iou
    return best


def _consistency_score(
    trajectory_regions: list[list[float]],
    predicted_next_regions: list,
) -> float:
    """Mean IoU between each round's promised next region and what was actually visited.

    Promise-diversity discount: if consecutive promises are nearly identical
    (IoU > STUCK_IOU_THRESH), apply diversity_factor = 1 - 0.5×stuck_rate
    to penalise the "always promise the same wide interval" hack.
    """
    if not predicted_next_regions or len(trajectory_regions) < 2:
        return 0.0

    valid_promises = [p for p in predicted_next_regions if p is not None]
    if not valid_promises:
        return 0.0

    # Base: promise → action IoU
    pairs, total_iou = 0, 0.0
    n = min(len(predicted_next_regions), len(trajectory_regions) - 1)
    for i in range(n):
        promise = predicted_next_regions[i]
        if promise is None:
            continue
        actual = trajectory_regions[i + 1]
        total_iou += _interval_iou(
            float(promise[0]), float(promise[1]),
            float(actual[0]),  float(actual[1]),
        )
        pairs += 1
    base = total_iou / pairs if pairs > 0 else 0.0

    # Promise-diversity discount
    stuck_pairs = 0
    for i in range(len(valid_promises) - 1):
        p1, p2 = valid_promises[i], valid_promises[i + 1]
        if _interval_iou(float(p1[0]), float(p1[1]),
                         float(p2[0]), float(p2[1])) > STUCK_IOU_THRESH:
            stuck_pairs += 1
    stuck_rate = stuck_pairs / (len(valid_promises) - 1) if len(valid_promises) > 1 else 0.0
    diversity_factor = 1.0 - PROMISE_DIVERSITY_DISCOUNT * stuck_rate
    return base * diversity_factor


def _verify_quality_score(verifications: list[str]) -> float:
    """Fraction of rounds with non-trivial <verification> (≥ VERIFY_MIN_CHARS chars)."""
    if not verifications:
        return 0.0
    good = sum(1 for v in verifications if len((v or "").strip()) >= VERIFY_MIN_CHARS)
    return good / len(verifications)


def _narrowing_bonus(
    trajectory_regions: list[list[float]],
    duration: float,
    num_rounds: int,
) -> float:
    """Broad→narrow search quality score (only called when answer is correct).

    Single-round (n=1):
        WARNING: legacy bug — a flat 1.0 × td gave full credit, and the model
        learned to "throw one enormous interval and finish in one shot".
        Fix: score tightness only (no shrink term when there is only 1 region).
        Reward "real one-shot precision" only when the single region is tight.
            tightness = clip(1 - (max(10, w)/duration)/0.5, 0, 1)
            return tightness × td

    Multi-round (n≥2):
        A. tightness (70%): how tight is the FINAL region?
           effective_final = max(MIN_FINAL_WIDTH_SEC, actual)  ← 10s floor
           tightness = clip(1 - (effective_final/duration)/0.5, 0, 1)
        B. overall_shrink × (1 - stagnation_rate) (30%):
           overall_shrink = max(0, 1 - width[-1]/width[0])  ← net progress
           stagnation penalises re-visiting the same interval (stuck loops).
           Intermediate widening (backtracking) is allowed.

    The result is multiplied by _time_discount(num_rounds).
    """
    if not trajectory_regions or duration <= 0:
        return 0.0

    td = _time_discount(num_rounds)
    widths = [float(e) - float(s) for s, e in trajectory_regions]

    # Tightness (shared by single- and multi-round)
    effective_final = max(MIN_FINAL_WIDTH_SEC, widths[-1])
    tightness = max(0.0, 1.0 - (effective_final / duration) / 0.5)

    # Single-round: ONLY tightness (no shrink to score)
    if len(trajectory_regions) == 1:
        return tightness * td

    # Multi-round: tightness + shrink × anti-stagnation
    overall_shrink = max(0.0, 1.0 - widths[-1] / max(widths[0], 1e-6))
    stagnation     = _stagnation_rate(trajectory_regions)
    component_b    = overall_shrink * (1.0 - stagnation)
    return (0.7 * tightness + 0.3 * component_b) * td


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_trajectory_reward(
    final_answer: str,
    gt_answer: str,
    trajectory_regions: list[list[float]],
    clue_intervals: list[list[float]] | None,
    num_rounds: int,
    max_rounds: int = 5,
    *,
    duration: float = 0.0,
    predicted_next_regions: list | None = None,
    verifications: list[str] | None = None,
    w_correct: float     = W_CORRECT,
    w_clue_iou: float    = W_CLUE_IOU,
    w_narrowing: float   = W_NARROWING,
    w_consistency: float = W_CONSISTENCY,
    w_verify: float      = W_VERIFY,
) -> float:
    """Compute one scalar reward per trajectory.

    v5.1 (correct-only + late time-tax):
        td_late = _time_discount_after_start(num_rounds)
                # n<3 → 1.0; n=3 → 0.9; n=4 → 0.8; n=5 → 0.7
        R       = w_correct * correct × td_late
                + (other w_* terms default to 0)

    All components are still computed and written to reward_trace.jsonl via
    the agent_loop's reward_extra_info, but by default only correct × td_late
    contributes to the total.
    """
    # Compute every component (even if the weight is 0) so reward_trace can
    # log them and later ablations can restore other terms.
    correct = float(
        bool(final_answer) and final_answer.upper() == gt_answer.upper()
    )

    td       = _time_discount(num_rounds)
    td_late  = _time_discount_after_start(num_rounds)

    clue_iou_score = _final_clue_iou(
        trajectory_regions, clue_intervals or []
    ) * td

    narrowing = _narrowing_bonus(
        trajectory_regions, float(duration), num_rounds
    )

    consistency = _consistency_score(
        trajectory_regions, predicted_next_regions or []
    )

    verify_q = _verify_quality_score(verifications or [])

    # v4: no longer performs "if not correct: return 0.0"
    # v5.1: multiply the correct term by td_late (mild time tax that only
    # activates for n >= 3)
    reward = (
        w_correct     * correct * td_late
        + w_clue_iou    * clue_iou_score
        + w_narrowing   * narrowing
        + w_consistency * consistency
        + w_verify      * verify_q
    )
    return float(reward)


# ─────────────────────────────────────────────────────────────────────────────
# verl RewardFunction interface
# ─────────────────────────────────────────────────────────────────────────────

class V9K2RewardFunction:
    """Plug-in reward class for verl's ray_trainer (legacy interface)."""

    def __call__(self, data_batch) -> list[float]:
        rewards: list[float] = []
        for i in range(len(data_batch)):
            item = data_batch[i]
            rm      = item.non_tensor_batch.get("reward_model", {})
            extra   = item.non_tensor_batch.get("extra_info", {})
            metrics = item.non_tensor_batch.get("agent_metrics", {})

            r = compute_trajectory_reward(
                final_answer       = metrics.get("final_answer", ""),
                gt_answer          = rm.get("ground_truth", ""),
                trajectory_regions = extra.get("trajectory_regions", []),
                clue_intervals     = extra.get("clue_intervals", []),
                num_rounds         = int(metrics.get("n_rounds", 1)),
                duration           = float(extra.get("duration", 0.0) or 0.0),
                predicted_next_regions = extra.get("predicted_next_regions", []),
                verifications      = extra.get("verifications", []),
            )
            rewards.append(r)
        return rewards


# ─────────────────────────────────────────────────────────────────────────────
# Standalone evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

def _parse_search_regions(result: dict) -> list[list[float]]:
    traj = result.get("trajectory", {})
    regions = traj.get("phase3_regions_per_round", [])
    out = []
    for r in regions:
        if isinstance(r, (list, tuple)) and len(r) >= 2:
            out.append([float(r[0]), float(r[1])])
    return out


def evaluate_file(result_path: str, source_path: str) -> dict:
    """Aggregate reward stats over a result JSONL (offline, no consistency/verify)."""
    with open(source_path, encoding="utf-8") as f:
        src_raw = f.read().strip()
    src_items = json.loads(src_raw) if src_raw.startswith("[") else [
        json.loads(l) for l in src_raw.splitlines() if l.strip()
    ]
    qid_to_clue = {str(x.get("qid", "")): x.get("clue_intervals", []) for x in src_items}
    qid_to_dur  = {str(x.get("qid", "")): float(x.get("duration", 0.0) or 0.0) for x in src_items}

    rewards = []
    with open(result_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = str(rec.get("qid", ""))
            r = compute_trajectory_reward(
                final_answer       = rec.get("answer", ""),
                gt_answer          = rec.get("gt", ""),
                trajectory_regions = _parse_search_regions(rec),
                clue_intervals     = qid_to_clue.get(qid, []),
                num_rounds         = int(rec.get("num_rounds", 1)),
                duration           = qid_to_dur.get(qid, 0.0),
            )
            rewards.append(r)

    if not rewards:
        return {}
    import statistics
    return {
        "n":      len(rewards),
        "mean":   round(statistics.mean(rewards), 4),
        "median": round(statistics.median(rewards), 4),
        "min":    round(min(rewards), 4),
        "max":    round(max(rewards), 4),
        "stdev":  round(statistics.stdev(rewards) if len(rewards) > 1 else 0.0, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--source", required=True)
    args = ap.parse_args()
    print(json.dumps(evaluate_file(args.result, args.source), indent=2))


if __name__ == "__main__":
    main()
