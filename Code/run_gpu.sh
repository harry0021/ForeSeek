set -e

# ─── Paths ───
export VIDEO_ROOT="${VIDEO_ROOT:-/path/to/aaai/eva/our_project/video_data/videos}"
export VIDEO_ROOT_LVB="${VIDEO_ROOT_LVB:-/path/to/datasets/LongVideoBench/videos}"
export VIDEO_ROOT_LVBENCH="${VIDEO_ROOT_LVBENCH:-/path/to/datasets/LVBench/videos}"
export VIDEO_ROOT_HOLMES="${VIDEO_ROOT_HOLMES:-/path/to/datasets/Video-Holmes/videos}"
export VIDEO_ROOT_MMU="${VIDEO_ROOT_MMU:-/path/to/datasets/VideoMMMU/videos}"
# DATA_DIR: directory containing all JSONL dataset files.
# Defaults to "data/" relative to the Foresight directory; override as needed:
#   DATA_DIR=/group/.../aaai/eva/our_project/baseEVA bash run_gpu.sh
export DATA_DIR="${DATA_DIR:-$(dirname "$0")/data}"

# ─── Local vLLM mode ─────────────────────────────────────────────────────────
LOCAL="${LOCAL:-0}"                              # set LOCAL=1 to use local vLLM
LOCAL_URL="${LOCAL_URL:-http://localhost:8000/v1}"
LOCAL_MODEL="${LOCAL_MODEL:-}"                   # leave empty to auto-detect

# ─── Cloud API (used when LOCAL=0) ───────────────────────────────────────────
export OPENAI_API_BASE="${OPENAI_API_BASE:-YOUR_API_BASE_URL}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-YOUR_API_KEY}"

# When LOCAL=1, override API settings to point at local vLLM before any preflight
if [ "$LOCAL" = "1" ]; then
    export OPENAI_API_BASE="$LOCAL_URL"
    export OPENAI_API_KEY="no"
fi

# ─── Run settings ───
DATASET="${DATASET:-videomme_test}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/output_gpu}"
NUM_GPUS="${NUM_GPUS:-1}"
SAMPLE="${SAMPLE:-}"
VERBOSE="${VERBOSE:-1}"

cd "$(dirname "$0")"

EXTRA_ARGS=""
if [ -n "$SAMPLE" ]; then
    EXTRA_ARGS="--sample $SAMPLE"
fi
if [ "$VERBOSE" = "1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --verbose"
fi
if [ "$LOCAL" = "1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --local --local-url $LOCAL_URL"
    if [ -n "$LOCAL_MODEL" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --local-model $LOCAL_MODEL"
    fi
fi
# Multi-machine sharding: set SHARD_ID and NUM_SHARDS env vars to activate.
if [ "${NUM_SHARDS:-1}" -gt 1 ]; then
    EXTRA_ARGS="$EXTRA_ARGS --shard-id ${SHARD_ID:-0} --num-shards ${NUM_SHARDS}"
fi

echo "═══════════════════════════════════════════"
echo " ForesightAgent (Unified VLM Mode)"
echo "═══════════════════════════════════════════"
echo " Dataset:     $DATASET"
echo " Data dir:    $DATA_DIR"
echo " Video root:  $VIDEO_ROOT"
echo " LVB root:    $VIDEO_ROOT_LVB"
echo " Output:      $OUTPUT_DIR"
if [ "$LOCAL" = "1" ]; then
echo " API:         [LOCAL] $LOCAL_URL"
echo " Model:       ${LOCAL_MODEL:-auto-detect}"
else
echo " API:         $OPENAI_API_BASE"
fi
echo " Workers:     $NUM_GPUS"
echo " Prompts:     ${PROMPTS_MODULE:-prompts (default)}"
echo "═══════════════════════════════════════════"

# Hard assert: the requested prompts module must actually be importable AND
# must be the world-model variant when v9e/v9f are requested. Catches the env-var
# typo class of bugs (silent fallback to baseline prompts.py) before burning GPU hours.
if [ -n "${PROMPTS_MODULE:-}" ] && [ "${PROMPTS_MODULE}" != "prompts" ]; then
    python3 -c "
import importlib, sys
m = importlib.import_module('${PROMPTS_MODULE}')
print(f'  → loaded ${PROMPTS_MODULE} from {m.__file__}')
# v9e/v9f use replacement schema with [OBSERVED at T] inside <structure>
if '${PROMPTS_MODULE}' in ('prompts_v9e','prompts_v9f'):
    assert 'OBSERVED at T' in m.PHASE1_STRUCTURE_USER_TEMPLATE, \
        'PROMPTS_MODULE=${PROMPTS_MODULE} but template is NOT the world-model variant!'
    print('  → world-model template confirmed (v9e/f schema)')
# v9g keeps baseline <captions> + adds <world_state> and <gap_inference> on top
if '${PROMPTS_MODULE}' == 'prompts_v9g':
    assert '<world_state>' in m.PHASE1_STRUCTURE_USER_TEMPLATE, \
        'PROMPTS_MODULE=prompts_v9g but <world_state> tag is missing!'
    assert '<gap_inference>' in m.PHASE1_STRUCTURE_USER_TEMPLATE, \
        'PROMPTS_MODULE=prompts_v9g but <gap_inference> tag is missing!'
    assert '<captions>' in m.PHASE1_STRUCTURE_USER_TEMPLATE, \
        'PROMPTS_MODULE=prompts_v9g but baseline <captions> tag is missing!'
    print('  → v9g world-model-on-baseline template confirmed')
" || { echo 'ABORT: prompts module check failed'; exit 1; }
fi


# Preflight check — verify API connectivity
python3 -c "
from openai import OpenAI
import os
base = os.environ.get('OPENAI_API_BASE','')
key  = os.environ.get('OPENAI_API_KEY','')
if not base:
    print('WARNING: OPENAI_API_BASE not set')
else:
    print(f'API base: {base}')
    print('Preflight OK')
" 2>/dev/null || true

echo ""
echo "Starting evaluation..."

mkdir -p "$OUTPUT_DIR" 2>/dev/null || true
chmod 777 "$OUTPUT_DIR" 2>/dev/null || true

# Only pass --video-root when VIDEO_ROOT is explicitly overridden by the user.
# For dataset-specific roots (e.g. VIDEO_ROOT_LVB for longvideobench), config.py
# picks them up from the environment automatically — don't override with VIDEO_ROOT.
VIDEO_ROOT_ARG=""
if [ -n "${VIDEO_ROOT_OVERRIDE:-}" ]; then
    VIDEO_ROOT_ARG="--video-root $VIDEO_ROOT_OVERRIDE"
fi

python3 eval_foresight.py \
    --dataset "$DATASET" \
    $VIDEO_ROOT_ARG \
    --output-dir "$OUTPUT_DIR" \
    --num-gpus "$NUM_GPUS" \
    $EXTRA_ARGS

echo ""
echo "═══════════════════════════════════════════"
echo " Done!"
echo "═══════════════════════════════════════════"
