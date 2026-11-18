set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
PARQUET_IN=/path/to/datasets/CG-Bench/grpo_parquet_multi/train.parquet
JSONL_OUT=/path/to/datasets/CG-Bench/grpo_parquet_multi/train_eval.jsonl

K=${K:-8}                         # Number of rollout rounds per shard
NUM_GPUS=${NUM_GPUS:-40}
TEMPERATURE=${TEMPERATURE:-0.7}   # >0 for stochastic rollouts
OUTPUT_BASE=${OUTPUT_BASE:-/path/to/foresight/output_passk}
PROMPTS_MODULE=${PROMPTS_MODULE:-prompts_v9k2}
AGENT_MODULE=${AGENT_MODULE:-foresight_agent_v9k2}
MAX_TOKENS_PHASE1=${MAX_TOKENS_PHASE1:-2400}
DATA_DIR=${DATA_DIR:-/path/to/aaai/eva/our_project/data}

# Multi-machine sharding: set SHARD_ID (0/1/2) and NUM_SHARDS (e.g. 3) on each machine.
# Results are saved with a shard tag in the filename so they can be merged later.
SHARD_ID=${SHARD_ID:-0}
NUM_SHARDS=${NUM_SHARDS:-1}

PROJ=/path/to/foresight

# ─── Step 1: Convert parquet → JSONL (idempotent) ────────────────────────────
if [ ! -f "$JSONL_OUT" ]; then
    echo "[Step 1] Converting parquet → JSONL ..."
    python3 "$PROJ/scripts/convert_train_parquet_to_jsonl.py" \
        --input  "$PARQUET_IN" \
        --output "$JSONL_OUT"
else
    echo "[Step 1] JSONL already exists: $JSONL_OUT"
fi

# ─── Step 2: K evaluation rounds ─────────────────────────────────────────────
echo ""
echo "[Step 2] Running K=$K rounds  shard=${SHARD_ID}/${NUM_SHARDS}  TEMPERATURE=$TEMPERATURE ..."

for k in $(seq 1 "$K"); do
    ROUND_DIR="$OUTPUT_BASE/round_${k}"
    mkdir -p "$ROUND_DIR"

    echo ""
    echo "=== Round $k / $K  shard $SHARD_ID/$NUM_SHARDS  →  $ROUND_DIR ==="

    PROMPTS_MODULE=$PROMPTS_MODULE \
    AGENT_MODULE=$AGENT_MODULE \
    LOCAL=1 \
    MAX_TOKENS_PHASE1=$MAX_TOKENS_PHASE1 \
    DATASET=train_grpo \
    NUM_GPUS=$NUM_GPUS \
    VERBOSE=0 \
    TEMPERATURE=$TEMPERATURE \
    TRAIN_GRPO_JSONL=$JSONL_OUT \
    DATA_DIR=$DATA_DIR \
    OUTPUT_DIR=$ROUND_DIR \
    SHARD_ID=$SHARD_ID \
    NUM_SHARDS=$NUM_SHARDS \
    bash "$PROJ/run_gpu.sh"
done

echo ""
echo "=== Shard $SHARD_ID: all $K rounds completed ==="
if [ "$NUM_SHARDS" -gt 1 ]; then
    echo "After collecting results from ALL $NUM_SHARDS machines, merge and filter:"
    echo "  python3 $PROJ/scripts/filter_by_passk.py \\"
    echo "      --rounds-dirs $OUTPUT_BASE [machine1_output_passk] [machine2_output_passk] \\"
    echo "      --k $K \\"
    echo "      --min-pass 1 --max-pass $((K-1)) \\"
    echo "      --parquet-in $PARQUET_IN \\"
    echo "      --parquet-out /path/to/datasets/CG-Bench/grpo_parquet_multi/train_select.parquet"
else
    echo "Run the filtering script:"
    echo "  python3 $PROJ/scripts/filter_by_passk.py \\"
    echo "      --rounds-dirs $OUTPUT_BASE \\"
    echo "      --k $K \\"
    echo "      --min-pass 1 --max-pass $((K-1)) \\"
    echo "      --parquet-in $PARQUET_IN \\"
    echo "      --parquet-out /path/to/datasets/CG-Bench/grpo_parquet_multi/train_select.parquet"
fi
