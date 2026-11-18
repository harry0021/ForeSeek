#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_v9k2_grpo_8gpu_1node.sh
# GRPO fine-tuning of v9k2 (Qwen3-VL-8B) on 1 node × 8 H20
#
# 启动顺序（一台机器即可）：
#   1) export PYTHONPATH=... + NCCL_DEBUG=WARN（其它 NCCL 单机不用）
#   2) ray start --head --num-gpus=8 --num-cpus=64 --include-dashboard=false
#   3) ray status   # 期望: 1 node, 8.0 GPU
#   4) bash run_v9k2_grpo_8gpu_1node.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ────────────────────────── paths ───────────────────────────────────────────
GRPO_DIR=${GRPO_DIR:-"/path/to/foresight/grpo"}
VERL_DIR=${VERL_DIR:-"/path/to/verl"}

MODEL_PATH=${MODEL_PATH:-"/path/to/sft/checkpoint_unlock24f"}
GRPO_DATA=${GRPO_DATA:-"/path/to/datasets/CG-Bench/grpo_parquet_multi"}
VIDEO_ROOT=${VIDEO_ROOT:-"/path/to/datasets/CG-Bench/videos"}
OUTPUT_DIR=${OUTPUT_DIR:-"/path/to/foresight/grpo/outputs/v9k2_gspo_unlock24f"}

AGENT_LOOP_CONFIG="${GRPO_DIR}/v9k2_agent_config_unlock24f.yaml"

# ────────────────────────── project / experiment ────────────────────────────
PROJECT_NAME=${PROJECT_NAME:-"v9k2_grpo"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"qwen3vl_8b_8gpu_1node_gspo_$(date +%m%d_%H%M)"}

# ────────────────────────── hyper-params (8 GPU) ────────────────────────────
# 4 GPU → 8 GPU: batch 翻倍
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}        # 4→8
PPO_MINI_BATCH=${PPO_MINI_BATCH:-8}             # = TRAIN_BATCH_SIZE（避免内层 PPO epoch）
ROLLOUT_N=${ROLLOUT_N:-8}                       # GRPO 组大小不变
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-40960}             # 24帧clamp：worst-case图像token量低于32帧/64帧版，
                                                     # 沿用32帧档的保守值(40960)即有充足余量，无需按64帧档去过量预留
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-2048}
PPO_MAX_TOK_PER_GPU=${PPO_MAX_TOK_PER_GPU:-24576}   # 保持不变：24帧下实际 token 用量增幅比32/64帧档更小，OOM 风险更低
                                                     # 若仍出现 OOM，兜底：actor_rollout_ref.actor.fsdp_config.offload_policy=True
                                                     # （用 D2H/H2D 换显存，见下方 ACTOR 配置注释）

# Anti entropy-collapse 配置（v2：观察到 entropy 反向爆涨到 2.5+ 后调整）：
#   现象：entropy_coeff=0.005 + kl_loss_type=low_var_kl(k3) 同时存在时，
#         k3 估计器在 r≈1 处梯度坍缩 → KL 拉力实际为 0
#         entropy bonus 单方面把 entropy 从 0.5 推到 2.5（策略接近随机化）
#   修复：
#     1) entropy_coeff 0.005 → 0  : 关掉 entropy bonus，避免"loss hacking"
#     2) kl_loss_type   low_var_kl(k3) → kl(==k1) : 换成 MC 无偏估计，梯度不衰减
#         "kl" 和 "k1" 在 verl 里完全等价（都走 logprob - ref_logprob），
#         yaml 默认用 "kl" 这个别名，更稳
#     3) kl_loss_coef   0.01 → 0.02 : 略加 KL 拉力（k1 比 k3 准，不用大幅加）
#     4) lr 维持 5e-7 (从 1e-6 下调)
ACTOR_LR=${ACTOR_LR:-1e-6}
# GSPO + KL 双层保护：
#   - GSPO 的 clip 防"单步漂移"（trust region）
#   - KL(π_θ ‖ π_ref=SFT) 防"累计漂移"（anchor）
# 实验经验：纯 GSPO 不带 KL 时，entropy 不再崩，但 250+ step 后策略累计漂得很远，
# eval 反而跌破 SFT base。所以保留 KL 锚点。
USE_KL_LOSS=${USE_KL_LOSS:-True}
KL_COEF=${KL_COEF:-0.02}
ENTROPY_COEFF=${ENTROPY_COEFF:-0.0}
KL_LOSS_TYPE=${KL_LOSS_TYPE:-kl}
# Policy loss 形式：vanilla(默认 PPO) / gspo(序列级 importance ratio) / clip_cov / kl_cov / ...
# 切到 gspo：抑制 token-level 极端 ratio，配合 multi-turn agent 更稳，是 GRPO 标准的"升级版"
# 论文：https://arxiv.org/abs/2507.18071
POLICY_LOSS_MODE=${POLICY_LOSS_MODE:-gspo}
# GSPO 要求 seq-mean-token-mean（源码强制）
LOSS_AGG_MODE=${LOSS_AGG_MODE:-seq-mean-token-mean}
# GSPO clip：序列几何平均 ratio 抖动按 1/|y| 衰减，必须远小于 PPO 的 0.2
# 论文用 (3e-4, 4e-4)：非对称，上抬空间略大于下压
GSPO_CLIP_LOW=${GSPO_CLIP_LOW:-1e-3}
GSPO_CLIP_HIGH=${GSPO_CLIP_HIGH:-1.5e-3}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
SAVE_FREQ=${SAVE_FREQ:-50}
# val 暂时关闭：vLLM v1 在高并发 multimodal 场景下 mm_receiver_cache race
# 会把 EngineCore daemon 线程打死，导致整个训练 hang。先专注于 train，
# val 等训练完后用独立 eval pipeline 跑（更可控）。
TEST_FREQ=${TEST_FREQ:-99999}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-50}             # 100→50：降低 vLLM 并发以减少 mm_hash race condition

MAX_ROUNDS=${MAX_ROUNDS:-5}

# ────────────────────────── PYTHONPATH ──────────────────────────────────────
FORESIGHT_DIR="$(dirname "${GRPO_DIR}")"
export PYTHONPATH="${GRPO_DIR}:${FORESIGHT_DIR}:${VERL_DIR}:${PYTHONPATH:-}"

# ────────────────────────── attention / vLLM compat ─────────────────────────
export ATTN_IMPL=sdpa
export VLLM_SKIP_VLLM_CUSTOM_CONFIG_REGISTRATION=1
# 关闭 vLLM v1 的 multimodal 输入缓存，避免高并发 race 触发
# "AssertionError: Expected a cached item for mm_hash=..."
# 跟 enable_prefix_caching 不是同一个 cache
export VLLM_MM_INPUT_CACHE_GIB=0

# ────────────────────────── NCCL（单节点只需基础设定）──────────────────────
# 单机 8 卡走 NVLink，**不需要**跨节点的 bond1/IB/Simple/Ring/NSOCKS 配置
# 那些是为 LACP bond 多链路包乱序设的 workaround，单机反而可能降速
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 接入已有 Ray 集群
export RAY_ADDRESS=auto

# ────────────────────────── derived ─────────────────────────────────────────
NGPUS_PER_NODE=8     # 8 卡 H20
NNODES=1             # 单节点
ROLLOUT_TP=2         # vLLM TP=2 → DP=4（4 个 vLLM 实例，每实例 2 卡，sleep 时只占模型 shard ~8GB）

# ────────────────────────── parameter arrays ────────────────────────────────

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False

    data.train_files="${GRPO_DATA}/train_select.parquet"
    data.val_files="${GRPO_DATA}/test.parquet"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.val_batch_size=${VAL_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LEN}
    data.max_response_length=${MAX_RESPONSE_LEN}
    data.filter_overlong_prompts=True
    data.truncation=error
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.enable_activation_offload=False
    actor_rollout_ref.model.use_fused_kernels=False
)

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOK_PER_GPU}
    # ⚠️ use_kl_loss True：actor loss 加 KL(π_θ || π_ref) 项，约束策略不偏离 SFT
    # kl_loss_type=k1：MC 无偏估计 logπ_θ - logπ_ref，比 k3/low_var_kl 梯度更忠实
    # （k3 在 r≈1 时梯度为 0，实测在 entropy 飞涨时完全失效）
    # GSPO：用序列级 importance ratio (π_θ(y)/π_old(y))^(1/|y|) + 序列级 clip
    # advantage estimator 仍然是 grpo（GSPO 只换 pg_loss 算法，不换 advantage）
    actor_rollout_ref.actor.policy_loss.loss_mode=${POLICY_LOSS_MODE}
    actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE}
    # GSPO clip 范围必须远小于 PPO 默认 0.2，否则序列几何平均 ratio 永远在 [0.8,1.2] 内
    # → clip 失效 → 等于 vanilla PG 无 trust region
    # 论文推荐非对称：上抬空间略大于下压（Clip-Higher，鼓励探索）
    actor_rollout_ref.actor.clip_ratio_low=${GSPO_CLIP_LOW}
    actor_rollout_ref.actor.clip_ratio_high=${GSPO_CLIP_HIGH}
    actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS}
    actor_rollout_ref.actor.kl_loss_coef=${KL_COEF}
    actor_rollout_ref.actor.kl_loss_type=${KL_LOSS_TYPE}
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    # 8 GPU FSDP2: params/grads 各 1.1 GB/卡（8.77B/8），optimizer CPU offload
    # vLLM TP=2 时与 actor 共享 GPU（colocated），sleep mode 释放 ~14 GB
    # ⚠️ OOM 兜底：若训练中出现 backward OOM，第一步就是把这里改成 True
    # （用 D2H/H2D 换显存，牺牲吞吐但避免崩溃），无需改动其他参数
    actor_rollout_ref.actor.fsdp_config.offload_policy=False
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    # vLLM TP=2 × DP=4 = 4 个 vLLM 实例，最大化 rollout 并发
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=0.50
    actor_rollout_ref.rollout.max_model_len=45056         # 24帧clamp：沿用32帧档的45056 (40960+2048+余量)，需 >= max_prompt+max_response
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.calculate_log_probs=False
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOK_PER_GPU}
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True

    actor_rollout_ref.rollout.multi_turn.enable=True
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_ROUNDS}
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${MAX_ROUNDS}

    actor_rollout_ref.rollout.agent.default_agent_loop=v9k2_agent
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_CONFIG}"
)

REF=(
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOK_PER_GPU}
)

RESUME_MODE=${RESUME_MODE:-"disable"}          # "disable" | "auto" | "resume_path"
RESUME_PATH=${RESUME_PATH:-""}                 # 仅 resume_mode=resume_path 时使用

# ─── RESUME 显著提示，防止悄悄从 step 0 重训 ────────────────────────────────
echo ""
echo "▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓"
if [ "${RESUME_MODE}" = "disable" ]; then
    echo "▓ RESUME_MODE = disable  →  将从 STEP 0 重新开始训练 (不加载任何 checkpoint)"
elif [ "${RESUME_MODE}" = "auto" ]; then
    echo "▓ RESUME_MODE = auto     →  自动从 ${OUTPUT_DIR} 找最新 ckpt"
elif [ "${RESUME_MODE}" = "resume_path" ]; then
    echo "▓ RESUME_MODE = resume_path"
    echo "▓ RESUME_PATH = ${RESUME_PATH}"
    if [ ! -d "${RESUME_PATH}/actor" ]; then
        echo "▓ ❌ 错误：${RESUME_PATH}/actor 不存在！将无法 resume！"
        exit 1
    fi
    echo "▓ ✅ ckpt actor 目录存在，将从该 step resume"
else
    echo "▓ ❌ 未知 RESUME_MODE=${RESUME_MODE}"
    exit 1
fi
echo "▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓"
echo ""

REWARD=(
    # val 阶段需要 RewardLoopWorker 才能 compute_score；
    # 我们的 data_source=grpo_multi_* 不在 verl 默认注册表里，
    # 用这个 shim 提供最小可用的 letter-match accuracy
    #
    # ⚠️ 注意：main_ppo_sync.py 没有调用 migrate_legacy_reward_impl()，
    # 所以必须直接写到 reward.custom_reward_function.*，
    # 而 NOT 顶层 custom_reward_function.*（那个只有 main_ppo.py 才会迁移）
    reward.custom_reward_function.path="${GRPO_DIR}/grpo_val_reward.py"
    reward.custom_reward_function.name=compute_score
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.val_before_train=False
    trainer.default_local_dir="${OUTPUT_DIR}"
    trainer.resume_mode=${RESUME_MODE}
    trainer.resume_from_path="${RESUME_PATH}"
)

# ────────────────────────── launch ──────────────────────────────────────────
echo "================================================================="
echo " V9K2 GRPO | Qwen3-VL-8B | 1 node × 8 H20"
echo " model     : ${MODEL_PATH}"
echo " data      : ${GRPO_DATA}"
echo " output    : ${OUTPUT_DIR}"
echo " batch     : ${TRAIN_BATCH_SIZE}  mini_batch: ${PPO_MINI_BATCH}  rollout_n: ${ROLLOUT_N}"
echo " topology  : vLLM TP=${ROLLOUT_TP} × DP=$((NGPUS_PER_NODE / ROLLOUT_TP)) instances  |  FSDP across ${NGPUS_PER_NODE} GPU"
echo " ray addr  : ${RAY_ADDRESS}"
echo "================================================================="

python3 -m verl.trainer.main_ppo_sync \
    "${DATA[@]}"    \
    "${MODEL[@]}"   \
    "${ACTOR[@]}"   \
    "${ROLLOUT[@]}" \
    "${REF[@]}"     \
    "${REWARD[@]}"  \
    "${TRAINER[@]}" \
    "$@"

# ─────────────────────────────────────────────────────────────────────────────
# 启动顺序（单机一气呵成）
# ─────────────────────────────────────────────────────────────────────────────
#
# ── Step 1: 设置环境变量（注意 ray start 必须在 export 之后）──────────────
#
#   conda activate grpo
#   export PYTHONPATH="/path/to/foresight/grpo:/path/to/foresight:/path/to/verl:${PYTHONPATH:-}"
#   export NCCL_DEBUG=WARN
#   export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
#
# ── Step 2: 启动 Ray head（同机即 head 即 worker）─────────────────────────
#
#   ray start --head --num-gpus=8 --num-cpus=64 --include-dashboard=false
#
# ── Step 3: 验证集群 ──────────────────────────────────────────────────────
#
#   ray status
#   # 期望：Active: 1 node; Resources: 8.0 GPU
#
# ── Step 4: 启动训练 ──────────────────────────────────────────────────────
#
#   bash /path/to/foresight/grpo/run_v9k2_gspo_8gpu_1node-24f.sh
#
#   # 或后台跑 + 日志：
#   nohup bash /path/to/foresight/grpo/run_v9k2_gspo_8gpu_1node-24f.sh \
#     > /tmp/v9k2_8gpu.log 2>&1 &
#
# ── Step 5: 关闭集群（训练结束后）─────────────────────────────────────────
#
#   ray stop
#
# ─────────────────────────────────────────────────────────────────────────────
