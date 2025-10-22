#!/bin/bash


export WANDB_MODE=offline
export WANDB_DIR=
mkdir -p "$WANDB_DIR"

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export RAY_LOGGING_LEVEL=DEBUG
export HYDRA_FULL_ERROR=1

adv_estimator=grpo

max_prompt_length=1024
max_response_length=8192
train_batch_size=256
ppo_epochs=2
ppo_mini_batch_size=64
ppo_micro_batch_size_per_gpu=4
rollout_engine=vllm
max_model_len=$((max_prompt_length + max_response_length))

low_start=0.8
low_max=0.95
high_start=1.2
high_max=2.0
target_pos_ratio=0.5

project_name=Qwen-target-distribution
base_name="${model_name}_${data_name}_bzs${ppo_mini_batch_size}_staleness$((ppo_epochs * train_batch_size / ppo_mini_batch_size))"
exp_name="${base_name}_${low_start}_${low_max}_${high_start}_${high_max}_target_${target_pos_ratio}"


RAY_DATA_HOME=${RAY_DATA_HOME:-""}
MODEL_PATH=${MODEL_PATH:-""}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/${data_name}.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/aime-2025.parquet"}

# Algorithm
temperature=1
top_p=1
top_k=100
kl_coef=0.0

# Performance Related Parameter
sp_size=1
use_dynamic_bsz=True
offload=True
gen_tp=1

# 生成日志文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR=
LOG_FILE=$LOG_DIR/${exp_name}_${TIMESTAMP}.log

mkdir -p "$LOG_DIR"

set -x

python3 -m recipe.bapo.main_bapo \
    algorithm.adv_estimator=${adv_estimator} \
    data.train_files="${TRAIN_FILE}" \
    data.val_files=${TEST_FILE} \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_epochs=${ppo_epochs} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_model_len} \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.name=${rollout_engine} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=False \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=256 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    trainer.critic_warmup=0 \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.total_training_steps=1000 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    +trainer.train_run_dir="${DETAIL_IO_DIR}" \
    actor_rollout_ref.actor.policy_loss.loss_mode=bapo \
    actor_rollout_ref.actor.policy_loss.adv_ratio_target=${target_pos_ratio} \
    actor_rollout_ref.actor.policy_loss.ratio_upper_start=${high_start} \
    actor_rollout_ref.actor.policy_loss.ratio_upper_max=${high_max} \
    actor_rollout_ref.actor.policy_loss.ratio_upper_step=0.1 \
    actor_rollout_ref.actor.policy_loss.ratio_lower_start=${low_start} \
    actor_rollout_ref.actor.policy_loss.ratio_lower_max=${low_max} \
    actor_rollout_ref.actor.policy_loss.ratio_lower_step=0.05 \
    2>&1 | tee -a "$LOG_FILE"


