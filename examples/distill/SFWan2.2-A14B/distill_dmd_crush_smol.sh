#!/bin/bash

# Basic Info
# export WANDB_MODE="online"
# export NCCL_P2P_DISABLE=1
# export TORCH_NCCL_ENABLE_MONITORING=0
# export MASTER_PORT=29500
# export TOKENIZERS_PARALLELISM=false
# export WANDB_BASE_URL="https://api.wandb.ai"
# export FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN

export WANDB_MODE="disabled"
export WANDB_DISABLED="true"

export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export MASTER_PORT=29500
export TOKENIZERS_PARALLELISM=false
export FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN


# Configs
NUM_GPUS=8

# Model paths for Wan2.2 A14B distillation
MODEL_PATH="Wan-AI/Wan2.2-T2V-A14B-Diffusers"
REAL_SCORE_MODEL_PATH="Wan-AI/Wan2.2-T2V-A14B-Diffusers"
FAKE_SCORE_MODEL_PATH="Wan-AI/Wan2.2-T2V-A14B-Diffusers"

# Dataset paths - using crush_smol dataset
DATA_DIR="data/crush-smol_processed_t2v/combined_parquet_dataset/"
VALIDATION_DATASET_FILE="examples/distill/SFWan2.2-A14B/validation.json"
OUTPUT_DIR="checkpoints/wan_a14b_dmd_crush_smol"

# Training arguments
training_args=(
  --tracker_project_name wan_a14b_distill_dmd_VSA_crush_smol
  --output_dir "$OUTPUT_DIR"
  --max_train_steps 4000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --num_latent_t 21
  --num_height 448
  --num_width 832
  --num_frames 81
  --enable_gradient_checkpointing_type "full"
  --training_state_checkpointing_steps 500
  --weight_only_checkpointing_steps 500
)

# Parallel arguments
parallel_args=(
  --num_gpus "$NUM_GPUS"
  --sp_size 1
  --tp_size 1
  # --hsdp_replicate_dim 1
  # --hsdp_shard_dim 1
)

# Model arguments
model_args=(
  --model_path $MODEL_PATH
  --pretrained_model_name_or_path $MODEL_PATH
  --real_score_model_path $REAL_SCORE_MODEL_PATH
  --fake_score_model_path $FAKE_SCORE_MODEL_PATH
)

# Dataset arguments
dataset_args=(
  --data_path "$DATA_DIR"
  --dataloader_num_workers 4
)

# Validation arguments
validation_args=(
  --log_validation
  --validation_dataset_file "$VALIDATION_DATASET_FILE"
  --validation_steps 200
  --validation_sampling_steps "4"
  --validation_guidance_scale "6.0"
)

# Optimizer arguments
optimizer_args=(
  --learning_rate 2e-6
  --mixed_precision "bf16"
  --weight_decay 0.01
  --max_grad_norm 1.0
)

# Miscellaneous arguments
miscellaneous_args=(
  --inference_mode False
  --checkpoints_total_limit 3
  --training_cfg_rate 0.0
  --dit_precision "fp32"
  --ema_start_step 0
  --flow_shift 5
  --seed 1000
)

# DMD arguments
dmd_args=(
  --dmd_denoising_steps '1000,750,500,250'
  --min_timestep_ratio 0.02
  --max_timestep_ratio 0.98
  --generator_update_interval 5
  --real_score_guidance_scale 3.5
  --VSA_sparsity 0.8
)

torchrun \
--nnodes 1 \
--nproc_per_node $NUM_GPUS \
--master_port $MASTER_PORT \
    fastvideo/training/wan_distillation_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}" \
    "${dmd_args[@]}"
