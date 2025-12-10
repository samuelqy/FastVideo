#!/bin/bash

GPU_NUM=1 # 2,4,8
MODEL_PATH="Wan-AI/Wan2.2-T2V-A14B-Diffusers"
MODEL_TYPE="wan"
DATA_MERGE_PATH="data/crush-smol/merge.txt"
OUTPUT_DIR="data/crush-smol_processed_t2v/"

torchrun --nproc_per_node=$GPU_NUM \
    fastvideo/pipelines/preprocess/v1_preprocess.py \
    --model_path $MODEL_PATH \
    --data_merge_path $DATA_MERGE_PATH \
    --preprocess_video_batch_size 2 \
    --seed 42 \
    --max_height 448 \
    --max_width 832 \
    --num_frames 81 \
    --dataloader_num_workers 0 \
    --output_dir=$OUTPUT_DIR \
    --train_fps 24 \
    --samples_per_file 8 \
    --flush_frequency 8 \
    --video_length_tolerance_range 5 \
    --preprocess_task "t2v"
