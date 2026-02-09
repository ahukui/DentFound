#!/bin/bash
#SBATCH --job-name=DentFound             # Custom job name
#SBATCH --partition=aisc               # Partition name
#SBATCH --nodelist=aisct[01-04]        # Specific nodes to use
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --cpus-per-task=64             # Number of CPU cores
#SBATCH --mem=256G                     # Total memory
#SBATCH --time=100:00:00                # Max run time (12 hours)
#SBATCH --gres=gpu:8                   # Request 4 GPUs
#SBATCH --output=logs/%x_%j.out        # Save stdout to logs/jobname_jobid.out
#SBATCH --error=logs/%x_%j.err         # Save stderr to logs/jobname_jobid.err

# Ensure the logs directory exists
mkdir -p logs

echo "=== Job started at $(date) on $(hostname) ==="

# Load your conda environment properly
source ~/.bashrc  # Or your shell config
conda activate dentfound

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH


export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((10000 + RANDOM % 40000))

deepspeed --master_addr $MASTER_ADDR --master_port $MASTER_PORT train_mem.py \
  --lora_enable True \
  --lora_r 128 \
  --lora_alpha 256 \
  --mm_projector_lr 2e-5 \
  --deepspeed ./scripts/zero3.json \
  --model_name_or_path llava_v1.5_7b \
  --version v1 \
  --data_path ToothQue.json \
  --image_folder "" \
  --vision_tower openai/clip-vit-large-patch14-336 \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length True \
  --bf16 True \
  --output_dir ./checkpoints/lora \
  --num_train_epochs 6 \
  --per_device_train_batch_size 24 \
  --per_device_eval_batch_size 24 \
  --gradient_accumulation_steps 8 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 500 \
  --save_total_limit 10 \
  --learning_rate 2e-4 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 500 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 8 \
  --lazy_preprocess True
