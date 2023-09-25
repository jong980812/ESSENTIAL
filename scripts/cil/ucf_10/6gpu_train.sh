#!/bin/bash
#SBATCH --job-name=video
#SBATCH --gres=gpu:6
#SBATCH --time=7-0  # 10 hour
#SBATCH --partition=batch_grad
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=160G
#SBATCH --nodelist=ariel-v6


eval "$(conda shell.bash hook)"
hostname

# Enter your environment variables
conda activate videoMAE
cd /data/bkh178/framework/VideoMAE_cil
PROJECT_PATH="/data/bkh178/framework/VideoMAE_cil"
SAVE_DIR="logs/ucf_10"
cd "$PROJECT_PATH" || exit

if [ ! -d "$SAVE_DIR" ]; then
  mkdir -p "$SAVE_DIR"
fi
PORT=$((48000 + $RANDOM % 999))

EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo  $SAVE_DIR/$EXP_NUM
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=6 \
    --master_port $PORT  \
    run_cil.py \
    --model vit_base_patch16_224 \
    --data_path "data/ucf_10_task" \
    --finetune '/data/bkh178/pretrain/timesformer/videoMAE.pth' \
    --log_dir  $SAVE_DIR/$EXP_NUM \
    --output_dir  $SAVE_DIR/$EXP_NUM \
    --data_set UCF101 \
    --nb_classes 101 \
    --batch_size 16 \
    --input_size 224 \
    --grad_from_block -1 \
    --short_side_size 224 \
    --save_ckpt_freq 50 \
    --num_frames 16 \
    --sampling_rate 4 \
    --num_sample 1 \
    --opt adamw \
    --lr 5e-4 \
    --memory_size 1010 \
    --warmup_lr 1e-8 \
    --min_lr 1e-5 \
    --layer_decay 0.7 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 50 \
    --rehearsal_epochs 30 \
    --test_num_segment 1 \
    --test_num_crop 1 \
    --fc_drop_rate 0.5 \
    --drop_path 0.2 \
    --use_checkpoint \
    --dist_eval \
    --enable_deepspeed 

