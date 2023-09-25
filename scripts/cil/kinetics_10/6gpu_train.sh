#!/bin/bash
#SBATCH --job-name=video
#SBATCH --gres=gpu:6
#SBATCH --time=7-0  # 10 hour
#SBATCH --partition=batch_grad
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=140G
#SBATCH --nodelist=ariel-v9


eval "$(conda shell.bash hook)"
hostname

# Enter your environment variables
conda activate videoMAE
cd /data/bkh178/framework/VideoMAE_cil
PROJECT_PATH="/data/bkh178/framework/VideoMAE_cil"
SAVE_DIR="logs/kinetics_10"
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
    --data_set Kinetics-400 \
    --nb_classes 400 \
    --data_path "data/kinetics_10_task" \
    --finetune '/data/bkh178/pretrain/timesformer/videoMAE.pth' \
    --log_dir  $SAVE_DIR/$EXP_NUM \
    --output_dir  $SAVE_DIR/$EXP_NUM \
    --batch_size 16 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt adamw \
    --lr 1e-3 \
    --memory_size 4000 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 50 \
    --rehearsal_epochs 30 \
    --grad_from_block -1 \
    --dist_eval \
    --test_num_segment 5 \
    --test_num_crop 3 \
    --enable_deepspeed 

