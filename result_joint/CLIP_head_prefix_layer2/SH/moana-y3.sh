#!/bin/bash
#SBATCH -p batch_ce_ugrad 
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=25G
#SBATCH --time=4-00:00:0

socket_ifname=$(cat /etc/hosts | grep $(hostname) | grep -Eo 'en\w+')

export NCCL_SOCKET_IFNAME=$socket_ifname


OUTPUT_DIR=$4 # weight저장.
MASTER_NODE=$1
OMP_NUM_THREADS=1 python -m torch.distributed.launch \
    --nproc_per_node=$6 \
    --master_port $3 --nnodes=$5 \
    --node_rank=$2 --master_addr=${MASTER_NODE} \
    /data/jong980812/project/videoCIL/run_joint.py \
    --model CLIP \
    --data_set ActivityNet_joint \
    --anno_path /data/jong980812/project/videoCIL/data/activitynet \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 10 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224 \
    --num_frames 16 \
    --opt adamw \
    --lr 1e-3 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --dist_eval \
    --enable_deepspeed \
    --num_workers 4 \
    --epochs 50 \
    --warmup_epochs 5 \
    --unfreeze_layers head prefix \
    --prefix \
    --prefix_layers 0 1 \
    --task joint
 


