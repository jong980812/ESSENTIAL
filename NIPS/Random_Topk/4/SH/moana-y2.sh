#!/bin/bash
#SBATCH -p batch_ce_ugrad 
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=25G
#SBATCH --time=4-00:00:0

socket_ifname=$(cat /etc/hosts | grep $(hostname) | grep -Eo 'en\w+')
export NCCL_SOCKET_IFNAME=$socket_ifname

DATA_PATH=/local_datasets/Epickitchens100_clips/video
VMAE_PATH=/data/datasets/Epickitchens100_clips/epic_checkpoint-2400.pth
# VMAE_PATH=/data/datasets/Epickitchens100_clips/epic_checkpoint-800.pth 
# VMAE_PATH=/data/jong980812/project/VideoMAE_experiments/dataset/ssv2/ssv2_1600.pth
# VMAE_PATH=/data/datasets/Epickitchens100_clips/vit_b_hybrid_pt_800e.pth
# VMAE_PATH=/data/jong980812/project/VideoMAE_experiments/dataset/ssv2/ssv2_1600.pth
CLIP_PATH=/data/datasets/Epickitchens100_clips/ViT-B-16.pt
OUTPUT_DIR=$4 # weight저장.
MASTER_NODE=$1
OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node=$6 \
    --master_port $3 --nnodes=$5 \
    --node_rank=$2 --master_addr=${MASTER_NODE} \
    /data/jong980812/project/cil/videoCIL/run_cil.py \
    --model AIM_base_decoder \
    --data_set SSV2 \
    --data_path /local_datasets/something-something/something-something-v2-mp4 \
    --anno_path /data/jong980812/project/cil/videoCIL/data/ssv2/annotation/ssv2_data_tasks_109_2.pkl \
    --num_tasks 10 \
    --log_dir  ${OUTPUT_DIR} \
    --output_dir  ${OUTPUT_DIR} \
    --batch_size 24 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224 \
    --num_frames 8 \
    --opt adamw \
    --lr 1e-3 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 50 \
    --warmup_epochs 5 \
    --dist_eval \
    --memory_size 3480 \
    --rehearsal_epochs 50 \
    --num_workers 12 \
    --dim_mlp 192 \
    --mixup_prob 0 \
    --mixup_switch_prob 0 \
    --cutmix 0. \
    --unfreeze_layers head decoder \
    --unfreeze_layers_after_base head decoder \
    --ba_layers 2 \
    --temp_mode attention \
    --fs_topk 8 \
    --uniform_ratio 2. \
    --sampling_rate -1 \
    --ssv2_first_finetune /data/jong980812/project/cil/videoCIL/result/ssv2_AIM_first_and_decoder/OUT/checkpoint/task1_checkpoint.pth \
    --use_aim_weight /data/yuri1255/project/videoCIL/result/rehearsal_fix/ssv2/AIM_3480/OUT/checkpoint/task1_checkpoint.pth \
    --rehearsal_samples_per_class 40 \
    --get_frame_index \
    --handcrafted_selection

    # --ssv2_first_finetune /data/yuri1255/project/videoCIL/result/rehearsal_fix/TCD/temporal_order/OUT/checkpoint/task1_checkpoint.pth \




    # --set_selection_frame \

    # --fs_density \


    # --cls_aug \
    # --use_aim_weight \

    

    # --finetune /data/jong980812/project/cil/videoCIL/ssv2_1600.pth
    # ${OUTPUT_DIR}/checkpoint-best.pth \
    
    
    # /data/jong980812/project/cil/videoCIL_ssv2/result/new_pkl/192_0/OUT/checkpoint-best.pth
    #  /data/jong980812/project/cil/videoCIL_ssv2/result/new_pkl/192_0/OUT/checkpoint-best.pth



    # --eval \
    # --fine_tune /data/jong980812/project/CAST-2/result/epic_prompt/test/debug/checkpoint-49/mp_rank_00_model_states.pt