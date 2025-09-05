
OUTPUT_DIR='./results'
MASTER_NODE=#'your master node ip'
OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node=1 \
    --master_port=#your master port \
    --nnodes=#number of nodes, if you use multi-node \
    --node_rank=# node rank if you use multi-node \
    --master_addr=${MASTER_NODE} \
    ESSENTIAL/run_cil.py \
    --model CLIPs \
    --data_set 'UCF101' \
    --anno_path 'ESSENTIAL/data/ucf101/UCF101_data_20.pkl' \
    --num_tasks 20 \
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
    --epochs 30 \
    --warmup_epochs 5 \
    --dist_eval \
    --rehearsal_epochs 20 \
    --num_workers 8 \
    --dim_mlp 192 \
    --unfreeze_layers head decoder mr_module \
    --unfreeze_layers_after_base_task head decoder mr_module  \
    --replay_token \
    --temp_mode attention \
    --adapter_layers -1 \
    --rehearsal_samples_per_class 16 \
    --fs_topk 2 \
    --static_matching \
    --temporal_matching \
    --prompt_mode cross \
    --memory_mode task \
    --static_matching_weight 1.0 \
    --temporal_matching_weight 1.0 \
    --get_frame_index \



    # --imagenet


# OMP_NUM_THREADS=1 torchrun \
#     --nproc_per_node=$6 \
#     --master_port $3 --nnodes=$5 \
#     --node_rank=$2 --master_addr=${MASTER_NODE} \
#     /data/jongseo/project/cil/videoCIL/run_cil.py \
#     --model AIM_final \
#     --data_set 'UCF101' \
#     --anno_path '/data/jongseo/project/cil/videoCIL/data/ucf101/UCF101_data_51-5_tcd.pkl' \
#     --num_tasks 11 \
#     --log_dir  ${OUTPUT_DIR} \
#     --output_dir  ${OUTPUT_DIR} \
#     --batch_size 10 \
#     --num_sample 1 \
#     --input_size 224 \
#     --short_side_size 224 \
#     --num_frames 8 \
#     --opt adamw \
#     --lr 1e-3 \
#     --opt_betas 0.9 0.999 \
#     --weight_decay 0.05 \
#     --epochs 30 \
#     --warmup_epochs 5 \
#     --dist_eval \
#     --memory_size 2020 \
#     --rehearsal_epochs 15 \
#     --num_workers 8 \
#     --dim_mlp 192 \
#     --mixup_prob 0 \
#     --mixup_switch_prob 0 \
#     --cutmix 0. \
#     --unfreeze_layers head decoder associator \
#     --unfreeze_layers_after_base head decoder associator \
#     --replay_token \
#     --temp_mode attention \
#     --handcrafted_selection \
#     --adapter_layers -1 \
#     --get_frame_index \
#     --rehearsal_samples_per_class 10 \
#     --fs_topk 1 \
#     --frame_matching \
#     --token_matching \
#     --prompt_mode cross \
#     --memory_mode global \
#     --init_scale 1.0 \
#     # --imagenet


# OMP_NUM_THREADS=1 torchrun \
#     --nproc_per_node=$6 \
#     --master_port $3 --nnodes=$5 \
#     --node_rank=$2 --master_addr=${MASTER_NODE} \
#     /data/jongseo/project/cil/videoCIL/run_cil.py \
#     --model AIM_final \
#     --data_set 'UCF101' \
#     --anno_path '/data/jongseo/project/cil/videoCIL/data/ucf101/UCF101_data_51-2_tcd.pkl' \
#     --num_tasks 26 \
#     --log_dir  ${OUTPUT_DIR} \
#     --output_dir  ${OUTPUT_DIR} \
#     --batch_size 10 \
#     --num_sample 1 \
#     --input_size 224 \
#     --short_side_size 224 \
#     --num_frames 8 \
#     --opt adamw \
#     --lr 1e-3 \
#     --opt_betas 0.9 0.999 \
#     --weight_decay 0.05 \
#     --epochs 30 \
#     --warmup_epochs 5 \
#     --dist_eval \
#     --memory_size 2020 \
#     --rehearsal_epochs 15 \
#     --num_workers 8 \
#     --dim_mlp 192 \
#     --mixup_prob 0 \
#     --mixup_switch_prob 0 \
#     --cutmix 0. \
#     --unfreeze_layers head decoder associator \
#     --unfreeze_layers_after_base head decoder associator \
#     --replay_token \
#     --temp_mode attention \
#     --handcrafted_selection \
#     --adapter_layers -1 \
#     --get_frame_index \
#     --rehearsal_samples_per_class 10 \
#     --fs_topk 1 \
#     --frame_matching \
#     --token_matching \
#     --prompt_mode cross \
#     --memory_mode global \
#     --init_scale 1.0 \
    # --imagenet



    # --cos_temp 2 \
    # --cos \

# OMP_NUM_THREADS=1 torchrun \
#     --nproc_per_node=$6 \
#     --master_port $3 --nnodes=$5 \
#     --node_rank=$2 --master_addr=${MASTER_NODE} \
#     /data/jongseo/project/cil/videoCIL/run_cil.py \
#     --model AIM_final \
#     --data_set 'UCF101' \
#     --anno_path '/data/jongseo/project/cil/videoCIL/data/ucf101/UCF101_data_20.pkl' \
#     --num_tasks 20 \
#     --log_dir  ${OUTPUT_DIR} \
#     --output_dir  ${OUTPUT_DIR} \
#     --batch_size 24 \
#     --num_sample 1 \
#     --input_size 224 \
#     --short_side_size 224 \
#     --num_frames 8 \
#     --opt adamw \
#     --lr 1e-3 \
#     --opt_betas 0.9 0.999 \
#     --weight_decay 0.05 \
#     --epochs 50 \
#     --warmup_epochs 5 \
#     --dist_eval \
#     --memory_size 6960 \
#     --rehearsal_epochs 15 \
#     --num_workers 8 \
#     --dim_mlp 192 \
#     --mixup_prob 0 \
#     --mixup_switch_prob 0 \
#     --cutmix 0. \
#     --unfreeze_layers head decoder associator  \
#     --unfreeze_layers_after_base head decoder associator  \
#     --ba_layers 2 \
#     --temp_mode attention \
#     --uniform_ratio 2 \
#     --sampling_rate 3 \
#     --get_frame_index \
#     --handcrafted_selection \
#     --token_matching \
#     --replay_token \
#     --adapter_layers -1 \
#     --virtual_weight 0.5 \
#     --cos \
#     --cos_temp 2 \
#     --rehearsal_samples_per_class 10\
#     --fs_topk 1 \