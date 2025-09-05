
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
    --data_set SSV2 \
    --anno_path ESSENTIAL/data/TCD/ssv2/ssv2_data_tasks_109_2.pkl \
    --num_tasks 10 \
    --log_dir  ${OUTPUT_DIR} \
    --output_dir  ${OUTPUT_DIR} \
    --batch_size 24 \
    --num_sample 1 \
    --save_ckpt_freq 30 \
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
    --rehearsal_epochs 25 \
    --num_workers 8 \
    --dim_mlp 192 \
    --unfreeze_layers head decoder mr_module \
    --unfreeze_layers_after_base_task head decoder mr_module  \
    --temporal_layer 3 \
    --temp_mode attention \
    --rehearsal_samples_per_class 4 \
    --fs_topk 4 \
    --static_matching \
    --temporal_matching \
    --replay_token \
    --prompt_mode cross \
    --memory_mode task \
    --static_matching_weight 1.0 \
    --temporal_matching_weight 1.0 \
    --get_frame_index \
    --use_clip_temporal ESSENTIAL/data/clip_temporal.pth \
