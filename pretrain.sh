set -x
# # Set the path to save checkpoints
# OUTPUT_DIR='exp/pretrain/mae_800ep_bs4096_base_size224_patch16_mask75_decdepth8_decdim512_unnorm_pos2d_mocoinit'
# # path to imagenet-1k train set
# DATA_PATH='./datasets/lmdb/train.lmdb'


# # batch_size can be adjusted according to the graphics card
# OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 run_mae_pretraining.py \
#         --data_path ${DATA_PATH} \
#         --normlize_target False \
#         --mask_ratio 0.75 \
#         --model pretrain_mae_base_patch16_224 \
#         --batch_size 512 \
# 	  --num_workers 8 \
#         --opt adamw \
#         --opt_betas 0.9 0.95 \
#         --warmup_epochs 40 \
#         --epochs 800 \
#         --save_ckpt_freq 80 \
#         --output_dir ${OUTPUT_DIR}


# Set the path to save checkpoints
OUTPUT_DIR='exp/e2e_finetune/mae_800ep_bs4096_base_size224_patch16_mask75_decdepth8_decdim512_unnorm_pos2d_mocoinit_v2'
# path to imagenet-1k set
DATA_PATH='./datasets/lmdb'
# path to pretrain model
MODEL_PATH='./exp/pretrain/mae_800ep_bs4096_base_size224_patch16_mask75_decdepth8_decdim512_unnorm_pos2d_mocoinit/checkpoint-799.pth'

# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 python -u -W ignore \
    -m torch.distributed.launch --nproc_per_node=8 run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 128 \
    --opt adamw \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 100 \
    --reprob 0 \
    --use_cls \
    --dist_eval