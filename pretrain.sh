set -x
# # Set the path to save checkpoints
# OUTPUT_DIR='exp/pretrain/mae_800ep_bs4096_base_size224_patch16_mask75_decdepth8_decdim512_norm_pos2d_mmseg'
# # path to imagenet-1k set
# DATA_PATH='./data/lmdb'

# torchrun --standalone --nnodes=1 --nproc_per_node=4 main_unsup.py \
#         --data_path ${DATA_PATH} \
#         --normlize_target True \
#         --mask_ratio 0.75 \
#         --model pretrain_mae_base_patch16_224 \
#         --batch_size 256 \
# 	  --num_workers 8 \
#         --opt adamw \
#         --opt_betas 0.9 0.95 \
#         --warmup_epochs 40 \
#         --epochs 800 \
#         --save_ckpt_freq 1 \
#         --output_dir ${OUTPUT_DIR}


# # Set the path to save checkpoints
# OUTPUT_DIR='exp/e2e/mae_800ep_bs4096_base_size224_patch16_mask75_decdepth8_decdim512_norm_pos2d_mmseg'
# # path to imagenet-1k set
# DATA_PATH='./data/lmdb'
# # path to pretrain model
# MODEL_PATH='./exp/pretrain/mae_800ep_bs4096_base_size224_patch16_mask75_decdepth8_decdim512_norm_pos2d_mmseg/checkpoint-0.pth'

# # batch_size can be adjusted according to the graphics card
# torchrun --standalone --nnodes=1 --nproc_per_node=4 main_e2e.py \
#     --model vit_base_patch16_224 \
#     --data_path ${DATA_PATH} \
#     --finetune ${MODEL_PATH} \
#     --output_dir ${OUTPUT_DIR} \
#     --batch_size 128 \
#     --opt adamw \
#     --opt_betas 0.9 0.999 \
#     --weight_decay 0.05 \
#     --epochs 100 \
#     --reprob 0 \
#     --use_cls \
#     --dist_eval


# Set the path to save checkpoints
OUTPUT_DIR='exp/linear/mae_800ep_bs4096_base_size224_patch16_mask75_decdepth8_decdim512_norm_pos2d_mmseg'
# path to imagenet-1k set
DATA_PATH='./data/lmdb'
# path to pretrain model
MODEL_PATH='./exp/pretrain/mae_800ep_bs4096_base_size224_patch16_mask75_decdepth8_decdim512_norm_pos2d_mmseg/checkpoint-0.pth'

# batch_size can be adjusted according to the graphics card
torchrun --standalone --nnodes=1 --nproc_per_node=4 main_linear.py \
    --model vit_base_patch16_224 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 2048 \
    --opt lars \
    --weight_decay 0. \
    --epochs 90 \
    --warmup_epochs 40 \
    --reprob 0 \
    --use_cls \
    --dist_eval