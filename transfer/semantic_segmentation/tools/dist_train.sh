#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}


# bash tools/dist_train.sh \
#     configs/mae/upernet/upernet_mae_base_12_512_slide_160k_ade20k_pt.py 1 \
#     --work-dir exp/mae_base_800ep_ade20k_512_160k_bs16 --seed 0  --deterministic \
#     --options model.pretrained=pretrain/mae_800ep_bs2048_base_size224_patch16_mask75_decdepth8_decdim512_norm_os1d_xavierinit/checkpoint-799.pth