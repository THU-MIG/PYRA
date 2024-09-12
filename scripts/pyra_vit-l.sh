#!/usr/bin/env bash

set -x

currenttime=`date "+%Y%m%d_%H%M%S"`

CONFIG="ViT-L_prompt_lora_12"
CONFIG_DIR="experiments/LoRA/"${CONFIG}".yaml"
CKPT="weights/imagenet21k_ViT-L_16.npz"
WEIGHT_DECAY=0.0001

device=0
PYRA_LR=(0.0001)                       # 0.00001 0.00003 0.0001 0.0003 0.001 0.003
merge_schedule="high"
DATASETS=(cifar100 caltech101 dtd oxford_flowers102 svhn sun397 oxford_pet patch_camelyon eurosat resisc45 diabetic_retinopathy clevr_count clevr_dist dmlab kitti dsprites_loc dsprites_ori smallnorb_azi smallnorb_ele)

for pyra_lr in "${PYRA_LR[@]}"
do
    for DATASET in "${DATASETS[@]}"
    do
        for LR in 0.001
        do
            LOG_DIR=outputs/${currenttime}_${CONFIG}_compress_${merge_schedule}_PYRA_LR_${pyra_lr}
            
            TARGET_DIR=${LOG_DIR}/${DATASET}_lr-${LR}_wd-${WEIGHT_DECAY}
            if [ ! -d ${TARGET_DIR} ]
            then
                mkdir -p ${LOG_DIR}
                mkdir -p ${TARGET_DIR}
            else
                echo "Dir already exists, skipping ${TARGET_DIR}"
                continue
            fi
            CUDA_VISIBLE_DEVICES=${device} python train.py --data-path=./data/vtab-1k/${DATASET} --data-set=${DATASET}\
                    --cfg=${CONFIG_DIR} --resume=${CKPT} --output_dir=${TARGET_DIR}\
                    --batch-size=32 --lr=${LR} --epochs=100 --weight-decay=${WEIGHT_DECAY}\
                    --no_aug --mixup=0 --cutmix=0 --direct_resize --smoothing=0\
                    --token_merging --merging_schedule=${merge_schedule}\
                    --pyra --separate_lr_for_pyra --pyra_lr=${pyra_lr}\
            2>&1 | tee -a ${LOG_DIR}/${DATASET}_lr-${LR}_wd-${WEIGHT_DECAY}.log > /dev/null 
        done
    done
done