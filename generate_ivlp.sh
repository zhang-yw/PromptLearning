#!/bin/bash

#cd ../..

# custom config
DATA="/nobackup3/yiwei/data/"
TRAINER=IVLP

# DATASET=$1
OUTPUT=$1
SEED=$2
TEXT_WEIGHT=$3
VISUAL_WEIGHT=$4

# CFG=vit_b16_c2_ep5_batch4_2+2ctx
CFG=vit_b16_c2_ep10_batch8_4+4ctx
SHOTS=16
LOADEP=10
SUB=new

for DATASET in caltech101 food101 dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft stanford_cars sun397 eurosat
do
    DIR=/nobackup3/yiwei/output_ivlp/${OUTPUT}/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --text-weight ${TEXT_WEIGHT} \
    --visual-weight ${VISUAL_WEIGHT} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base

    COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
    MODEL_DIR=/nobackup3/yiwei/output_ivlp/${OUTPUT}/base2new/train_base/${COMMON_DIR}
    DIR=/nobackup3/yiwei/output_ivlp/${OUTPUT}/base2new/test_${SUB}/${COMMON_DIR}
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}
done