#!/bin/bash

#cd ../..

# custom config
DATA="/nobackup3/yiwei/data/"
TRAINER=IVLP

DATASET=$1
SEED=$2
TEXT_WEIGHT=$3
VISUAL_WEIGHT=$4

# CFG=vit_b16_c2_ep5_batch4_2+2ctx
CFG=test
SHOTS=16


DIR=/nobackup3/yiwei/output_minmax/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Resuming..."
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
else
    echo "Run this job and save the output to ${DIR}"
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
fi