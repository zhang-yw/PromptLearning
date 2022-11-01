OUTPUT=$1
CATEGORY=$2
DATASET=$3
CFG=vit_b16_c2_ep10_batch8_4+4ctx
SHOTS=16


python parse_test_res.py /nobackup3/yiwei/output_ivlp/${OUTPUT}/base2new/${CATEGORY}/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/ --test-log