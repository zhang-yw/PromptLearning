for SEED in 1 2 3
do
    bash scripts/maple/xd_train_maple.sh caltech101 ${SEED}
    bash scripts/maple/xd_train_maple.sh oxford_pets ${SEED}
    bash scripts/maple/xd_train_maple.sh stanford_cars ${SEED}
done