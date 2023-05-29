TRAIN_CONFIGS='configs/v1_hrnet_flag3d_ft.yml'
# TRAIN_CONFIGS='configs/v1_hrnet_3dpw_ft.yml'

GPUS=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.gpu)
DATASET=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.dataset)
TAB=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.tab)

# CUDA_VISIBLE_DEVICES=${GPUS} python -u -m romp.train --configs_yml=${TRAIN_CONFIGS}
CUDA_VISIBLE_DEVICES=${GPUS} python -u -m debugpy --wait-for-client --listen 0.0.0.0:5678 -m romp.train --configs_yml=${TRAIN_CONFIGS} 

# > 'log/'${TAB}'_'${DATASET}'_g'${GPUS}.log 2
