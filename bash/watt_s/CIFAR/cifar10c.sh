GPU_ID=0
DATASET=cifar10
DATA_DIR=./data/CIFAR10/
BATCH_SIZE=128
LR=1e-3
METHOD=watt
WATT_TEMPLATE_DIR=./templates.yaml
WATT_TYPE=sequential
WATT_L=2
WATT_M=5

ALL_CORRUPTIONS="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"



BACKBONE=ViT-B/32
SAVE_DIR=./save/${DATASET}/${DOMAIN}/${BACKBONE}/watt-${WATT_TYPE}-l${WATT_L}-m${WATT_M}
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --data_dir $DATA_DIR --dataset $DATASET --adapt --method $METHOD --corruptions_list $ALL_CORRUPTIONS --save_dir $SAVE_DIR --backbone $BACKBONE --batch-size $BATCH_SIZE --lr $LR  --watt_temps $WATT_TEMPLATE_DIR --watt_type $WATT_TYPE --watt_l $WATT_L --watt_m $WATT_M



BACKBONE=ViT-B/16
SAVE_DIR=./save/${DATASET}/${DOMAIN}/${BACKBONE}/watt-${WATT_TYPE}-l${WATT_L}-m${WATT_M}
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --data_dir $DATA_DIR --dataset $DATASET --adapt --method $METHOD --corruptions_list $ALL_CORRUPTIONS --save_dir $SAVE_DIR --backbone $BACKBONE --batch-size $BATCH_SIZE --lr $LR  --watt_temps $WATT_TEMPLATE_DIR --watt_type $WATT_TYPE --watt_l $WATT_L --watt_m $WATT_M



BACKBONE=ViT-L/14
SAVE_DIR=./save/${DATASET}/${DOMAIN}/${BACKBONE}/watt-${WATT_TYPE}-l${WATT_L}-m${WATT_M}
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --data_dir $DATA_DIR --dataset $DATASET --adapt --method $METHOD --corruptions_list $ALL_CORRUPTIONS --save_dir $SAVE_DIR --backbone $BACKBONE --batch-size $BATCH_SIZE --lr $LR  --watt_temps $WATT_TEMPLATE_DIR --watt_type $WATT_TYPE --watt_l $WATT_L --watt_m $WATT_M
