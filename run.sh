#!/bin/bash
# Set up environment variables for distributed training
export PYTHONPATH="/workspace/Image_classification_Matry_NC"
export LOCAL_RANK='-1'
export MASTER_ADDR='127.0.0.1'
export MASTER_PORT='12355'
export RANK='0'
export WORLD_SIZE='1'

# Combined NC (NC1 + NC2) Knowledge Distillation with CIFAR ResNet110 teacher
python tools/train.py -c configs/strategies/resnet/resnet.yaml \
    --model matryoshka_cifar_resnet20 \
    --experiment new_20_nc_combined_1 \
    --matryoshka-dims 8 16 32 64 \
    --kd nc \
    --teacher-model cifar_resnet110 \
    --teacher-ckpt experiments/new/checkpoint-323.pth.tar
