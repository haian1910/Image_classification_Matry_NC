#!/bin/bash
# Set up environment variables for distributed training
export PYTHONPATH="/workspace/Image_classification_Matry_NC"
export LOCAL_RANK='-1'
export MASTER_ADDR='127.0.0.1'
export MASTER_PORT='12355'
export RANK='0'
export WORLD_SIZE='1'

python tools/train.py \
    --model matryoshka_resnet18 \
    --dataset cifar10 \
    --kd nc1 \
    --teacher-model resnet101 \/workspace/Image_classification_Matry_NC/experiments/exp/checkpoint-0.pth.tar
    --teacher-pretrained \
    --matryoshka-dims 128 256 384 512 \
    --matryoshka-adaptive \
    --epochs 1 \
    --decay-rate 0.1 \
    --experiment matryoshka_nc1_test_50ep
# python tools/train.py -c configs/strategies/resnet/resnet.yaml --model resnet101
