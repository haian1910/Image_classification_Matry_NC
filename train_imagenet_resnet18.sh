#!/bin/bash
# Train ResNet18 on the existing ImageNet sample dataset
# The dataset is already available at data/imagenet_sample/

# Set up environment variables
export PYTHONPATH="/workspace/Image_classification_Matry_NC"
export LOCAL_RANK='-1'
export MASTER_ADDR='127.0.0.1'
export MASTER_PORT='12356'
export RANK='0'
export WORLD_SIZE='1'

# Check if dataset exists
DATA_PATH="data/imagenet_sample"
if [ ! -d "$DATA_PATH" ]; then
    echo "❌ ImageNet sample dataset not found at $DATA_PATH"
    echo "The sample dataset should already exist in your workspace."
    exit 1
fi

echo "🚀 Training ResNet18 on ImageNet sample dataset"
echo "Dataset path: $DATA_PATH"
echo "Dataset info: 10 classes, ~5000 train images, ~200 val images, ~12MB"

# Train ResNet18 on ImageNet sample
python tools/train.py \
    --model resnet18 \
    --dataset imagenet \
    --data-path $DATA_PATH \
    --experiment imagenet_resnet18_sample \
    --epochs 50 \
    --batch-size 64 \
    --lr 0.01 \
    --weight-decay 1e-4 \
    --workers 2 \
    --log-interval 10 \
    --opt sgd \
    --momentum 0.9 \
    --sched step \
    --decay-epochs 20 \
    --decay-rate 0.1 \
    --image-mean 0.485 0.456 0.406 \
    --image-std 0.229 0.224 0.225

echo "✅ Training completed! Check results in experiments/imagenet_resnet18_sample/"
