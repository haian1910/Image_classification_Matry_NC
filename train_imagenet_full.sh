#!/bin/bash
# Train ResNet18 on the full ImageNet dataset
# Run download_imagenet_full.py first to setup the dataset

# Set up environment variables  
export PYTHONPATH="/workspace/Image_classification_Matry_NC"
export LOCAL_RANK='-1'
export MASTER_ADDR='127.0.0.1'
export MASTER_PORT='12356'
export RANK='0'
export WORLD_SIZE='1'

# Dataset path (change this if you used a different path)
DATA_PATH="data/imagenet"

# Check if dataset exists
if [ ! -d "$DATA_PATH" ]; then
    echo "❌ ImageNet dataset not found at $DATA_PATH"
    echo "Please run first: python tools/download_imagenet_full.py --username YOUR_USER --accesskey YOUR_KEY"
    exit 1
fi

# Check if meta files exist
if [ ! -f "$DATA_PATH/meta/train.txt" ]; then
    echo "❌ ImageNet meta files not found!"
    echo "Please ensure the dataset was downloaded correctly."
    exit 1
fi

echo "🚀 Training ResNet18 on Full ImageNet Dataset"
echo "Dataset path: $DATA_PATH"
echo "Expected: 1000 classes, ~1.3M train images, 50K val images"

# Count classes and images for verification
TRAIN_CLASSES=$(ls -1 "$DATA_PATH/train" | wc -l)
VAL_CLASSES=$(ls -1 "$DATA_PATH/val" | wc -l)
TRAIN_LINES=$(wc -l < "$DATA_PATH/meta/train.txt")  
VAL_LINES=$(wc -l < "$DATA_PATH/meta/val.txt")

echo "Found: $TRAIN_CLASSES train classes, $VAL_CLASSES val classes"
echo "Found: $TRAIN_LINES train images, $VAL_LINES val images"

if [ "$TRAIN_CLASSES" -ne 1000 ] || [ "$VAL_CLASSES" -ne 1000 ]; then
    echo "⚠️  Warning: Expected 1000 classes, but found $TRAIN_CLASSES train, $VAL_CLASSES val"
    echo "   Continuing anyway..."
fi

# Full ImageNet training with standard settings
python tools/train.py \
    --model resnet18 \
    --dataset imagenet \
    --data-path $DATA_PATH \
    --experiment imagenet_resnet18_full \
    --epochs 90 \
    --batch-size 256 \
    --lr 0.1 \
    --weight-decay 1e-4 \
    --workers 8 \
    --log-interval 100 \
    --opt sgd \
    --momentum 0.9 \
    --sched step \
    --decay-epochs 30 \
    --decay-rate 0.1 \
    --warmup-epochs 5 \
    --warmup-lr 0.01 \
    --image-mean 0.485 0.456 0.406 \
    --image-std 0.229 0.224 0.225 \
    --mixup 0.2 \
    --cutmix 1.0 \
    --smoothing 0.1 \
    --aa rand-m9-mstd0.5-inc1 \
    --reprob 0.25 \
    --model-ema \
    --model-ema-decay 0.9999 \
    --amp

echo "✅ Training completed! Check results in experiments/imagenet_resnet18_full/"
echo "📊 Expected Top-1 accuracy: ~70% (ResNet18 on ImageNet)"
