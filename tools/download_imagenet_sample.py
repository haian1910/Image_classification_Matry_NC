#!/usr/bin/env python3
"""
Script to download a small sample of ImageNet dataset (~100MB) for testing purposes.
This creates a mini-ImageNet with 10 classes and ~100 images per class.
"""

import os
import sys
import shutil
import requests
import json
import zipfile
from pathlib import Path
from tqdm import tqdm
import argparse
from urllib.parse import urlparse
import random

# Configuration for small ImageNet sample
SAMPLE_CLASSES = [
    "n01440764",  # tench
    "n01443537",  # goldfish 
    "n01484850",  # great_white_shark
    "n01491361",  # tiger_shark
    "n01494475",  # hammerhead
    "n01496331",  # electric_ray
    "n01498041",  # stingray
    "n01514668",  # cock
    "n01514859",  # hen
    "n01518878",  # ostrich
]

# Alternative: Use a pre-existing mini-ImageNet dataset
MINI_IMAGENET_URL = "https://github.com/yaoyao-liu/mini-imagenet-tools/releases/download/v1.0/mini-imagenet.tar.gz"

# Global variables for paths
OUTPUT_DIR = "data/imagenet_sample"
TEMP_DIR = "data/temp_imagenet_sample"

def download_file(url, filename, chunk_size=8192):
    """Download a file with progress bar."""
    print(f"Downloading {filename} from {url}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as f, tqdm(
            desc=os.path.basename(filename),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"✓ Downloaded {filename}")
        return True
        
    except Exception as e:
        print(f"✗ Error downloading {filename}: {e}")
        if os.path.exists(filename):
            os.remove(filename)
        return False

def create_directory_structure():
    """Create the directory structure for ImageNet sample."""
    dirs = [
        OUTPUT_DIR,
        f"{OUTPUT_DIR}/train",
        f"{OUTPUT_DIR}/val", 
        f"{OUTPUT_DIR}/meta",
        TEMP_DIR
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print(f"Created directory structure in {OUTPUT_DIR}")

def create_synthetic_sample():
    """Create a synthetic ImageNet sample with random colored images."""
    print("Creating synthetic ImageNet sample...")
    
    from PIL import Image
    import numpy as np
    
    # Create class directories and synthetic images
    images_per_class = 100
    val_images_per_class = 20
    
    train_meta = []
    val_meta = []
    class_info = {}
    
    for i, class_id in enumerate(SAMPLE_CLASSES):
        # Create class info
        class_info[class_id] = {
            'ilsvrc_id': i + 1,
            'class_name': f'sample_class_{i}',
            'zero_based_id': i
        }
        
        # Create train directory and images
        train_dir = f"{OUTPUT_DIR}/train/{class_id}"
        Path(train_dir).mkdir(exist_ok=True)
        
        for j in range(images_per_class):
            # Create a random colored image
            img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            pil_img = Image.fromarray(img)
            
            image_name = f"{class_id}_{j:04d}.JPEG"
            pil_img.save(f"{train_dir}/{image_name}")
            train_meta.append(f"{class_id}/{image_name} {i}")
        
        # Create val directory and images
        val_dir = f"{OUTPUT_DIR}/val/{class_id}"
        Path(val_dir).mkdir(exist_ok=True)
        
        for j in range(val_images_per_class):
            # Create a random colored image
            img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            pil_img = Image.fromarray(img)
            
            image_name = f"ILSVRC2012_val_{i:05d}_{j:02d}.JPEG"
            pil_img.save(f"{val_dir}/{image_name}")
            val_meta.append(f"{class_id}/{image_name} {i}")
    
    # Create meta files
    with open(f"{OUTPUT_DIR}/meta/train.txt", 'w') as f:
        f.write('\n'.join(train_meta))
    
    with open(f"{OUTPUT_DIR}/meta/val.txt", 'w') as f:
        f.write('\n'.join(val_meta))
    
    with open(f"{OUTPUT_DIR}/meta/class_mapping.json", 'w') as f:
        json.dump(class_info, f, indent=2)
    
    print(f"✓ Created synthetic sample with {len(SAMPLE_CLASSES)} classes")
    print(f"  - Train: {len(SAMPLE_CLASSES) * images_per_class} images")
    print(f"  - Val: {len(SAMPLE_CLASSES) * val_images_per_class} images")
    
    return True

def download_tiny_imagenet():
    """Download Tiny ImageNet dataset as an alternative."""
    print("Downloading Tiny ImageNet dataset...")
    
    # Tiny ImageNet URL
    tiny_imagenet_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_file = f"{TEMP_DIR}/tiny-imagenet-200.zip"
    
    if not download_file(tiny_imagenet_url, zip_file):
        print("Failed to download Tiny ImageNet, falling back to synthetic data")
        return False
    
    # Extract zip file
    print("Extracting Tiny ImageNet...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(TEMP_DIR)
    
    # Reorganize to match our structure
    tiny_dir = f"{TEMP_DIR}/tiny-imagenet-200"
    
    if not os.path.exists(tiny_dir):
        print("Tiny ImageNet extraction failed")
        return False
    
    # Copy a subset of classes (10 classes to keep it small)
    train_classes = os.listdir(f"{tiny_dir}/train")[:10]
    
    train_meta = []
    val_meta = []
    class_info = {}
    
    # Process training data
    for i, class_id in enumerate(train_classes):
        class_info[class_id] = {
            'ilsvrc_id': i + 1,
            'class_name': class_id,
            'zero_based_id': i
        }
        
        src_train_dir = f"{tiny_dir}/train/{class_id}/images"
        dst_train_dir = f"{OUTPUT_DIR}/train/{class_id}"
        
        if os.path.exists(src_train_dir):
            shutil.copytree(src_train_dir, dst_train_dir)
            
            # Add to train meta
            for img_name in os.listdir(dst_train_dir):
                if img_name.endswith('.JPEG'):
                    train_meta.append(f"{class_id}/{img_name} {i}")
        
        # Create val directory (use some training images as validation)
        dst_val_dir = f"{OUTPUT_DIR}/val/{class_id}"
        Path(dst_val_dir).mkdir(exist_ok=True)
        
        # Copy first 20 training images as validation
        train_images = [f for f in os.listdir(dst_train_dir) if f.endswith('.JPEG')][:20]
        for img_name in train_images:
            shutil.copy(f"{dst_train_dir}/{img_name}", f"{dst_val_dir}/{img_name}")
            val_meta.append(f"{class_id}/{img_name} {i}")
    
    # Create meta files
    with open(f"{OUTPUT_DIR}/meta/train.txt", 'w') as f:
        f.write('\n'.join(train_meta))
    
    with open(f"{OUTPUT_DIR}/meta/val.txt", 'w') as f:
        f.write('\n'.join(val_meta))
    
    with open(f"{OUTPUT_DIR}/meta/class_mapping.json", 'w') as f:
        json.dump(class_info, f, indent=2)
    
    print(f"✓ Created Tiny ImageNet sample with {len(train_classes)} classes")
    return True

def get_dataset_size(directory):
    """Get the total size of a directory."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.isfile(filepath):
                total_size += os.path.getsize(filepath)
    return total_size

def verify_dataset():
    """Verify the ImageNet sample dataset structure."""
    print("\nVerifying dataset structure...")
    
    # Check directories
    required_dirs = [
        f"{OUTPUT_DIR}/train",
        f"{OUTPUT_DIR}/val",
        f"{OUTPUT_DIR}/meta"
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"✗ Missing directory: {dir_path}")
            return False
        print(f"✓ Found directory: {dir_path}")
    
    # Check meta files
    meta_files = [
        f"{OUTPUT_DIR}/meta/train.txt",
        f"{OUTPUT_DIR}/meta/val.txt",
        f"{OUTPUT_DIR}/meta/class_mapping.json"
    ]
    
    for meta_file in meta_files:
        if not os.path.exists(meta_file):
            print(f"✗ Missing meta file: {meta_file}")
            return False
        print(f"✓ Found meta file: {meta_file}")
    
    # Count classes and images
    train_classes = len([d for d in os.listdir(f"{OUTPUT_DIR}/train") if os.path.isdir(f"{OUTPUT_DIR}/train/{d}")])
    val_classes = len([d for d in os.listdir(f"{OUTPUT_DIR}/val") if os.path.isdir(f"{OUTPUT_DIR}/val/{d}")])
    
    # Count total images
    train_images = 0
    for class_dir in os.listdir(f"{OUTPUT_DIR}/train"):
        class_path = f"{OUTPUT_DIR}/train/{class_dir}"
        if os.path.isdir(class_path):
            train_images += len([f for f in os.listdir(class_path) if f.endswith('.JPEG')])
    
    val_images = 0
    for class_dir in os.listdir(f"{OUTPUT_DIR}/val"):
        class_path = f"{OUTPUT_DIR}/val/{class_dir}"
        if os.path.isdir(class_path):
            val_images += len([f for f in os.listdir(class_path) if f.endswith('.JPEG')])
    
    # Get dataset size
    dataset_size = get_dataset_size(OUTPUT_DIR)
    dataset_size_mb = dataset_size / (1024 * 1024)
    
    print(f"\nDataset Summary:")
    print(f"- Training classes: {train_classes}")
    print(f"- Validation classes: {val_classes}")
    print(f"- Training images: {train_images}")
    print(f"- Validation images: {val_images}")
    print(f"- Total size: {dataset_size_mb:.1f} MB")
    print(f"- Dataset location: {OUTPUT_DIR}")
    
    if train_classes >= 10 and val_classes >= 10:
        print("✅ ImageNet sample dataset created successfully!")
        return True
    else:
        print(f"⚠️  Warning: Expected at least 10 classes, found {train_classes} train, {val_classes} val")
        return False

def main():
    """Main function to create ImageNet sample."""
    global OUTPUT_DIR, TEMP_DIR
    
    parser = argparse.ArgumentParser(description='Create a small ImageNet sample dataset (~100MB)')
    parser.add_argument('--method', choices=['synthetic', 'tiny-imagenet'], default='tiny-imagenet',
                       help='Method to create sample: synthetic (random images) or tiny-imagenet')
    parser.add_argument('--output-dir', default=OUTPUT_DIR,
                       help='Output directory for ImageNet sample')
    parser.add_argument('--temp-dir', default=TEMP_DIR,
                       help='Temporary directory for downloads')
    
    args = parser.parse_args()
    
    # Update global paths if provided
    OUTPUT_DIR = args.output_dir
    TEMP_DIR = args.temp_dir
    
    print("🚀 ImageNet Sample Creation Script")
    print(f"Method: {args.method}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Target size: ~100MB")
    
    # Step 1: Create directory structure
    create_directory_structure()
    
    # Step 2: Create the sample dataset
    success = False
    if args.method == 'tiny-imagenet':
        success = download_tiny_imagenet()
        if not success:
            print("Tiny ImageNet failed, falling back to synthetic method")
            success = create_synthetic_sample()
    else:
        success = create_synthetic_sample()
    
    if not success:
        print("✗ Failed to create sample dataset!")
        return 1
    
    # Step 3: Clean up temp directory
    if os.path.exists(TEMP_DIR):
        print(f"Cleaning up temporary files...")
        shutil.rmtree(TEMP_DIR)
    
    # Step 4: Verify dataset
    if verify_dataset():
        print("\n🎉 ImageNet sample dataset created successfully!")
        print(f"\nTo use this dataset, run:")
        print(f"python tools/train_imagenet.py --data-path {OUTPUT_DIR}")
        return 0
    else:
        print("\n❌ Dataset creation completed with warnings!")
        return 1

if __name__ == "__main__":
    exit(main())
