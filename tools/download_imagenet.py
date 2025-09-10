#!/usr/bin/env python3
"""
Script to download and prepare ImageNet dataset.
This script handles downloading ImageNet ILSVRC2012 from official sources
and organizing it in the format expected by your codebase.
"""

import os
import sys
import shutil
import tarfile
import requests
import json
from pathlib import Path
from tqdm import tqdm
import argparse
from urllib.parse import urlparse
import hashlib

# Configuration
IMAGENET_BASE_URL = "https://image-net.org/data/ILSVRC/2012/"

# Global variables for paths
OUTPUT_DIR = "data/imagenet"
TEMP_DIR = "data/temp_imagenet"

# Expected file sizes and checksums (for verification)
EXPECTED_FILES = {
    "ILSVRC2012_img_train.tar": {
        "size": 147897477120,  # ~138GB
        "url": IMAGENET_BASE_URL + "ILSVRC2012_img_train.tar"
    },
    "ILSVRC2012_img_val.tar": {
        "size": 6744924160,  # ~6.3GB  
        "url": IMAGENET_BASE_URL + "ILSVRC2012_img_val.tar"
    },
    "ILSVRC2012_devkit_t12.tar.gz": {
        "size": 2568634,  # ~2.5MB
        "url": IMAGENET_BASE_URL + "ILSVRC2012_devkit_t12.tar.gz"
    }
}

def download_file(url, filename, chunk_size=8192):
    """Download a file with progress bar."""
    print(f"Downloading {filename} from {url}")
    
    # Check if file already exists
    if os.path.exists(filename):
        file_size = os.path.getsize(filename)
        expected_size = EXPECTED_FILES[os.path.basename(filename)]["size"]
        if file_size == expected_size:
            print(f"✓ {filename} already exists and has correct size. Skipping download.")
            return True
        else:
            print(f"File exists but has wrong size ({file_size} vs {expected_size}). Re-downloading...")
    
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

def extract_tarfile(tar_path, extract_to, desc="Extracting"):
    """Extract tar file with progress bar."""
    print(f"Extracting {tar_path} to {extract_to}")
    
    with tarfile.open(tar_path, 'r') as tar:
        members = tar.getmembers()
        with tqdm(total=len(members), desc=desc) as pbar:
            for member in members:
                tar.extract(member, extract_to)
                pbar.update(1)
    
    print(f"✓ Extracted {tar_path}")

def create_directory_structure():
    """Create the directory structure for ImageNet."""
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

def download_devkit():
    """Download and extract the ImageNet development kit."""
    devkit_file = f"{TEMP_DIR}/ILSVRC2012_devkit_t12.tar.gz"
    
    if not download_file(EXPECTED_FILES["ILSVRC2012_devkit_t12.tar.gz"]["url"], devkit_file):
        return False
    
    # Extract devkit
    extract_tarfile(devkit_file, TEMP_DIR, "Extracting devkit")
    
    return True

def parse_devkit():
    """Parse the development kit to get class mappings."""
    devkit_path = f"{TEMP_DIR}/ILSVRC2012_devkit_t12"
    
    # Read validation ground truth
    val_ground_truth_file = f"{devkit_path}/data/ILSVRC2012_validation_ground_truth.txt"
    if not os.path.exists(val_ground_truth_file):
        print("✗ Validation ground truth file not found!")
        return None
    
    with open(val_ground_truth_file, 'r') as f:
        val_labels = [int(line.strip()) - 1 for line in f]  # Convert to 0-based indexing
    
    # Read class info (synsets)
    meta_file = f"{devkit_path}/data/meta.mat"
    if os.path.exists(meta_file):
        try:
            import scipy.io
            meta = scipy.io.loadmat(meta_file)
            synsets = meta['synsets']
            
            # Extract class information
            class_info = {}
            for i, synset in enumerate(synsets[0]):
                ilsvrc_id = synset[0][0]  # ILSVRC ID (1-1000)
                wnid = synset[1][0]       # WordNet ID (n########)
                words = synset[2][0]      # Class name
                
                class_info[wnid] = {
                    'ilsvrc_id': int(ilsvrc_id),
                    'class_name': words,
                    'zero_based_id': int(ilsvrc_id) - 1
                }
            
            return val_labels, class_info
            
        except ImportError:
            print("scipy not found. Using alternative method for class mapping...")
    
    # Alternative: create a simple mapping
    print("Creating simplified class mapping...")
    class_info = {}
    for i in range(1000):
        wnid = f"n{i:08d}"  # Simplified synset ID
        class_info[wnid] = {
            'ilsvrc_id': i + 1,
            'class_name': f'class_{i}',
            'zero_based_id': i
        }
    
    return val_labels, class_info

def download_and_extract_train():
    """Download and extract training data."""
    train_file = f"{TEMP_DIR}/ILSVRC2012_img_train.tar"
    
    print("⚠️  WARNING: Training data is ~138GB. Make sure you have enough space!")
    response = input("Continue with training data download? [y/N]: ")
    if response.lower() != 'y':
        print("Skipping training data download.")
        return False
    
    if not download_file(EXPECTED_FILES["ILSVRC2012_img_train.tar"]["url"], train_file):
        return False
    
    # Extract training data
    print("Extracting training data...")
    train_temp_dir = f"{TEMP_DIR}/train"
    Path(train_temp_dir).mkdir(exist_ok=True)
    
    extract_tarfile(train_file, train_temp_dir, "Extracting training data")
    
    # Each class is in a separate tar file, extract them
    print("Extracting individual class tar files...")
    for tar_file in tqdm(os.listdir(train_temp_dir)):
        if tar_file.endswith('.tar'):
            class_name = tar_file.replace('.tar', '')
            class_dir = f"{OUTPUT_DIR}/train/{class_name}"
            Path(class_dir).mkdir(exist_ok=True)
            
            class_tar_path = f"{train_temp_dir}/{tar_file}"
            extract_tarfile(class_tar_path, class_dir, f"Extracting {class_name}")
            
            # Clean up class tar file
            os.remove(class_tar_path)
    
    # Clean up train temp directory
    shutil.rmtree(train_temp_dir)
    print("✓ Training data extracted and organized")
    return True

def download_and_extract_val():
    """Download and extract validation data.""" 
    val_file = f"{TEMP_DIR}/ILSVRC2012_img_val.tar"
    
    if not download_file(EXPECTED_FILES["ILSVRC2012_img_val.tar"]["url"], val_file):
        return False
    
    # Extract validation data
    print("Extracting validation data...")
    val_temp_dir = f"{TEMP_DIR}/val_temp"
    Path(val_temp_dir).mkdir(exist_ok=True)
    
    extract_tarfile(val_file, val_temp_dir, "Extracting validation data")
    
    return val_temp_dir

def organize_validation_data(val_temp_dir, val_labels, class_info):
    """Organize validation data into class directories."""
    print("Organizing validation data by class...")
    
    # Create class directories
    wnid_to_dirname = {}
    for wnid, info in class_info.items():
        class_dir = f"{OUTPUT_DIR}/val/{wnid}"
        Path(class_dir).mkdir(exist_ok=True)
        wnid_to_dirname[info['zero_based_id']] = wnid
    
    # Move validation images to appropriate class directories
    val_images = sorted([f for f in os.listdir(val_temp_dir) if f.endswith('.JPEG')])
    
    for i, image_name in enumerate(tqdm(val_images, desc="Organizing val images")):
        if i < len(val_labels):
            class_id = val_labels[i]
            wnid = wnid_to_dirname[class_id]
            
            src_path = f"{val_temp_dir}/{image_name}"
            dst_path = f"{OUTPUT_DIR}/val/{wnid}/{image_name}"
            
            shutil.move(src_path, dst_path)
    
    # Clean up temp directory
    shutil.rmtree(val_temp_dir)
    print("✓ Validation data organized")

def create_meta_files(class_info):
    """Create meta files required by the dataset loader."""
    print("Creating meta files...")
    
    # Create train.txt
    train_txt_path = f"{OUTPUT_DIR}/meta/train.txt"
    with open(train_txt_path, 'w') as f:
        for wnid, info in class_info.items():
            class_dir = f"{OUTPUT_DIR}/train/{wnid}"
            if os.path.exists(class_dir):
                for image_name in os.listdir(class_dir):
                    if image_name.endswith('.JPEG'):
                        f.write(f"{wnid}/{image_name} {info['zero_based_id']}\n")
    
    # Create val.txt
    val_txt_path = f"{OUTPUT_DIR}/meta/val.txt"
    with open(val_txt_path, 'w') as f:
        for wnid, info in class_info.items():
            class_dir = f"{OUTPUT_DIR}/val/{wnid}"
            if os.path.exists(class_dir):
                for image_name in os.listdir(class_dir):
                    if image_name.endswith('.JPEG'):
                        f.write(f"{wnid}/{image_name} {info['zero_based_id']}\n")
    
    # Create class mapping JSON
    class_mapping_path = f"{OUTPUT_DIR}/meta/class_mapping.json"
    with open(class_mapping_path, 'w') as f:
        json.dump(class_info, f, indent=2)
    
    print(f"✓ Created meta files in {OUTPUT_DIR}/meta/")

def verify_dataset():
    """Verify the ImageNet dataset structure."""
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
    
    print(f"\nDataset Summary:")
    print(f"- Training classes: {train_classes}")
    print(f"- Validation classes: {val_classes}")
    print(f"- Dataset location: {OUTPUT_DIR}")
    
    if train_classes == 1000 and val_classes == 1000:
        print("✅ ImageNet dataset structure verified!")
        return True
    else:
        print(f"⚠️  Warning: Expected 1000 classes, found {train_classes} train, {val_classes} val")
        return False

def main():
    """Main function to download and prepare ImageNet."""
    parser = argparse.ArgumentParser(description='Download and prepare ImageNet dataset')
    parser.add_argument('--skip-train', action='store_true', 
                       help='Skip downloading training data (validation only)')
    parser.add_argument('--output-dir', default=OUTPUT_DIR,
                       help='Output directory for ImageNet data')
    parser.add_argument('--temp-dir', default=TEMP_DIR,
                       help='Temporary directory for downloads')
    
    args = parser.parse_args()
    
    # Update global paths if provided
    global OUTPUT_DIR, TEMP_DIR
    OUTPUT_DIR = args.output_dir
    TEMP_DIR = args.temp_dir
    
    print("🚀 ImageNet Download and Preparation Script")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Temporary directory: {TEMP_DIR}")
    
    # Check available space
    statvfs = os.statvfs('.')
    free_space = statvfs.f_frsize * statvfs.f_available
    required_space = 200 * 1024 * 1024 * 1024  # 200GB
    
    if free_space < required_space:
        print(f"⚠️  Warning: Available space ({free_space/1024**3:.1f}GB) may be insufficient.")
        print(f"   Recommended: {required_space/1024**3:.1f}GB")
        response = input("Continue anyway? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return 1
    
    # Step 1: Create directory structure
    create_directory_structure()
    
    # Step 2: Download and parse devkit
    if not download_devkit():
        print("✗ Failed to download devkit!")
        return 1
    
    val_labels, class_info = parse_devkit()
    if val_labels is None or class_info is None:
        print("✗ Failed to parse devkit!")
        return 1
    
    # Step 3: Download and extract validation data
    val_temp_dir = download_and_extract_val()
    if not val_temp_dir:
        print("✗ Failed to download validation data!")
        return 1
    
    # Step 4: Organize validation data
    organize_validation_data(val_temp_dir, val_labels, class_info)
    
    # Step 5: Download and extract training data (optional)
    if not args.skip_train:
        if not download_and_extract_train():
            print("⚠️  Training data download failed or skipped.")
            print("   You can continue with validation data only for testing.")
    
    # Step 6: Create meta files
    create_meta_files(class_info)
    
    # Step 7: Clean up
    if os.path.exists(TEMP_DIR):
        print(f"Cleaning up temporary files in {TEMP_DIR}...")
        shutil.rmtree(TEMP_DIR)
    
    # Step 8: Verify dataset
    if verify_dataset():
        print("\n🎉 ImageNet dataset preparation completed successfully!")
        print(f"\nTo use this dataset, run:")
        print(f"python tools/train.py --dataset imagenet --data-path {OUTPUT_DIR}")
        return 0
    else:
        print("\n❌ Dataset preparation completed with warnings!")
        return 1

if __name__ == "__main__":
    exit(main())
