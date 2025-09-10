#!/usr/bin/env python3
"""
Download and setup full ImageNet ILSVRC2012 dataset in the format expected by the codebase.
This script downloads the real ImageNet dataset and organizes it exactly like the working sample.

Requirements:
1. You need to register at http://image-net.org and get download credentials
2. About 200GB of free disk space
3. Fast internet connection (dataset is ~150GB)

Usage:
    python tools/download_imagenet_full.py --username YOUR_USERNAME --accesskey YOUR_ACCESS_KEY
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
import time
import concurrent.futures
from multiprocessing import Pool, cpu_count

# Configuration for ImageNet ILSVRC2012
IMAGENET_BASE_URL = "https://image-net.org/data/ILSVRC/2012/"
ALTERNATIVE_URLS = {
    # Alternative mirrors if official site is down
    "academic_torrents": "https://academictorrents.com/download/",
    "kaggle": "https://www.kaggle.com/c/imagenet-object-localization-challenge/data"
}

# Global variables for paths
OUTPUT_DIR = "data/imagenet"
TEMP_DIR = "data/temp_imagenet_full"

# Expected files for full ImageNet
IMAGENET_FILES = {
    "ILSVRC2012_img_train.tar": {
        "size": 147897477120,  # ~138GB
        "description": "Training images (1000 tar files, one per class)"
    },
    "ILSVRC2012_img_val.tar": {
        "size": 6744924160,   # ~6.3GB
        "description": "Validation images (50,000 images)"
    },
    "ILSVRC2012_devkit_t12.tar.gz": {
        "size": 2568634,      # ~2.5MB
        "description": "Development kit with ground truth and metadata"
    }
}

def check_disk_space():
    """Check if there's enough disk space for ImageNet."""
    statvfs = os.statvfs('.')
    free_space = statvfs.f_frsize * statvfs.f_available
    required_space = 200 * 1024 * 1024 * 1024  # 200GB
    
    print(f"Available disk space: {free_space / (1024**3):.1f}GB")
    print(f"Required disk space: {required_space / (1024**3):.1f}GB")
    
    if free_space < required_space:
        print("❌ Insufficient disk space!")
        print("You need at least 200GB of free space to download and process ImageNet.")
        return False
    return True

def download_with_auth(url, filename, username, accesskey, chunk_size=8192):
    """Download ImageNet files with authentication."""
    print(f"🔽 Downloading {filename}")
    print(f"   Size: {IMAGENET_FILES[os.path.basename(filename)]['size'] / (1024**3):.1f}GB")
    
    # Check if file already exists and has correct size
    if os.path.exists(filename):
        file_size = os.path.getsize(filename)
        expected_size = IMAGENET_FILES[os.path.basename(filename)]["size"]
        if file_size == expected_size:
            print(f"✅ {filename} already exists with correct size. Skipping.")
            return True
        else:
            print(f"⚠️  File exists but wrong size ({file_size} vs {expected_size}). Re-downloading...")
            os.remove(filename)
    
    try:
        # Create session with authentication
        session = requests.Session()
        
        # For ImageNet, you typically need to authenticate with username and access key
        auth_data = {
            'username': username,
            'accesskey': accesskey
        }
        
        # First, get the download URL (some sites require this step)
        response = session.get(url, auth=(username, accesskey), stream=True)
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
        
        # Verify file size
        actual_size = os.path.getsize(filename)
        expected_size = IMAGENET_FILES[os.path.basename(filename)]["size"]
        
        if actual_size == expected_size:
            print(f"✅ Successfully downloaded {filename}")
            return True
        else:
            print(f"❌ Size mismatch for {filename}: {actual_size} vs {expected_size}")
            os.remove(filename)
            return False
            
    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")
        if os.path.exists(filename):
            os.remove(filename)
        return False

def extract_tar_parallel(tar_path, extract_to, desc="Extracting", max_workers=4):
    """Extract tar file with parallel processing."""
    print(f"📦 {desc}: {tar_path}")
    
    def extract_member(args):
        tar_path, member, extract_to = args
        with tarfile.open(tar_path, 'r') as tar:
            tar.extract(member, extract_to)
        return member.name
    
    with tarfile.open(tar_path, 'r') as tar:
        members = tar.getmembers()
        
        # Use parallel processing for large files
        if len(members) > 1000:
            print(f"   Using {max_workers} workers for parallel extraction...")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                tasks = [(tar_path, member, extract_to) for member in members]
                
                with tqdm(total=len(members), desc=desc) as pbar:
                    for result in executor.map(extract_member, tasks):
                        pbar.update(1)
        else:
            # Sequential extraction for smaller files
            with tqdm(total=len(members), desc=desc) as pbar:
                for member in members:
                    tar.extract(member, extract_to)
                    pbar.update(1)
    
    print(f"✅ Extracted {tar_path}")

def create_directory_structure():
    """Create the directory structure for ImageNet."""
    dirs = [
        OUTPUT_DIR,
        f"{OUTPUT_DIR}/train",
        f"{OUTPUT_DIR}/val", 
        f"{OUTPUT_DIR}/meta",
        TEMP_DIR,
        f"{TEMP_DIR}/train_raw",
        f"{TEMP_DIR}/val_raw"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Created directory structure in {OUTPUT_DIR}")

def download_devkit(username, accesskey):
    """Download and extract the ImageNet development kit."""
    devkit_file = f"{TEMP_DIR}/ILSVRC2012_devkit_t12.tar.gz"
    devkit_url = IMAGENET_BASE_URL + "ILSVRC2012_devkit_t12.tar.gz"
    
    if not download_with_auth(devkit_url, devkit_file, username, accesskey):
        return False
    
    # Extract devkit
    extract_tar_parallel(devkit_file, TEMP_DIR, "Extracting devkit")
    return True

def parse_devkit():
    """Parse the development kit to get class mappings."""
    devkit_path = f"{TEMP_DIR}/ILSVRC2012_devkit_t12"
    
    # Read validation ground truth
    val_ground_truth_file = f"{devkit_path}/data/ILSVRC2012_validation_ground_truth.txt"
    if not os.path.exists(val_ground_truth_file):
        print("❌ Validation ground truth file not found!")
        return None, None
    
    print("📖 Parsing validation ground truth...")
    with open(val_ground_truth_file, 'r') as f:
        val_labels = [int(line.strip()) - 1 for line in f]  # Convert to 0-based indexing
    
    # Parse synset information
    print("📖 Parsing class information...")
    synsets_file = f"{devkit_path}/data/meta.mat"
    
    class_info = {}
    try:
        import scipy.io
        meta = scipy.io.loadmat(synsets_file)
        synsets = meta['synsets']
        
        for i, synset in enumerate(synsets[0]):
            ilsvrc_id = int(synset[0][0])  # ILSVRC ID (1-1000)
            wnid = synset[1][0]           # WordNet ID (n########)
            words = synset[2][0]          # Class name
            
            class_info[wnid] = {
                'ilsvrc_id': ilsvrc_id,
                'class_name': words,
                'zero_based_id': ilsvrc_id - 1
            }
            
    except ImportError:
        print("⚠️  scipy not found, creating simplified class mapping...")
        # Create mapping from synset directories if scipy is not available
        train_dirs = []
        if os.path.exists(f"{TEMP_DIR}/train_raw"):
            train_dirs = [d for d in os.listdir(f"{TEMP_DIR}/train_raw") if d.startswith('n')]
        
        for i, wnid in enumerate(sorted(train_dirs)):
            class_info[wnid] = {
                'ilsvrc_id': i + 1,
                'class_name': f'class_{wnid}',
                'zero_based_id': i
            }
    
    print(f"✅ Found {len(class_info)} classes")
    return val_labels, class_info

def download_and_process_training_data(username, accesskey):
    """Download and process training data."""
    train_file = f"{TEMP_DIR}/ILSVRC2012_img_train.tar"
    train_url = IMAGENET_BASE_URL + "ILSVRC2012_img_train.tar"
    
    print("🚨 WARNING: Training data is ~138GB!")
    print("This will take several hours to download and process.")
    
    if not download_with_auth(train_url, train_file, username, accesskey):
        return False
    
    # Extract main training tar file
    print("📦 Extracting training data (this may take a while)...")
    train_raw_dir = f"{TEMP_DIR}/train_raw"
    extract_tar_parallel(train_file, train_raw_dir, "Extracting training tar", max_workers=2)
    
    # Process individual class tar files
    print("🔄 Processing individual class tar files...")
    class_tar_files = [f for f in os.listdir(train_raw_dir) if f.endswith('.tar')]
    
    def process_class_tar(tar_file):
        class_name = tar_file.replace('.tar', '')
        class_dir = f"{OUTPUT_DIR}/train/{class_name}"
        Path(class_dir).mkdir(exist_ok=True)
        
        class_tar_path = f"{train_raw_dir}/{tar_file}"
        extract_tar_parallel(class_tar_path, class_dir, f"Extracting {class_name}", max_workers=1)
        
        # Clean up individual tar file
        os.remove(class_tar_path)
        return class_name
    
    # Process class tar files in parallel
    max_workers = min(cpu_count() // 2, 8)  # Don't overwhelm the system
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(class_tar_files), desc="Processing class files") as pbar:
            for result in executor.map(process_class_tar, class_tar_files):
                pbar.update(1)
    
    # Clean up main training tar and raw directory
    os.remove(train_file)
    shutil.rmtree(train_raw_dir)
    
    print("✅ Training data processed")
    return True

def download_and_process_validation_data(username, accesskey, val_labels, class_info):
    """Download and process validation data."""
    val_file = f"{TEMP_DIR}/ILSVRC2012_img_val.tar"
    val_url = IMAGENET_BASE_URL + "ILSVRC2012_img_val.tar"
    
    if not download_with_auth(val_url, val_file, username, accesskey):
        return False
    
    # Extract validation data
    print("📦 Extracting validation data...")
    val_raw_dir = f"{TEMP_DIR}/val_raw"
    extract_tar_parallel(val_file, val_raw_dir, "Extracting validation data")
    
    # Create class directories
    print("🗂️  Organizing validation data by class...")
    wnid_to_dirname = {}
    for wnid, info in class_info.items():
        class_dir = f"{OUTPUT_DIR}/val/{wnid}"
        Path(class_dir).mkdir(exist_ok=True)
        wnid_to_dirname[info['zero_based_id']] = wnid
    
    # Move validation images to appropriate class directories
    val_images = sorted([f for f in os.listdir(val_raw_dir) if f.endswith('.JPEG')])
    
    with tqdm(total=len(val_images), desc="Organizing validation images") as pbar:
        for i, image_name in enumerate(val_images):
            if i < len(val_labels):
                class_id = val_labels[i]
                wnid = wnid_to_dirname[class_id]
                
                src_path = f"{val_raw_dir}/{image_name}"
                dst_path = f"{OUTPUT_DIR}/val/{wnid}/{image_name}"
                
                shutil.move(src_path, dst_path)
            pbar.update(1)
    
    # Clean up
    os.remove(val_file)
    shutil.rmtree(val_raw_dir)
    
    print("✅ Validation data organized")
    return True

def create_meta_files(class_info):
    """Create meta files in the same format as the working sample."""
    print("📝 Creating meta files...")
    
    # Create train.txt
    train_txt_path = f"{OUTPUT_DIR}/meta/train.txt"
    with open(train_txt_path, 'w') as f:
        for wnid, info in sorted(class_info.items()):
            class_dir = f"{OUTPUT_DIR}/train/{wnid}"
            if os.path.exists(class_dir):
                images = sorted([img for img in os.listdir(class_dir) if img.endswith('.JPEG')])
                for image_name in images:
                    f.write(f"{wnid}/{image_name} {info['zero_based_id']}\n")
    
    # Create val.txt
    val_txt_path = f"{OUTPUT_DIR}/meta/val.txt"
    with open(val_txt_path, 'w') as f:
        for wnid, info in sorted(class_info.items()):
            class_dir = f"{OUTPUT_DIR}/val/{wnid}"
            if os.path.exists(class_dir):
                images = sorted([img for img in os.listdir(class_dir) if img.endswith('.JPEG')])
                for image_name in images:
                    f.write(f"{wnid}/{image_name} {info['zero_based_id']}\n")
    
    # Create class mapping JSON (same format as sample)
    class_mapping_path = f"{OUTPUT_DIR}/meta/class_mapping.json"
    with open(class_mapping_path, 'w') as f:
        json.dump(class_info, f, indent=2)
    
    print(f"✅ Created meta files in {OUTPUT_DIR}/meta/")

def verify_full_dataset():
    """Verify the full ImageNet dataset."""
    print("\n🔍 Verifying dataset structure...")
    
    # Check directories exist
    required_dirs = [f"{OUTPUT_DIR}/train", f"{OUTPUT_DIR}/val", f"{OUTPUT_DIR}/meta"]
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"❌ Missing directory: {dir_path}")
            return False
        print(f"✅ Found directory: {dir_path}")
    
    # Check meta files
    meta_files = [
        f"{OUTPUT_DIR}/meta/train.txt",
        f"{OUTPUT_DIR}/meta/val.txt", 
        f"{OUTPUT_DIR}/meta/class_mapping.json"
    ]
    for meta_file in meta_files:
        if not os.path.exists(meta_file):
            print(f"❌ Missing meta file: {meta_file}")
            return False
        print(f"✅ Found meta file: {meta_file}")
    
    # Count classes and images
    train_classes = len([d for d in os.listdir(f"{OUTPUT_DIR}/train") if os.path.isdir(f"{OUTPUT_DIR}/train/{d}")])
    val_classes = len([d for d in os.listdir(f"{OUTPUT_DIR}/val") if os.path.isdir(f"{OUTPUT_DIR}/val/{d}")])
    
    # Count images
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
    
    # Calculate dataset size
    total_size = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, dirnames, filenames in os.walk(OUTPUT_DIR)
        for filename in filenames
    )
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Training classes: {train_classes}")
    print(f"   Validation classes: {val_classes}")
    print(f"   Training images: {train_images:,}")
    print(f"   Validation images: {val_images:,}")
    print(f"   Total size: {total_size / (1024**3):.1f}GB")
    print(f"   Location: {OUTPUT_DIR}")
    
    if train_classes == 1000 and val_classes == 1000:
        print("✅ Full ImageNet dataset verified successfully!")
        return True
    else:
        print(f"⚠️  Warning: Expected 1000 classes, found {train_classes} train, {val_classes} val")
        return False

def main():
    """Main function to download and setup full ImageNet."""
    global OUTPUT_DIR, TEMP_DIR
    
    parser = argparse.ArgumentParser(
        description='Download and setup full ImageNet ILSVRC2012 dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tools/download_imagenet_full.py --username your_user --accesskey your_key
    
Prerequisites:
    1. Register at http://image-net.org
    2. Get your download credentials  
    3. Ensure ~200GB free disk space
    4. Fast internet connection recommended
        """
    )
    
    parser.add_argument('--username', required=True, 
                       help='Your ImageNet username')
    parser.add_argument('--accesskey', required=True,
                       help='Your ImageNet access key')
    parser.add_argument('--output-dir', default="data/imagenet",
                       help='Output directory (default: data/imagenet)')
    parser.add_argument('--temp-dir', default="data/temp_imagenet_full", 
                       help='Temporary directory for downloads')
    parser.add_argument('--skip-train', action='store_true',
                       help='Skip training data (validation only, ~7GB)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers for extraction')
    
    args = parser.parse_args()
    
    # Update global paths
    OUTPUT_DIR = args.output_dir  
    TEMP_DIR = args.temp_dir
    
    print("🚀 ImageNet ILSVRC2012 Full Dataset Downloader")
    print(f"📁 Output: {OUTPUT_DIR}")
    print(f"🔧 Temp: {TEMP_DIR}")
    print(f"👤 User: {args.username}")
    
    # Check prerequisites
    if not check_disk_space():
        return 1
    
    # Estimate download time
    total_size_gb = 145  # Approximate
    if not args.skip_train:
        total_size_gb = 145
    else:
        total_size_gb = 7
    
    print(f"\n⏱️  Estimated download: ~{total_size_gb}GB")
    print("   On a 100Mbps connection: ~3-4 hours")
    print("   On a 1Gbps connection: ~20-30 minutes")
    
    response = input("\n🔄 Continue with download? [y/N]: ")
    if response.lower() != 'y':
        print("❌ Aborted.")
        return 1
    
    start_time = time.time()
    
    try:
        # Step 1: Create directories  
        create_directory_structure()
        
        # Step 2: Download devkit
        print("\n📋 Step 1/4: Downloading development kit...")
        if not download_devkit(args.username, args.accesskey):
            print("❌ Failed to download devkit!")
            return 1
        
        # Step 3: Parse devkit
        print("\n📖 Step 2/4: Parsing development kit...")
        val_labels, class_info = parse_devkit()
        if not val_labels or not class_info:
            print("❌ Failed to parse devkit!")
            return 1
        
        # Step 4: Download validation data
        print("\n📥 Step 3/4: Downloading validation data...")
        if not download_and_process_validation_data(args.username, args.accesskey, val_labels, class_info):
            print("❌ Failed to process validation data!")
            return 1
        
        # Step 5: Download training data (optional)
        if not args.skip_train:
            print("\n📥 Step 4/4: Downloading training data (this will take a while)...")
            if not download_and_process_training_data(args.username, args.accesskey):
                print("❌ Failed to process training data!")
                return 1
        else:
            print("\n⏭️  Step 4/4: Skipping training data")
        
        # Step 6: Create meta files
        print("\n📝 Creating meta files...")
        create_meta_files(class_info)
        
        # Step 7: Verify dataset
        print("\n🔍 Verifying dataset...")
        if verify_full_dataset():
            elapsed = time.time() - start_time
            print(f"\n🎉 SUCCESS! ImageNet dataset ready in {elapsed/3600:.1f} hours")
            print(f"\n🚀 You can now train with:")
            print(f"   bash train_imagenet_resnet18.sh")
            print(f"   # (Update DATA_PATH to '{OUTPUT_DIR}' in the script)")
        else:
            print("\n⚠️  Dataset setup completed with warnings")
        
        # Cleanup temp directory
        if os.path.exists(TEMP_DIR):
            print(f"🧹 Cleaning up {TEMP_DIR}...")
            shutil.rmtree(TEMP_DIR)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Download interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
