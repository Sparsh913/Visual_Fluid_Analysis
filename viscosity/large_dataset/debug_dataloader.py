import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

from model.dataloader import FluidViscosityDataset

# First let's test the dataloader whether all the data is being read correctly or not

train_dataset  =FluidViscosityDataset(
    root_dir='/ocean/projects/agr240001p/mqureshi/sparsh/liquid_detection/sam2/test_liq4',
    config_json='configs/test_config.json',
    vial_label_csv='vial_label.csv',
    split='train',
    sequence_length=11,
    mask_format='png',
    transform=None
)
print("length of train dataset", len(train_dataset))
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
sample_1 = train_loader.__iter__().__next__()
sample_2 = train_loader.__iter__().__next__()
print("sample 1")
for key, val in sample_1.items():
    if isinstance(val, torch.Tensor):
        print(key, val.shape)
    else:
        print(key, val)
print("sample 2")
for key, val in sample_2.items():
    if isinstance(val, torch.Tensor):
        print(key, val.shape)
    else:
        print(key, val)
        

def validate_masks(dataset):
    """
    Pre-validate all masks in a dataset to avoid runtime errors.
    
    Args:
        dataset: FluidViscosityDataset instance
    
    Returns:
        list: Indices of samples with invalid masks
    """
    invalid_indices = []
    
    print(f"Validating masks for {len(dataset)} samples...")
    for idx, sample in enumerate(dataset.samples):
        if idx % 100 == 0:
            print(f"Validating sample {idx}/{len(dataset.samples)}...")
            
        has_invalid = False
        for path in sample['mask_paths']:
            if dataset.mask_format == 'png':
                mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    print(f"Invalid mask at index {idx}: {path}")
                    has_invalid = True
                    break
            elif dataset.mask_format == 'npy':
                try:
                    mask = np.load(path)
                except Exception as e:
                    print(f"Failed to load NPY mask at index {idx}: {path} - {e}")
                    has_invalid = True
                    break
                    
        if has_invalid:
            invalid_indices.append(idx)
    
    print(f"Found {len(invalid_indices)} samples with invalid masks out of {len(dataset.samples)}")
    return invalid_indices


def validate_and_sample_logs(config_json, root_dir, n_samples=10):
    """
    Validate and sample from img_map and joint_log dictionaries across train, val, test splits.
    
    Args:
        config_json: str, path to JSON file containing train/val/test split info
        root_dir: str, path to the data root containing vial directories
        n_samples: int, number of random indices to sample per split
    """
    import random
    import json
    import os
    
    # Load config
    with open(config_json, 'r') as f:
        config = json.load(f)
    
    # Helper to process a single vial profile and sample indices
    def process_vial_profile(vial_id, profile, split):
        mp_path = os.path.join(root_dir, vial_id, profile)
        
        # Get log files
        try:
            joint_log_files = [f for f in os.listdir(mp_path) if f.startswith("joint_log")]
            image_log_files = [f for f in os.listdir(mp_path) if f.startswith("image_log")]
            
            if not joint_log_files or not image_log_files:
                print(f"[Error] Missing log files in {mp_path}")
                return None
                
            joint_log_path = os.path.join(mp_path, joint_log_files[0])
            image_log_path = os.path.join(mp_path, image_log_files[0])
            
            # Load image log
            img_map = {}
            with open(image_log_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        idx = int(parts[0])
                        timestamp = float(parts[1])
                        img_map[idx] = timestamp
            
            # Load joint log
            joint_log = {}
            with open(joint_log_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    try:
                        if len(parts) >= 2:
                            idx = int(parts[0])
                            timestamp = float(parts[1])
                            
                            if "Actual Angles:" in line and "Actual Speeds:" in line and "Actual Accelerations:" in line:
                                def parse_section(label):
                                    start = line.index(f"{label}:") + len(f"{label}:")
                                    end = line.index(';', start)
                                    return list(map(float, line[start:end].split(',')))
                                
                                try:
                                    angles = parse_section("Actual Angles")
                                    speeds = parse_section("Actual Speeds")
                                    accels = parse_section("Actual Accelerations")
                                    
                                    joint_log[idx] = {
                                        'timestamp': timestamp,
                                        'angle': angles[5] if len(angles) > 5 else None,
                                        'speed': speeds[5] if len(speeds) > 5 else None,
                                        'accel': accels[5] if len(accels) > 5 else None
                                    }
                                except Exception as e:
                                    print(f"[Error] Parsing sections in {joint_log_path}, line starts with: {line[:50]}...: {e}")
                            else:
                                print(f"[Warning] Missing expected sections in line: {line[:50]}...")
                    except Exception as e:
                        print(f"[Error] Processing joint log line in {joint_log_path}: {e}")
                        print(f"  Problematic line: {line[:100]}...")
            
            # Find common indices
            common_indices = sorted(set(img_map.keys()) & set(joint_log.keys()))
            
            if not common_indices:
                print(f"[Warning] No common indices between img_map and joint_log in {mp_path}")
                return None
            
            # Sample random indices
            if len(common_indices) <= n_samples:
                sample_indices = common_indices
            else:
                sample_indices = sorted(random.sample(common_indices, n_samples))
            
            # Prepare sample data
            sample_data = []
            for idx in sample_indices:
                sample_data.append({
                    'idx': idx,
                    'img_timestamp': img_map[idx],
                    'joint_timestamp': joint_log[idx]['timestamp'],
                    'angle': joint_log[idx]['angle'],
                    'speed': joint_log[idx]['speed'],
                    'accel': joint_log[idx]['accel']
                })
            
            return {
                'vial_id': vial_id,
                'profile': profile,
                'total_img_indices': len(img_map),
                'total_joint_indices': len(joint_log),
                'common_indices': len(common_indices),
                'samples': sample_data
            }
            
        except Exception as e:
            print(f"[Error] Processing {mp_path}: {e}")
            return None
    
    # Process each split
    results = {}
    
    for split in ['train', 'val', 'test']:
        print(f"\n==== Validating {split} split ====")
        results[split] = []
        
        if split in ['train', 'val']:
            # For train/val, process specific profiles
            for vial_id, profiles in config[split].items():
                for profile in profiles:
                    result = process_vial_profile(vial_id, profile, split)
                    if result:
                        results[split].append(result)
        else:
            # For test, process all profiles for each vial
            for vial_id in config['test']:
                vial_path = os.path.join(root_dir, vial_id)
                if os.path.isdir(vial_path):
                    profiles = sorted(os.listdir(vial_path))
                    for profile in profiles:
                        result = process_vial_profile(vial_id, profile, split)
                        if result:
                            results[split].append(result)
        
        # Print results for this split
        for result in results[split]:
            print(f"\nVial: {result['vial_id']}, Profile: {result['profile']}")
            print(f"Total indices - Images: {result['total_img_indices']}, Joints: {result['total_joint_indices']}, Common: {result['common_indices']}")
            
            print("\nSample data:")
            for i, sample in enumerate(result['samples']):
                print(f"  Sample {i+1}:")
                print(f"    Index: {sample['idx']}")
                print(f"    Image timestamp: {sample['img_timestamp']}")
                print(f"    Joint timestamp: {sample['joint_timestamp']}")
                print(f"    Angle: {sample['angle']}")
                print(f"    Speed: {sample['speed']}")
                print(f"    Accel: {sample['accel']}")
    
    return results