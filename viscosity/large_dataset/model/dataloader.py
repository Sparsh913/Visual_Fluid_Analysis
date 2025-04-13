import os
import json
import torch
import numpy as np
import cv2
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class FluidViscosityDataset(Dataset):
    def __init__(self, 
                 root_dir,
                 config_json,
                 vial_label_csv,
                 split='train',
                 sequence_length=5,
                 mask_format='png',
                 transform=None,
                 normalize_robot_data=True,
                 normalize_timestamps=True,
                 robot_mean_std=None,
                 deg_convert=True,
                 task='classification',  # 'classification' or 'regression'
                 regression_csv=None     # path to regression values CSV
                 ):
        """
        Args:
            root_dir: str, path to the data root containing vial directories
            config_json: str, path to JSON file containing train/val/test split info
            vial_label_csv: str, path to CSV mapping vial_id -> label
            split: 'train' | 'val' | 'test'
            sequence_length: int, number of frames per sequence
            mask_format: 'png' or 'npy'
            transform: optional transform for masks
            normalize_robot_data: bool, whether to normalize robot joint data
            normalize_timestamps: bool, whether to scale timestamps per sequence
            robot_mean_std: dict with normalization parameters for robot data
            deg_convert: bool, whether to convert angles from radians to degrees
            task: 'classification' or 'regression'
            regression_csv: path to CSV with viscosity regression values
        """
        self.root_dir = root_dir
        self.split = split
        self.sequence_length = sequence_length
        self.mask_format = mask_format
        self.transform = transform
        self.normalize_robot_data = normalize_robot_data
        self.normalize_timestamps = normalize_timestamps
        self.robot_mean_std = robot_mean_std
        self.deg_convert = deg_convert
        self.task = task
        
        # Load vial label CSV for classification
        df_cls = pd.read_csv(vial_label_csv)
        self.vial_to_label = {row['vial_id']: row['label'].lower() for _, row in df_cls.iterrows()}
        self.label_to_int = {'low': 0, 'medium': 1, 'high': 2}
        
        # Load regression values if in regression mode
        self.vial_to_value = {}
        if task == 'regression' and regression_csv is not None:
            df_reg = pd.read_csv(regression_csv)
            self.vial_to_value = {row['vial_id']: row['value'] for _, row in df_reg.iterrows()}
            
            # Calculate regression statistics for normalization
            reg_values = np.array(list(self.vial_to_value.values()))
            self.reg_mean = float(np.mean(reg_values))
            self.reg_std = float(np.std(reg_values)) + 1e-6
            print(f"Regression value statistics - Mean: {self.reg_mean:.4f}, Std: {self.reg_std:.4f}")

        # Load config JSON
        with open(config_json, 'r') as f:
            self.config = json.load(f)

        self.samples = []
        self.all_robot_vals = [] if split == 'train' and normalize_robot_data and robot_mean_std is None else None
        self._prepare_samples()

        if self.split == 'train' and self.normalize_robot_data and self.robot_mean_std is None:
            all_vals = torch.tensor(self.all_robot_vals, dtype=torch.float32)
            self.robot_mean = all_vals.mean(dim=0)
            self.robot_std = all_vals.std(dim=0) + 1e-6
        elif self.robot_mean_std:
            self.robot_mean = torch.tensor(self.robot_mean_std['mean'], dtype=torch.float32)
            self.robot_std = torch.tensor(self.robot_mean_std['std'], dtype=torch.float32)
        else:
            self.robot_mean, self.robot_std = None, None

    def _prepare_samples(self):
        print(f"Preparing samples for split: {self.split}")

        if self.split in ['train', 'val']:
            vial_profiles = self.config[self.split]
        elif self.split == 'test':
            if isinstance(self.config['test'], dict):
                vial_profiles = self.config[self.split] # load selected profiles
            else:
                vial_profiles = {vial: None for vial in self.config['test']}  # load all profiles
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

        for vial_id, profile_list in vial_profiles.items():
            # Skip if vial doesn't have required data for the task
            if self.task == 'classification' and vial_id not in self.vial_to_label:
                print(f"[Skip] Vial {vial_id} not found in classification labels")
                continue
            if self.task == 'regression' and vial_id not in self.vial_to_value:
                print(f"[Skip] Vial {vial_id} not found in regression values")
                continue
                
            vial_path = os.path.join(self.root_dir, vial_id)
            if not os.path.isdir(vial_path):
                print(f"[Warning] Vial path {vial_path} does not exist.")
                continue

            motion_profiles = profile_list or sorted(os.listdir(vial_path))

            for profile in motion_profiles:
                mp_path = os.path.join(vial_path, profile)
                mask_dir = os.path.join(mp_path, "masks")

                joint_log_file = [f for f in os.listdir(mp_path) if f.startswith("joint_log")][0]
                image_log_file = [f for f in os.listdir(mp_path) if f.startswith("image_log")][0]

                joint_log_path = os.path.join(mp_path, joint_log_file)
                image_log_path = os.path.join(mp_path, image_log_file)

                if not (os.path.exists(mask_dir) and os.path.exists(joint_log_path) and os.path.exists(image_log_path)):
                    print(f"[Skipping] Incomplete profile: {mp_path}")
                    continue

                img_map = {}
                with open(image_log_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        idx = int(parts[0])
                        timestamp = float(parts[1])
                        img_map[idx] = timestamp

                joint_log = {}
                with open(joint_log_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        try:
                            idx = int(parts[0])
                            timestamp = float(parts[1])

                            def parse_section(label):
                                start = line.index(f"{label}:") + len(f"{label}:")
                                end = line.index(';', start)
                                return list(map(float, line[start:end].split(',')))

                            angles = parse_section("Actual Angles")
                            speeds = parse_section("Actual Speeds")
                            accels = parse_section("Actual Accelerations")

                            if not self.deg_convert:
                                joint_log[idx] = {
                                    'timestamp': timestamp,
                                    'angle': angles[5],
                                    'speed': speeds[5],
                                    'accel': accels[5]
                                }
                            else:
                                joint_log[idx] = {
                                    'timestamp': timestamp,
                                    'angle': np.rad2deg(angles[5]),
                                    'speed': np.rad2deg(speeds[5]),
                                    'accel': np.rad2deg(accels[5])
                                }

                        except Exception as e:
                            print(f"[Error] Parsing joint log at {mp_path}: {e}")
                            continue

                sorted_ids = sorted(set(img_map.keys()) & set(joint_log.keys()))
                for i in range(len(sorted_ids) - self.sequence_length + 1):
                    ids_seq = sorted_ids[i:i + self.sequence_length]
                    timestamps = [joint_log[idx]['timestamp'] for idx in ids_seq]
                    angles = [joint_log[idx]['angle'] for idx in ids_seq]
                    speeds = [joint_log[idx]['speed'] for idx in ids_seq]
                    accels = [joint_log[idx]['accel'] for idx in ids_seq]

                    masks_exist = all([
                        os.path.exists(os.path.join(mask_dir, f"{idx}.{self.mask_format}")) 
                        for idx in ids_seq
                    ])
                    if not masks_exist:
                        print(f"[Skip] Missing masks in sequence starting at {ids_seq[0]} in {mp_path}")
                        continue

                    if self.all_robot_vals is not None:
                        self.all_robot_vals.extend(list(zip(angles, speeds, accels)))
                    
                    # Prepare sample with appropriate label based on task
                    sample_data = {
                        'mask_paths': [os.path.join(mask_dir, f"{idx}.{self.mask_format}") for idx in ids_seq],
                        'angles': angles,
                        'speeds': speeds,
                        'accels': accels,
                        'timestamps': timestamps,
                        'vial_id': vial_id,
                    }
                    
                    # Add task-specific target
                    if self.task == 'classification':
                        sample_data['label'] = self.label_to_int[self.vial_to_label[vial_id]]
                    else:  # regression
                        sample_data['value'] = self.vial_to_value[vial_id]
                    
                    self.samples.append(sample_data)

        print(f"Total sequences for split '{self.split}' ({self.task}): {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        mask_seq = []
        for path in sample['mask_paths']:
            if self.mask_format == 'png':
                # Load the mask with OpenCV
                mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise ValueError(f"Could not read mask image: {path}")
                    
                # Apply transforms if needed (assumes transforms don't include ToTensor)
                if self.transform:
                    mask_pil = Image.fromarray(mask)
                    mask_pil = self.transform(mask_pil)
                    mask = np.array(mask_pil)
                    
                # Convert to tensor and normalize once
                mask = torch.tensor(mask, dtype=torch.float32) / 255.0
                    
            elif self.mask_format == 'npy':
                mask = np.load(path)
                
                # For NPY files, assume they're already in the right format
                # but still apply transforms if needed
                if self.transform:
                    # Convert to PIL for transforms
                    mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
                    mask_pil = self.transform(mask_pil)
                    mask = np.array(mask_pil) / 255.0
                    
                mask = torch.tensor(mask, dtype=torch.float32)
            else:
                raise ValueError("Unsupported mask format.")
                
            mask_seq.append(mask.unsqueeze(0))
            
        mask_tensor = torch.stack(mask_seq) # (T,1,H,W)
        robot_tensor = torch.tensor(
            list(zip(sample['angles'], sample['speeds'], sample['accels'])),
            dtype=torch.float32
        )  # (T,3)

        if self.normalize_robot_data:
            if self.robot_mean is not None:
                robot_tensor = (robot_tensor - self.robot_mean) / self.robot_std
            else:
                robot_tensor = (robot_tensor - robot_tensor.mean(dim=0)) / (robot_tensor.std(dim=0) + 1e-6)

        ts = np.array(sample['timestamps'], dtype=np.float32)
        ts -= ts[0]  # normalize to start at zero
        if self.normalize_timestamps:
            ts /= (ts[-1] + 1e-6)  # normalize to [0, 1] per sequence

        timestamp_tensor = torch.tensor(ts, dtype=torch.float32)  # (T,)
        
        # Prepare output dictionary
        output = {
            'masks': mask_tensor,
            'robot': robot_tensor,
            'timestamps': timestamp_tensor,
            'vial_id': sample['vial_id'],
        }
        
        # Add task-specific target
        if self.task == 'classification':
            output['label'] = torch.tensor(sample['label'], dtype=torch.long)
        else:  # regression
            # Normalize regression values
            raw_value = sample['value']
            normalized_value = (raw_value - self.reg_mean) / self.reg_std
            output['value'] = torch.tensor(normalized_value, dtype=torch.float32)
            # Also include raw value for evaluation
            output['raw_value'] = torch.tensor(raw_value, dtype=torch.float32)
        
        return output