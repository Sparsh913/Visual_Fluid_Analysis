import os
import json
import torch
import numpy as np
import cv2
import pandas as pd
from torch.utils.data import Dataset

class FluidViscosityDataset(Dataset):
    def __init__(self, 
                 root_dir,
                 config_json,
                 vial_label_csv,
                 split='train',
                 sequence_length=5,
                 mask_format='png',
                 transform=None):
        """
        Args:
            root_dir: str, path to the data root containing vial directories
            config_json: str, path to JSON file containing train/val/test split info
            vial_label_csv: str, path to CSV mapping vial_id -> label
            split: 'train' | 'val' | 'test'
            sequence_length: int, number of frames per sequence
            mask_format: 'png' or 'npy'
            transform: optional transform for masks
        """
        self.root_dir = root_dir
        self.split = split
        self.sequence_length = sequence_length
        self.mask_format = mask_format
        self.transform = transform

        # Load vial label CSV
        df = pd.read_csv(vial_label_csv)
        self.vial_to_label = {row['vial_id']: row['label'].lower() for _, row in df.iterrows()}
        self.label_to_int = {'low': 0, 'medium': 1, 'high': 2}

        # Load config JSON
        with open(config_json, 'r') as f:
            self.config = json.load(f)

        self.samples = []
        self._prepare_samples()

    def _prepare_samples(self):
        print(f"Preparing samples for split: {self.split}")

        if self.split in ['train', 'val']:
            vial_profiles = self.config[self.split]
        elif self.split == 'test':
            vial_profiles = {vial: None for vial in self.config['test']}  # load all profiles
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

        for vial_id, profile_list in vial_profiles.items():
            vial_path = os.path.join(self.root_dir, vial_id)
            if not os.path.isdir(vial_path):
                print(f"[Warning] Vial path {vial_path} does not exist.")
                continue

            # If None, load all motion profiles in this vial
            motion_profiles = profile_list or sorted(os.listdir(vial_path))

            for profile in motion_profiles:
                mp_path = os.path.join(vial_path, profile)
                mask_dir = os.path.join(mp_path, "masks")

                # Infer log file names
                joint_log_file = [f for f in os.listdir(mp_path) if f.startswith("joint_log")][0]
                image_log_file = [f for f in os.listdir(mp_path) if f.startswith("image_log")][0]

                joint_log_path = os.path.join(mp_path, joint_log_file)
                image_log_path = os.path.join(mp_path, image_log_file)

                if not (os.path.exists(mask_dir) and os.path.exists(joint_log_path) and os.path.exists(image_log_path)):
                    print(f"[Skipping] Incomplete profile: {mp_path}")
                    continue

                # Load image timestamp mapping
                img_map = {}
                with open(image_log_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        idx = int(parts[0])
                        timestamp = float(parts[1])
                        img_map[idx] = timestamp

                # Parse joint log
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

                            joint_log[idx] = {
                                'timestamp': timestamp,
                                'angle': angles[5],
                                'speed': speeds[5],
                                'accel': accels[5]
                            }
                        except Exception as e:
                            print(f"[Error] Parsing joint log at {mp_path}: {e}")
                            continue

                # Build valid sequences
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

                    self.samples.append({
                        'mask_paths': [os.path.join(mask_dir, f"{idx}.{self.mask_format}") for idx in ids_seq],
                        'angles': angles,
                        'speeds': speeds,
                        'accels': accels,
                        'timestamps': timestamps,
                        'label': self.label_to_int[self.vial_to_label[vial_id]]
                    })

        print(f"Total sequences for split '{self.split}': {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load masks
        mask_seq = []
        mask_paths = [] # return mask paths for debugging
        for path in sample['mask_paths']:
            if self.mask_format == 'png':
                mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                mask = torch.tensor(mask, dtype=torch.float32) / 255.0
            elif self.mask_format == 'npy':
                mask = np.load(path)
                mask = torch.tensor(mask, dtype=torch.float32)
            else:
                raise ValueError("Unsupported mask format.")
            if self.transform:
                mask = self.transform(mask)
            mask_seq.append(mask.unsqueeze(0))  # (1,H,W)
            mask_paths.append(path)

        mask_tensor = torch.stack(mask_seq)  # (T,1,H,W)
        robot_tensor = torch.tensor(
            list(zip(sample['angles'], sample['speeds'], sample['accels'])),
            dtype=torch.float32
        )  # (T,3)

        ts = np.array(sample['timestamps'], dtype=np.float32)
        # ts -= ts[0]  # normalize
        timestamp_tensor = torch.tensor(ts, dtype=torch.float32)  # (T,)
        label = torch.tensor(sample['label'], dtype=torch.int8)  # (1,)

        return {
            'masks': mask_tensor,
            'robot': robot_tensor,
            'timestamps': timestamp_tensor,
            'label': label,
            'mask_paths': mask_paths
        }
