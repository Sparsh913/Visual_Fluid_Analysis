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

from dataloader import FluidViscosityDataset

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