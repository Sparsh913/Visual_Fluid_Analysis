from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from .mask_time_series import MaskTimeSeries


class VisModel(nn.Module):
    def __init__(self, embed_dim=160, num_classes=3, task='classification'):
        super().__init__()
        self.transformer = MaskTimeSeries(embed_dim)
        self.task = task
        if task == 'classification':
            self.out_layer = nn.Sequential(
                nn.Linear(embed_dim, 32), nn.ReLU(),
                nn.Linear(32, num_classes)
            )
        else:
            self.out_layer = nn.Sequential(
                nn.Linear(embed_dim, 32), nn.ReLU(),
                nn.Linear(32, 1)
            )
        
    def forward(self, mask_seq, robot_seq, timestamps, return_attn=False, apply_attn_reg=False):
        if return_attn:
            seq_emb, attn_weights, attn_penalty = self.transformer(
                mask_seq, robot_seq, timestamps, return_attn=True
            )
            output = self.out_layer(seq_emb)
            
            # Adjust output shape for regression
            if self.task == 'regression':
                output = output.squeeze(-1)  # Remove last dimension
            if apply_attn_reg:
                return output, attn_weights, attn_penalty
            return output, attn_weights
        else:
            seq_emb, attn_penalty = self.transformer(
                mask_seq, robot_seq, timestamps, return_attn=False, apply_attn_reg=True
            )
            output = self.out_layer(seq_emb)
            
            # Adjust output shape for regression
            if self.task == 'regression':
                output = output.squeeze(-1)  # Remove last dimension
            if apply_attn_reg:
                return output, attn_penalty
            return output