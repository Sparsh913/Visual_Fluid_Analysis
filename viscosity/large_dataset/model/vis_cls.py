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


class VisCls(nn.Module):
    def __init__(self, embed_dim=160):
        super().__init__()
        self.transformer = MaskTimeSeries(embed_dim)
        # num_classes = 3
        self.cls = nn.Sequential(
            nn.Linear(embed_dim, 32), nn.ReLU(),
            nn.Linear(32, 3)
        )
        
    def forward(self, mask_seq, robot_seq, timestamps, return_attn=False):
        if return_attn:
            seq_emb, attn_weights, attn_penalty = self.transformer(
                mask_seq, robot_seq, timestamps, return_attn=True
            )
            cls = self.cls(seq_emb)
            return cls, attn_weights, attn_penalty
        else:
            seq_emb, attn_penalty = self.transformer(
                mask_seq, robot_seq, timestamps, return_attn=False, apply_attn_reg=True
            )
            cls = self.cls(seq_emb)
            return cls, attn_penalty