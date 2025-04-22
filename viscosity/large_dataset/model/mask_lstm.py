from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mask_encoder import MaskEncoder
from .mask_resnet import MaskResnet
from .robot_encoder import RobotEncoder

class MaskLSTM(nn.Module):
    def __init__(self, embed_dim=160, num_heads=4, num_layers=1):
        super().__init__()
        self.mask_encoder = MaskResnet(int(0.8*embed_dim))
        self.robot_encoder = RobotEncoder(embedding_dim=embed_dim-int(0.8*embed_dim)) 
        self.timestamp_encoder = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(),
            nn.Linear(32, embed_dim)
        )
        self.embed_dim = embed_dim
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=embed_dim // (2 if embed_dim % 2 == 0 else 1),
            num_layers=num_layers,
            dropout=0.2,
            batch_first=True,
            bidirectional=True
        )

    
    def forward(self, mask_seq, robot_seq, timestamps, return_attn=False, apply_attn_reg=False):
        B, T, C, H, W = mask_seq.shape
        # B, T, _ = mask_seq.shape
        mask_feats = self.mask_encoder(mask_seq)  # (B,T,E1)
        robot_feats = self.robot_encoder(robot_seq)  # (B,T,E2)
        combined_feats = torch.cat([mask_feats, robot_feats], dim=-1)  # (B,T,E1+E2)
        
        out, (h_n, c_n) = self.lstm(combined_feats)

        # if bidirectional, h_n is (2*num_layers, B, hidden), else (num_layers, B, hidden)
        # we take the last layer’s hidden state(s):
        if self.lstm.bidirectional:
            # concat final forward + backward
            last_fwd = h_n[-2]  # (B, hidden)
            last_bwd = h_n[-1]  # (B, hidden)
            seq_emb = torch.cat([last_fwd, last_bwd], dim=-1)  # (B, 2*hidden)==embed_dim
        else:
            seq_emb = h_n[-1]  # (B, hidden)==embed_dim

        return seq_emb