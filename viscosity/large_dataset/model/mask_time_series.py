from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mask_encoder import MaskEncoder
from .mask_resnet import MaskResnet
from .robot_encoder import RobotEncoder

class MaskTimeSeries(nn.Module):
    def __init__(self, embed_dim=160, num_heads=4, num_layers=1):
        super().__init__()
        self.mask_encoder = MaskResnet() 
        self.robot_encoder = RobotEncoder() 
        self.timestamp_encoder = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(),
            nn.Linear(32, embed_dim)
        )
        self.embed_dim = embed_dim
        self.layer_norm = nn.LayerNorm(embed_dim)
        transformer_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, dropout=0.3)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers)

    def forward(self, mask_seq, robot_seq, timestamps):
        B, T, C, H, W = mask_seq.shape
        mask_feats = self.mask_encoder(mask_seq) # (B,T,E1)
        robot_feats = self.robot_encoder(robot_seq) # (B,T,E2)
        combined_feats = torch.cat([mask_feats, robot_feats],dim=-1) # (B,T,E1+E2)
        # timestamps = timestamps - timestamps[:,[0]]  # normalized timestamps start at zero
        timestamp_feats = self.timestamp_encoder(timestamps.unsqueeze(-1)) # (B,T,embed_dim) This is the positional embedding
        transformer_in = self.layer_norm(combined_feats + timestamp_feats)  # (B,T,E1+E2+32)
        # transformer_in = self.layer_norm(combined_feats)
        transformer_out = self.transformer(transformer_in.permute(1,0,2)) #(T,B,E)
        transformer_pooled = transformer_out.mean(dim=0) #(B,E)
        return transformer_pooled #combined_feats.permute(1,0,2).mean(dim=0) #
        