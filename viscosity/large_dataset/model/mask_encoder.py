from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # global pooling
            nn.Flatten(),             # -> shape (B,32)
            nn.Linear(32, embedding_dim),       # final embedding
            nn.ReLU()
        )

    def forward(self, masks):  # shape: (B, C, T, H, W) # C = 1
        B, T, C, H, W = masks.shape
        embeddings = []
        for t in range(T):
            emb_t = self.encoder(masks[:,t,:,:,:])  # shape (B, embedding_dim)
            embeddings.append(emb_t)
        embeddings = torch.stack(embeddings, dim=1)  # shape (B, T, embedding_dim)
        return embeddings
    