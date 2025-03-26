from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class MaskResnet(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.cnn = models.resnet18(pretrained=True)
        # Modify the first convolutional layer to accept single-channel input
        self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Remove the final fully connected layer
        self.cnn.fc = nn.Linear(512, embedding_dim)

    def forward(self, masks):  # shape: (B, C, T, H, W) # C = 1
        B, T, C, H, W = masks.shape
        embeddings = []
        for t in range(T):
            x = self.cnn(masks[:,t,:,:,:])  # shape (B, 512)
            embeddings.append(x)
        embeddings = torch.stack(embeddings, dim=1)
        return embeddings