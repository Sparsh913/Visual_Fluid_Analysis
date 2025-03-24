from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class RobotEncoder(nn.Module):
    def __init__(self, input_dim=3):  # [pos, vel, acc] per timestep
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim,64), 
            nn.ReLU(),
            nn.Linear(64,32)
        )
    def forward(self, robot_seq):  # (B,T,3)
        B,T,_ = robot_seq.shape
        robot_emb = self.fc(robot_seq.reshape(B*T,-1))
        robot_emb = robot_emb.reshape(B,T,-1) # (B,T,32)
        return robot_emb
