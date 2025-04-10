from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mask_encoder import MaskEncoder
from .mask_resnet import MaskResnet
from .robot_encoder import RobotEncoder


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim: int, temperature: float = 10000.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, T, embed_dim)
        Returns:
            Tensor of shape (B, T, embed_dim)
        """
        B, T, _ = x.shape
        position = torch.arange(T, dtype=torch.float32, device=x.device).unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2, dtype=torch.float32, device=x.device) * -(np.log(self.temperature) / self.embed_dim))  # (embed_dim // 2)
        pos_embedding = torch.zeros((B, T, self.embed_dim), device=x.device)  # (B, T, embed_dim)
        pos_embedding[:, :, 0::2] = torch.sin(position * div_term)  # (B, T, embed_dim // 2)
        pos_embedding[:, :, 1::2] = torch.cos(position * div_term)  # (B, T, embed_dim // 2)
        return pos_embedding
    

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
        transformer_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, dropout=0.2)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers)#, norm=self.layer_norm)
        self.pe = SinusoidalPositionalEmbedding(embed_dim)

    # def forward(self, mask_seq, robot_seq, timestamps):
    #     B, T, C, H, W = mask_seq.shape
    #     mask_feats = self.mask_encoder(mask_seq) # (B,T,E1)
    #     robot_feats = self.robot_encoder(robot_seq) # (B,T,E2)
    #     combined_feats = torch.cat([mask_feats, robot_feats],dim=-1) # (B,T,E1+E2)
    #     # timestamps = timestamps - timestamps[:,[0]]  # normalized timestamps start at zero
    #     timestamp_feats = self.timestamp_encoder(timestamps.unsqueeze(-1)) # (B,T,embed_dim) This is the positional embedding
    #     # transformer_in = combined_feats + timestamp_feats  # (B,T,E1+E2+32) # pe as timestamp embedding
    #     transformer_in = combined_feats + self.pe(combined_feats) # (B,T,E1+E2)
    #     # transformer_in = self.layer_norm(combined_feats)
    #     transformer_out = self.transformer(transformer_in.permute(1,0,2)) #(T,B,E)
    #     transformer_pooled = transformer_out.mean(dim=0) #(B,E)
    #     return transformer_pooled #combined_feats.permute(1,0,2).mean(dim=0) #transformer_pooled
    
    def forward(self, mask_seq, robot_seq, timestamps, return_attn=False, apply_attn_reg=True):
        B, T, C, H, W = mask_seq.shape
        mask_feats = self.mask_encoder(mask_seq)  # (B,T,E1)
        robot_feats = self.robot_encoder(robot_seq)  # (B,T,E2)
        combined_feats = torch.cat([mask_feats, robot_feats], dim=-1)  # (B,T,E1+E2)
        
        # Add positional encoding
        transformer_in = combined_feats + self.pe(combined_feats)  # (B,T,E1+E2)
        
        # Transformer expects (T,B,E) format
        transformer_in = transformer_in.permute(1, 0, 2)  # (T,B,E)
        
        # Get transformer output and attention weights
        attn_penalty = 0.0
        
        if return_attn or apply_attn_reg:
            # Directly access the transformer layers to get attention weights
            transformer_out = transformer_in
            attn_weights = []
            
            # Iterate through transformer layers
            for layer in self.transformer.layers:
                # Apply self attention
                src2, attn_weights_layer = layer.self_attn(
                    transformer_out, transformer_out, transformer_out,
                    need_weights=True, average_attn_weights=False
                )
                
                if apply_attn_reg:
                    # Calculate attention concentration penalty using entropy
                    # First apply softmax to get proper attention probabilities
                    attn_probs = F.softmax(attn_weights_layer, dim=-1)
                    
                    # Calculate entropy of attention distribution (across target sequence)
                    # Higher entropy means more distributed attention -> We want to maximize the entropy -> add negative entropy to penalty
                    entropy = -torch.sum(attn_probs * torch.log(attn_probs + 1e-10), dim=-1)  # (B, num_heads, T) # sum(p*log(p))
                    
                    # We want to maximize entropy, so we minimize negative entropy
                    # Average across batch, heads, and source positions
                    attn_penalty += -torch.mean(entropy) # mean over all dims
                
                if return_attn:
                    attn_weights.append(attn_weights_layer.detach())  # (B, num_heads, T, T)
                
                # Continue with the rest of the layer
                transformer_out = transformer_out + layer.dropout1(src2)
                transformer_out = layer.norm1(transformer_out)
                src2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(transformer_out))))
                transformer_out = transformer_out + layer.dropout2(src2)
                transformer_out = layer.norm2(transformer_out)
            
            # Mean pooling
            transformer_pooled = transformer_out.mean(dim=0)  # (B,E)
            
            if return_attn:
                return transformer_pooled, attn_weights, attn_penalty
            else:
                return transformer_pooled, attn_penalty
        else:
            # Use the regular forward pass
            transformer_out = self.transformer(transformer_in)  # (T,B,E)
            transformer_pooled = transformer_out.mean(dim=0)  # (B,E)
            return transformer_pooled