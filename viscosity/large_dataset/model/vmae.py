import torch
import torch.nn as nn
from transformers import VideoMAEModel, VideoMAEConfig
from .robot_encoder import RobotEncoder

class VideoMAEVisModel(nn.Module):
    """
    A VideoMAE backbone + simple MLP head that also fuses in robot_seq.
    - mask_seq: (B, T, 1, H, W) binary masks in [0,1]
    - robot_seq: (B, T, 2) - Note: expecting 2 values per timestep (angle, speed)
    - timestamps: (B, T) - Optional timestamp information
    - labels: (B,) integers for classification or floats for regression
    
    Returns:
    - (loss, logits) when labels is passed in,
    - logits otherwise.
    """
    def __init__(
        self,
        pretrained_ckpt: str = "MCG-NJU/videomae-base",
        task: str = "classification",  # "classification" or "regression"
        num_classes: int = 3,
        embed_dim: int = 768  # must match the VideoMAE hidden size
    ):
        super().__init__()
        assert task in {"classification", "regression"}, f"task {task} not supported"
        self.task = task
        
        # 1) load the MAE backbone (no head)
        # we want VideoMAEModel, not the classification wrapper
        self.backbone = VideoMAEModel.from_pretrained(pretrained_ckpt)
        # self.backbone.config.num_frames = 10
        hidden_size = self.backbone.config.hidden_size
        
        # 2) robot encoder: avg over time, then MLP → same hidden_size
        # Note: Changed input dim from 3 to 2 to match your data format (angle, speed)
        # self.robot_encoder = nn.Sequential(
        #     nn.Linear(2, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, hidden_size)
        # )
        self.robot_encoder = RobotEncoder(input_dim=2, embed_dim=hidden_size)
        
        # 3) final head
        head_out = num_classes if task == "classification" else 1
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, head_out)
        )
        
        # if regression, we'll squeeze the final dim
        if task == "regression":
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, mask_seq, robot_seq, timestamps=None, labels=None):
        # Check input shapes for debugging
        batch_size, seq_len, ch, height, width = mask_seq.shape
        
        # VideoMAE expects pixel_values with shape (batch_size, num_channels, num_frames, height, width)
        # where num_channels is typically 3 for RGB
        
        # 1) prepare pixel_values for VideoMAE: from (B,T,1,H,W) → (B,3,T,H,W)
        # Repeat the grayscale channel 3 times to create RGB-like input
        x = mask_seq#.permute(0, 2, 1, 3, 4)  # (B,1,T,H,W)
        x = x.repeat(1, 1, 3, 1, 1)  # (B,3,T,H,W) - using repeat instead of expand
        
        # 2) run through backbone
        out = self.backbone(pixel_values=x).last_hidden_state
        
        # take the CLS token (first token) as video embedding
        # VideoMAE uses a patchified input, CLS is at index 0
        video_emb = out[:, 0, :]  # (B, hidden_size)
        
        # 3) robot embedding: mean over time → (B,2) → robot_emb (B,hidden_size)
        # robot_mean = robot_seq.mean(dim=1)  # (B,2)
        robot_emb = self.robot_encoder(robot_seq)  # (B,T,hidden_size)
        robot_emb = robot_emb.mean(dim=1)  # (B,hidden_size)
        
        # 4) fuse and head
        fused = torch.cat([video_emb, robot_emb], dim=1)  # (B, 2*hidden)
        logits = self.head(fused)
        
        if labels is not None:
            # compute loss
            if self.task == "classification":
                loss = self.loss_fn(logits, labels)*100
            else:
                # regression: logits shape (B,1) → (B,)
                preds = logits.squeeze(-1)
                loss = self.loss_fn(preds, labels)
            return loss, logits
        else:
            # no loss
            if self.task == "regression":
                return logits.squeeze(-1)
            return logits