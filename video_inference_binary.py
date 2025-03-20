import torch
import os
import numpy as np
import cv2
from sam2.build_sam import build_sam2_video_predictor

# Enable MPS fallback for unsupported ops if using Apple MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Paths
sam2_checkpoint = "sam2_logs/configs/sam2.1_training/sam2.1_hiera_t_liq_finetune_copy.yaml/checkpoints_2/checkpoint.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
video_dir = 'training/dataset/valid_liq_flipped/JPEGImages/stream_1_flipped/'
output_dir = 'training/dataset/valid_liq_flipped/masks/'

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Initialize Model
device = "cuda" if torch.cuda.is_available() else "cpu"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

# Get sorted frame names
frame_names = sorted(
    [p for p in os.listdir(video_dir) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]],
    key=lambda p: int(os.path.splitext(p)[0])
)

# Load first frame to get resolution
sample_img = cv2.imread(os.path.join(video_dir, frame_names[0]), cv2.IMREAD_GRAYSCALE)
orig_height, orig_width = sample_img.shape

# Initialize segmentation inference
inference_state = predictor.init_state(video_path=video_dir)

# Define bounding box for segmentation
ann_frame_idx = 0  # Frame index where interaction occurs
ann_obj_id = 1  # Unique object ID
box = np.array([134, 141, 267, 309], dtype=np.float32)  # Bounding box coordinates

# Add bounding box for segmentation
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    box=box,
)

# Propagate segmentation throughout the video
video_segments = {}  # Stores per-frame segmentation masks
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy().astype(np.uint8) * 255
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# Save binary masks resized to original resolution
for frame_idx, obj_masks in video_segments.items():
    for obj_id, mask in obj_masks.items():
        # Resize mask to match original image resolution
        # print("mask shape: ", mask.shape) # (1, height, width)
        mask = mask.squeeze()
        # mask_resized = cv2.resize(mask, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)
        
        # Save as binary PNG
        cv2.imwrite(os.path.join(output_dir, f"{frame_idx}.png"), mask)
