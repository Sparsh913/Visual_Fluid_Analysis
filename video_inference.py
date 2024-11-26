import torch
import os
# Enable MPS fallback for unsupported ops if using Apple MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.build_sam import build_sam2_video_predictor

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

sam2_checkpoint = "sam2_logs/configs/sam2.1_training/sam2.1_hiera_t_liq_finetune_copy.yaml/checkpoints/checkpoint.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
device = "cuda" if torch.cuda.is_available() else "cpu"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

video_dir = 'test_liq2/vialw'
output_dir = 'test_liq2/masks_w'
os.makedirs(output_dir, exist_ok=True)

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

inference_state = predictor.init_state(video_path=video_dir)

ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

box = np.array([134, 141, 267, 309], dtype=np.float32) # put the correct bounding box here
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    box=box,
)

# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }
    
# save the segmentation results
for frame_idx, obj_masks in video_segments.items():
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.axis("off")
    # ax.set_title(f"Frame {frame_idx}")
    for obj_id, mask in obj_masks.items():
        show_mask(mask, ax, obj_id=obj_id, random_color=True)
    plt.savefig(os.path.join(output_dir, f"{frame_idx}.png"), bbox_inches="tight")
    plt.close()    
