import torch
import os
import numpy as np
import cv2
from sam2.build_sam import build_sam2_video_predictor
import tqdm

# Enable MPS fallback for unsupported ops if using Apple MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Initialize Model
sam2_checkpoint = "sam2_logs/configs/sam2.1_training/sam2.1_hiera_t_liq_finetune_copy.yaml/checkpoints_2/checkpoint.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
device = "cuda" if torch.cuda.is_available() else "cpu"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

# Root directory containing all vial data
root_dir = "test_liq4"

# Set flag to save masks as .png and/or .npy
SAVE_PNG = True  # Set to False if PNG is not needed
SAVE_NPY = False  # Set to False if NumPy array is not needed


def process_images(image_dir):
    """
    Processes images in the given directory, predicts segmentation masks, and saves them.
    """
    print(f"Processing images in: {image_dir}")

    # Output directory for masks (same level as `images/`)
    parent_dir = os.path.dirname(image_dir)  # Get parent directory of images/
    output_dir = os.path.join(parent_dir, "masks")  # Create masks/ at the same level as images/
    os.makedirs(output_dir, exist_ok=True)

    # Get sorted list of image files
    frame_names = sorted(
        [p for p in os.listdir(image_dir) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]],
        key=lambda p: int(os.path.splitext(p)[0])  # Sort numerically if filenames are numbered
    )

    if not frame_names:
        print(f"No images found in {image_dir}, skipping...")
        return

    # Load first frame to get image resolution
    sample_img = cv2.imread(os.path.join(image_dir, frame_names[0]), cv2.IMREAD_GRAYSCALE)
    orig_height, orig_width = sample_img.shape

    # Initialize segmentation inference state
    inference_state = predictor.init_state(video_path=image_dir)

    # Define bounding box for segmentation (adjust this as needed)
    ann_frame_idx = 0  # First frame for annotation
    ann_obj_id = 1  # Unique ID for the object
    box = np.array([134, 141, 267, 309], dtype=np.float32)  # Sample bounding box

    # Add bounding box to the predictor
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        box=box,
    )

    # Propagate segmentation across frames
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy().astype(np.uint8) * 255
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # Save masks as binary PNGs and/or NumPy arrays
    for frame_idx, obj_masks in video_segments.items():
        for obj_id, mask in obj_masks.items():
            mask = mask.squeeze()  # Remove extra channel dimension
            mask_resized = cv2.resize(mask, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)

            # Save as PNG
            if SAVE_PNG:
                cv2.imwrite(os.path.join(output_dir, f"{frame_idx}.png"), mask)

            # Save as NumPy array
            if SAVE_NPY:
                np.save(os.path.join(output_dir, f"{frame_idx}.npy"), mask)


def traverse_and_process(root_directory):
    """
    Recursively searches for 'images/' directories and processes them.
    """
    # include tqdm
    for dirpath, dirnames, filenames in tqdm.tqdm(os.walk(root_directory)):
        if "images" in dirnames:
            image_dir = os.path.join(dirpath, "images")
            torch.cuda.empty_cache()
            process_images(image_dir)


# Start processing
traverse_and_process(root_dir)
print("Processing complete!")
