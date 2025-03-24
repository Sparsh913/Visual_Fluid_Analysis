import torch
import os
# Enable MPS fallback for unsupported ops if using Apple MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import cv2

np.random.seed(3)

def show_anns(anns, output_path, borders=True):
    if len(anns) == 0:
        return
    
    # Sort and prepare canvas for annotations
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    
    # Draw each mask on the canvas
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    # Save the mask
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Configuration
image_folder = 'test_liq3/images_constant/'  # Folder containing images
output_folder = 'test_liq3/masks_constant/'  # Folder to save masks
os.makedirs(output_folder, exist_ok=True)

sam2_checkpoint = "sam2_logs/configs/sam2.1_training/sam2.1_hiera_t_liq_finetune_copy.yaml/checkpoints/checkpoint.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

# Initialize the model
device = "cuda" if torch.cuda.is_available() else "cpu"
sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
mask_generator = SAM2AutomaticMaskGenerator(sam2)

# Process each image in the folder
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load the image
        image_path = os.path.join(image_folder, filename)
        image = Image.open(image_path)
        image = np.array(image.convert("RGB"))

        # Generate masks
        masks = mask_generator.generate(image)

        # Save the annotated image with masks
        output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_mask.jpg")
        show_anns(masks, output_path)
        print(f"Processed and saved mask for {filename}")
