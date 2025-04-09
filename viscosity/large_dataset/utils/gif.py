import os
import numpy as np
import imageio
from PIL import Image
import tqdm

def load_images_from_folder(folder, exts=[".jpg", ".png"]):
    # Natural sorting function for filenames with numbers
    def natural_sort_key(s):
        import re
        return [int(text) if text.isdigit() else text.lower() 
                for text in re.split(r'(\d+)', s)]
    
    files = [f for f in os.listdir(folder) if os.path.splitext(f)[-1].lower() in exts]
    # Use natural sorting instead of simple alphabetical sorting
    files = sorted(files, key=natural_sort_key)
    paths = [os.path.join(folder, f) for f in files]
    return [Image.open(p).convert("RGB") for p in paths]

def process_all(root_dir, fps=10):
    # use tqdm
    for dirpath, dirnames, _ in tqdm.tqdm(os.walk(root_dir)):
        if 'images' in dirnames and 'masks' in dirnames:
            images_dir = os.path.join(dirpath, 'images')
            masks_dir = os.path.join(dirpath, 'masks')

            rgb_images = load_images_from_folder(images_dir, [".jpg"])
            mask_images = load_images_from_folder(masks_dir, [".png"])

            # Ensure matching length
            min_len = min(len(rgb_images), len(mask_images))
            rgb_images = rgb_images[:min_len]
            mask_images = mask_images[:min_len]

            combined_frames = []
            for img, mask in zip(rgb_images, mask_images):
                mask = mask.convert("RGB").resize(img.size)
                combined = Image.new("RGB", (img.width * 2, img.height))
                combined.paste(img, (0, 0))
                combined.paste(mask, (img.width, 0))
                combined_frames.append(np.array(combined))  # Convert to numpy array for imageio

            # Save GIF using imageio
            save_path = os.path.join(dirpath, "animation.gif")
            imageio.mimsave(save_path, combined_frames, fps=fps, loop=100)
            print(f"Saved GIF to: {save_path}")

if __name__ == "__main__":
    root = "data/"  # e.g., "./" or "vial_1"
    process_all(root, 30)
