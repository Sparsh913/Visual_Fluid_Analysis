import numpy as np
import os
from PIL import Image
import cv2

def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute the Intersection over Union (IoU) for two binary masks.
    
    Args:
        mask1 (np.ndarray): First binary mask.
        mask2 (np.ndarray): Second binary mask.
    
    Returns:
        float: IoU score between 0 and 1.
    """
    # Convert masks to boolean arrays if they aren't already.
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    
    # Calculate intersection and union.
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    # If union is 0 (i.e. both masks are empty), we can define IoU as 1.
    if union == 0:
        return 1.0
    return intersection / union

def average_iou_score(ground_truth_masks: list[np.ndarray], predicted_masks: list[np.ndarray]) -> float:
    """
    Compute the average IoU score over multiple pairs of masks.
    
    Args:
        ground_truth_masks (list[np.ndarray]): List of ground truth binary masks.
        predicted_masks (list[np.ndarray]): List of predicted binary masks.
    
    Returns:
        float: The average IoU score.
    """
    assert len(ground_truth_masks) == len(predicted_masks), "Mismatch in number of masks."
    
    iou_scores = []
    for gt, pred in zip(ground_truth_masks, predicted_masks):
        iou = compute_iou(gt, pred)
        iou_scores.append(iou)
    return np.mean(iou_scores)

if __name__ == "__main__":
    ground_truth_masks_path = 'training/dataset/valid_liq/Annotations/stream_1/'
    predicted_masks_path = 'training/dataset/valid_liq_flipped/masks/'
    
    # Load ground truth and predicted masks
    ground_truth_masks = []
    predicted_masks = []
    for frame_name in os.listdir(ground_truth_masks_path):
        if frame_name.endswith('.png'):
            gt_mask = np.array(Image.open(os.path.join(ground_truth_masks_path, frame_name)))
            # flip the mask horizontally and keep the same binary. Don't add channels
            gt_mask = np.flip(gt_mask, axis=1)  # flip the mask horizontally
            ground_truth_masks.append(gt_mask)
            
    for frame_name in os.listdir(predicted_masks_path):
        if frame_name.endswith('.png'):
            pred_mask = np.array(Image.open(os.path.join(predicted_masks_path, frame_name)))
            # print(f"pred_mask shape: {pred_mask.shape}")
            predicted_masks.append(pred_mask)
            
    avg_iou = average_iou_score(ground_truth_masks, predicted_masks)
    print(f"Average IoU Score: {avg_iou:.4f}")