import os
from PIL import Image
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from regressor import EnhancedSequenceViscosityRegressor
from regressor import SequenceViscosityDataset
# from regressor import CyclicSampler
from torch.utils.data import Dataset, DataLoader

# Test Function
def test_model(checkpoint_path, base_dir, labels_dict, vis_dict, velocities, transform, criterion, batch_size=2, device='cpu', k=5):
    """
    Args:
        checkpoint_path (str): Path to the saved checkpoint.
        base_dir (str): Path to the dataset directory.
        labels_dict (dict): Mapping of folder numbers to viscosity labels.
        velocities (dict): Mapping of folder numbers to angular velocities.
        transform (callable): Transformations for the dataset.
        batch_size (int): Batch size for DataLoader.
        device (str): Device to run the model on.
    """
    # Load the test dataset
    test_dataset = SequenceViscosityDataset(base_dir, labels_dict, vis_dict, velocities, transform, seq_len=k, mode='val')
    test_loader = DataLoader(test_dataset, batch_size=batch_size)  # No sampler for testing

    # Create a reverse mapping to track original keys
    idx_to_label_key = list(labels_dict.keys())

    # Load the model
    model = EnhancedSequenceViscosityRegressor().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # Set the model to evaluation mode

    # Testing loop
    total = 0
    predictions_reg = []
    ground_truths_reg = []
    label_keys = []
    loss = 0

    with torch.no_grad():
        for idx, (masks, velocities, labels, vis) in enumerate(test_loader):
            masks = masks.to(device)
            velocities = velocities.to(device)
            labels = labels.to(device)
            vis = vis.to(device)

            # Forward pass
            output_reg = model(masks, velocities)
            predicted_reg = output_reg.to(device) #* (vis_max - vis_min) + vis_min
            loss += criterion(predicted_reg, vis)

            # Record predictions, ground truths, and label keys
            predictions_reg.extend(predicted_reg.cpu().numpy())
            ground_truths_reg.extend(vis.cpu().numpy())
            label_keys.extend(idx_to_label_key[idx * batch_size:(idx + 1) * batch_size])

            # Accuracy calculation
            total += labels.size(0)
    # denormalize predictions
    pred_mean = np.mean(list(vis_dict.values()))
    pred_std = np.std(list(vis_dict.values()))
    predictions_reg = [p * std + mean for p in predictions_reg]
    ground_truths_reg = [g * std + mean for g in ground_truths_reg]

    # Print results
    print(f"\nResults:{k}")
    print(f"{'Label Key':<10}{'Ground Truth':<15}{'Prediction':<15}")
    print("=" * 40)
    for i in range(len(predictions_reg)):
        print(f"{label_keys[i]:<10}{ground_truths_reg[i].item():<15.4f}{predictions_reg[i].item():<15.4f}")
    print("loss:", loss.item() / total)

    return predictions_reg, ground_truths_reg

def normalize_viscosity(vis_dict):
    values = list(vis_dict.values())
    mean = np.mean(values)
    std = np.std(values)
    return {k: (v - mean) / std for k, v in vis_dict.items()}, mean, std

def denormalize_viscosity(vis_dict, mean, std):
    return {k: v * std + mean for k, v in vis_dict.items()}

def mixed_viscosity_loss(output, target):
    # Ensure matching shapes
    output = output.squeeze()
    target = target.squeeze()
    
    mse_loss = F.mse_loss(output, target)
    l1_loss = F.l1_loss(output, target)
    return 10*(mse_loss + l1_loss)

# Main Script
if __name__ == "__main__":
    # Dataset Configuration
    base_dir = "test_liq2"  # Path to the test dataset
    # labels_dict = {6: 0, 7: 0, 8: 1, 9: 2, 10: 2}  # Viscosity labels
    # velocities = {6: 30, 7: 30, 8: 20, 9: 10, 10: 10}  # Angular velocities
    labels_dict = {3:0, 4:0, 2:1, 1:2, 6:2, 7:2, 8:2, 9:2}#, 10:0}  # Viscosity labels
    vis_dict = {3:1.477, 4:1.176, 2:0.0414, 1:-1.6382, 6:-0.4691, 7:-1.0154, 8:-1.3187, 9: -3}#, 10:1.4771}  # Viscosity values
    velocities = {1: 15, 2: 15, 6: 15, 7: 15, 8: 15, 9:15, 3:15, 4:15}#, 10:15}  # Angular velocities
    # labels_dict = {8:0, 9:0, 10:1}  # Viscosity labels
    # vis_dict = {8:0.048, 9:0.001, 10:1.1}  # Viscosity values
    # velocities = {8:15, 9:15, 10:15}  # Angular velocities
    normalized_vis_dict, mean, std = normalize_viscosity(vis_dict)
    
    # labels_dict = {7:2, 8:2, 9:2}#, 10:0}
    # vis_dict = {7:-1.0154, 8:-1.3187, 9: -3}#, 10:30}
    # velocities = {7: 15, 8: 15, 9:15}#, 10:15}
    
    # labels_dict = {3:0, 4:0, 2:1, 1:2, 6:2}
    # vis_dict = {1:0.023, 2:1.1, 3:30, 4:15, 6:0.3395}
    # velocities = {1: 15, 2: 15, 6: 15, 3:15, 4:15}
    

    # Data Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # No 3-channel normalization
        # transforms.Normalize(mean=[0.5], std=[0.5]),
        # transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2),
    ])

    # for k in range(1, 55, 5):
        
    # Test Configuration
    checkpoint_path = f"viscosity/new_data_11_24/k_reg/sequence_model_epoch_reg_50_950.pt"  # Replace with the actual path to your checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = mixed_viscosity_loss

    # Test the model
    predictions, ground_truths = test_model(
        checkpoint_path, base_dir, labels_dict, normalized_vis_dict, velocities, transform, criterion, batch_size=64, device=device, k=50
    )

    # # Print results
    # print("Predictions:", predictions)
    # print(f"label:{}
