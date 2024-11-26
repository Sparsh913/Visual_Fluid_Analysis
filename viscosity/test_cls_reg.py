import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from classifier import SequenceViscosityClassifier
from classifier import SequenceViscosityDataset

# Test Function
def test_model(checkpoint_path, base_dir, labels_dict_cls, labels_dict_vis, velocities, transform, batch_size=2, device='cpu'):
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
    test_dataset = SequenceViscosityDataset(base_dir, labels_dict_cls, labels_dict_vis, velocities, transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load the model
    model = SequenceViscosityClassifier(num_classes=3).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # Set the model to evaluation mode

    # Testing loop
    total = 0
    correct_cls = 0
    predictions_cls = []
    predictions_reg = []
    ground_truths_cls = []
    ground_truths_reg = []

    with torch.no_grad():
        for masks, velocities, labels, vis in test_loader:
            masks = masks.to(device)
            velocities = velocities.to(device)
            labels = labels.to(device)
            vis = vis.to(device)

            # Forward pass
            output_cls, output_reg = model(masks, velocities)
            # print("Outputs:", output_cls)
            _, predicted_cls = torch.max(output_cls, 1)

            # Record predictions and ground truths
            predictions_cls.extend(predicted_cls.cpu().numpy())
            ground_truths_cls.extend(labels.cpu().numpy())
            
            # Record predictions and ground truths reg
            predictions_reg.extend(output_reg.cpu().numpy())
            ground_truths_reg.extend(vis.cpu().numpy())

            # Accuracy calculation
            total += labels.size(0)
            correct_cls += (predicted_cls == labels).sum().item()

    accuracy = 100 * correct_cls / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    return predictions_cls, ground_truths_cls, predictions_reg, ground_truths_reg

# Main Script
if __name__ == "__main__":
    # Dataset Configuration
    base_dir = "test_liq2"  # Path to the test dataset
    # labels_dict = {6: 0, 7: 0, 8: 1, 9: 2, 10: 2}  # Viscosity labels
    # velocities = {6: 30, 7: 30, 8: 20, 9: 10, 10: 10}  # Angular velocities
    labels_dict = {1:2, 2:1, 3:0, 4:0, 6:2, 7:2, 8:2, 9:2}  # Viscosity labels
    vis_dict = {1:0.023, 2:1.1, 3:30, 4:15, 6:0.3395, 7:0.0965, 8:0.048, 9: 0.001}  # Viscosity values
    velocities = {1: 15, 2: 15, 3:15, 4:15, 6: 15, 7: 15, 8: 15, 9:15}  # Angular velocities
    # labels_dict = {8:0, 9:0, 10:1}  # Viscosity labels
    # velocities = {8:15, 9:15, 10:15}  # Angular velocities

    # Data Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # No 3-channel normalization
    ])

    # Test Configuration
    checkpoint_path = "viscosity/new_data_11_24/sequence_model_epoch_cls_reg_760.pt"  # Replace with the actual path to your checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test the model
    predictions_cls, ground_truths_cls, predictions_reg, ground_truths_reg  = test_model(
        checkpoint_path, base_dir, labels_dict, vis_dict, velocities, transform, batch_size=2, device=device
    )

    # Print results
    print("Predictions cls:", predictions_cls)
    print("Ground Truths cls:", ground_truths_cls)
    print("Predictions reg:", predictions_reg)
    print("Ground Truths reg:", ground_truths_reg)
