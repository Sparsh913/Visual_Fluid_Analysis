import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from classifier import SequenceViscosityClassifier
from classifier import SequenceViscosityDataset
from classifier import CyclicSampler
from torch.utils.data import Dataset, DataLoader


# class SequenceViscosityDataset(Dataset):
#     def __init__(self, base_dir, labels_dict_cls, labels_dict_vis, velocities, transform=None, seq_len=50):
#         self.base_dir = base_dir
#         self.labels_dict_cls = labels_dict_cls
#         self.labels_dict_vis = labels_dict_vis
#         self.velocities = velocities
#         self.transform = transform
#         self.seq_len = seq_len

#         # Collect all sequences (folders)
#         self.sequences = [
#             (f"masks_{folder}", labels_dict_cls[folder], velocities[folder], labels_dict_vis[folder])
#             for folder in labels_dict_cls.keys()
#         ]

#         # Preload all mask paths for each folder
#         self.folder_data = {
#             folder: sorted(
#                 [os.path.join(base_dir, folder, img)  # Use folder directly
#                 for img in os.listdir(os.path.join(base_dir, folder))  # Use folder directly
#                 if img.endswith(('.png'))]
#             )
#             for folder, _, _, _ in self.sequences
#         }

#     def __len__(self):
#         # Number of sequences is equal to the number of keys in labels_dict
#         return len(self.sequences)

#     def __getitem__(self, idx):
#         folder, label, velocity, viscosity = self.sequences[idx]

#         mask_paths = self.folder_data[folder]
#         num_masks = len(mask_paths)

#         # Ensure there are enough masks for the sequence
#         if num_masks < self.seq_len:
#             raise ValueError(f"Folder {folder} has fewer than {self.seq_len} masks.")

#         # Sample a single sequence deterministically (e.g., from the start)
#         start_idx = 0  # Alternatively, use np.random.randint for randomness
#         sampled_paths = mask_paths[start_idx:start_idx + self.seq_len]

#         # Load and transform masks
#         masks = []
#         for img_path in sampled_paths:
#             image = Image.open(img_path).convert("L")
#             if self.transform:
#                 image = self.transform(image)
#             masks.append(image)

#         # Stack masks into a tensor of shape (seq_len, 1, H, W)
#         masks = torch.stack(masks, dim=0)

#         return masks, torch.tensor(velocity, dtype=torch.float32), label, torch.tensor(viscosity, dtype=torch.float32)

# Test Function
def test_model(checkpoint_path, base_dir, labels_dict, vis_dict, velocities, transform, batch_size=2, device='cpu', k=5):
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
    sampler = CyclicSampler(dataset=test_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)  # No sampler for testing

    # Create a reverse mapping to track original keys
    idx_to_label_key = list(labels_dict.keys())

    # Load the model
    model = SequenceViscosityClassifier(num_classes=3).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # Set the model to evaluation mode

    # Testing loop
    total = 0
    correct = 0
    predictions = []
    ground_truths = []
    label_keys = []

    with torch.no_grad():
        for idx, (masks, velocities, labels, _) in enumerate(test_loader):
            masks = masks.to(device)
            velocities = velocities.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(masks, velocities)
            _, predicted = torch.max(outputs, 1)

            # Record predictions, ground truths, and label keys
            predictions.extend(predicted.cpu().numpy())
            ground_truths.extend(labels.cpu().numpy())
            label_keys.extend(idx_to_label_key[idx * batch_size:(idx + 1) * batch_size])

            # Accuracy calculation
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Print results
    print(f"\nResults:{k}")
    print(f"{'Label Key':<10}{'Ground Truth':<15}{'Prediction':<15}")
    print("=" * 40)
    for i in range(len(predictions)):
        print(f"{label_keys[i]:<10}{ground_truths[i]:<15}{predictions[i]:<15}")

    return predictions, ground_truths

# Main Script
if __name__ == "__main__":
    # Dataset Configuration
    base_dir = "test_liq2"  # Path to the test dataset
    # labels_dict = {6: 0, 7: 0, 8: 1, 9: 2, 10: 2}  # Viscosity labels
    # velocities = {6: 30, 7: 30, 8: 20, 9: 10, 10: 10}  # Angular velocities
    labels_dict = {3:0, 4:0, 2:1, 1:2, 6:2, 7:2, 8:2, 9:2, 10:0}  # Viscosity labels
    vis_dict = {1:0.023, 2:1.1, 3:30, 4:15, 6:0.3395, 7:0.0965, 8:0.048, 9: 0.001, 10:30}  # Viscosity values
    velocities = {1: 15, 2: 15, 6: 15, 7: 15, 8: 15, 9:15, 3:15, 4:15, 10:15}  # Angular velocities
    # labels_dict = {8:0, 9:0, 10:1}  # Viscosity labels
    # vis_dict = {8:0.048, 9:0.001, 10:1.1}  # Viscosity values
    # velocities = {8:15, 9:15, 10:15}  # Angular velocities
    
    labels_dict = {7:2, 8:2, 9:2}#, 10:0}
    vis_dict = {7:0.0965, 8:0.048, 9: 0.001}#, 10:30}
    velocities = {7: 15, 8: 15, 9:15}#, 10:15}
    
    # labels_dict = {3:0, 4:0, 2:1, 1:2, 6:2}
    # vis_dict = {1:0.023, 2:1.1, 3:30, 4:15, 6:0.3395}
    # velocities = {1: 15, 2: 15, 6: 15, 3:15, 4:15}

    # Data Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # No 3-channel normalization
    ])

    for k in range(5, 75, 5):
        
        # Test Configuration
        checkpoint_path = f"viscosity/new_data_11_24/k_ablation/sequence_model_epoch_cls_k_{k}_100.pt"  # Replace with the actual path to your checkpoint
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Test the model
        predictions, ground_truths = test_model(
            checkpoint_path, base_dir, labels_dict, vis_dict, velocities, transform, batch_size=1, device=device, k=k
        )

    # # Print results
    # print("Predictions:", predictions)
    # print(f"label:{}
