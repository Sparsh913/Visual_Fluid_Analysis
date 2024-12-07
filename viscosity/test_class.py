import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from classifier import SequenceViscosityClassifier
from classifier import SequenceViscosityDataset
from classifier import CyclicSampler
from torch.utils.data import Dataset, DataLoader
import numpy as np


# class SequenceViscosityDataset(Dataset):
#     def __init__(self, base_dir, labels_dict_cls, labels_dict_vis, velocities, transform=None, seq_len=43, mode="train"):
#         """
#         Args:
#             base_dir (str): Path to the base directory containing masks_* folders.
#             labels_dict_cls (dict): Mapping of folder numbers to viscosity labels.
#             labels_dict_vis (dict): Mapping of folder numbers to viscosity values.
#             velocities (dict): Mapping of folder numbers to angular velocities.
#             transform (callable, optional): Transform to apply to the images.
#             seq_len (int): Fixed length of the sampled sequence.
#             mode (str): "train" or "val", determines which split to use.
#         """
#         self.base_dir = base_dir
#         self.labels_dict_cls = labels_dict_cls
#         self.labels_dict_vis = labels_dict_vis
#         self.velocities = velocities
#         self.transform = transform
#         self.seq_len = seq_len
#         self.mode = mode

#         # Collect all sequences (folders)
#         self.sequences = [
#             (f"masks_{folder}", labels_dict_cls[folder], velocities[folder], labels_dict_vis[folder])
#             for folder in labels_dict_cls.keys()
#         ]

#         # Preload all mask paths for each folder
#         self.folder_data = {
#             folder: sorted(
#                 [os.path.join(base_dir, folder, img)
#                  for img in os.listdir(os.path.join(base_dir, f"{folder}"))
#                  if img.endswith(('.png', '.jpg', '.jpeg'))]
#             )
#             for folder, _, _, _ in self.sequences
#         }

#         # Compute train and val splits for each folder
#         self.train_data = {}
#         self.val_data = {}
#         for folder, _, _, _ in self.sequences:
#             mask_paths = self.folder_data[folder]
#             num_masks = len(mask_paths)

#             # Ensure there are enough masks for at least one training and one validation sequence
#             if num_masks < 2 * seq_len:
#                 raise ValueError(f"Folder {folder} has fewer than {2 * seq_len} masks.")

#             # Reserve the last `seq_len` masks for validation
#             # self.val_data[folder] = mask_paths[-seq_len:]
#             self.train_data[folder] = mask_paths[:-seq_len]
#             val_start_idx = np.random.randint(0, num_masks - seq_len + 1)
#             # print("val_start_idx", val_start_idx)
#             self.val_data[folder] = mask_paths[val_start_idx:val_start_idx + seq_len]

#     def __len__(self):
#         # Total sequences depends on the mode
#         if self.mode == "train":
#             return sum(len(paths) // self.seq_len for paths in self.train_data.values())
#         elif self.mode == "val":
#             # return sum(len(paths) // self.seq_len for paths in self.val_data.values())
#             return len(self.val_data)

#     def __getitem__(self, idx):
#         # Determine which data split to use
#         data_split = self.train_data if self.mode == "train" else self.val_data

#         # Calculate folder index based on total length
#         folder_idx = idx % len(self.sequences)
#         folder, label, velocity, viscosity = self.sequences[folder_idx]

#         mask_paths = data_split[folder]
#         num_masks = len(mask_paths)
#         # print("num_masks", num_masks)

#         # Ensure there are enough masks for the sequence
#         if num_masks < self.seq_len:
#             raise ValueError(f"Folder {folder} has fewer than {self.seq_len} masks for {self.mode}.")

#         # Randomly sample a sequence for training, or take the full reserved sequence for validation
#         if self.mode == "train":
#             start_idx = np.random.randint(0, num_masks - self.seq_len + 1)
#             sampled_paths = mask_paths[start_idx:start_idx + self.seq_len]
#         else:
#             # For validation, use the reserved chunk as-is
#             sampled_paths = mask_paths

#         # Load and transform masks
#         masks = []
#         for img_path in sampled_paths:
#             image = Image.open(img_path).convert("L")  # Convert to single channel
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
    test_dataset = SequenceViscosityDataset(base_dir, labels_dict, vis_dict, velocities, transform, seq_len=k, mode='extrapolate')
    # sampler = CyclicSampler(dataset=test_dataset, batch_size=32)
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
    correct_cls = 0
    predictions_cls = []
    ground_truths_cls = []
    
    # Initialize dictionaries to track per-label accuracy
    label_keys = list(test_loader.dataset.labels_dict_cls.keys())
    vial_predictions = {key: [] for key in label_keys}
    vial_ground_truths = {key: [] for key in label_keys}
    vial_total = {key: 0 for key in label_keys}
    vial_correct = {key: 0 for key in label_keys}

    with torch.no_grad():
        for idx, (masks, velocities, labels, *_) in enumerate(test_loader):
            masks = masks.to(device)
            velocities = velocities.to(device)
            labels = labels.to(device)
            
            # Forward pass
            output_cls = model(masks, velocities)
            _, predicted_cls = torch.max(output_cls, 1)
            
            # Overall accuracy calculation
            total += labels.size(0)
            correct_cls += (predicted_cls == labels).sum().item()
            
            # Per-label key accuracy calculation
            for i in range(len(labels)):
                # Get the folder index (label key) for this specific sample
                folder_idx = int(test_loader.dataset.vial_tracker[idx * test_loader.batch_size + i])
                # print("folder_idx", folder_idx)
                
                # Store predictions and ground truths for each label key
                vial_predictions[folder_idx].append(predicted_cls[i].item())
                vial_ground_truths[folder_idx].append(labels[i].item())
                vial_total[folder_idx] += 1
                
                # Check correctness for this specific sample
                if predicted_cls[i] == labels[i]:
                    vial_correct[folder_idx] += 1
            
            # Store predictions for potential later analysis
            predictions_cls.extend(predicted_cls.cpu().numpy())
            ground_truths_cls.extend(labels.cpu().numpy())
        
        # print("len of predictions_cls", len(predictions_cls))
        # Calculate and log overall accuracy
        overall_accuracy = 100 * correct_cls / total
        print(f"======================{k}======================")
        print(f"Extrapolation Accuracy: {overall_accuracy:.2f}%")
        
        # Calculate and log per-label key accuracy
        for key in label_keys:
            if vial_total[key] > 0:
                vial_accuracy = 100 * vial_correct[key] / vial_total[key]
                print(f"Extrapolation Accuracy for masks_{key}: {vial_accuracy:.2f}%")
                # Optionally, you can also log the confusion details
                print(f'Predictions for masks_{key}: {vial_predictions[key]}')
                print(f'Ground Truths for masks_{key}: {vial_ground_truths[key]}')

# Main Script
if __name__ == "__main__":
    # Dataset Configuration
    base_dir = "test_liq2"  # Path to the test dataset
    # labels_dict = {6: 0, 7: 0, 8: 1, 9: 2, 10: 2}  # Viscosity labels
    # velocities = {6: 30, 7: 30, 8: 20, 9: 10, 10: 10}  # Angular velocities
    labels_dict = {3:0, 4:0, 2:1, 1:2, 6:2, 7:2, 8:2, 9:2}#, 10:0}  # Viscosity labels
    vis_dict = {1:0.023, 2:1.1, 3:30, 4:15, 6:0.3395, 7:0.0965, 8:0.048, 9: 0.001}#, 10:30}  # Viscosity values
    velocities = {1: 15, 2: 15, 6: 15, 7: 15, 8: 15, 9:15, 3:15, 4:15}#, 10:15}  # Angular velocities
    # labels_dict = {8:2, 9:2, 10:2}  # Viscosity labels
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

    for k in range(2, 22, 2):
        
        # Test Configuration
        checkpoint_path = f"viscosity/new_data_11_24/k_ablation/sequence_model_epoch_cls_k_{k}_100.pt"  # Replace with the actual path to your checkpoint
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Test the model
        test_model(
            checkpoint_path, base_dir, labels_dict, vis_dict, velocities, transform, batch_size=150, device=device, k=k
        )
        torch.cuda.empty_cache()

    # # Print results
    # print("Predictions:", predictions)
    # print(f"label:{}
