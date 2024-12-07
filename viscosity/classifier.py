import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torch.nn.functional as F
from torch.utils.data import Sampler
import random
import argparse
import numpy as np
import wandb
import tqdm
os.environ["WANDB_START_METHOD"] = "thread"

# Dataset Definition
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
#             self.val_data[folder] = mask_paths[-seq_len:]
#             self.train_data[folder] = mask_paths[:-seq_len]

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

class SequenceViscosityDataset(Dataset):
    def __init__(self, base_dir, labels_dict_cls, labels_dict_vis, velocities, transform=None, seq_len=43, mode="train", datapoints=5):
        """
        Args:
            base_dir (str): Path to the base directory containing masks_* folders.
            labels_dict_cls (dict): Mapping of folder numbers to viscosity labels.
            labels_dict_vis (dict): Mapping of folder numbers to viscosity values.
            velocities (dict): Mapping of folder numbers to angular velocities.
            transform (callable, optional): Transform to apply to the images.
            seq_len (int): Fixed length of the sampled sequence.
            mode (str): "train", "val", or "test", determines which split to use.
        """
        self.base_dir = base_dir
        self.labels_dict_cls = labels_dict_cls
        self.labels_dict_vis = labels_dict_vis
        self.velocities = velocities
        self.transform = transform
        self.data_points = datapoints
        self.seq_len = seq_len + datapoints-1  # Include extra masks for test/val data points
        self.chunk_len = seq_len
        self.mode = mode
        self.vial_tracker = []

        # Collect all sequences (folders)
        self.sequences = [
            (f"masks_{folder}", labels_dict_cls[folder], velocities[folder], labels_dict_vis[folder])
            for folder in labels_dict_cls.keys()
        ]

        # Preload all mask paths for each folder
        self.folder_data = {
            folder: sorted(
                [os.path.join(base_dir, folder, img)
                 for img in os.listdir(os.path.join(base_dir, f"{folder}"))
                 if img.endswith((".png", ".jpg", ".jpeg"))]
            )
            for folder, _, _, _ in self.sequences
        }

        # Split data into train, val, and test for each folder
        self.train_data = {}
        self.val_data = {}
        self.test_data = {}
        for folder, _, _, _ in self.sequences:
            mask_paths = self.folder_data[folder]
            num_masks = len(mask_paths)
            # print("length of mask_paths", len(mask_paths))

            # Ensure there are enough masks for train, val, and test
            if num_masks < 2 * self.seq_len:
                raise ValueError(f"Folder {folder} has fewer than {2 * self.seq_len} masks.")

            # Split into first and second halves
            # First half - train, val, test
            # Second half - train, val, test
            self.midpoint = num_masks // 2
            self.first_half = mask_paths[:self.midpoint]
            self.second_half = mask_paths[self.midpoint:]

            # Assign train, val, and test splits
            self.train_data[folder] = self.first_half[:- 2 * self.seq_len] + self.second_half[:- 2 * self.seq_len]
            # print("len of train_data", len(self.train_data[folder]))
            self.val_data[folder] = self.first_half[- 2 * self.seq_len:- self.seq_len] + self.second_half[- 2 * self.seq_len:- self.seq_len]
            # print("length of val_data", len(self.val_data[folder]))
            self.test_data[folder] = self.first_half[- self.seq_len:] + self.second_half[- self.seq_len:]
            # print("length of test_data", len(self.test_data[folder]))
            

    def __len__(self):
        # Total sequences depends on the mode
        if self.mode == "train":
            return sum(len(paths) // self.chunk_len for paths in self.train_data.values())
        elif self.mode == "val":
            # return sum(len(paths) // self.chunk_len for paths in self.val_data.values())
            # return len(self.val_data)
            # print("self.data_points * len(self.val_data)", self.data_points * len(self.val_data))
            return self.data_points * 2 * len(self.labels_dict_cls)
        elif self.mode == "test":
            # return sum(len(paths) // self.chunk_len for paths in self.test_data.values())
            # return len(self.test_data)
            return self.data_points * 2 * len(self.labels_dict_cls)

    def __getitem__(self, idx):
        # Determine which data split to use
        if self.mode == "train":
            data_split = self.train_data
        elif self.mode == "val":
            data_split = self.val_data
        elif self.mode == "test":
            data_split = self.test_data
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        # Calculate folder index based on total length
        folder_idx = idx % len(self.sequences)
        # print("folder_idx", folder_idx)
        # print("length of sequences", len(self.sequences))
        # print("length of data_split", len(data_split))
        folder, label, velocity, viscosity = self.sequences[folder_idx]
        # extract the folder number from the folder name
        self.vial_tracker.append(folder.split("_")[1])
        # print("folder", folder)

        mask_paths = data_split[folder]
        num_masks = len(mask_paths)

        # Ensure there are enough masks for the sequence
        if num_masks < self.seq_len:
            raise ValueError(f"Folder {folder} has fewer than {self.seq_len} masks for {self.mode}.")

        # For train, randomly sample a sequence; for val/test, take the entire reserved chunk
        if self.mode == "train":
            # start index could be picked from the first half of train data or the second half
            start_idx_first_half = np.random.randint(0, len(self.train_data[folder])//2 - self.seq_len + 1)
            start_idx_second_half = np.random.randint(len(self.train_data[folder])//2, len(self.train_data[folder]) - self.seq_len + 1)
            start_idx = start_idx_first_half if np.random.rand() < 0.5 else start_idx_second_half
            sampled_paths = mask_paths[start_idx:start_idx + self.chunk_len]
        # elif self.mode == "val":
        #     # In validation, the start idx should be selected in order as there are a total of "datapoints" number of sequences to take
        #     # so start_idx can be selected from first "datapoints" indices
        #     start_idx_1 = idx % self.data_points
        #     start_idx_2 = idx % self.data_points + self.seq_len
        #     start_idx = start_idx_1 if idx % 2 == 0 else start_idx_2
        #     sampled_paths = mask_paths[start_idx:start_idx + self.chunk_len]
        else:
            # select start_idx from the first half and then the second half and then repeat for the next test
            start_idx_1 = idx % self.data_points
            # print("start_idx_1", start_idx_1)
            start_idx_2 = idx % self.data_points + self.seq_len
            # print("start_idx_2", start_idx_2)
            start_idx = start_idx_1 if idx % 2 == 0 else start_idx_2
            # print("length of mask_paths", len(mask_paths))
            sampled_paths = mask_paths[start_idx:start_idx + self.chunk_len]
            # print("length of sampled_paths", len(sampled_paths))

        # Load and transform masks
        masks = []
        for img_path in sampled_paths:
            image = Image.open(img_path).convert("L")  # Convert to single channel
            if self.transform:
                image = self.transform(image)
            masks.append(image)

        # Stack masks into a tensor of shape (seq_len, 1, H, W)
        masks = torch.stack(masks, dim=0)

        return masks, torch.tensor(velocity, dtype=torch.float32), label, torch.tensor(viscosity, dtype=torch.float32)


class CyclicSampler(Sampler):
    def __init__(self, dataset, batch_size):
        """
        Args:
            dataset (Dataset): Instance of the SequenceViscosityDataset.
            batch_size (int): Batch size for the DataLoader.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_folders = len(dataset.sequences)

    def __iter__(self):
        # Generate indices for cyclic sampling
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)  # Shuffle to ensure randomness
        repeated_indices = indices * (self.batch_size // self.num_folders + 1)  # Repeat to cover larger batches
        return iter(repeated_indices[:len(self.dataset)])  # Return as iterator

    def __len__(self):
        return len(self.dataset)

# Model Definition
class SequenceViscosityClassifier(nn.Module):
    def __init__(self, num_classes=3, cnn_feature_dim=512, velocity_dim=32):
        super(SequenceViscosityClassifier, self).__init__()

        # Use a pre-trained CNN (e.g., ResNet) for feature extraction
        self.cnn = models.resnet18(pretrained=True)

        # Modify the first convolutional layer to accept single-channel input
        self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.cnn.fc = nn.Identity()  # Remove the final classification layer
        self.cnn_feature_dim = cnn_feature_dim

        # Temporal feature aggregation
        self.feature_pool = nn.AdaptiveAvgPool1d(1)  # Aggregate over sequence dimension

        # Linear layer to embed the angular velocity
        self.velocity_embedding = nn.Linear(1, velocity_dim)

        # Fully connected layers for classification
        self.fc1 = nn.Linear(cnn_feature_dim + velocity_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dp = nn.Dropout(0.4)

    def forward(self, masks, angular_velocity):
        # Flatten masks into batch of individual images
        batch_size, seq_len, C, H, W = masks.shape
        masks = masks.view(batch_size * seq_len, C, H, W)

        # Extract features for each mask
        features = self.cnn(masks)  # Shape: (batch_size * seq_len, cnn_feature_dim)
        features = features.view(batch_size, seq_len, -1)  # Shape: (batch_size, seq_len, cnn_feature_dim)

        # Aggregate features over the sequence
        features = features.permute(0, 2, 1)  # Shape: (batch_size, cnn_feature_dim, seq_len)
        aggregated_features = self.feature_pool(features).squeeze(-1)  # Shape: (batch_size, cnn_feature_dim)

        # Embed angular velocity
        angular_velocity = angular_velocity.unsqueeze(1)  # Shape: (batch_size, 1)
        velocity_embedded = F.relu(self.velocity_embedding(angular_velocity))  # Shape: (batch_size, velocity_dim)

        # Combine features and velocity
        combined_features = torch.cat((aggregated_features, velocity_embedded), dim=1)

        # Classification
        x = self.dp(F.relu(self.fc1(combined_features)))
        output_cls = self.fc2(x)  # Output logits for each class

        return output_cls


# Training Loop
def train_sequence_model(
    model, train_dataloader, valid_dataloader, criterion_cls, criterion_reg, optimizer, scheduler, num_epochs=10, device="cpu", start_epoch=0, k=50
):
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0

        for masks, velocities, labels, _ in train_dataloader:
            masks = masks.to(device)
            velocities = velocities.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            output_cls = model(masks, velocities)
            loss = 100 * criterion_cls(output_cls, labels)
            wandb.log({"total training loss": loss.item()})

            # Backward pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimization step
            optimizer.step()

            # Track loss
            running_loss += loss.item()

        scheduler.step()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Total Loss: {running_loss / len(train_dataloader):.4f}")

        # Save the model checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "loss": running_loss / len(train_dataloader),
                },
                f"viscosity/new_data_11_24/k_ablation/sequence_model_epoch_cls_k_{k}_{epoch+1}.pt",
            )

            # Test the model on the validation set
            model.eval()
            total = 0
            correct_cls = 0
            predictions_cls = []
            ground_truths_cls = []
            
            # Initialize dictionaries to track per-label accuracy
            label_keys = list(valid_dataloader.dataset.labels_dict_cls.keys())
            vial_predictions = {key: [] for key in label_keys}
            vial_ground_truths = {key: [] for key in label_keys}
            vial_total = {key: 0 for key in label_keys}
            vial_correct = {key: 0 for key in label_keys}
            # print("vial_predictions", vial_predictions)

            with torch.no_grad():
                for idx, (masks, velocities, labels, *_) in enumerate(valid_dataloader):
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
                        folder_idx = int(valid_dataloader.dataset.vial_tracker[idx * valid_dataloader.batch_size + i])
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
                
                # Calculate and log overall accuracy
                overall_accuracy = 100 * correct_cls / total
                print(f"======================{k}======================")
                print(f"Validation Accuracy: {overall_accuracy:.2f}%")
                wandb.log({"validation_accuracy": overall_accuracy})
                
                # Calculate and log per-label key accuracy
                for key in label_keys:
                    if vial_total[key] > 0:
                        vial_accuracy = 100 * vial_correct[key] / vial_total[key]
                        print(f"Validation Accuracy for masks_{key}: {vial_accuracy:.2f}%")
                        wandb.log({
                            f"validation_accuracy_masks_{key}": vial_accuracy,
                            f"total_samples_masks_{key}": vial_total[key]
                        })
                        
                        # Optionally, you can also log the confusion details
                        print(f'Predictions for masks_{key}: {vial_predictions[key]}')
                        print(f'Ground Truths for masks_{key}: {vial_ground_truths[key]}')
            model.train()
                    

                    
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
            

# Main Script
if __name__ == "__main__":
    set_seed(42)
    
    parser = argparse.ArgumentParser(description='Train a sequence-based viscosity classifier')
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to a checkpoint to load and resume training")
    
    args = parser.parse_args()
    wandb.login()
    
    # Dataset Configuration
    base_dir = "test_liq2"
    '''
    viscosity labels:
    0: High
    1: Medium
    2: Low
    '''
    labels_dict = {3:0, 4:0, 2:1, 1:2, 6:2}  # Viscosity labels
    vis_dict = {3:30, 4:15, 2:1.1, 1:0.023, 6:0.3395}
    velocities = {3:15, 4:15, 2:15, 1:15, 6:15}  # Angular velocities
    
    labels_val = {7:2, 8:2, 9:2}
    vis_val = {7:0.0965, 8:0.048, 9:0.001}
    velocities_val = {7:15, 8:15, 9:15}

    # Data Transforms
    # Translation, rotation, and scaling transforms
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # No need for 3-channel normalization
    # transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    # transforms.GaussianBlur(kernel_size=3),
    # transforms.RandomGrayscale(p=0.1),
    # transforms.RandomApply([transforms.ElasticTransform(alpha=1.0)], p=0.3),
])
    
    transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # No need for 3-channel normalization
])


    # Dataset and DataLoader
    # dataset = SequenceViscosityDataset(base_dir, labels_dict, vis_dict, velocities, transform, seq_len=50, mode="train")
    # sampler = CyclicSampler(dataset, batch_size=64)
    # train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)# sampler=sampler)
    
    # dataset_val = SequenceViscosityDataset(base_dir, labels_dict, vis_dict, velocities, transform, seq_len=50, mode="val")
    # valid_dataloader = DataLoader(dataset_val, batch_size=64)

    # Model, Loss, and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SequenceViscosityClassifier(num_classes=3).to(device)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    # set different lrs for cls and reg heads
    # optimizer = optim.Adam([
    #     {"params": model.cnn.parameters(), "lr": 0.001},
    #     {"params": model.fc1.parameters()},
    #     {"params": model.fc2.parameters()},
    #     {"params": model.fc3.parameters(), "lr": 0.05},
    #     {"params": model.velocity_embedding.parameters()}
    # ], lr=0.001)
    # freeze velocity embedding
    # for param in model.velocity_embedding.parameters():
    #     param.requires_grad = False
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    if args.load_checkpoint:
        checkpoint = torch.load(args.load_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)  # Start from the saved epoch
        print(f"Loaded checkpoint from {args.load_checkpoint}")
    else:
        start_epoch = 0
        
    wandb.init(project="sequence-viscosity-classification", config={
        "base_dir": base_dir,
        "labels_dict": labels_dict,
        "velocities": velocities,
        "transform": transform,
        "num_classes": 3,
        "cnn_feature_dim": 512,
        "velocity_dim": 32,
        "batch_size": 7,
        "learning_rate": 0.001,
        "device": device.type,
        "start_epoch": start_epoch
    })  # Initialize
    wandb.watch(model, log="all")
    
    dataset = SequenceViscosityDataset(base_dir, labels_dict, vis_dict, velocities, transform, seq_len=1, mode="train")
#     sampler = CyclicSampler(dataset, batch_size=32)
    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    dataset_val = SequenceViscosityDataset(base_dir, labels_dict, vis_dict, velocities, transform, seq_len=1, mode="val")
    valid_dataloader = DataLoader(dataset_val, batch_size=32)
    
    # train_sequence_model(model, train_dataloader, valid_dataloader, criterion_cls, criterion_reg, optimizer, scheduler, num_epochs=100, device=device, start_epoch=start_epoch, k=1)
    # torch.cuda.empty_cache()

    # Train the Model
    # for k in range(5, 50, 5):
    # use tqdm for progress bar
    for k in tqdm.tqdm(range(2, 52, 2)):
        model = SequenceViscosityClassifier(num_classes=3)
        model.to(device)
        
        # Reinitialize the optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        if k >= 15:
            d=1
        else:
            d=5

        dataset = SequenceViscosityDataset(base_dir, labels_dict, vis_dict, velocities, transform, seq_len=k, mode="train", datapoints=d)
    #     sampler = CyclicSampler(dataset, batch_size=32)
        train_dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        
        dataset_val = SequenceViscosityDataset(base_dir, labels_dict, vis_dict, velocities, transform_val, seq_len=k, mode="val", datapoints=d)
        valid_dataloader = DataLoader(dataset_val, batch_size=256)
        train_sequence_model(model, train_dataloader, valid_dataloader, criterion_cls, criterion_reg, optimizer, scheduler, num_epochs=100, device=device, start_epoch=start_epoch, k=k)
        torch.cuda.empty_cache()
