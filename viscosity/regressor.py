import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torch.nn.functional as F
import argparse
import numpy as np
import wandb
os.environ["WANDB_START_METHOD"] = "thread"

# Dataset Definition
# class SequenceViscosityDataset(Dataset):
#     def __init__(self, base_dir, labels_dict_cls, labels_dict_vis, velocities, transform=None):
#         """
#         Args:
#             base_dir (str): Path to the base directory containing masks_* folders.
#             labels_dict_cls (dict): Mapping of folder numbers to viscosity labels.
#             labels_dict_vis (dict): Mapping of folder numbers to viscosity values.
#             velocities (dict): Mapping of folder numbers to angular velocities.
#             transform (callable, optional): Transform to apply to the images.
#         """
#         self.base_dir = base_dir
#         self.labels_dict_cls = labels_dict_cls
#         self.labels_dict_vis = labels_dict_vis
#         self.velocities = velocities
#         self.transform = transform

#         # Compute the minimum sequence length across all folders
#         self.max_seq_len = self._compute_min_seq_length()

#         # Collect all sequences (folders)
#         self.sequences = [
#             (f"masks_{folder}", labels_dict_cls[folder], velocities[folder], labels_dict_vis[folder])
#             for folder in labels_dict_cls.keys()
#         ]

#     def _compute_min_seq_length(self):
#         min_len = float("inf")
#         for folder in self.labels_dict_cls.keys():
#             folder_path = os.path.join(self.base_dir, f"masks_{folder}")
#             num_images = len(
#                 [img for img in os.listdir(folder_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
#             )
#             min_len = min(min_len, num_images)
#         return min_len

#     def __len__(self):
#         return len(self.sequences)

#     def __getitem__(self, idx):
#         folder, label, velocity, viscosity = self.sequences[idx]

#         folder_path = os.path.join(self.base_dir, folder)
#         mask_paths = sorted(
#             [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
#         )

#         # Load and transform masks (truncate to max_seq_len)
#         masks = []
#         for img_path in mask_paths[:self.max_seq_len]:
#             image = Image.open(img_path).convert("L")  # Convert to single channel
#             if self.transform:
#                 image = self.transform(image)
#             masks.append(image)

#         # Stack masks into a tensor of shape (max_seq_len, 1, H, W)
#         masks = torch.stack(masks, dim=0)
#         # vis_max = max(self.labels_dict_vis.values())
#         # vis_min = min(self.labels_dict_vis.values())
#         # viscosity = (viscosity - vis_min) / (vis_max - vis_min)

#         return masks, torch.tensor(velocity, dtype=torch.float32), label, torch.tensor(viscosity, dtype=torch.float32)

class SequenceViscosityDataset(Dataset):
    def __init__(self, base_dir, labels_dict_cls, labels_dict_vis, velocities, transform=None, seq_len=5, mode="train"):
        """
        Args:
            base_dir (str): Path to the base directory containing masks_* folders.
            labels_dict_cls (dict): Mapping of folder numbers to viscosity labels.
            labels_dict_vis (dict): Mapping of folder numbers to viscosity values.
            velocities (dict): Mapping of folder numbers to angular velocities.
            transform (callable, optional): Transform to apply to the images.
            seq_len (int): Fixed length of the sampled sequence.
            mode (str): "train" or "val", determines which split to use.
        """
        self.base_dir = base_dir
        self.labels_dict_cls = labels_dict_cls
        self.labels_dict_vis = labels_dict_vis
        self.velocities = velocities
        self.transform = transform
        self.seq_len = seq_len
        self.mode = mode

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
                 if img.endswith(('.png', '.jpg', '.jpeg'))]
            )
            for folder, _, _, _ in self.sequences
        }

        # Compute train and val splits for each folder
        self.train_data = {}
        self.val_data = {}
        for folder, _, _, _ in self.sequences:
            mask_paths = self.folder_data[folder]
            num_masks = len(mask_paths)

            # Ensure there are enough masks for at least one training and one validation sequence
            if num_masks < 2 * seq_len:
                raise ValueError(f"Folder {folder} has fewer than {2 * seq_len} masks.")

            # Reserve the last `seq_len` masks for validation
            self.val_data[folder] = mask_paths[-seq_len:]
            self.train_data[folder] = mask_paths[:-seq_len]

    def __len__(self):
        # Total sequences depends on the mode
        if self.mode == "train":
            return sum(len(paths) // self.seq_len for paths in self.train_data.values())
        elif self.mode == "val":
            # return sum(len(paths) // self.seq_len for paths in self.val_data.values())
            return len(self.val_data)

    def __getitem__(self, idx):
        # Determine which data split to use
        data_split = self.train_data if self.mode == "train" else self.val_data

        # Calculate folder index based on total length
        folder_idx = idx % len(self.sequences)
        folder, label, velocity, viscosity = self.sequences[folder_idx]

        mask_paths = data_split[folder]
        num_masks = len(mask_paths)

        # Ensure there are enough masks for the sequence
        if num_masks < self.seq_len:
            raise ValueError(f"Folder {folder} has fewer than {self.seq_len} masks for {self.mode}.")

        # Randomly sample a sequence for training, or take the full reserved sequence for validation
        if self.mode == "train":
            start_idx = np.random.randint(0, num_masks - self.seq_len + 1)
            sampled_paths = mask_paths[start_idx:start_idx + self.seq_len]
        else:
            # For validation, use the reserved chunk as-is
            sampled_paths = mask_paths

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


# Model Definition
class SequenceViscosityRegressor(nn.Module):
    def __init__(self, num_classes=3, cnn_feature_dim=512, velocity_dim=32):
        super(SequenceViscosityRegressor, self).__init__()

        # Use a pre-trained CNN (e.g., ResNet) for feature extraction
        self.cnn = models.resnet50(pretrained=True)

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
        # add a regression head as well
        self.reg_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.Linear(64, 1)
        )
        self.dp = nn.Dropout(0.4)
        self.layernorm = nn.LayerNorm(128)
        

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
        # x = F.relu(self.fc1(combined_features))
        # output_cls = self.fc2(x)  # Output logits for each class
        
        # Regression
        x = F.relu(self.fc1(combined_features))
        x = self.dp(x)
        x = self.layernorm(x)
        output_reg = self.reg_head(x)

        return output_reg
    
    
class EnhancedSequenceViscosityRegressor(nn.Module):
    def __init__(self, num_classes=3, cnn_feature_dim=512, velocity_dim=32):
        super().__init__()
        
        # CNN Backbone
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn.fc = nn.Identity()
        
        # Advanced Sequence Processing
        self.lstm = nn.LSTM(
            input_size=cnn_feature_dim, 
            hidden_size=256, 
            num_layers=2, 
            batch_first=True, 
            dropout=0.3
        )
        
        # Velocity Embedding
        self.velocity_embedding = nn.Sequential(
            nn.Linear(1, velocity_dim),
            # nn.BatchNorm1d(velocity_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Regression Head
        self.reg_head = nn.Sequential(
            nn.Linear(256 + velocity_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
        # Weight initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.reg_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, masks, angular_velocity):
        batch_size, seq_len, C, H, W = masks.shape
        
        # Extract CNN features for each mask
        masks = masks.view(batch_size * seq_len, C, H, W)
        features = self.cnn(masks)
        features = features.view(batch_size, seq_len, -1)
        
        # Process sequence with LSTM
        lstm_out, (hidden, _) = self.lstm(features)
        
        # Use last hidden state
        sequence_features = hidden[-1]
        
        # Embed velocity
        velocity_embedding = self.velocity_embedding(angular_velocity.unsqueeze(1))
        
        # Combine features
        combined_features = torch.cat([sequence_features, velocity_embedding], dim=1)
        
        # Predict viscosity
        viscosity = self.reg_head(combined_features)
        
        return viscosity

# Training Loop
def train_sequence_model(
    model, train_dataloader, valid_dataloader, criterion_cls, criterion_reg, optimizer, scheduler, num_epochs=10, device="cpu", start_epoch=0
):
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        total_cls_loss = 0.0
        stop_cls = False  # Flag to stop optimizing classification head

        for masks, velocities, _, vis in train_dataloader:
            masks = masks.to(device)
            velocities = velocities.to(device)
            # labels = labels.to(device)
            vis = vis.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            output_reg = model(masks, velocities)
            # loss1 = criterion_cls(output_cls, labels)
            loss = criterion_reg(output_reg, vis.view(-1,1))

            wandb.log({
                "training_loss_reg": loss.item(),
                "training_loss_mse": F.mse_loss(output_reg, vis.view(-1, 1)).item(),
                "training_loss_l1": F.l1_loss(output_reg, vis.view(-1, 1)).item(),
            })

            # Backward pass
            loss.backward()

            # Gradient Clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimization step
            optimizer.step()

            # Track loss
            running_loss += loss.item()
        scheduler.step(running_loss)    

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
                f"viscosity/new_data_11_24/sequence_model_epoch_reg_{epoch+1}.pt",
            )

            # Test the model on the validation set
            model.eval()
            total = 0
            predictions_reg = []
            ground_truths_reg = []
            with torch.no_grad():
                for masks, velocities, labels, vis in valid_dataloader:
                    masks = masks.to(device)
                    velocities = velocities.to(device)
                    labels = labels.to(device)
                    vis = vis.to(device)
                    vis_max = max(vis)
                    vis_min = min(vis)

                    # Forward pass
                    output_reg = model(masks, velocities)
                    predicted_reg = output_reg.to(device) #* (vis_max - vis_min) + vis_min

                    # Log Validation Loss
                    loss = 10*criterion_reg(output_reg, vis.view(-1,1))
                    wandb.log({
                        "val_loss_reg": loss.item(),
                        "val_loss_mse": F.mse_loss(output_reg, vis.view(-1, 1)).item(),
                        "val_loss_l1": F.l1_loss(output_reg, vis.view(-1, 1)).item(),
                    })

                    # Record predictions and ground truths
                    predictions_reg.extend(predicted_reg.cpu().numpy())
                    ground_truths_reg.extend(vis.cpu().numpy())

                    # Accuracy calculation
                    total += labels.size(0)
                    # mse_reg = F.mse_loss(predicted_reg, vis, reduction="sum").item()
                    l1_reg = F.l1_loss(predicted_reg, vis, reduction="sum").item()
                    wandb.log({"validation L1 reg": l1_reg})

            print(f"Predictions reg: {predictions_reg}")
            print(f"Ground Truths reg: {ground_truths_reg}")
            model.train()

def diagnose_dataset(dataloader):
    viscosity_values = []
    velocities_values = []

    for idx, (masks, velocities, labels, vis) in enumerate(dataloader):
        viscosity_values.extend(vis.numpy())
        velocities_values.extend(velocities.numpy())
        print(f"Batch {idx + 1}: Viscosity: {vis.numpy()}, Velocities: {velocities.numpy()}")

    print("Viscosity Statistics:")
    print(f"Mean: {np.mean(viscosity_values)}")
    print(f"Std Dev: {np.std(viscosity_values)}")
    print(f"Min: {np.min(viscosity_values)}")
    print(f"Max: {np.max(viscosity_values)}")

    print("\nVelocity Statistics:")
    print(f"Mean: {np.mean(velocities_values)}")
    print(f"Std Dev: {np.std(velocities_values)}")
    print(f"Min: {np.min(velocities_values)}")
    print(f"Max: {np.max(velocities_values)}")
    
def mixed_viscosity_loss(output, target):
    # Ensure matching shapes
    output = output.squeeze()
    target = target.squeeze()
    
    mse_loss = F.mse_loss(output, target)
    l1_loss = F.l1_loss(output, target)
    return 10*(mse_loss + l1_loss)

def normalize_viscosity(vis_dict):
    values = list(vis_dict.values())
    mean = np.mean(values)
    std = np.std(values)
    return {k: (v - mean) / std for k, v in vis_dict.items()}

# Main Script
if __name__ == "__main__":
    
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
    vis_dict = {3:1.477, 4:1.176, 2:0.0414, 1:-1.6382, 6:-0.4691}
    velocities = {3:15, 4:15, 2:15, 1:15, 6:15}  # Angular velocities
    
    normalized_vis_dict = normalize_viscosity(vis_dict)
    
    labels_val = {7:2, 8:2, 9:2}
    vis_val = {7:0.0965, 8:0.048, 9:0.001}
    velocities_val = {7:15, 8:15, 9:15}

    # Data Transforms
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # No need for 3-channel normalization
    transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])


    # Dataset and DataLoader
    dataset = SequenceViscosityDataset(base_dir, labels_dict, normalized_vis_dict, velocities, transform, seq_len=20, mode="train")
    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    dataset_val = SequenceViscosityDataset(base_dir, labels_dict, normalized_vis_dict, velocities, transform, seq_len=20, mode="val")
    valid_dataloader = DataLoader(dataset_val, batch_size=64, shuffle=True)

    # Model, Loss, and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedSequenceViscosityRegressor().to(device)
    criterion_cls = nn.CrossEntropyLoss()
    # criterion_reg = nn.MSELoss()
    # criterion_reg = nn.L1Loss()
    # criterion_reg = nn.SmoothL1Loss() # Huber loss
    criterion_reg = mixed_viscosity_loss
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,  # Initial restart
        T_mult=2,  # Exponential restart
        eta_min=1e-5
    )
    # set different lrs for cls and reg heads
    # optimizer = optim.Adam([
    #     {"params": model.cnn.parameters(), "lr": 0.001},
    #     {"params": model.fc1.parameters()},
    #     {"params": model.fc2.parameters()},
    #     {"params": model.fc3.parameters(), "lr": 0.05},
    #     {"params": model.velocity_embedding.parameters()}
    # ], lr=0.001)
    
    # freeze velocity embedding layer
    # for param in model.velocity_embedding.parameters():
    #     param.requires_grad = False
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
    
    # Call this before training
    # diagnose_dataset(train_dataloader)
    # exit()
    # Train the Model
    train_sequence_model(model, train_dataloader, valid_dataloader, criterion_cls, criterion_reg, optimizer, scheduler, num_epochs=1000, device=device, start_epoch=start_epoch)
