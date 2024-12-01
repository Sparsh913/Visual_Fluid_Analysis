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
class SequenceViscosityDataset(Dataset):
    def __init__(self, base_dir, labels_dict_cls, labels_dict_vis, velocities, transform=None):
        """
        Args:
            base_dir (str): Path to the base directory containing masks_* folders.
            labels_dict_cls (dict): Mapping of folder numbers to viscosity labels.
            labels_dict_vis (dict): Mapping of folder numbers to viscosity values.
            velocities (dict): Mapping of folder numbers to angular velocities.
            transform (callable, optional): Transform to apply to the images.
        """
        self.base_dir = base_dir
        self.labels_dict_cls = labels_dict_cls
        self.labels_dict_vis = labels_dict_vis
        self.velocities = velocities
        self.transform = transform

        # Compute the minimum sequence length across all folders
        self.max_seq_len = self._compute_min_seq_length()

        # Collect all sequences (folders)
        self.sequences = [
            (f"masks_{folder}", labels_dict_cls[folder], velocities[folder], labels_dict_vis[folder])
            for folder in labels_dict_cls.keys()
        ]

    def _compute_min_seq_length(self):
        min_len = float("inf")
        for folder in self.labels_dict_cls.keys():
            folder_path = os.path.join(self.base_dir, f"masks_{folder}")
            num_images = len(
                [img for img in os.listdir(folder_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
            )
            min_len = min(min_len, num_images)
        return min_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        folder, label, velocity, viscosity = self.sequences[idx]

        folder_path = os.path.join(self.base_dir, folder)
        mask_paths = sorted(
            [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
        )

        # Load and transform masks (truncate to max_seq_len)
        masks = []
        for img_path in mask_paths[:self.max_seq_len]:
            image = Image.open(img_path).convert("L")  # Convert to single channel
            if self.transform:
                image = self.transform(image)
            masks.append(image)

        # Stack masks into a tensor of shape (max_seq_len, 1, H, W)
        masks = torch.stack(masks, dim=0)
        vis_max = max(self.labels_dict_vis.values())
        vis_min = min(self.labels_dict_vis.values())
        viscosity = (viscosity - vis_min) / (vis_max - vis_min)

        return masks, torch.tensor(velocity, dtype=torch.float32), label, torch.tensor(viscosity, dtype=torch.float32)


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
            loss = criterion_reg(output_reg, vis)

            wandb.log({
                "training loss reg": loss.item(),
            })

            # Backward pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
                    loss = 10*criterion_reg(output_reg, vis)
                    wandb.log({
                        "validation loss reg": loss.item(),
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
    dataset = SequenceViscosityDataset(base_dir, labels_dict, vis_dict, velocities, transform)
    train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    dataset_val = SequenceViscosityDataset(base_dir, labels_dict, vis_dict, velocities, transform)
    valid_dataloader = DataLoader(dataset_val, batch_size=2, shuffle=True)

    # Model, Loss, and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SequenceViscosityClassifier().to(device)
    criterion_cls = nn.CrossEntropyLoss()
    # criterion_reg = nn.MSELoss()
    # criterion_reg = nn.L1Loss()
    criterion_reg = nn.SmoothL1Loss() # Huber loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    # set different lrs for cls and reg heads
    # optimizer = optim.Adam([
    #     {"params": model.cnn.parameters(), "lr": 0.001},
    #     {"params": model.fc1.parameters()},
    #     {"params": model.fc2.parameters()},
    #     {"params": model.fc3.parameters(), "lr": 0.05},
    #     {"params": model.velocity_embedding.parameters()}
    # ], lr=0.001)
    
    # freeze velocity embedding layer
    for param in model.velocity_embedding.parameters():
        param.requires_grad = False
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
