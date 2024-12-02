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
class SequenceViscosityDataset(Dataset):
    def __init__(self, base_dir, labels_dict_cls, labels_dict_vis, velocities, transform=None, seq_len=43, mode="train"):
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
            label_keys = []  # To store the unique label key indices
            idx_to_label_key = list(valid_dataloader.dataset.labels_dict_cls.keys())

            with torch.no_grad():
                for idx, (masks, velocities, labels, _) in enumerate(valid_dataloader):
                    masks = masks.to(device)
                    velocities = velocities.to(device)
                    labels = labels.to(device)

                    # Forward pass
                    output_cls = model(masks, velocities)
                    _, predicted_cls = torch.max(output_cls, 1)

                    # Log Validation Loss
                    loss = criterion_cls(output_cls, labels)
                    wandb.log({"validation loss cls": loss.item()})

                    # Record predictions, ground truths, and label keys
                    predictions_cls.extend(predicted_cls.cpu().numpy())
                    ground_truths_cls.extend(labels.cpu().numpy())

                    # Extract unique label keys based on dataset sequences
                    label_keys.extend(idx_to_label_key[idx * valid_dataloader.batch_size:(idx + 1) * valid_dataloader.batch_size])

                    # Accuracy calculation
                    total += labels.size(0)
                    correct_cls += (predicted_cls == labels).sum().item()

            # print("length predictions cls", len(predictions_cls))
            # print("label_keys", label_keys)
            # Calculate and display validation accuracy
            accuracy_cls = 100 * correct_cls / total
            print(f"Validation Accuracy cls: {accuracy_cls:.2f}%")

            # Print detailed results
            print("\nValidation Results:")
            print(f"{'Label Key':<10}{'Ground Truth':<15}{'Prediction':<15}")
            print("=" * 40)
            for i in range(len(predictions_cls)):
                print(f"{label_keys[i]:<10}{ground_truths_cls[i]:<15}{predictions_cls[i]:<15}")

            model.train()

            

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
    transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    # transforms.GaussianBlur(kernel_size=3),
    # transforms.RandomGrayscale(p=0.1),
    # transforms.RandomApply([transforms.ElasticTransform(alpha=1.0)], p=0.3),
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
    for k in tqdm.tqdm(range(5, 55, 5)):
        dataset = SequenceViscosityDataset(base_dir, labels_dict, vis_dict, velocities, transform, seq_len=k, mode="train")
    #     sampler = CyclicSampler(dataset, batch_size=32)
        train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        dataset_val = SequenceViscosityDataset(base_dir, labels_dict, vis_dict, velocities, transform, seq_len=k, mode="val")
        valid_dataloader = DataLoader(dataset_val, batch_size=32)
        train_sequence_model(model, train_dataloader, valid_dataloader, criterion_cls, criterion_reg, optimizer, scheduler, num_epochs=90, device=device, start_epoch=start_epoch, k=k)
        torch.cuda.empty_cache()
