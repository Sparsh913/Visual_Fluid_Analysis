import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import argparse
import sys
import wandb
import random
import uuid
from datetime import datetime as dt
from time import time
import shutil
import traceback
from model.vis_model import VisModel
from model.vmae import VideoMAEVisModel
from model.dataloader import FluidViscosityDataset
from debug_dataloader import validate_and_sample_logs, validate_masks

def move_batch_to_device(batch, device):
    """Helper function to move batch data to the specified device"""
    keys = ['masks', 'robot', 'timestamps']
    if 'label' in batch:
        keys.append('label')
    if 'value' in batch:
        keys.append('value')
        keys.append('raw_value')
    return {k: batch[k].to(device) for k in keys}

def log_gpu_memory():
    """Log GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        return f"GPU Memory: {allocated:.1f}MB allocated, {reserved:.1f}MB reserved"
    return "GPU not available"

def unfreeze_backbone_layers(model, num_layers_to_unfreeze=None):
    """
    Gradually unfreeze layers in the VideoMAE backbone.
    If num_layers_to_unfreeze is None, unfreeze all layers.
    """
    # Get all backbone parameters/layers organized by module
    backbone_layers = []
    
    # Access the main transformer blocks which contain most params
    if hasattr(model.backbone, 'encoder'):
        # Get the encoder layers (transformer blocks)
        transformer_blocks = list(model.backbone.encoder.layer)
        backbone_layers.extend(reversed(transformer_blocks))  # Reverse to unfreeze from last to first
    
    # Add embeddings module if it exists
    if hasattr(model.backbone, 'embeddings'):
        backbone_layers.append(model.backbone.embeddings)
    
    # Now unfreeze the appropriate number of layers
    if num_layers_to_unfreeze is None:
        # Unfreeze everything
        for param in model.backbone.parameters():
            param.requires_grad = True
        print(f"Unfrozen all backbone layers")
    else:
        # Unfreeze specific number of layers from the end
        layers_unfrozen = 0
        for i, layer in enumerate(backbone_layers):
            if layers_unfrozen >= num_layers_to_unfreeze:
                break
                
            print(f"Unfreezing layer: {type(layer).__name__}")
            for param in layer.parameters():
                param.requires_grad = True
            layers_unfrozen += 1
            
        print(f"Unfrozen {layers_unfrozen} backbone layers")
    
    return model

def main_worker(args, config, run_id):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Initialize wandb at the beginning of training
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or run_id,
            config=config,
            notes=args.notes
        )
        
        # Log system info
        wandb.log({"system_info": {
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        }})

        print("Loading datasets...")
        # Create training dataset and extract normalization stats
        try:
            train_dataset = FluidViscosityDataset(
                root_dir=args.root_dir,
                config_json=args.config_file,
                vial_label_csv='vial_label.csv',
                split='train',
                sequence_length=config['sequence_length'],
                mask_format='png',
                transform= transforms.Compose([
                    transforms.Resize((224, 224)),
                ]),
                normalize_robot_data = True,
                normalize_timestamps = True,
                task=args.task,
                regression_csv='data_reg.csv' if args.task == 'regression' else None
            )
            
            if train_dataset.robot_mean is not None:
                robot_stats = {
                    'mean': train_dataset.robot_mean.tolist(),
                    'std': train_dataset.robot_std.tolist()
                }

                # Save stats for test-time use
                stats_path = os.path.join(args.checkpoint_path, run_id, 'robot_stats.json')
                if args.load_ckpt:
                    stats_path = os.path.join(os.path.dirname(args.load_ckpt), 'robot_stats.json')
                with open(stats_path, 'w') as f:
                    json.dump(robot_stats, f)
                print(f"Saved normalization stats to {stats_path}")
                
            # If regression, also save regression normalization stats
            if args.task == 'regression':
                reg_stats = {
                    'mean': train_dataset.reg_mean,
                    'std': train_dataset.reg_std
                }
                reg_stats_path = os.path.join(args.checkpoint_path, run_id, 'reg_stats.json')
                if args.load_ckpt:
                    reg_stats_path = os.path.join(os.path.dirname(args.load_ckpt), 'reg_stats.json')
                with open(reg_stats_path, 'w') as f:
                    json.dump(reg_stats, f)
                print(f"Saved regression normalization stats to {reg_stats_path}")

            # Validation dataset uses training stats
            val_dataset = FluidViscosityDataset(
                root_dir=args.root_dir,
                config_json=args.config_file,
                vial_label_csv='vial_label.csv',
                split='val',
                sequence_length=config['sequence_length'],
                mask_format='png',
                transform= transforms.Compose([
                    transforms.Resize((224, 224)),
                ]),
                robot_mean_std=robot_stats,
                normalize_robot_data=True,
                normalize_timestamps=True,
                task=args.task,
                regression_csv='data_reg.csv' if args.task == 'regression' else None,
                reg_mean_std=reg_stats if args.task == 'regression' else None
            )
        except Exception as e:
            print(f"Error loading datasets: {e}")
            traceback.print_exc()
            wandb.finish()
            return

        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Val dataset size: {len(val_dataset)}")
        
        # invalid_indices = validate_masks(train_dataset)
        # if invalid_indices:
        #     print(f"Invalid masks found at indices: {invalid_indices}")
        #     wandb.finish()
        #     return
        # else:
        #     print("All masks are valid")
        
        # print("Validating log files across all splits...")
        # validate_and_sample_logs(args.config_file, args.root_dir, n_samples=10)

        # Set up data loaders with appropriate batch sizes
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True, 
            num_workers=4, 
            pin_memory=True,
            drop_last=False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config['batch_size'] * 2, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True
        )

        # Create model with appropriate task
        # model = VisModel(embed_dim=160, task=args.task).to(device)
        model = VideoMAEVisModel(task=args.task, embed_dim=config['embed_dim']).to(device)
        
        # Initially freeze the VideoMAE backbone
        for param in model.backbone.parameters():
            param.requires_grad = False
        
        # Get initially trainable parameters (non-frozen)
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        
        # Initialize model weights with Kaiming initialization
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Set up optimizer and scheduler
        optimizer = optim.AdamW(
            trainable_params, 
            lr=config['lr'], 
            weight_decay=config['weight_decay']
        )
        
        # scheduler = optim.lr_scheduler.StepLR(
        #     optimizer, 
        #     step_size=30, 
        #     gamma=0.1
        # )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config['epochs'], 
            eta_min=0
        )
        
        # Initialize training variables
        start_epoch = 0
        best_val_acc = 0  # For classification
        best_val_mae = float('inf')  # For regression
        ckpt_path = os.path.join(args.checkpoint_path, run_id)
        
        # Load checkpoint if provided
        if args.load_ckpt and os.path.exists(args.load_ckpt):
            try:
                print(f"Loading checkpoint from {args.load_ckpt}")
                checkpoint = torch.load(args.load_ckpt, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                if args.task == 'classification':
                    best_val_acc = checkpoint.get('best_val_acc', 0)
                    print(f"Current best validation accuracy: {best_val_acc:.4f}")
                else:
                    best_val_mae = checkpoint.get('best_val_mae', float('inf'))
                    print(f"Current best validation MAE: {best_val_mae:.4f}")
                ckpt_path = checkpoint.get('checkpoint_path', ckpt_path)
                
                # Restore batch size if available
                if 'current_batch_size' in checkpoint:
                    current_batch_size = checkpoint['current_batch_size']
                    print(f"Restored batch size: {current_batch_size}")
                    
                    # Recreate dataloaders with restored batch size
                    train_loader = DataLoader(
                        train_dataset, 
                        batch_size=current_batch_size, 
                        shuffle=True, 
                        num_workers=4, 
                        pin_memory=True,
                        drop_last=False
                    )
                    
                    val_loader = DataLoader(
                        val_dataset, 
                        batch_size=current_batch_size * 2, 
                        shuffle=False, 
                        num_workers=4, 
                        pin_memory=True
                    )
                
                print(f"Checkpoint loaded. Starting from epoch {start_epoch}")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                traceback.print_exc()
        
        # Track model in wandb
        wandb.watch(model, log="all", log_freq=100)
        
        # Start training timer
        start_time = time()
        epoch_times = []

        # Get attention regularization strength
        lambda_reg = config.get('attn_reg', 0.1)
        
        # Gradual unfreezing settings
        unfreeze_start_epoch = 6  # Starting epoch for unfreezing (0-indexed)
        total_layers = 12         # Typical number of transformer blocks in base models
        unfreeze_schedule = {     # Define which layers to unfreeze at each epoch
            unfreeze_start_epoch: 2,      # Unfreeze 2 layers at epoch 6
            unfreeze_start_epoch + 3: 4,  # Unfreeze 4 layers at epoch 9
            unfreeze_start_epoch + 6: 8,  # Unfreeze 8 layers at epoch 12
            unfreeze_start_epoch + 9: None  # Unfreeze all remaining layers at epoch 15
        }
        
        # Batch size reduction schedule based on unfreezing
        # As we unfreeze more layers, we'll reduce the batch size to prevent OOM errors
        batch_size_schedule = {
            unfreeze_start_epoch: config['batch_size'] // 2,      # Half the batch size when initial unfreezing starts
            unfreeze_start_epoch + 6: config['batch_size'] // 6,  # Quarter the batch size when unfreezing 8 layers
            unfreeze_start_epoch + 9: config['batch_size'] // 10   # Further reduce when unfreezing all layers
        }
        
        # Current batch size (will be adjusted during training)
        current_batch_size = config['batch_size']

        print("Starting training...")
        for epoch in range(start_epoch, start_epoch + config['epochs']):
            epoch_start = time()
            
            # Check if we need to unfreeze layers based on the schedule
            if epoch in unfreeze_schedule:
                num_layers = unfreeze_schedule[epoch]
                print(f"Epoch {epoch+1}: Unfreezing {num_layers if num_layers else 'all'} backbone layers")
                model = unfreeze_backbone_layers(model, num_layers)
                
                # Update optimizer with newly unfrozen parameters
                trainable_params = filter(lambda p: p.requires_grad, model.parameters())
                optimizer = optim.AdamW(
                    trainable_params, 
                    lr=config['lr'] * 0.1,  # Use a lower learning rate for fine-tuning
                    weight_decay=config['weight_decay']
                )
                
                # Reset scheduler with the remaining epochs
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, 
                    T_max=start_epoch + config['epochs'] - epoch, 
                    eta_min=0
                )
            
            # Check if we need to adjust batch size based on the schedule
            if epoch in batch_size_schedule:
                # Update the current batch size
                current_batch_size = batch_size_schedule[epoch]
                print(f"Epoch {epoch+1}: Adjusting batch size to {current_batch_size}")
                
                # Recreate dataloaders with new batch size
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=current_batch_size, 
                    shuffle=True, 
                    num_workers=4, 
                    pin_memory=True,
                    drop_last=False
                )
                
                val_loader = DataLoader(
                    val_dataset, 
                    batch_size=current_batch_size * 2, 
                    shuffle=False, 
                    num_workers=4, 
                    pin_memory=True
                )
            
            # Training phase
            torch.cuda.empty_cache()
            model.train()
            train_loss = 0
            train_acc = 0  # for classification
            train_mae = 0  # for regression
            train_samples = 0
            
            # Progress bar for training
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch+config['epochs']} [Train]")
            
            for data in train_pbar:
                # Move data to device
                batch = move_batch_to_device(data, device)
                
                # Forward pass
                optimizer.zero_grad()
                
                if args.task == 'classification':
                    loss, outputs = model(batch['masks'], batch['robot'], batch['timestamps'], batch['label'])
                    # Update metrics
                    batch_size = batch['label'].size(0)
                    train_acc += (outputs.argmax(1) == batch['label']).sum().item()
                else:  # regression
                    loss, outputs = model(batch['masks'], batch['robot'], batch['timestamps'], batch['value'])
                    # Update metrics - track MAE on raw values
                    batch_size = batch['value'].size(0)
                    with torch.no_grad():
                        pred_raw = outputs * train_dataset.reg_std + train_dataset.reg_mean
                        true_raw = batch['raw_value']
                        train_mae += torch.abs(pred_raw - true_raw).sum().item()
                
                # Backward pass
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Update total loss
                train_loss += loss.item() * batch_size
                train_samples += batch_size
                
                # Update progress bar with appropriate metrics
                if args.task == 'classification':
                    train_pbar.set_postfix({
                        'loss': loss.item(),
                        'acc': (outputs.argmax(1) == batch['label']).sum().item() / batch_size,
                        'mem': log_gpu_memory()
                    })
                else:
                    train_pbar.set_postfix({
                        'loss': loss.item(),
                        'mae': torch.abs(pred_raw - true_raw).mean().item(),
                        'mem': log_gpu_memory()
                    })
            
            # Calculate epoch metrics
            train_loss /= train_samples
            if args.task == 'classification':
                train_acc /= train_samples
            else:
                train_mae /= train_samples
            
            # Step the scheduler
            scheduler.step()
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_acc = 0  # for classification
            val_mae = 0  # for regression
            val_samples = 0
            
            # Progress bar for validation
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{start_epoch+config['epochs']} [Val]")
            
            with torch.no_grad():
                for val_data in val_pbar:
                    # Move data to device
                    batch = move_batch_to_device(val_data, device)
                    
                    # Forward pass
                    if args.task == 'classification':
                        loss, outputs = model(batch['masks'], batch['robot'], batch['timestamps'], batch['label'])
                        # Update metrics
                        batch_size = batch['label'].size(0)
                        val_acc += (outputs.argmax(1) == batch['label']).sum().item()
                    else:  # regression
                        loss, outputs = model(batch['masks'], batch['robot'], batch['timestamps'], batch['value'])
                        # Update metrics
                        batch_size = batch['value'].size(0)
                        pred_raw = outputs * val_dataset.reg_std + val_dataset.reg_mean
                        true_raw = batch['raw_value']
                        val_mae += torch.abs(pred_raw - true_raw).sum().item()
                    
                    # Update total loss
                    val_loss += loss.item() * batch_size
                    val_samples += batch_size
                    
                    # Update progress bar
                    if args.task == 'classification':
                        val_pbar.set_postfix({
                            'loss': loss.item(),
                            'acc': (outputs.argmax(1) == batch['label']).sum().item() / batch_size
                        })
                    else:
                        val_pbar.set_postfix({
                            'loss': loss.item(),
                            'mae': torch.abs(pred_raw - true_raw).mean().item()
                        })
            
            # Calculate validation metrics
            val_loss /= val_samples
            if args.task == 'classification':
                val_acc /= val_samples
            else:
                val_mae /= val_samples
            
            # Calculate epoch time
            epoch_end = time()
            epoch_time = epoch_end - epoch_start
            epoch_times.append(epoch_time)
            
            # Check for best model
            if args.task == 'classification':
                is_best = val_acc > best_val_acc
                if is_best:
                    best_val_acc = val_acc
            else:  # regression
                is_best = val_mae < best_val_mae
                if is_best:
                    best_val_mae = val_mae
            
            # Log metrics
            metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch": epoch,
                "epoch_time": epoch_time,
                "learning_rate": scheduler.get_last_lr()[0],
            }
            
            # Track frozen/unfrozen status
            frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            metrics["frozen_params_pct"] = 100 * frozen_params / total_params
            
            # Add task-specific metrics
            if args.task == 'classification':
                metrics.update({
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "best_val_acc": best_val_acc
                })
            else:
                metrics.update({
                    "train_mae": train_mae,
                    "val_mae": val_mae,
                    "best_val_mae": best_val_mae
                })
                
            wandb.log(metrics)
            
            # Print epoch summary
            if args.task == 'classification':
                print(f"Epoch {epoch+1} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_acc:.4f} | "
                      f"Frozen: {metrics['frozen_params_pct']:.1f}% | "
                      f"Time: {epoch_time:.2f}s")
            else:
                print(f"Epoch {epoch+1} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Train MAE: {train_mae:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val MAE: {val_mae:.4f} | "
                      f"Frozen: {metrics['frozen_params_pct']:.1f}% | "
                      f"Time: {epoch_time:.2f}s")
            
            # Save checkpoint
            if (epoch+1) % 5 == 0 or is_best:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'checkpoint_path': ckpt_path,
                    'task': args.task,
                    'frozen_params_pct': metrics['frozen_params_pct'],
                    'current_batch_size': current_batch_size,  # Save current batch size for resuming training
                    'unfrozen_layers': unfreeze_schedule.get(epoch, 0) if epoch in unfreeze_schedule else None
                }
                
                # Add task-specific metrics
                if args.task == 'classification':
                    checkpoint.update({
                        'train_acc': train_acc,
                        'val_acc': val_acc,
                        'best_val_acc': best_val_acc
                    })
                else:
                    checkpoint.update({
                        'train_mae': train_mae,
                        'val_mae': val_mae,
                        'best_val_mae': best_val_mae
                    })
                
                # Save periodic checkpoint
                if (epoch+1) % 5 == 0:
                    save_path = os.path.join(ckpt_path, f"epoch_{epoch+1}.pth")
                    torch.save(checkpoint, save_path)
                    print(f"Checkpoint saved at: {save_path}")
                
                # Save best model
                if is_best:
                    best_save_path = os.path.join(ckpt_path, "best_model.pth")
                    torch.save(checkpoint, best_save_path)
                    print(f"Best model saved at: {best_save_path}")
        
        # Training summary
        training_time = time() - start_time
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        
        summary = {
            "total_training_time": training_time,
            "average_epoch_time": avg_epoch_time,
            "final_train_loss": train_loss,
            "final_val_loss": val_loss
        }
        
        # Add task-specific metrics to summary
        if args.task == 'classification':
            summary["best_val_accuracy"] = best_val_acc
        else:
            summary["best_val_mae"] = best_val_mae
        
        wandb.log(summary)
        
        if args.task == 'classification':
            print(f"Training completed in {training_time:.2f} seconds")
            print(f"Average epoch time: {avg_epoch_time:.2f} seconds")
            print(f"Best validation accuracy: {best_val_acc:.4f}")
        else:
            print(f"Training completed in {training_time:.2f} seconds")
            print(f"Average epoch time: {avg_epoch_time:.2f} seconds")
            print(f"Best validation MAE: {best_val_mae:.4f}")
        
        # Finish wandb run
        wandb.finish()
        
    except Exception as e:
        print(f"Error during training: {e}")
        traceback.print_exc()
        wandb.finish()

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Single GPU Training Script for Fluid Viscosity Classification/Regression")
    parser.add_argument('--root_dir', type=str, default='/ocean/projects/agr240001p/mqureshi/sparsh/liquid_detection/sam2/viscosity/large_dataset/data')
    parser.add_argument('--config_file', type=str, default='configs/valid_train_config.json')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/')
    parser.add_argument('--wandb_project', type=str, default='viscosity_cls')
    parser.add_argument('--load_ckpt', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--notes', type=str, default='', help='Notes for the run')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Wandb run name')
    parser.add_argument('--task', type=str, default='classification', choices=['classification', 'regression'], 
                        help='Task type: classification or regression')
    parser.add_argument('--attn_reg', type=float, default=0.1, 
                        help='Attention regularization strength')
    args = parser.parse_args()

    # Login to wandb
    try:
        wandb.login()
    except Exception as e:
        print(f"Error logging in to wandb: {e}")
        print("Continuing without wandb logging")

    # Load configuration
    try:
        with open(args.config_file, "r") as f:
            config = json.load(f)
            # Add attention regularization to config if provided as argument
            if args.attn_reg is not None:
                config['attn_reg'] = args.attn_reg
    except Exception as e:
        print(f"Error loading config file: {e}")
        sys.exit(1)

    # Generate unique run ID
    run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-{uuid.uuid4().hex[:8]}"
    
    # Create checkpoint directory
    try:
        checkpoint_dir = os.path.join(args.checkpoint_path, run_id)
        if args.load_ckpt:
            checkpoint_dir = os.path.dirname(args.load_ckpt)
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoints will be saved in {checkpoint_dir}")
        
        # Save notes if provided
        if args.notes:
            with open(os.path.join(checkpoint_dir, 'notes.txt'), 'w') as f:
                f.write(args.notes)
        
        # Copy config file to checkpoint directory
        shutil.copy(args.config_file, os.path.join(checkpoint_dir))
    except Exception as e:
        print(f"Error creating checkpoint directory: {e}")
        sys.exit(1)

    # Set random seeds for reproducibility
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Print training information
    print(f"Starting training run: {run_id}")
    print(f"Task: {args.task}")
    print(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Start training
    main_worker(args, config, run_id)