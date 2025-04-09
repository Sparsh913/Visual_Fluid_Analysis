import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
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
from model.vis_cls import VisCls
from model.dataloader import FluidViscosityDataset

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def all_reduce_mean(tensor, world_size):
    torch.distributed.all_reduce(tensor)
    tensor.div_(world_size)
    return tensor

def main_worker(rank, world_size, args, config, run_id):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    
    is_main = rank == 0
    if is_main:
        os.makedirs(os.path.join(args.checkpoint_path, run_id), exist_ok=True)
        wandb.init(project=args.wandb_project, name=args.wandb_run_name or run_id, config=config, notes=args.notes)

    # Create training dataset and extract normalization stats
    train_dataset = FluidViscosityDataset(
        root_dir=args.root_dir,
        config_json=args.config_file,
        vial_label_csv='vial_label.csv',
        split='train',
        sequence_length=config['sequence_length'],
        mask_format='png',
        transform=None
    )
    robot_stats = {
        'mean': train_dataset.robot_mean.tolist(),
        'std': train_dataset.robot_std.tolist()
    }

    # Save stats for test-time use
    if is_main:
        with open(os.path.join(args.checkpoint_path, run_id, 'robot_stats.json'), 'w') as f:
            json.dump(robot_stats, f)

    # Make sure all processes have the same stats by broadcasting from rank 0
    robot_mean_tensor = torch.tensor(robot_stats['mean'], device=device)
    robot_std_tensor = torch.tensor(robot_stats['std'], device=device)
    dist.broadcast(robot_mean_tensor, 0)
    dist.broadcast(robot_std_tensor, 0)
    
    # Update robot_stats with broadcast values
    robot_stats['mean'] = robot_mean_tensor.cpu().numpy().tolist()
    robot_stats['std'] = robot_std_tensor.cpu().numpy().tolist()

    # Validation dataset uses training stats
    val_dataset = FluidViscosityDataset(
        root_dir=args.root_dir,
        config_json=args.config_file,
        vial_label_csv='vial_label.csv',
        split='val',
        sequence_length=config['sequence_length'],
        mask_format='png',
        transform=None,
        robot_mean_std=robot_stats
    )
    
    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    
    # Calculate batch sizes per GPU
    global_batch_size = config['batch_size']
    per_gpu_batch_size = global_batch_size // world_size
    val_batch_size = max(1, per_gpu_batch_size * 4)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=per_gpu_batch_size, 
        sampler=train_sampler, 
        num_workers=4, 
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=val_batch_size, 
        sampler=val_sampler, 
        num_workers=4, 
        pin_memory=True
    )

    # Print dataset sizes for debugging
    if is_main:
        print(f"Train dataset size: {len(train_dataset)}, batches: {len(train_loader)}")
        print(f"Val dataset size: {len(val_dataset)}, batches: {len(val_loader)}")

    # Ensure all processes have reached this point before continuing
    dist.barrier()

    # Create model and apply DDP
    model = VisCls(embed_dim=160).to(device)
    
    # Initialize model weights if not loading from checkpoint
    if not args.load_ckpt:
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    # Convert model to DDP
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Initialize tracking variables
    start_epoch = 0
    best_val_acc = 0
    ckpt_path = os.path.join(args.checkpoint_path, run_id)
    
    # Load checkpoint if provided
    if args.load_ckpt and os.path.exists(args.load_ckpt):
        if is_main:
            print(f"Loading checkpoint from {args.load_ckpt}")
        
        # Load on CPU first to avoid GPU synchronization issues
        checkpoint = torch.load(args.load_ckpt, map_location='cpu')
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0)
        ckpt_path = checkpoint.get('checkpoint_path', ckpt_path)
        
        if is_main:
            print(f"Checkpoint loaded. Starting from epoch {start_epoch}")
    
    # Ensure all processes have loaded the checkpoint
    dist.barrier()

    start = time()
    if is_main:
        wandb.watch(model, log_freq=100)

    for epoch in range(start_epoch, start_epoch + config['epochs']):
        # Set epoch for samplers (crucial for proper shuffling in distributed setting)
        train_sampler.set_epoch(epoch)
        
        model.train()
        train_loss_sum = torch.tensor(0.0, device=device)
        train_correct = torch.tensor(0, device=device)
        train_total = torch.tensor(0, device=device)

        if is_main:
            train_iterator = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        else:
            train_iterator = train_loader

        for data in train_iterator:
            # Move data to device
            mask_seq = data['masks'].to(device, non_blocking=True)
            robot_seq = data['robot'].to(device, non_blocking=True)
            timestamps = data['timestamps'].to(device, non_blocking=True)
            labels = data['label'].to(device, non_blocking=True)
            
            # Forward pass and loss computation
            optimizer.zero_grad()
            outputs = model(mask_seq, robot_seq, timestamps)
            loss = 100 * F.cross_entropy(outputs, labels, reduction='mean')
            
            # Backward pass
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Track metrics
            batch_size = labels.size(0)
            train_loss_sum += loss.detach() * batch_size
            train_correct += (outputs.argmax(1) == labels).sum()
            train_total += batch_size
        
        # Aggregate metrics across all processes
        dist.all_reduce(train_loss_sum)
        dist.all_reduce(train_correct)
        dist.all_reduce(train_total)
        
        train_loss = train_loss_sum.item() / train_total.item()
        train_acc = train_correct.item() / train_total.item()
        
        # Step the scheduler
        scheduler.step()

        # Validation phase
        model.eval()
        val_loss_sum = torch.tensor(0.0, device=device)
        val_correct = torch.tensor(0, device=device)
        val_total = torch.tensor(0, device=device)
        
        if is_main:
            val_iterator = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        else:
            val_iterator = val_loader
            
        with torch.no_grad():
            for val_data in val_iterator:
                # Move data to device
                mask_seq = val_data['masks'].to(device, non_blocking=True)
                robot_seq = val_data['robot'].to(device, non_blocking=True)
                timestamps = val_data['timestamps'].to(device, non_blocking=True)
                labels = val_data['label'].to(device, non_blocking=True)
                
                # Forward pass
                outputs = model(mask_seq, robot_seq, timestamps)
                loss = 100 * F.cross_entropy(outputs, labels, reduction='mean')
                
                # Track metrics
                batch_size = labels.size(0)
                val_loss_sum += loss.detach() * batch_size
                val_correct += (outputs.argmax(1) == labels).sum()
                val_total += batch_size
        
        # Aggregate validation metrics
        dist.all_reduce(val_loss_sum)
        dist.all_reduce(val_correct)
        dist.all_reduce(val_total)
        
        val_loss = val_loss_sum.item() / val_total.item()
        val_acc = val_correct.item() / val_total.item()
        
        # Ensure all processes have calculated metrics
        dist.barrier()
        
        # Log and save checkpoint (only in main process)
        if is_main:
            wandb.log({
                "train_loss": train_loss, 
                "train_acc": train_acc, 
                "val_loss": val_loss, 
                "val_acc": val_acc, 
                "epoch": epoch,
                "learning_rate": scheduler.get_last_lr()[0]
            })
            
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Save checkpoint periodically
            if (epoch + 1) % 5 == 0 or val_acc > best_val_acc:
                is_best = val_acc > best_val_acc
                best_val_acc = max(best_val_acc, val_acc)
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'best_val_acc': best_val_acc,
                    'checkpoint_path': ckpt_path
                }
                
                # Save checkpoint
                ckpt_filename = f"epoch_{epoch+1}.pth"
                if is_best:
                    ckpt_filename = f"best_model.pth"
                
                torch.save(checkpoint, os.path.join(ckpt_path, ckpt_filename))
                print(f"Checkpoint saved at epoch {epoch+1}" + (" (best model)" if is_best else ""))
        
        # Ensure checkpoint is saved before proceeding
        dist.barrier()
    
    # Finish training
    if is_main:
        end = time()
        print(f"Training took {end - start:.2f} seconds")
        wandb.finish()
    
    # Clean up distributed environment
    cleanup()

if __name__ == "__main__":
    # Login to wandb
    wandb.login()
    
    parser = argparse.ArgumentParser(description="Distributed Training Script for Fluid Viscosity Classification")
    parser.add_argument('--root_dir', type=str, default='/ocean/projects/agr240001p/mqureshi/sparsh/liquid_detection/sam2/viscosity/large_dataset/data')
    parser.add_argument('--config_file', type=str, default='configs/valid_train_config.json')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/')
    parser.add_argument('--wandb_project', type=str, default='viscosity_cls')
    parser.add_argument('--load_ckpt', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--notes', type=str, default='', help='Notes for the run')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Wandb run name')
    args = parser.parse_args()

    # Load configuration
    with open(args.config_file, "r") as f:
        config = json.load(f)

    # Generate unique run ID
    run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-{uuid.uuid4().hex[:8]}"
    os.makedirs(os.path.join(args.checkpoint_path, run_id), exist_ok=True)
    print("Checkpoints will be saved in", os.path.join(args.checkpoint_path, run_id))
    
    # Save notes if provided
    if args.notes:
        with open(os.path.join(args.checkpoint_path, run_id, 'notes.txt'), 'w') as f:
            f.write(args.notes)
    
    # Copy config file to checkpoint directory
    shutil.copy(args.config_file, os.path.join(args.checkpoint_path, run_id))

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

    # Check GPU availability
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No GPUs found. This script requires at least one GPU.")
    print(f"Using {world_size} GPUs for distributed training")

    # Launch distributed processes
    mp.spawn(main_worker, args=(world_size, args, config, run_id), nprocs=world_size)