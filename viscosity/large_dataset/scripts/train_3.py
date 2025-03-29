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
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main_worker(rank, world_size, args, config, run_id):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    
    is_main = rank == 0
    if is_main:
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
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    global_batch_size = config['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=global_batch_size//world_size, sampler=train_sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=global_batch_size//world_size * 4, sampler=val_sampler, num_workers=2, pin_memory=True)

    model = VisCls(embed_dim=160).to(device)
    model = DDP(model, device_ids=[rank])
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    optimizer.zero_grad()
    start_epoch = 0
    
    if args.load_ckpt and os.path.exists(args.load_ckpt):
        print(f"Loading checkpoint from {args.load_ckpt}")
        checkpoint = torch.load(args.load_ckpt)
        model.load_state_dict(checkpoint['model_state_dict'], map_location=device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        train_loss = checkpoint['train_loss']
        train_acc = checkpoint['train_acc']
        val_loss = checkpoint['val_loss']
        val_acc = checkpoint['val_acc']
        ckpt_path = checkpoint['checkpoint_path']
        print(f"Checkpoint loaded. Starting from epoch {start_epoch}")
    # else:
    #     model.apply(lambda m: nn.init.kaiming_normal_(m.weight) if hasattr(m, 'weight') else None)
    #     model.apply(lambda m: nn.init.constant_(m.bias, 0) if hasattr(m, 'bias') else None)

    start = time()
    wandb.init(project=args.wandb_project, name=run_id, config=config)
    wandb.watch(model)

    for epoch in tqdm(range(start_epoch, start_epoch + config['epochs'])):
        torch.cuda.empty_cache()
        model.train()
        train_sampler.set_epoch(epoch)
        train_loss = 0
        train_acc = 0

        for data in train_loader:
            mask_seq, robot_seq, timestamps, labels = [data[k].to(device) for k in ['masks', 'robot', 'timestamps', 'label']]
            optimizer.zero_grad()
            outputs = model(mask_seq, robot_seq, timestamps)
            loss = 100 * F.cross_entropy(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            train_acc += (outputs.argmax(1) == labels).sum().item()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader.dataset)
        scheduler.step()

        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for val_data in val_loader:
                mask_seq, robot_seq, timestamps, labels = [val_data[k].to(device) for k in ['masks', 'robot', 'timestamps', 'label']]
                outputs = model(mask_seq, robot_seq, timestamps)
                loss = 100 * F.cross_entropy(outputs, labels)
                val_loss += loss.item()
                val_acc += (outputs.argmax(1) == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc /= len(val_loader.dataset)

        if is_main:
            wandb.log({"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc, "epoch": epoch})
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            if epoch+1 % 5 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'checkpoint_path': os.path.join(args.checkpoint_path, run_id)
                }
                if args.load_ckpt and os.path.exists(args.load_ckpt):
                    torch.save(checkpoint, os.path.join(ckpt_path, f"epoch_{epoch+1}.pth"))
                else:
                    torch.save(checkpoint, os.path.join(args.checkpoint_path, run_id, f"epoch_{epoch+1}.pth"))
                print(f"Checkpoint saved at epoch {epoch}")
    
    if is_main:
        end = time()
        print(f"Training took {end - start:.2f} seconds")
        wandb.finish()
    cleanup()

if __name__ == "__main__":
    wandb.login()
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--root_dir', type=str, default='/ocean/projects/agr240001p/mqureshi/sparsh/liquid_detection/sam2/viscosity/large_dataset/data')
    parser.add_argument('--config_file', type=str, default='configs/valid_train_config.json')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/')
    parser.add_argument('--wandb_project', type=str, default='viscosity_cls')
    parser.add_argument('--load_ckpt', type=str, default=None)
    parser.add_argument('--notes', type=str, default='', help='Notes for the run')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Wandb run name')
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = json.load(f)

    run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-{uuid.uuid4()}"
    os.makedirs(os.path.join(args.checkpoint_path, run_id), exist_ok=True)
    print("checkpoints will be saved in ", os.path.join(args.checkpoint_path, run_id))
    if args.notes:
        with open(os.path.join(args.checkpoint_path, run_id, 'notes.txt'), 'w') as f:
            f.write(args.notes)
    shutil.copy(args.config_file, os.path.join(args.checkpoint_path, run_id))

    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    world_size = torch.cuda.device_count()
    mp.spawn(main_worker, args=(world_size, args, config, run_id), nprocs=world_size)  # Spawn processes
