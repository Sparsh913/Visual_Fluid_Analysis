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
from model.vis_cls import VisCls

from model.dataloader import FluidViscosityDataset

def main_worker(args, config, run_id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define the dataset
    train_dataset  =FluidViscosityDataset(
        root_dir='/ocean/projects/agr240001p/mqureshi/sparsh/liquid_detection/sam2/test_liq4',
        config_json='configs/test_config.json',
        vial_label_csv='vial_label.csv',
        split='train',
        sequence_length=10,
        mask_format='png',
        transform=None
    )
    val_dataset = FluidViscosityDataset(
        root_dir='/ocean/projects/agr240001p/mqureshi/sparsh/liquid_detection/sam2/test_liq4',
        config_json='configs/test_config.json',
        vial_label_csv='vial_label.csv',
        split='val',
        sequence_length=10,
        mask_format='png',
        transform=None
    )
    train_batch_sz = config['batch_size']
    val_batch_sz = config['batch_size'] * 4
    train_loader = DataLoader(train_dataset, batch_size=train_batch_sz, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_sz, shuffle=False, num_workers=2, pin_memory=True)

    ##############################
    ########### MODEL ############
    ##############################
    
    model = VisCls(embed_dim=160)
    model = model.to(device)
    
    ##############################
    ######### OPTIMIZER ##########
    ##############################
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    optimizer.zero_grad()
    
    ##############################
    ########## TRAINING ##########
    ##############################
    start = time()
    wandb.init(project=args.wandb_project, name=run_id, config=config)
    wandb.watch(model)
    # use tqdm in epoch loop
    for epoch in tqdm(range(config['epochs']), desc="Epochs"):
        torch.cuda.empty_cache()
        model.train()
        train_loss = 0
        train_acc = 0
        for i, data in enumerate(train_loader):
            mask_seq = data['masks']
            robot_seq = data['robot']
            timestamps = data['timestamps']
            labels = data['label']
            mask_seq = mask_seq.to(device)
            robot_seq = robot_seq.to(device)
            timestamps = timestamps.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(mask_seq, robot_seq, timestamps)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
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
            for i, val_data in enumerate(val_loader):
                mask_seq = val_data['masks']
                robot_seq = val_data['robot']
                timestamps = val_data['timestamps']
                labels = val_data['label']
                mask_seq = mask_seq.to(device)
                robot_seq = robot_seq.to(device)
                timestamps = timestamps.to(device)
                labels = labels.to(device)
                outputs = model(mask_seq, robot_seq, timestamps)
                loss = F.cross_entropy(outputs, labels)
                val_loss += loss.item()
                val_acc += (outputs.argmax(1) == labels).sum().item()
        val_loss /= len(val_loader)
        val_acc /= len(val_loader.dataset)
        wandb.log({"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc, "epoch": epoch})
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        if epoch % 20 == 0:
            torch.save(model.state_dict(), os.path.join(args.checkpoint_path, run_id, f"epoch_{epoch}.pth"))
    end = time()
    print(f"Training took {end - start:.2f} seconds")
    wandb.finish()
            

if __name__ == "__main__":
    wandb.login()
    
    # Define args
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--root_dir', type=str, default='/ocean/projects/agr240001p/mqureshi/sparsh/liquid_detection/sam2/test_liq4')
    parser.add_argument('--config_file', type=str, default='configs/test_config.json')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/')
    parser.add_argument('--wandb_project', type=str, default='viscosity_cls')
    
    args = parser.parse_args()
    with open(args.config_file, "r") as f:
        config = json.load(f)
    
    run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-{uuid.uuid4()}"
    os.makedirs(os.path.join(args.checkpoint_path, run_id), exist_ok=True)
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
    ngpus_per_node = torch.cuda.device_count()
    
    main_worker(args, config, run_id)