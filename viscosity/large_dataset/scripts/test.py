import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
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
from collections import defaultdict
import uuid
from datetime import datetime as dt
from time import time
import shutil
from model.vis_cls import VisCls
from model.dataloader import FluidViscosityDataset

def main_worker(args, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(args.load_ckpt)
    # robot stats from json
    with open(os.path.join(base_dir, 'robot_stats.json'), 'r') as f:
        robot_stats = json.load(f)
    
    # Create training dataset and input normalization stats
    train_dataset = FluidViscosityDataset(
        root_dir=args.root_dir,
        config_json=args.config_file,
        vial_label_csv='vial_label.csv',
        split='test',
        sequence_length=config['sequence_length'],
        mask_format='png',
        # transform=transforms.Compose([
        #     transforms.Resize((224, 224)),
        # ]),
        robot_mean_std=robot_stats
    )
    
    test_loader = DataLoader(train_dataset, batch_size=config['batch_size'] * 10, shuffle=False, num_workers=2, pin_memory=True)
    # Load model
    model = VisCls(embed_dim=160).to(device)
    model.to(device)
    model.load_state_dict(torch.load(args.load_ckpt, map_location=device)['model_state_dict'], strict=False)
    model.eval()
    all_preds = []
    all_labels = []
    all_vials = []
    out_dir = os.path.join(base_dir, 'output')
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        for data in tqdm(test_loader):
            mask_seq, robot_seq, timestamps, labels, vial_ids = [data[k].to(device) if isinstance(data[k], torch.Tensor) else data[k] for k in ['masks', 'robot', 'timestamps', 'label', 'vial_id']]
            outputs = model(mask_seq, robot_seq, timestamps)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_vials.extend(vial_ids)
            torch.cuda.empty_cache()
    preds = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)
    # accuracy
    acc = (preds == labels).float().mean()
    print(f'Accuracy: {acc.item():.4f}')
    
    vialwise_correct = defaultdict(int)
    vialwise_total = defaultdict(int)
    for vial_id, pred, label in zip(all_vials, preds, labels):
        vialwise_total[vial_id] += 1
        if pred == label:
            vialwise_correct[vial_id] += 1
    vialwise_acc = {vial: vialwise_correct[vial] / vialwise_total[vial] for vial in vialwise_total}
    vial_acc_df = pd.DataFrame([
        {'vial_id': vial, 'accuracy': acc} for vial, acc in vialwise_acc.items()
    ])
    vial_acc_df.to_csv(os.path.join(out_dir, 'vialwise_accuracy.csv'), index=False)

    # confusion matrix
    cm = confusion_matrix(labels.numpy(), preds.numpy())
    cm_df = pd.DataFrame(cm, index=['low', 'medium', 'high'], columns=['low', 'medium', 'high'])
    cm_df.to_csv(os.path.join(out_dir, 'confusion_matrix.csv'))
    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'))
    plt.close()
    # Classification report
    report = classification_report(labels.numpy(), preds.numpy(), target_names=['low', 'medium', 'high'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(out_dir, 'classification_report.csv'))
    
    # Also save vialwise confusion matrix
    # each vial will have its own confusion matrix
    vialwise_cm = defaultdict(lambda: defaultdict(int))
    for vial_id, pred, label in zip(all_vials, preds, labels):
        vialwise_cm[vial_id][(label.item(), pred.item())] += 1
    for vial_id, cm in vialwise_cm.items():
        cm_dict = {(str(k[0]), str(k[1])): v for k, v in cm.items()}
        cm_df = pd.DataFrame.from_dict(cm_dict, orient='index', columns=['count'])
        cm_df.index = pd.MultiIndex.from_tuples(cm_df.index, names=['True', 'Predicted'])
        cm_df = cm_df.unstack(fill_value=0)
        cm_df.columns = cm_df.columns.droplevel(0)
        cm_df.to_csv(os.path.join(out_dir, f'vial_{vial_id}_confusion_matrix.csv'))
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for Vial {vial_id}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(out_dir, f'vial_{vial_id}_confusion_matrix.png'))
        plt.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fluid Viscosity Classification")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory of the dataset")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the config file")
    parser.add_argument("--load_ckpt", type=str, required=True, help="Path to the checkpoint file")
    
    args = parser.parse_args()
    
    with open(args.config_file, 'r') as f:
        config = json.load(f)
        
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    main_worker(args, config)