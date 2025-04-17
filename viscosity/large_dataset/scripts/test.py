import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error, r2_score
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
from model.vis_model import VisModel
from model.dataloader import FluidViscosityDataset
from utils.attn_vis import visualize_attention

def analyze_attention(args, model, test_dataset, num_samples=5, seq_len=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(args.load_ckpt)
    model.eval()
    attn_dir = os.path.join(base_dir, 'attention_viz')
    
    # Visualize attention maps
    visualize_attention(model, test_dataset, attn_dir, device, num_samples=num_samples, sequence_length=seq_len)

def move_batch_to_device(batch, device):
    """Helper function to move batch data to the specified device"""
    keys = ['interfaces', 'robot', 'timestamps', 'vial_id']
    if 'label' in batch:
        keys.append('label')
    if 'value' in batch:
        keys.append('value')
        keys.append('raw_value')
    return {k: batch[k].to(device) if k in batch and isinstance(batch[k], torch.Tensor) else batch[k] for k in keys}

def main_worker(args, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(args.load_ckpt)
    
    # Load robot stats from json
    with open(os.path.join(base_dir, 'robot_stats.json'), 'r') as f:
        robot_stats = json.load(f)
    
    # Load checkpoint to get task type
    checkpoint = torch.load(args.load_ckpt, map_location=device)
    task = checkpoint.get('task', args.task)  # Fallback to args.task if not in checkpoint
    
    # Load regression stats if needed
    reg_stats = None
    if task == 'regression':
        try:
            with open(os.path.join(base_dir, 'reg_stats.json'), 'r') as f:
                reg_stats = json.load(f)
                print(f"Loaded regression stats - Mean: {reg_stats['mean']:.4f}, Std: {reg_stats['std']:.4f}")
        except FileNotFoundError:
            print("Warning: Regression stats file not found, using defaults from dataset")
            
    # load global bounds
    try:
        with open(os.path.join(base_dir, 'global_bounds.json'), 'r') as f:
            global_bounds = json.load(f)
            print(f"Loaded global bounds - Min: {global_bounds['y_min']:.4f}, Max: {global_bounds['y_max']:.4f}")
    except FileNotFoundError:
        print("Warning: Global bounds file not found, using defaults from dataset")
        global_bounds = {'y_min': 0, 'y_max': 261}  # Default values
    
    # Create test dataset
    test_dataset = FluidViscosityDataset(
        root_dir=args.root_dir,
        config_json=args.config_file,
        vial_label_csv='vial_label.csv',
        split='test',
        sequence_length=config['sequence_length'],
        mask_format='png',
        robot_mean_std=robot_stats,
        task=task,
        regression_csv='data_reg.csv' if task == 'regression' else None,
        global_bounds=global_bounds,
        num_interface_points = config['interface_points']
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'] * 10, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )
    
    # Load model
    model = VisModel(embed_dim=config["embedding_dim"], task=task, num_points=config["interface_points"]).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    # Create output directory
    out_dir = os.path.join(base_dir, 'output')
    os.makedirs(out_dir, exist_ok=True)
    
    # Handle task-specific evaluation
    if task == 'classification':
        evaluate_classification(model, test_loader, device, out_dir)
    else:  # regression
        evaluate_regression(model, test_loader, device, out_dir, test_dataset)
    
    # analyze attention
    analyze_attention(args, model, test_dataset, num_samples=10, seq_len=config['sequence_length'])

def evaluate_classification(model, test_loader, device, out_dir):
    all_preds = []
    all_labels = []
    all_vials = []
    
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Evaluating classification"):
            batch = move_batch_to_device(data, device)
            outputs = model(batch['interfaces'], batch['robot'], batch['timestamps'])
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu())
            all_labels.append(batch['label'].cpu())
            all_vials.extend(batch['vial_id'])
            torch.cuda.empty_cache()
    
    preds = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    # Overall accuracy
    acc = (preds == labels).float().mean()
    print(f'Classification Accuracy: {acc.item():.4f}')
    
    # Vial-wise accuracy
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

    # Confusion matrix
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
    
    # Vial-wise confusion matrices
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

def evaluate_regression(model, test_loader, device, out_dir, dataset):
    all_preds = []
    all_values = []
    all_raw_preds = []
    all_raw_values = []
    all_vials = []
    
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Evaluating regression"):
            batch = move_batch_to_device(data, device)
            outputs = model(batch['interfaces'], batch['robot'], batch['timestamps'])
            
            # Store normalized predictions and targets
            all_preds.append(outputs.cpu().unsqueeze(-1))
            all_values.append(batch['value'].cpu().unsqueeze(-1))
            
            # Convert to raw values
            raw_preds = outputs.cpu() * dataset.reg_std + dataset.reg_mean
            all_raw_preds.append(raw_preds.unsqueeze(-1))
            all_raw_values.append(batch['raw_value'].cpu().unsqueeze(-1))
            
            all_vials.extend(batch['vial_id'])
            torch.cuda.empty_cache()
    
    # Concatenate results
    preds = torch.cat(all_preds, dim=0).numpy()
    values = torch.cat(all_values, dim=0).numpy()
    assert len(all_raw_preds) == len(all_raw_values), f"Mismatched batch count: {len(all_raw_preds)} vs {len(all_raw_values)}"
    raw_preds = torch.cat(all_raw_preds, dim=0).numpy()
    raw_values = torch.cat(all_raw_values, dim=0).numpy()
    print(f"Raw Predictions Shape: {raw_preds.shape}, Raw Values Shape: {raw_values.shape}")
    
    # Calculate metrics on raw values
    mae = mean_absolute_error(raw_values, raw_preds)
    mse = np.mean((raw_values - raw_preds) ** 2)
    rmse = np.sqrt(mse)
    r2 = r2_score(raw_values, raw_preds)
    
    print(f"Regression Results:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Save metrics
    metrics_df = pd.DataFrame({
        'Metric': ['MAE', 'MSE', 'RMSE', 'R²'],
        'Value': [mae, mse, rmse, r2]
    })
    metrics_df.to_csv(os.path.join(out_dir, 'regression_metrics.csv'), index=False)
    
    # Calculate vial-wise metrics
    vial_metrics = defaultdict(lambda: {'preds': [], 'values': []})
    for vial_id, pred, value in zip(all_vials, raw_preds, raw_values):
        vial_metrics[vial_id]['preds'].append(pred)
        vial_metrics[vial_id]['values'].append(value)
    
    vial_results = []
    for vial_id, data in vial_metrics.items():
        vial_preds = np.array(data['preds'])
        vial_values = np.array(data['values'])
        vial_mae = mean_absolute_error(vial_values, vial_preds)
        vial_mse = np.mean((vial_values - vial_preds) ** 2)
        vial_rmse = np.sqrt(vial_mse)
        vial_r2 = r2_score(vial_values, vial_preds) if len(vial_values) > 1 else float('nan')
        
        vial_results.append({
            'vial_id': vial_id,
            'mae': vial_mae,
            'mse': vial_mse,
            'rmse': vial_rmse,
            'r2': vial_r2
        })
    
    # Save vial-wise metrics
    vial_metrics_df = pd.DataFrame(vial_results)
    vial_metrics_df.to_csv(os.path.join(out_dir, 'vialwise_regression_metrics.csv'), index=False)
    
    # Plot predictions vs actual values
    plt.figure(figsize=(10, 8))
    plt.scatter(raw_values, raw_preds, alpha=0.6)
    
    # Add perfect prediction line
    min_val = min(np.min(raw_values), np.min(raw_preds))
    max_val = max(np.max(raw_values), np.max(raw_preds))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title('Predicted vs Actual Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'pred_vs_actual.png'))
    plt.close()
    print(f"Saved Predicted vs Actual plot to {os.path.join(out_dir, 'pred_vs_actual.png')}")
    
    # Save raw predictions
    results_df = pd.DataFrame({
        'vial_id': all_vials,
        'true_value': raw_values.squeeze(-1),
        'predicted_value': raw_preds.squeeze,
        'error': (raw_values - raw_preds).squeeze(-1)
    })
    results_df.to_csv(os.path.join(out_dir, 'regression_predictions.csv'), index=False)
    print(f"Saved regression predictions to {os.path.join(out_dir, 'regression_predictions.csv')}")
    
    # Plot error distribution
    plt.figure(figsize=(10, 6))
    errors = raw_values - raw_preds
    plt.hist(errors, bins=30, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Error Distribution')
    plt.xlabel('Error (True - Predicted)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'error_distribution.png'))
    plt.close()
    print(f"Saved error distribution plot to {os.path.join(out_dir, 'error_distribution.png')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fluid Viscosity Classification/Regression Test")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory of the dataset")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the config file")
    parser.add_argument("--load_ckpt", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--task", type=str, default='classification', 
                        choices=['classification', 'regression'],
                        help="Task type (will be overridden by checkpoint if available)")
    
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