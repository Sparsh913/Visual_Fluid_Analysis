import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from matplotlib.colors import LinearSegmentedColormap

def visualize_attention(model, dataset, output_dir, device, num_samples=5, sequence_length=10):
    """
    Visualize attention maps from transformer layers with probability values
    
    Args:
        model: The trained model
        dataset: The dataset to sample from
        output_dir: Directory to save visualizations
        device: Device to run the model on
        num_samples: Number of samples to visualize
        sequence_length: Length of each sequence
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a dataloader with batch size 1 for easier handling
    sampler = torch.utils.data.RandomSampler(dataset, replacement=False, num_samples=num_samples)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=sampler)
    
    # Set model to evaluation mode
    model.eval()
    
    # Define a custom colormap for attention probabilities (0 to 1)
    cmap = plt.cm.viridis
    
    # Determine task type
    task = 'classification' if hasattr(model, 'task') and model.task == 'classification' else 'regression'
    print(f"Visualizing attention for {task} task")
    
    with torch.no_grad():
        for i, data in enumerate(loader):
            # Move data to device
            keys = ['interfaces', 'robot', 'timestamps']
            if task == 'classification':
                keys.append('label')
            else:
                keys.append('value')
                keys.append('raw_value')
                
            batch = {k: data[k].to(device) if k in data and isinstance(data[k], torch.Tensor) else data[k] 
                     for k in keys}
            
            # Get model outputs with attention weights
            outputs, attention_weights = model(batch['interfaces'], batch['robot'], batch['timestamps'], return_attn=True)
            
            # Process prediction based on task
            if task == 'classification':
                pred = outputs.argmax(1).item()
                true_label = batch['label'].item()
                label_names = {0: 'low', 1: 'medium', 2: 'high'}
                sample_title = f"Vial: {data['vial_id'][0]}, True: {label_names[true_label]}, Pred: {label_names[pred]}"
            else:
                # For regression
                pred_value = outputs.item() * dataset.reg_std + dataset.reg_mean
                true_value = batch['raw_value'].item()
                sample_title = f"Vial: {data['vial_id'][0]}, True: {true_value:.4f}, Pred: {pred_value:.4f}"
            
            # Process robot data for visualization
            robot_data = batch['robot'][0].cpu().numpy()  # Shape: (T, 3)
            
            # Visualize attention maps for each layer and head
            for layer_idx, layer_attention in enumerate(attention_weights):
                layer_attention = layer_attention[0]  # Remove batch dimension
                num_heads = layer_attention.shape[0]
                
                # Apply softmax to get proper attention probabilities
                layer_attention_prob = torch.nn.functional.softmax(layer_attention, dim=-1)
                
                # Create a figure with subplots for each attention head
                fig, axes = plt.subplots(num_heads + 1, 1, figsize=(12, 3*num_heads + 3))
                fig.suptitle(f"Sample {i+1}: {sample_title}", fontsize=16)
                
                # Handle case when there's only one head
                if num_heads == 1:
                    axes = [axes[0], axes[1]]
                
                # First plot: robot motion data
                ax = axes[0]
                ax.plot(range(sequence_length), robot_data[:, 0], 'r-', label='Angle (deg)')
                ax.plot(range(sequence_length), robot_data[:, 1], 'g-', label='Speed (deg/s)')
                ax.plot(range(sequence_length), robot_data[:, 2], 'b-', label='Accel (deg/s²)')
                ax.set_title("Robot Motion Data")
                ax.set_xlabel("Time Step")
                ax.set_ylabel("Value")
                ax.legend()
                ax.grid(True)
                
                # Then plot attention maps for each head
                for head_idx in range(num_heads):
                    ax = axes[head_idx + 1]
                    attn = layer_attention_prob[head_idx].cpu().numpy()
                    im = ax.imshow(attn, cmap=cmap, vmin=0, vmax=1)
                    ax.set_title(f"Layer {layer_idx+1}, Head {head_idx+1} Attention")
                    ax.set_xlabel("Target Sequence Position")
                    ax.set_ylabel("Source Sequence Position")
                    
                    # Add probability values as text in each cell
                    # Only add text for larger values to avoid cluttering
                    for src_idx in range(attn.shape[0]):
                        for tgt_idx in range(attn.shape[1]):
                            # Format the probability value
                            val = attn[src_idx, tgt_idx]
                            text = f"{val:.2f}" if val >= 0.1 else ""
                            
                            # Choose text color with better contrast
                            # White text on dark background, black text on light background
                            # with more distinct threshold for better visibility
                            text_color = 'white' if val > 0.3 else 'black'
                            
                            # Only add text if value is significant
                            if text:
                                ax.text(tgt_idx, src_idx, text, ha="center", va="center", 
                                        color=text_color, fontsize=6, fontweight='bold')
                    
                    # Add colorbar
                    plt.colorbar(im, ax=ax)
                
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.savefig(os.path.join(output_dir, f"sample_{i+1}_layer_{layer_idx+1}_attention.png"), dpi=200)
                plt.close(fig)
                
            # Also create a figure with just the combined attention across all heads
            fig, axes = plt.subplots(len(attention_weights) + 1, 1, figsize=(12, 4*len(attention_weights) + 3))
            fig.suptitle(f"Sample {i+1}: {sample_title} - Combined Attention", fontsize=16)
            
            # Handle case when there's only one layer
            if len(attention_weights) == 1:
                axes = [axes[0], axes[1]]
            
            # First plot: robot motion data
            ax = axes[0]
            ax.plot(range(sequence_length), robot_data[:, 0], 'r-', label='Angle (deg)')
            ax.plot(range(sequence_length), robot_data[:, 1], 'g-', label='Speed (deg/s)')
            ax.plot(range(sequence_length), robot_data[:, 2], 'b-', label='Accel (deg/s²)')
            ax.set_title("Robot Motion Data")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True)
            
            # Then plot combined attention for each layer
            for layer_idx, layer_attention in enumerate(attention_weights):
                ax = axes[layer_idx + 1]
                # Apply softmax to get proper attention probabilities
                layer_attention_prob = torch.nn.functional.softmax(layer_attention[0], dim=-1)
                # Average attention across heads
                attn = layer_attention_prob.mean(dim=0).cpu().numpy()
                im = ax.imshow(attn, cmap=cmap, vmin=0, vmax=1)
                ax.set_title(f"Layer {layer_idx+1} Combined Attention")
                ax.set_xlabel("Target Sequence Position")
                ax.set_ylabel("Source Sequence Position")
                
                # Add probability values as text in each cell
                for src_idx in range(attn.shape[0]):
                    for tgt_idx in range(attn.shape[1]):
                        # Format the probability value
                        val = attn[src_idx, tgt_idx]
                        text = f"{val:.2f}" if val >= 0.05 else ""
                        
                        # Choose text color with better contrast
                        text_color = 'white' if val > 0.3 else 'black'
                        
                        # Only add text if value is significant
                        if text:
                            ax.text(tgt_idx, src_idx, text, ha="center", va="center", 
                                    color=text_color, fontsize=7, fontweight='bold')
                
                # Add colorbar
                plt.colorbar(im, ax=ax)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(os.path.join(output_dir, f"sample_{i+1}_combined_attention.png"), dpi=200)
            plt.close(fig)
            
            # Also create a plot of just the interface positions
            if 'interfaces' in batch:
                fig, ax = plt.subplots(figsize=(10, 5))
                interface_data = batch['interfaces'][0].cpu().numpy()  # Shape: (T, num_points)
                
                # Plot each timestep's interface as a line
                for t in range(interface_data.shape[0]):
                    points_x = np.arange(interface_data.shape[1])
                    ax.plot(points_x, interface_data[t], 
                            label=f"t={t}", alpha=0.7, linewidth=2)
                
                ax.set_title(f"Fluid Interface Positions - {data['vial_id'][0]}")
                ax.set_xlabel("Horizontal Position (points)")
                ax.set_ylabel("Normalized Vertical Position")
                ax.set_ylim(0, 1.1)  # Normalized coordinates
                ax.grid(True)
                ax.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"sample_{i+1}_interfaces.png"), dpi=200)
                plt.close(fig)
            
    print(f"Attention visualizations saved to {output_dir}")