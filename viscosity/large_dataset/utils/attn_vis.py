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
    
    with torch.no_grad():
        for i, data in enumerate(loader):
            # Move data to device
            mask_seq, robot_seq, timestamps, label, vial_id = [
                data[k].to(device) if isinstance(data[k], torch.Tensor) else data[k] 
                for k in ['masks', 'robot', 'timestamps', 'label', 'vial_id']
            ]
            
            # Get model outputs with attention weights
            outputs, attention_weights, _ = model(mask_seq, robot_seq, timestamps, return_attn=True)
            pred = outputs.argmax(1).item()
            true_label = label.item()
            
            label_names = {0: 'low', 1: 'medium', 2: 'high'}
            sample_title = f"Vial: {vial_id[0]}, True: {label_names[true_label]}, Pred: {label_names[pred]}"
            
            # Process robot data for visualization
            robot_data = robot_seq[0].cpu().numpy()  # Shape: (T, 3)
            
            # Visualize attention maps for each layer and head
            for layer_idx, layer_attention in enumerate(attention_weights):
                layer_attention = layer_attention[0]  # Remove batch dimension
                num_heads = layer_attention.shape[0]
                
                # Apply softmax to get proper attention probabilities
                layer_attention_prob = torch.nn.functional.softmax(layer_attention, dim=-1)
                
                # Create a figure with subplots for each attention head
                fig, axes = plt.subplots(num_heads + 1, 1, figsize=(12, 3*num_heads + 3))
                fig.suptitle(f"Sample {i+1}: {sample_title}", fontsize=16)
                
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
                    for src_idx in range(attn.shape[0]):
                        for tgt_idx in range(attn.shape[1]):
                            # Format the probability value
                            text = f"{attn[src_idx, tgt_idx]:.2f}"
                            # Choose text color based on background darkness
                            text_color = 'white' if attn[src_idx, tgt_idx] > 0.5 else 'black'
                            # Add text to the cell
                            ax.text(tgt_idx, src_idx, text, ha="center", va="center", 
                                    color=text_color, fontsize=8)
                    
                    # Add colorbar
                    plt.colorbar(im, ax=ax)
                
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.savefig(os.path.join(output_dir, f"sample_{i+1}_layer_{layer_idx+1}_attention.png"), dpi=150)
                plt.close(fig)
                
            # Also create a figure with just the combined attention across all heads
            fig, axes = plt.subplots(len(attention_weights) + 1, 1, figsize=(12, 4*len(attention_weights) + 3))
            fig.suptitle(f"Sample {i+1}: {sample_title} - Combined Attention", fontsize=16)
            
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
                        text = f"{attn[src_idx, tgt_idx]:.2f}"
                        # Choose text color based on background darkness
                        text_color = 'white' if attn[src_idx, tgt_idx] > 0.5 else 'black'
                        # Add text to the cell
                        ax.text(tgt_idx, src_idx, text, ha="center", va="center", 
                                color=text_color, fontsize=8)
                
                # Add colorbar
                plt.colorbar(im, ax=ax)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(os.path.join(output_dir, f"sample_{i+1}_combined_attention.png"), dpi=150)
            plt.close(fig)
            
    print(f"Attention visualizations saved to {output_dir}")