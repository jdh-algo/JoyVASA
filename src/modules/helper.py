
import asyncio
import websockets
import json
import torch
import torch.nn.functional as F
from logger import logger, TorchDebugger
from typing import *
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
from logger import logger

import torch
import numpy as np
import matplotlib.pyplot as plt

def detailed_model_output_visualization(target, output, save_dir='visualizations/model_outputs', iteration=None):
    """
    Enhanced visualization of model outputs with detailed error metrics
    
    Args:
        target: Target tensor 
        output: Predicted output tensor
        save_dir: Directory to save visualizations
        iteration: Current training iteration (for filename)
    """
    from pathlib import Path
    
    # Ensure save directory exists
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Convert to numpy arrays and move to CPU if needed
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    if torch.is_tensor(output):
        output = output.detach().cpu().numpy()
    
    # Take first sample from batch
    target_sample = target[0]
    output_sample = output[0]
    
    # Compute error metrics
    difference = output_sample - target_sample
    max_abs_diff = np.max(np.abs(difference))
    mean_abs_diff = np.mean(np.abs(difference))
    rmse = np.sqrt(np.mean(difference**2))
    
    # Create a figure with four subplots
    fig, axs = plt.subplots(2, 2, figsize=(20, 15))
    
    # Plot target 
    im1 = axs[0, 0].imshow(target_sample.reshape(1, -1), 
                            aspect='auto', 
                            cmap='RdBu_r',
                            vmin=target_sample.min(), 
                            vmax=target_sample.max())
    axs[0, 0].set_title('Target Output')
    plt.colorbar(im1, ax=axs[0, 0])
    axs[0, 0].set_yticks([])
    
    # Plot predicted output
    im2 = axs[0, 1].imshow(output_sample.reshape(1, -1), 
                            aspect='auto', 
                            cmap='RdBu_r',
                            vmin=output_sample.min(), 
                            vmax=output_sample.max())
    axs[0, 1].set_title('Predicted Output')
    plt.colorbar(im2, ax=axs[0, 1])
    axs[0, 1].set_yticks([])
    
    # Plot difference
    max_diff = np.abs(difference).max()
    im3 = axs[1, 0].imshow(difference.reshape(1, -1),
                            aspect='auto', 
                            cmap='RdBu_r',
                            vmin=-max_diff, 
                            vmax=max_diff)
    axs[1, 0].set_title('Difference (Predicted - Target)')
    plt.colorbar(im3, ax=axs[1, 0])
    axs[1, 0].set_yticks([])
    
    # Plot error distribution
    axs[1, 1].hist(difference.flatten(), bins=50, color='skyblue', edgecolor='black')
    axs[1, 1].set_title('Error Distribution')
    axs[1, 1].set_xlabel('Error Value')
    axs[1, 1].set_ylabel('Frequency')
    
    # Add metrics text
    metrics_text = (
        f"Max Absolute Difference: {max_abs_diff:.6f}\n"
        f"Mean Absolute Difference: {mean_abs_diff:.6f}\n"
        f"Root Mean Square Error: {rmse:.6f}"
    )
    fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.5))
    
    # Add titles and adjust layout
    if iteration is not None:
        plt.suptitle(f'Model Output Comparison - Iteration {iteration}', fontsize=16)
    plt.tight_layout()
    
    # Save the figure
    filename = f'detailed_model_output_comparison_{iteration if iteration else "latest"}.png'
    plt.savefig(Path(save_dir) / filename, bbox_inches='tight', dpi=300)
    plt.close()

    # Log metrics
    print(f"Iteration {iteration or 'N/A'}:")
    print(f"Max Absolute Difference: {max_abs_diff}")
    print(f"Mean Absolute Difference: {mean_abs_diff}")
    print(f"Root Mean Square Error: {rmse}")
    
    return {
        'max_abs_diff': max_abs_diff,
        'mean_abs_diff': mean_abs_diff,
        'rmse': rmse
    }
def visualize_model_outputs(target, output, save_dir='visualizations/model_outputs', iteration=None):
    """
    Visualize comparison between target and model outputs.
    
    Args:
        target: Target tensor 
        output: Predicted output tensor
        save_dir: Directory to save visualizations
        iteration: Current training iteration (for filename)
    """
    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Convert to numpy arrays and move to CPU if needed
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    if torch.is_tensor(output):
        output = output.detach().cpu().numpy()
    
    # Take first sample from batch
    target_sample = target[0]
    output_sample = output[0]
    
    # Create a figure with three subplots side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot target 
    im1 = ax1.imshow(target_sample.reshape(1, -1), 
                     aspect='auto', 
                     cmap='RdBu_r',
                     vmin=target_sample.min(), 
                     vmax=target_sample.max())
    ax1.set_title('Target Output')
    plt.colorbar(im1, ax=ax1)
    ax1.set_yticks([])
    
    # Plot predicted output
    im2 = ax2.imshow(output_sample.reshape(1, -1), 
                     aspect='auto', 
                     cmap='RdBu_r',
                     vmin=output_sample.min(), 
                     vmax=output_sample.max())
    ax2.set_title('Predicted Output')
    plt.colorbar(im2, ax=ax2)
    ax2.set_yticks([])
    
    # Plot difference
    difference = output_sample - target_sample
    max_diff = np.abs(difference).max()
    im3 = ax3.imshow(difference.reshape(1, -1),
                     aspect='auto', 
                     cmap='RdBu_r',
                     vmin=-max_diff, 
                     vmax=max_diff)
    ax3.set_title('Difference (Predicted - Target)')
    plt.colorbar(im3, ax=ax3)
    ax3.set_yticks([])
    
    # Add titles and adjust layout
    if iteration is not None:
        plt.suptitle(f'Model Output Comparison - Iteration {iteration}')
    plt.tight_layout()
    
    # Save the figure
    filename = f'model_output_comparison_{iteration if iteration else "latest"}.png'
    plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()