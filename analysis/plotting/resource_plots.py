from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ..utils.data_loading import ensure_directory

def create_resource_plots(
    metrics: Dict,
    output_dir: Path
) -> List[Path]:
    """
    Create all resource utilization plots.
    
    Args:
        metrics: Dictionary containing resource metrics
        output_dir: Directory to save plots
        
    Returns:
        List of paths to generated plot files
    """
    ensure_directory(output_dir)
    generated_plots = []
    
    # Memory utilization plots
    memory_plot = plot_memory_utilization(
        metrics['memory_data'],
        output_dir / 'memory_utilization.png'
    )
    generated_plots.append(memory_plot)
    
    # CPU utilization plots
    cpu_plot = plot_cpu_utilization(
        metrics['cpu_data'],
        output_dir / 'cpu_utilization.png'
    )
    generated_plots.append(cpu_plot)
    
    # Resource efficiency plots
    efficiency_plot = plot_resource_efficiency(
        metrics['efficiency_data'],
        output_dir / 'resource_efficiency.png'
    )
    generated_plots.append(efficiency_plot)
    
    return generated_plots

def plot_memory_utilization(
    memory_data: Dict,
    output_path: Path,
    figsize: tuple = (12, 6)
) -> Path:
    """
    Create line plot of memory utilization over time.
    
    Args:
        memory_data: Dictionary with device IDs and memory utilization data
        output_path: Path to save the plot
        figsize: Figure size tuple
        
    Returns:
        Path to saved plot
    """
    plt.figure(figsize=figsize)
    
    # Plot line for each device
    for device_id, data in memory_data.items():
        timestamps = data['timestamps']
        utilization = data['utilization']
        
        plt.plot(timestamps, utilization, 
                label=f'Device {device_id}', alpha=0.7)
        
        # Add capacity line
        capacity = data['capacity']
        plt.axhline(y=capacity, linestyle='--', alpha=0.3)
    
    plt.title('Memory Utilization Over Time')
    plt.xlabel('Generation Step')
    plt.ylabel('Memory Usage (GB)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add warning threshold
    plt.axhline(y=0.9, color='r', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def plot_cpu_utilization(
    cpu_data: Dict,
    output_path: Path,
    figsize: tuple = (12, 6)
) -> Path:
    """
    Create line plot of CPU utilization over time.
    
    Args:
        cpu_data: Dictionary with device IDs and CPU utilization data
        output_path: Path to save the plot
        figsize: Figure size tuple
        
    Returns:
        Path to saved plot
    """
    plt.figure(figsize=figsize)
    
    # Plot line for each device
    for device_id, data in cpu_data.items():
        timestamps = data['timestamps']
        utilization = data['utilization']
        
        plt.plot(timestamps, utilization, 
                label=f'Device {device_id}', alpha=0.7)
        
        # Add moving average
        window = 5
        moving_avg = np.convolve(utilization, 
                               np.ones(window)/window, 
                               mode='valid')
        plt.plot(timestamps[window-1:], moving_avg, 
                linestyle='--', alpha=0.3)
    
    plt.title('CPU Utilization Over Time')
    plt.xlabel('Generation Step')
    plt.ylabel('CPU Usage (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def plot_resource_efficiency(
    efficiency_data: Dict,
    output_path: Path,
    figsize: tuple = (10, 6)
) -> Path:
    """
    Create bar plot of resource efficiency metrics.
    
    Args:
        efficiency_data: Dictionary with efficiency metrics per algorithm
        output_path: Path to save the plot
        figsize: Figure size tuple
        
    Returns:
        Path to saved plot
    """
    plt.figure(figsize=figsize)
    
    algorithms = list(efficiency_data.keys())
    metrics = list(efficiency_data[algorithms[0]].keys())
    
    # Create grouped bar plot
    x = np.arange(len(algorithms))
    width = 0.35
    multiplier = 0
    
    for metric in metrics:
        offset = width * multiplier
        values = [efficiency_data[algo][metric] for algo in algorithms]
        
        plt.bar(x + offset, values, width, label=metric)
        multiplier += 1
    
    plt.title('Resource Efficiency Comparison')
    plt.xlabel('Algorithm')
    plt.ylabel('Efficiency Score')
    plt.xticks(x + width/2, algorithms, rotation=45)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def plot_resource_summary(
    resource_metrics: Dict,
    output_path: Path,
    figsize: tuple = (15, 10)
) -> Path:
    """
    Create a comprehensive resource usage summary plot.
    
    Args:
        resource_metrics: Dictionary containing all resource metrics
        output_path: Path to save the plot
        figsize: Figure size tuple
        
    Returns:
        Path to saved plot
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2)
    
    # Memory utilization
    ax1 = fig.add_subplot(gs[0, 0])
    memory_data = resource_metrics['memory_data']
    for device_id, data in memory_data.items():
        ax1.plot(data['timestamps'], data['utilization'], 
                label=f'Device {device_id}')
    ax1.set_title('Memory Utilization')
    ax1.set_ylabel('Memory Usage (GB)')
    ax1.legend()
    
    # CPU utilization
    ax2 = fig.add_subplot(gs[0, 1])
    cpu_data = resource_metrics['cpu_data']
    for device_id, data in cpu_data.items():
        ax2.plot(data['timestamps'], data['utilization'], 
                label=f'Device {device_id}')
    ax2.set_title('CPU Utilization')
    ax2.set_ylabel('CPU Usage (%)')
    ax2.legend()
    
    # Resource efficiency
    ax3 = fig.add_subplot(gs[1, :])
    efficiency_data = resource_metrics['efficiency_data']
    
    algorithms = list(efficiency_data.keys())
    metrics = list(efficiency_data[algorithms[0]].keys())
    x = np.arange(len(algorithms))
    width = 0.35
    multiplier = 0
    
    for metric in metrics:
        offset = width * multiplier
        values = [efficiency_data[algo][metric] for algo in algorithms]
        ax3.bar(x + offset, values, width, label=metric)
        multiplier += 1
        
    ax3.set_title('Resource Efficiency')
    ax3.set_xlabel('Algorithm')
    ax3.set_ylabel('Efficiency Score')
    ax3.set_xticks(x + width/2)
    ax3.set_xticklabels(algorithms, rotation=45)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path