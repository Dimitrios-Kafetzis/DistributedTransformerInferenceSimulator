from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ..utils.data_loading import ensure_directory

def create_comparison_plots(
    metrics: Dict,
    output_dir: Path
) -> List[Path]:
    """
    Create all comparison plots between algorithms and scenarios.
    
    Args:
        metrics: Dictionary containing all metrics
        output_dir: Directory to save plots
        
    Returns:
        List of paths to generated plot files
    """
    ensure_directory(output_dir)
    generated_plots = []
    
    # Algorithm comparison plots
    algo_plot = plot_algorithm_comparison(
        metrics['algorithm_comparison'],
        output_dir / 'algorithm_comparison.png'
    )
    generated_plots.append(algo_plot)
    
    # Scenario comparison plots
    scenario_plot = plot_scenario_comparison(
        metrics['scenario_comparison'],
        output_dir / 'scenario_comparison.png'
    )
    generated_plots.append(scenario_plot)
    
    # Scaling behavior plots
    scaling_plot = plot_scaling_behavior(
        metrics['scaling_data'],
        output_dir / 'scaling_behavior.png'
    )
    generated_plots.append(scaling_plot)
    
    # Create summary plot
    summary_plot = plot_comparison_summary(
        metrics,
        output_dir / 'comparison_summary.png'
    )
    generated_plots.append(summary_plot)
    
    return generated_plots

def plot_algorithm_comparison(
    comparison_data: Dict,
    output_path: Path,
    figsize: tuple = (12, 8)
) -> Path:
    """
    Create detailed algorithm comparison visualization.
    
    Args:
        comparison_data: Dictionary with algorithm comparison metrics
        output_path: Path to save the plot
        figsize: Figure size tuple
        
    Returns:
        Path to saved plot
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Latency comparison
    sns.boxplot(data=pd.DataFrame(comparison_data['latency']), 
                ax=axes[0, 0])
    axes[0, 0].set_title('Latency Distribution')
    axes[0, 0].set_xlabel('Algorithm')
    axes[0, 0].set_ylabel('Latency (ms)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Resource utilization
    resource_data = pd.DataFrame(comparison_data['resource_utilization'])
    resource_data.plot(kind='bar', ax=axes[0, 1])
    axes[0, 1].set_title('Resource Utilization')
    axes[0, 1].set_xlabel('Algorithm')
    axes[0, 1].set_ylabel('Utilization (%)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Communication overhead
    comm_data = comparison_data['communication_overhead']
    x = range(len(comm_data))
    axes[1, 0].bar(x, [d['data_transferred'] for d in comm_data.values()],
                   alpha=0.7, label='Data Transferred')
    axes[1, 0].bar(x, [d['num_messages'] for d in comm_data.values()],
                   alpha=0.7, label='Number of Messages')
    axes[1, 0].set_title('Communication Overhead')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(comm_data.keys(), rotation=45)
    axes[1, 0].legend()
    
    # Efficiency score
    efficiency_data = comparison_data['efficiency_score']
    axes[1, 1].bar(efficiency_data.keys(), efficiency_data.values())
    axes[1, 1].set_title('Overall Efficiency Score')
    axes[1, 1].set_xlabel('Algorithm')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def plot_scenario_comparison(
    scenario_data: Dict,
    output_path: Path,
    figsize: tuple = (15, 6)
) -> Path:
    """
    Create comparison visualization across different scenarios.
    
    Args:
        scenario_data: Dictionary with scenario comparison data
        output_path: Path to save the plot
        figsize: Figure size tuple
        
    Returns:
        Path to saved plot
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Edge cluster performance
    edge_data = pd.DataFrame(scenario_data['edge_cluster'])
    sns.boxplot(data=edge_data, ax=axes[0])
    axes[0].set_title('Edge Cluster')
    axes[0].set_xlabel('Metric')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Distributed edge performance
    dist_data = pd.DataFrame(scenario_data['distributed_edge'])
    sns.boxplot(data=dist_data, ax=axes[1])
    axes[1].set_title('Distributed Edge')
    axes[1].set_xlabel('Metric')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Hybrid cloud-edge performance
    hybrid_data = pd.DataFrame(scenario_data['hybrid_cloud_edge'])
    sns.boxplot(data=hybrid_data, ax=axes[2])
    axes[2].set_title('Hybrid Cloud-Edge')
    axes[2].set_xlabel('Metric')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def plot_scaling_behavior(
    scaling_data: Dict,
    output_path: Path,
    figsize: tuple = (12, 6)
) -> Path:
    """
    Create visualization of scaling behavior across configurations.
    
    Args:
        scaling_data: Dictionary with scaling behavior metrics
        output_path: Path to save the plot
        figsize: Figure size tuple
        
    Returns:
        Path to saved plot
    """
    plt.figure(figsize=figsize)
    
    # Plot scaling curves for each algorithm
    for algo, data in scaling_data.items():
        sequence_lengths = data['sequence_lengths']
        latencies = data['latencies']
        
        plt.plot(sequence_lengths, latencies, 
                marker='o', label=algo, alpha=0.7)
        
        # Add trend line
        z = np.polyfit(sequence_lengths, latencies, 2)
        p = np.poly1d(z)
        plt.plot(sequence_lengths, p(sequence_lengths), 
                linestyle='--', alpha=0.3)
    
    plt.title('Scaling Behavior Analysis')
    plt.xlabel('Sequence Length')
    plt.ylabel('Latency (ms)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def plot_comparison_summary(
    all_metrics: Dict,
    output_path: Path,
    figsize: tuple = (15, 10)
) -> Path:
    """
    Create comprehensive comparison summary plot.
    
    Args:
        all_metrics: Dictionary containing all comparison metrics
        output_path: Path to save the plot
        figsize: Figure size tuple
        
    Returns:
        Path to saved plot
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3)
    
    # Latency comparison
    ax1 = fig.add_subplot(gs[0, 0])
    latency_data = pd.DataFrame(all_metrics['algorithm_comparison']['latency'])
    sns.boxplot(data=latency_data, ax=ax1)
    ax1.set_title('Latency Distribution')
    ax1.tick_params(axis='x', rotation=45)
    
    # Resource utilization
    ax2 = fig.add_subplot(gs[0, 1])
    resource_data = pd.DataFrame(
        all_metrics['algorithm_comparison']['resource_utilization']
    )
    resource_data.plot(kind='bar', ax=ax2)
    ax2.set_title('Resource Utilization')
    ax2.tick_params(axis='x', rotation=45)
    
    # Communication overhead
    ax3 = fig.add_subplot(gs[0, 2])
    comm_data = all_metrics['algorithm_comparison']['communication_overhead']
    x = range(len(comm_data))
    ax3.bar(x, [d['data_transferred'] for d in comm_data.values()],
            alpha=0.7, label='Data Transferred')
    ax3.set_title('Communication Overhead')
    ax3.set_xticks(x)
    ax3.set_xticklabels(comm_data.keys(), rotation=45)
    
    # Scaling behavior
    ax4 = fig.add_subplot(gs[1, :])
    scaling_data = all_metrics['scaling_data']
    for algo, data in scaling_data.items():
        ax4.plot(data['sequence_lengths'], data['latencies'],
                marker='o', label=algo, alpha=0.7)
    ax4.set_title('Scaling Behavior')
    ax4.set_xlabel('Sequence Length')
    ax4.set_ylabel('Latency (ms)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def plot_statistical_comparison(
    statistical_data: Dict,
    output_path: Path,
    figsize: tuple = (10, 6)
) -> Path:
    """
    Create visualization of statistical comparison results.
    
    Args:
        statistical_data: Dictionary with statistical test results
        output_path: Path to save the plot
        figsize: Figure size tuple
        
    Returns:
        Path to saved plot
    """
    plt.figure(figsize=figsize)
    
    # Create heatmap of p-values
    p_values = pd.DataFrame(statistical_data['p_values'])
    sns.heatmap(p_values, annot=True, cmap='RdYlGn_r',
                vmin=0, vmax=0.05)
    
    plt.title('Statistical Significance (p-values)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path