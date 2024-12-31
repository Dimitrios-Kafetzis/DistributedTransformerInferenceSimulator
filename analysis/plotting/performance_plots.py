# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Dimitrios Kafetzis
#
# This file is part of the Transformer Inference Simulator project.
# Licensed under the MIT License; you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#   https://opensource.org/licenses/MIT
#
# Author: Dimitrios Kafetzis (dimitrioskafetzis@gmail.com)
# File: analysis/plotting/performance_plots.py
# Description:
#   Provides plotting utilities for visualizing various performance metrics
#   (latency, throughput, timeline) related to distributed transformer
#   inference simulations.
#
# ---------------------------------------------------------------------------

"""
Contains functions and classes to create performance-related plots, such as
latency distribution plots, throughput comparisons, and latency evolution
over time. These visualizations aid in diagnosing performance bottlenecks
and evaluating the efficiency of inference algorithms.
"""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ..utils.data_loading import ensure_directory

def create_performance_plots(
    metrics: Dict,
    output_dir: Path
) -> List[Path]:
    """
    Create all performance-related plots.
    
    Args:
        metrics: Dictionary containing performance metrics
        output_dir: Directory to save plots
        
    Returns:
        List of paths to generated plot files
    """
    ensure_directory(output_dir)
    generated_plots = []
    
    # Generate latency distribution plot
    latency_plot = plot_latency_distribution(
        metrics['latency_data'],
        output_dir / 'latency_distribution.png'
    )
    generated_plots.append(latency_plot)
    
    # Generate throughput comparison
    throughput_plot = plot_throughput_comparison(
        metrics['throughput_data'],
        output_dir / 'throughput_comparison.png'
    )
    generated_plots.append(throughput_plot)
    
    # Generate latency over time
    timeline_plot = plot_latency_over_time(
        metrics['timeline_data'],
        output_dir / 'latency_timeline.png'
    )
    generated_plots.append(timeline_plot)
    
    return generated_plots

def plot_latency_distribution(
    latency_data: Dict,
    output_path: Path,
    figsize: tuple = (10, 6)
) -> Path:
    """
    Create violin plot of latency distribution per algorithm.
    
    Args:
        latency_data: Dictionary with algorithm names as keys and latency lists as values
        output_path: Path to save the plot
        figsize: Figure size tuple
        
    Returns:
        Path to saved plot
    """
    plt.figure(figsize=figsize)
    
    # Create violin plot
    data = [
        (algo, lat) 
        for algo, latencies in latency_data.items()
        for lat in latencies
    ]
    df = pd.DataFrame(data, columns=['Algorithm', 'Latency'])
    
    sns.violinplot(data=df, x='Algorithm', y='Latency')
    plt.title('Latency Distribution by Algorithm')
    plt.xlabel('Algorithm')
    plt.ylabel('Latency (ms)')
    plt.xticks(rotation=45)
    
    # Add median lines
    plt.axhline(y=df['Latency'].median(), color='r', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def plot_throughput_comparison(
    throughput_data: Dict,
    output_path: Path,
    figsize: tuple = (10, 6)
) -> Path:
    """
    Create bar plot comparing throughput across algorithms.
    
    Args:
        throughput_data: Dictionary with algorithm names as keys and throughput values
        output_path: Path to save the plot
        figsize: Figure size tuple
        
    Returns:
        Path to saved plot
    """
    plt.figure(figsize=figsize)
    
    algorithms = list(throughput_data.keys())
    throughputs = [throughput_data[algo] for algo in algorithms]
    
    # Create bar plot
    bars = plt.bar(algorithms, throughputs)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom')
    
    plt.title('Throughput Comparison')
    plt.xlabel('Algorithm')
    plt.ylabel('Throughput (tokens/sec)')
    plt.xticks(rotation=45)
    
    # Add mean line
    plt.axhline(y=np.mean(throughputs), color='r', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def plot_latency_over_time(
    timeline_data: Dict,
    output_path: Path,
    figsize: tuple = (12, 6)
) -> Path:
    """
    Create line plot showing latency evolution over time.
    
    Args:
        timeline_data: Dictionary with timestamps and latencies per algorithm
        output_path: Path to save the plot
        figsize: Figure size tuple
        
    Returns:
        Path to saved plot
    """
    plt.figure(figsize=figsize)
    
    # Plot line for each algorithm
    for algorithm, data in timeline_data.items():
        timestamps = data['timestamps']
        latencies = data['latencies']
        
        plt.plot(timestamps, latencies, label=algorithm, alpha=0.7)
        
        # Add trend line
        z = np.polyfit(range(len(timestamps)), latencies, 1)
        p = np.poly1d(z)
        plt.plot(timestamps, p(range(len(timestamps))), 
                linestyle='--', alpha=0.3)
    
    plt.title('Latency Evolution Over Time')
    plt.xlabel('Generation Step')
    plt.ylabel('Latency (ms)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def plot_performance_summary(
    performance_metrics: Dict,
    output_path: Path,
    figsize: tuple = (15, 10)
) -> Path:
    """
    Create a comprehensive performance summary plot.
    
    Args:
        performance_metrics: Dictionary containing all performance metrics
        output_path: Path to save the plot
        figsize: Figure size tuple
        
    Returns:
        Path to saved plot
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2)
    
    # Latency distribution
    ax1 = fig.add_subplot(gs[0, 0])
    df_latency = pd.DataFrame(performance_metrics['latency_data'])
    sns.violinplot(data=df_latency, ax=ax1)
    ax1.set_title('Latency Distribution')
    ax1.set_ylabel('Latency (ms)')
    
    # Throughput comparison
    ax2 = fig.add_subplot(gs[0, 1])
    throughput_data = performance_metrics['throughput_data']
    ax2.bar(throughput_data.keys(), throughput_data.values())
    ax2.set_title('Throughput Comparison')
    ax2.set_ylabel('Throughput (tokens/sec)')
    
    # Latency over time
    ax3 = fig.add_subplot(gs[1, :])
    timeline_data = performance_metrics['timeline_data']
    for algo, data in timeline_data.items():
        ax3.plot(data['timestamps'], data['latencies'], 
                label=algo, alpha=0.7)
    ax3.set_title('Latency Evolution')
    ax3.set_xlabel('Generation Step')
    ax3.set_ylabel('Latency (ms)')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path