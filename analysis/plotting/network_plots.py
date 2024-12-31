from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from ..utils.data_loading import ensure_directory

def create_network_plots(
    metrics: Dict,
    output_dir: Path
) -> List[Path]:
    """
    Create all network-related plots.
    
    Args:
        metrics: Dictionary containing network metrics
        output_dir: Directory to save plots
        
    Returns:
        List of paths to generated plot files
    """
    ensure_directory(output_dir)
    generated_plots = []
    
    # Network topology plot
    topology_plot = plot_network_topology(
        metrics['topology_data'],
        output_dir / 'network_topology.png'
    )
    generated_plots.append(topology_plot)
    
    # Communication overhead plot
    comm_plot = plot_communication_overhead(
        metrics['communication_data'],
        output_dir / 'communication_overhead.png'
    )
    generated_plots.append(comm_plot)
    
    # Bandwidth utilization plot
    bandwidth_plot = plot_bandwidth_utilization(
        metrics['bandwidth_data'],
        output_dir / 'bandwidth_utilization.png'
    )
    generated_plots.append(bandwidth_plot)
    
    return generated_plots

def plot_network_topology(
    topology_data: Dict,
    output_path: Path,
    figsize: tuple = (12, 8)
) -> Path:
    """
    Create network topology visualization.
    
    Args:
        topology_data: Dictionary containing network topology information
        output_path: Path to save the plot
        figsize: Figure size tuple
        
    Returns:
        Path to saved plot
    """
    plt.figure(figsize=figsize)
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes
    for node_id, node_data in topology_data['nodes'].items():
        G.add_node(node_id, **node_data)
        
    # Add edges
    for edge in topology_data['edges']:
        G.add_edge(edge['source'], edge['target'], 
                  bandwidth=edge['bandwidth'])
    
    # Set up layout
    pos = nx.spring_layout(G)
    
    # Draw nodes
    node_colors = [G.nodes[node].get('type', 'default') 
                  for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=1000, alpha=0.7)
    
    # Draw edges with width proportional to bandwidth
    edge_widths = [G[u][v]['bandwidth'] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5)
    
    # Add labels
    nx.draw_networkx_labels(G, pos)
    
    plt.title('Network Topology')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def plot_communication_overhead(
    communication_data: Dict,
    output_path: Path,
    figsize: tuple = (10, 6)
) -> Path:
    """
    Create plot showing communication overhead between components.
    
    Args:
        communication_data: Dictionary with communication metrics
        output_path: Path to save the plot
        figsize: Figure size tuple
        
    Returns:
        Path to saved plot
    """
    plt.figure(figsize=figsize)
    
    algorithms = list(communication_data.keys())
    metrics = ['data_transferred', 'num_messages']
    
    # Create grouped bar plot
    x = np.arange(len(algorithms))
    width = 0.35
    multiplier = 0
    
    for metric in metrics:
        offset = width * multiplier
        values = [communication_data[algo][metric] 
                 for algo in algorithms]
        
        plt.bar(x + offset, values, width, label=metric)
        multiplier += 1
    
    plt.title('Communication Overhead by Algorithm')
    plt.xlabel('Algorithm')
    plt.ylabel('Communication Metrics')
    plt.xticks(x + width/2, algorithms, rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def plot_bandwidth_utilization(
    bandwidth_data: Dict,
    output_path: Path,
    figsize: tuple = (12, 6)
) -> Path:
    """
    Create plot showing bandwidth utilization over time.
    
    Args:
        bandwidth_data: Dictionary with bandwidth utilization data
        output_path: Path to save the plot
        figsize: Figure size tuple
        
    Returns:
        Path to saved plot
    """
    plt.figure(figsize=figsize)
    
    # Plot line for each link
    for link_id, data in bandwidth_data.items():
        timestamps = data['timestamps']
        utilization = data['utilization']
        
        plt.plot(timestamps, utilization, 
                label=f'Link {link_id}', alpha=0.7)
        
        # Add capacity threshold
        capacity = data['capacity']
        plt.axhline(y=capacity, linestyle='--', alpha=0.3)
    
    plt.title('Bandwidth Utilization Over Time')
    plt.xlabel('Generation Step')
    plt.ylabel('Bandwidth Usage (Gbps)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def plot_network_summary(
    network_metrics: Dict,
    output_path: Path,
    figsize: tuple = (15, 10)
) -> Path:
    """
    Create comprehensive network metrics summary plot.
    
    Args:
        network_metrics: Dictionary containing all network metrics
        output_path: Path to save the plot
        figsize: Figure size tuple
        
    Returns:
        Path to saved plot
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2)
    
    # Network topology
    ax1 = fig.add_subplot(gs[0, 0])
    G = nx.Graph()
    topology_data = network_metrics['topology_data']
    for node_id, node_data in topology_data['nodes'].items():
        G.add_node(node_id, **node_data)
    for edge in topology_data['edges']:
        G.add_edge(edge['source'], edge['target'])
    nx.draw(G, ax=ax1, with_labels=True)
    ax1.set_title('Network Topology')
    
    # Communication overhead
    ax2 = fig.add_subplot(gs[0, 1])
    comm_data = network_metrics['communication_data']
    algorithms = list(comm_data.keys())
    data_transferred = [comm_data[algo]['data_transferred'] 
                       for algo in algorithms]
    ax2.bar(algorithms, data_transferred)
    ax2.set_title('Data Transferred')
    ax2.set_ylabel('Data (GB)')
    plt.xticks(rotation=45)
    
    # Bandwidth utilization
    ax3 = fig.add_subplot(gs[1, :])
    bandwidth_data = network_metrics['bandwidth_data']
    for link_id, data in bandwidth_data.items():
        ax3.plot(data['timestamps'], data['utilization'], 
                label=f'Link {link_id}')
    ax3.set_title('Bandwidth Utilization')
    ax3.set_xlabel('Generation Step')
    ax3.set_ylabel('Bandwidth Usage (Gbps)')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path