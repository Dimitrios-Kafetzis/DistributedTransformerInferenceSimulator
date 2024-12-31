from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import pandas as pd
from scipy import stats
import networkx as nx
from collections import defaultdict

def calculate_network_metrics(data: Dict) -> Dict:
    """
    Calculate comprehensive network metrics from experimental data.
    
    Args:
        data: Dictionary containing network measurements
        
    Returns:
        Dictionary containing calculated network metrics
    """
    metrics = {}
    
    # Calculate communication overhead
    metrics['communication'] = calculate_communication_overhead(
        data['communication_data']
    )
    
    # Calculate bandwidth utilization
    metrics['bandwidth'] = calculate_bandwidth_utilization(
        data['bandwidth_data']
    )
    
    # Analyze network patterns
    metrics['patterns'] = analyze_network_patterns(
        data['network_data']
    )
    
    return metrics

def calculate_communication_overhead(data: Dict) -> Dict:
    """
    Calculate communication overhead metrics.
    
    Args:
        data: Dictionary containing communication measurements
        
    Returns:
        Dictionary containing communication metrics
    """
    comm_metrics = defaultdict(dict)
    
    for algo_id, measurements in data.items():
        transfers = np.array(measurements['data_transfers'])
        messages = np.array(measurements['message_counts'])
        
        # Basic transfer metrics
        comm_metrics[algo_id].update({
            'total_data_transferred': np.sum(transfers),
            'mean_transfer_size': np.mean(transfers),
            'total_messages': np.sum(messages),
            'mean_messages_per_step': np.mean(messages)
        })
        
        # Calculate transfer patterns
        patterns = analyze_transfer_patterns(transfers, messages)
        comm_metrics[algo_id]['patterns'] = patterns
        
        # Communication efficiency
        if 'computation_time' in measurements:
            efficiency = calculate_communication_efficiency(
                transfers,
                messages,
                measurements['computation_time']
            )
            comm_metrics[algo_id]['efficiency'] = efficiency
            
    return dict(comm_metrics)

def calculate_bandwidth_utilization(data: Dict) -> Dict:
    """
    Calculate bandwidth utilization metrics.
    
    Args:
        data: Dictionary containing bandwidth measurements
        
    Returns:
        Dictionary containing bandwidth metrics
    """
    bandwidth_metrics = defaultdict(dict)
    
    for link_id, measurements in data.items():
        usage = np.array(measurements['usage'])
        capacity = measurements['capacity']
        
        # Basic utilization metrics
        bandwidth_metrics[link_id].update({
            'mean_utilization': np.mean(usage) / capacity * 100,
            'peak_utilization': np.max(usage) / capacity * 100,
            'min_utilization': np.min(usage) / capacity * 100,
            'std_utilization': np.std(usage) / capacity * 100
        })
        
        # Analyze congestion
        congestion = analyze_link_congestion(usage, capacity)
        bandwidth_metrics[link_id]['congestion'] = congestion
        
        # Analyze utilization patterns
        patterns = analyze_bandwidth_patterns(usage, capacity)
        bandwidth_metrics[link_id]['patterns'] = patterns
        
    return dict(bandwidth_metrics)

def analyze_network_patterns(data: Dict) -> Dict:
    """
    Analyze network communication patterns.
    
    Args:
        data: Dictionary containing network measurements
        
    Returns:
        Dictionary containing pattern analysis
    """
    pattern_metrics = {}
    
    # Analyze topology characteristics
    pattern_metrics['topology'] = analyze_topology_characteristics(
        data['topology']
    )
    
    # Analyze communication patterns
    pattern_metrics['communication'] = analyze_communication_patterns(
        data['transfers']
    )
    
    # Analyze temporal patterns
    if 'timestamps' in data:
        pattern_metrics['temporal'] = analyze_temporal_patterns(
            data['timestamps'],
            data['transfers']
        )
        
    return pattern_metrics

def analyze_transfer_patterns(
    transfers: np.ndarray,
    messages: np.ndarray
) -> Dict:
    """
    Analyze patterns in data transfers.
    
    Args:
        transfers: Array of transfer sizes
        messages: Array of message counts
        
    Returns:
        Dictionary containing pattern analysis
    """
    return {
        'transfer_size_distribution': analyze_distribution(transfers),
        'message_patterns': analyze_message_patterns(messages),
        'burstiness': calculate_burstiness(transfers),
        'temporal_correlation': calculate_temporal_correlation(transfers)
    }

def analyze_link_congestion(
    usage: np.ndarray,
    capacity: float
) -> Dict:
    """
    Analyze link congestion patterns.
    
    Args:
        usage: Array of bandwidth usage
        capacity: Link capacity
        
    Returns:
        Dictionary containing congestion metrics
    """
    utilization = usage / capacity
    
    return {
        'congestion_events': detect_congestion_events(utilization),
        'congestion_duration': calculate_congestion_duration(utilization),
        'congestion_severity': calculate_congestion_severity(utilization)
    }

def analyze_bandwidth_patterns(
    usage: np.ndarray,
    capacity: float
) -> Dict:
    """
    Analyze bandwidth usage patterns.
    
    Args:
        usage: Array of bandwidth usage
        capacity: Link capacity
        
    Returns:
        Dictionary containing pattern analysis
    """
    utilization = usage / capacity
    
    return {
        'usage_distribution': analyze_distribution(utilization),
        'temporal_patterns': analyze_temporal_patterns(usage),
        'stability': calculate_stability(utilization),
        'predictability': calculate_predictability(utilization)
    }

def analyze_topology_characteristics(topology_data: Dict) -> Dict:
    """
    Analyze network topology characteristics.
    
    Args:
        topology_data: Dictionary containing topology information
        
    Returns:
        Dictionary containing topology metrics
    """
    # Create NetworkX graph
    G = nx.Graph()
    for edge in topology_data['edges']:
        G.add_edge(edge['source'], edge['target'], 
                  bandwidth=edge['bandwidth'])
    
    return {
        'diameter': nx.diameter(G),
        'average_path_length': nx.average_shortest_path_length(G),
        'clustering_coefficient': nx.average_clustering(G),
        'edge_connectivity': nx.edge_connectivity(G),
        'assortativity': nx.degree_assortativity_coefficient(G)
    }

def analyze_communication_patterns(transfers: List[Dict]) -> Dict:
    """
    Analyze communication patterns between components.
    
    Args:
        transfers: List of transfer records
        
    Returns:
        Dictionary containing pattern analysis
    """
    # Create communication graph
    G = nx.DiGraph()
    for transfer in transfers:
        source = transfer['source']
        target = transfer['target']
        size = transfer['size']
        G.add_edge(source, target, weight=size)
    
    return {
        'hotspots': identify_communication_hotspots(G),
        'communities': detect_communication_communities(G),
        'flow_patterns': analyze_flow_patterns(G)
    }

def analyze_temporal_patterns(
    timestamps: np.ndarray,
    transfers: np.ndarray
) -> Dict:
    """
    Analyze temporal patterns in network usage.
    
    Args:
        timestamps: Array of timestamps
        transfers: Array of transfer sizes
        
    Returns:
        Dictionary containing temporal analysis
    """
    return {
        'periodicity': detect_periodicity(timestamps, transfers),
        'trends': analyze_temporal_trends(timestamps, transfers),
        'changepoints': detect_temporal_changepoints(timestamps, transfers)
    }

def calculate_communication_efficiency(
    transfers: np.ndarray,
    messages: np.ndarray,
    computation_time: np.ndarray
) -> Dict:
    """
    Calculate communication efficiency metrics.
    
    Args:
        transfers: Array of transfer sizes
        messages: Array of message counts
        computation_time: Array of computation times
        
    Returns:
        Dictionary containing efficiency metrics
    """
    total_time = np.sum(computation_time)
    total_data = np.sum(transfers)
    total_messages = np.sum(messages)
    
    return {
        'data_transfer_rate': total_data / total_time,
        'message_rate': total_messages / total_time,
        'communication_overhead_ratio': total_data / total_messages,
        'efficiency_score': calculate_efficiency_score(
            transfers, computation_time
        )
    }

# Helper functions
def analyze_distribution(values: np.ndarray) -> Dict:
    """Analyze statistical distribution of values"""
    return {
        'mean': np.mean(values),
        'std': np.std(values),
        'skewness': stats.skew(values),
        'kurtosis': stats.kurtosis(values),
        'percentiles': {
            'p50': np.percentile(values, 50),
            'p90': np.percentile(values, 90),
            'p99': np.percentile(values, 99)
        }
    }

def calculate_burstiness(values: np.ndarray) -> float:
    """Calculate burstiness of network traffic"""
    std = np.std(values)
    mean = np.mean(values)
    return (std - mean) / (std + mean)

def calculate_temporal_correlation(values: np.ndarray) -> float:
    """Calculate temporal correlation in network traffic"""
    if len(values) < 2:
        return 0.0
    return np.corrcoef(values[:-1], values[1:])[0, 1]

def detect_congestion_events(utilization: np.ndarray) -> List[Dict]:
    """Detect congestion events in network traffic"""
    threshold = 0.9
    is_congested = utilization > threshold
    
    # Find continuous congestion periods
    changes = np.diff(is_congested.astype(int))
    start_indices = np.where(changes == 1)[0] + 1
    end_indices = np.where(changes == -1)[0] + 1
    
    events = []
    for start, end in zip(start_indices, end_indices):
        events.append({
            'start_index': start,
            'end_index': end,
            'duration': end - start,
            'severity': np.mean(utilization[start:end])
        })
        
    return events

def calculate_stability(values: np.ndarray) -> float:
    """Calculate stability score for network metrics"""
    if len(values) < 2:
        return 1.0
    return 1.0 / (1.0 + np.std(np.diff(values)))

def calculate_predictability(values: np.ndarray) -> float:
    """Calculate predictability score for network metrics"""
    if len(values) < 2:
        return 0.0
        
    # Use autocorrelation as predictability measure
    autocorr = np.correlate(values, values, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    return np.sum(autocorr[1:] / autocorr[0])

def calculate_congestion_severity(utilization: np.ndarray) -> float:
    """
    Calculate severity of network congestion.
    
    Args:
        utilization: Array of utilization values
        
    Returns:
        Congestion severity score
    """
    # Calculate weighted average of over-utilization
    threshold = 0.8  # 80% utilization threshold
    over_util = np.maximum(0, utilization - threshold)
    severity = np.mean(over_util) / (1 - threshold)
    return severity

def calculate_congestion_duration(utilization: np.ndarray) -> float:
    """
    Calculate duration of congestion periods.
    
    Args:
        utilization: Array of utilization values
        
    Returns:
        Proportion of time in congestion
    """
    threshold = 0.8
    congested_periods = utilization > threshold
    return np.mean(congested_periods)

def analyze_message_patterns(messages: np.ndarray) -> Dict:
    """
    Analyze patterns in message exchanges.
    
    Args:
        messages: Array of message counts
        
    Returns:
        Dictionary containing message pattern analysis
    """
    return {
        'burst_frequency': detect_message_bursts(messages),
        'temporal_distribution': analyze_message_distribution(messages),
        'pattern_regularity': calculate_pattern_regularity(messages)
    }

def detect_periodicity(timestamps: np.ndarray, transfers: np.ndarray) -> Dict:
    """
    Detect periodic patterns in network traffic.
    
    Args:
        timestamps: Array of timestamps
        transfers: Array of transfer values
        
    Returns:
        Dictionary containing periodicity analysis
    """
    # Compute FFT for frequency analysis
    fft_vals = np.abs(np.fft.fft(transfers))
    freqs = np.fft.fftfreq(len(transfers))
    
    # Find dominant frequencies
    threshold = np.percentile(fft_vals, 90)
    peaks = fft_vals > threshold
    
    return {
        'has_periodicity': np.any(peaks[1:]),  # Exclude DC component
        'dominant_frequencies': freqs[peaks],
        'strength': fft_vals[peaks] / np.sum(fft_vals)
    }

def analyze_temporal_trends(timestamps: np.ndarray, transfers: np.ndarray) -> Dict:
    """
    Analyze temporal trends in network traffic.
    
    Args:
        timestamps: Array of timestamps
        transfers: Array of transfer values
        
    Returns:
        Dictionary containing trend analysis
    """
    # Fit linear trend
    coeffs = np.polyfit(range(len(transfers)), transfers, 1)
    trend = np.poly1d(coeffs)
    
    return {
        'trend_coefficient': coeffs[0],
        'baseline': coeffs[1],
        'trend_strength': calculate_trend_strength(transfers, trend)
    }

def detect_temporal_changepoints(timestamps: np.ndarray, transfers: np.ndarray) -> List[Dict]:
    """
    Detect significant changes in network traffic patterns.
    
    Args:
        timestamps: Array of timestamps
        transfers: Array of transfer values
        
    Returns:
        List of detected changepoints
    """
    window_size = max(20, len(transfers) // 10)
    changepoints = []
    
    for i in range(window_size, len(transfers) - window_size):
        before = transfers[i-window_size:i]
        after = transfers[i:i+window_size]
        
        # Perform statistical test
        stat, pval = stats.ttest_ind(before, after)
        
        if pval < 0.05:  # Significant change
            changepoints.append({
                'timestamp': timestamps[i],
                'index': i,
                'statistic': stat,
                'p_value': pval,
                'change_magnitude': np.mean(after) - np.mean(before)
            })
            
    return changepoints

def identify_communication_hotspots(G: nx.Graph) -> Dict:
    """
    Identify communication hotspots in the network.
    
    Args:
        G: NetworkX graph of communication
        
    Returns:
        Dictionary containing hotspot analysis
    """
    # Calculate node centrality measures
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    
    # Identify high-traffic nodes
    traffic_threshold = np.percentile(list(degree_centrality.values()), 75)
    hotspots = {
        node: {
            'degree_centrality': degree_centrality[node],
            'betweenness_centrality': betweenness_centrality[node],
            'traffic_load': G.degree(node, weight='weight')
        }
        for node in G.nodes()
        if degree_centrality[node] > traffic_threshold
    }
    
    return hotspots

def detect_communication_communities(G: nx.Graph) -> Dict:
    """
    Detect communication communities in the network.
    
    Args:
        G: NetworkX graph of communication
        
    Returns:
        Dictionary containing community analysis
    """
    # Detect communities using Louvain method
    try:
        communities = nx.community.louvain_communities(G)
    except:
        # Fallback to simpler community detection
        communities = list(nx.connected_components(G))
    
    return {
        'num_communities': len(communities),
        'community_sizes': [len(c) for c in communities],
        'modularity': calculate_modularity(G, communities)
    }

def analyze_flow_patterns(G: nx.Graph) -> Dict:
    """
    Analyze network flow patterns.
    
    Args:
        G: NetworkX graph of communication
        
    Returns:
        Dictionary containing flow pattern analysis
    """
    return {
        'flow_concentration': calculate_flow_concentration(G),
        'path_diversity': calculate_path_diversity(G),
        'bottleneck_analysis': identify_bottlenecks(G)
    }

def calculate_efficiency_score(transfers: np.ndarray, computation_time: np.ndarray) -> float:
    """
    Calculate communication efficiency score.
    
    Args:
        transfers: Array of transfer sizes
        computation_time: Array of computation times
        
    Returns:
        Efficiency score between 0 and 1
    """
    # Calculate ratio of useful work to total time
    total_time = np.sum(computation_time)
    transfer_overhead = np.sum(transfers) / total_time
    
    # Normalize to [0, 1] range (lower overhead is better)
    return 1.0 / (1.0 + transfer_overhead)

# Additional helper functions
def detect_message_bursts(messages: np.ndarray) -> Dict:
    """Detect bursts in message patterns"""
    # Calculate message rate changes
    rate_changes = np.diff(messages)
    threshold = np.std(rate_changes) * 2
    
    bursts = rate_changes > threshold
    return {
        'burst_count': np.sum(bursts),
        'burst_frequency': np.mean(bursts),
        'avg_burst_size': np.mean(rate_changes[bursts]) if any(bursts) else 0
    }

def analyze_message_distribution(messages: np.ndarray) -> Dict:
    """Analyze temporal distribution of messages"""
    return {
        'mean_rate': np.mean(messages),
        'peak_rate': np.max(messages),
        'variation': np.std(messages) / np.mean(messages) if np.mean(messages) > 0 else 0
    }

def calculate_pattern_regularity(messages: np.ndarray) -> float:
    """Calculate regularity of message patterns"""
    if len(messages) < 2:
        return 1.0
    
    # Use coefficient of variation as regularity measure
    cv = np.std(messages) / np.mean(messages) if np.mean(messages) > 0 else 0
    return 1.0 / (1.0 + cv)  # Transform to [0,1] range, higher is more regular

def calculate_trend_strength(values: np.ndarray, trend: np.poly1d) -> float:
    """Calculate strength of temporal trend"""
    trend_values = trend(range(len(values)))
    residuals = values - trend_values
    return 1.0 - (np.var(residuals) / np.var(values))

def calculate_modularity(G: nx.Graph, communities: List[Set]) -> float:
    """Calculate modularity of community structure"""
    if not G.edges():
        return 0.0
    
    modularity = 0
    m = G.number_of_edges()
    
    for community in communities:
        for u in community:
            for v in community:
                if G.has_edge(u, v):
                    modularity += 1 - (G.degree(u) * G.degree(v)) / (2 * m)
                    
    return modularity / (2 * m)

def calculate_flow_concentration(G: nx.Graph) -> float:
    """Calculate concentration of network flows"""
    edge_weights = [d['weight'] for _, _, d in G.edges(data=True)]
    if not edge_weights:
        return 0.0
    return np.std(edge_weights) / np.mean(edge_weights)

def calculate_path_diversity(G: nx.Graph) -> Dict:
    """Calculate diversity of communication paths"""
    # Sample node pairs for path analysis
    sample_size = min(100, len(G.nodes()) * (len(G.nodes()) - 1) // 2)
    nodes = list(G.nodes())
    path_lengths = []
    
    for _ in range(sample_size):
        u, v = np.random.choice(nodes, 2, replace=False)
        try:
            paths = list(nx.all_simple_paths(G, u, v, cutoff=10))
            path_lengths.append(len(paths))
        except nx.NetworkXNoPath:
            continue
            
    return {
        'average_paths': np.mean(path_lengths) if path_lengths else 0,
        'path_variance': np.var(path_lengths) if path_lengths else 0
    }

def identify_bottlenecks(G: nx.Graph) -> List[Dict]:
    """Identify potential bottlenecks in the network"""
    edge_betweenness = nx.edge_betweenness_centrality(G, weight='weight')
    
    # Identify high betweenness edges
    threshold = np.percentile(list(edge_betweenness.values()), 90)
    
    bottlenecks = [
        {
            'edge': edge,
            'betweenness': value,
            'capacity': G.edges[edge].get('weight', 1.0)
        }
        for edge, value in edge_betweenness.items()
        if value > threshold
    ]
    
    return bottlenecks