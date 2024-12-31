from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict

def calculate_performance_metrics(data: Dict) -> Dict:
    """
    Calculate comprehensive performance metrics from experimental data.
    
    Args:
        data: Dictionary containing raw experimental data
        
    Returns:
        Dictionary containing calculated performance metrics
    """
    metrics = {}
    
    # Calculate latency metrics
    metrics['latency'] = calculate_latency_metrics(data)
    
    # Calculate throughput metrics
    metrics['throughput'] = calculate_throughput_metrics(data)
    
    # Analyze performance trends
    metrics['trends'] = analyze_performance_trends(data)
    
    return metrics

def calculate_latency_metrics(data: Dict) -> Dict:
    """
    Calculate latency-related metrics.
    
    Args:
        data: Dictionary containing latency measurements
        
    Returns:
        Dictionary containing latency metrics
    """
    latency_metrics = defaultdict(dict)
    
    for algo, measurements in data['latency_measurements'].items():
        latencies = np.array(measurements)
        
        # Basic statistics
        latency_metrics[algo].update({
            'mean_latency': np.mean(latencies),
            'median_latency': np.median(latencies),
            'std_latency': np.std(latencies),
            'min_latency': np.min(latencies),
            'max_latency': np.max(latencies),
            'p95_latency': np.percentile(latencies, 95),
            'p99_latency': np.percentile(latencies, 99)
        })
        
        # Calculate jitter
        latency_metrics[algo]['jitter'] = calculate_jitter(latencies)
        
        # Analyze tail latencies
        tail_metrics = analyze_tail_latencies(latencies)
        latency_metrics[algo]['tail_metrics'] = tail_metrics
        
    return dict(latency_metrics)

def calculate_throughput_metrics(data: Dict) -> Dict:
    """
    Calculate throughput-related metrics.
    
    Args:
        data: Dictionary containing throughput measurements
        
    Returns:
        Dictionary containing throughput metrics
    """
    throughput_metrics = defaultdict(dict)
    
    for algo, measurements in data['throughput_measurements'].items():
        throughputs = np.array(measurements)
        
        # Basic statistics
        throughput_metrics[algo].update({
            'mean_throughput': np.mean(throughputs),
            'median_throughput': np.median(throughputs),
            'std_throughput': np.std(throughputs),
            'min_throughput': np.min(throughputs),
            'max_throughput': np.max(throughputs),
            'sustained_throughput': calculate_sustained_throughput(throughputs)
        })
        
        # Calculate stability metrics
        throughput_metrics[algo]['stability'] = analyze_throughput_stability(throughputs)
        
        # Analyze scaling efficiency
        if 'sequence_lengths' in data:
            scaling = analyze_scaling_efficiency(
                throughputs,
                data['sequence_lengths']
            )
            throughput_metrics[algo]['scaling'] = scaling
            
    return dict(throughput_metrics)

def analyze_performance_trends(data: Dict) -> Dict:
    """
    Analyze performance trends over time and sequence lengths.
    
    Args:
        data: Dictionary containing performance measurements
        
    Returns:
        Dictionary containing trend analysis results
    """
    trends = defaultdict(dict)
    
    for algo, measurements in data['performance_measurements'].items():
        # Time-based trends
        time_trends = analyze_time_trends(
            measurements['timestamps'],
            measurements['metrics']
        )
        trends[algo]['time_trends'] = time_trends
        
        # Sequence length scaling
        if 'sequence_lengths' in measurements:
            scaling_trends = analyze_scaling_trends(
                measurements['sequence_lengths'],
                measurements['metrics']
            )
            trends[algo]['scaling_trends'] = scaling_trends
            
        # Workload sensitivity
        if 'workload_metrics' in measurements:
            sensitivity = analyze_workload_sensitivity(
                measurements['workload_metrics']
            )
            trends[algo]['workload_sensitivity'] = sensitivity
            
    return dict(trends)

def calculate_jitter(latencies: np.ndarray) -> float:
    """Calculate latency jitter (variation)"""
    # Calculate differences between consecutive latencies
    diffs = np.diff(latencies)
    return np.std(diffs)

def analyze_tail_latencies(latencies: np.ndarray) -> Dict:
    """Analyze tail latency behavior"""
    percentiles = [90, 95, 99, 99.9]
    tail_metrics = {
        f'p{p}': np.percentile(latencies, p)
        for p in percentiles
    }
    
    # Calculate tail-to-median ratio
    tail_metrics['tail_to_median_ratio'] = (
        tail_metrics['p99'] / np.median(latencies)
    )
    
    return tail_metrics

def calculate_sustained_throughput(throughputs: np.ndarray) -> float:
    """Calculate sustained throughput (excluding outliers)"""
    # Remove outliers (values outside 2 standard deviations)
    mean = np.mean(throughputs)
    std = np.std(throughputs)
    valid_throughputs = throughputs[
        (throughputs >= mean - 2*std) & 
        (throughputs <= mean + 2*std)
    ]
    return np.mean(valid_throughputs)

def analyze_throughput_stability(throughputs: np.ndarray) -> Dict:
    """Analyze throughput stability"""
    return {
        'coefficient_of_variation': stats.variation(throughputs),
        'stability_score': calculate_stability_score(throughputs),
        'trend_coefficient': calculate_trend_coefficient(throughputs)
    }

def analyze_scaling_efficiency(
    throughputs: np.ndarray,
    sequence_lengths: np.ndarray
) -> Dict:
    """Analyze throughput scaling with sequence length"""
    # Calculate scaling factor
    log_throughputs = np.log(throughputs)
    log_lengths = np.log(sequence_lengths)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        log_lengths,
        log_throughputs
    )
    
    return {
        'scaling_exponent': slope,
        'scaling_constant': np.exp(intercept),
        'correlation': r_value,
        'significance': p_value,
        'std_error': std_err
    }

def analyze_time_trends(
    timestamps: np.ndarray,
    metrics: np.ndarray
) -> Dict:
    """Analyze performance trends over time"""
    # Calculate trend line
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        timestamps,
        metrics
    )
    
    # Detect any significant changes
    change_points = detect_change_points(timestamps, metrics)
    
    return {
        'trend_slope': slope,
        'trend_intercept': intercept,
        'correlation': r_value,
        'significance': p_value,
        'std_error': std_err,
        'change_points': change_points
    }

def analyze_scaling_trends(
    sequence_lengths: np.ndarray,
    metrics: np.ndarray
) -> Dict:
    """Analyze performance scaling with sequence length"""
    # Fit polynomial for non-linear scaling
    coeffs = np.polyfit(sequence_lengths, metrics, deg=2)
    
    # Calculate goodness of fit
    fitted_values = np.polyval(coeffs, sequence_lengths)
    residuals = metrics - fitted_values
    r_squared = 1 - (np.sum(residuals**2) / 
                     np.sum((metrics - np.mean(metrics))**2))
    
    return {
        'scaling_coefficients': coeffs.tolist(),
        'r_squared': r_squared,
        'residuals_std': np.std(residuals)
    }

def analyze_workload_sensitivity(workload_metrics: Dict) -> Dict:
    """Analyze sensitivity to different workload characteristics"""
    sensitivity = {}
    
    # Analyze sensitivity to each workload parameter
    for param, values in workload_metrics.items():
        correlation = stats.spearmanr(
            values['parameter_values'],
            values['performance_values']
        )
        
        sensitivity[param] = {
            'correlation_coefficient': correlation.correlation,
            'significance': correlation.pvalue,
            'effect_size': calculate_effect_size(
                values['parameter_values'],
                values['performance_values']
            )
        }
        
    return sensitivity

def calculate_stability_score(values: np.ndarray) -> float:
    """Calculate stability score based on variance and trends"""
    # Combine coefficient of variation with trend analysis
    cv = stats.variation(values)
    trend_coeff = calculate_trend_coefficient(values)
    
    # Lower score is better (more stable)
    return np.sqrt(cv**2 + trend_coeff**2)

def calculate_trend_coefficient(values: np.ndarray) -> float:
    """Calculate trend coefficient (how much values change over time)"""
    # Use linear regression to detect systematic changes
    times = np.arange(len(values))
    slope, _, r_value, _, _ = stats.linregress(times, values)
    
    # Normalize by mean value
    return abs(slope * len(values) / np.mean(values))

def calculate_effect_size(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Cohen's d effect size"""
    pooled_std = np.sqrt((np.var(x) + np.var(y)) / 2)
    return abs(np.mean(x) - np.mean(y)) / pooled_std

def detect_change_points(
    timestamps: np.ndarray,
    metrics: np.ndarray
) -> List[Dict]:
    """Detect significant changes in performance metrics"""
    change_points = []
    window_size = len(metrics) // 10  # 10% of data points
    
    for i in range(window_size, len(metrics) - window_size):
        before = metrics[i-window_size:i]
        after = metrics[i:i+window_size]
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(before, after)
        
        if p_value < 0.05:  # Significant change
            change_points.append({
                'timestamp': timestamps[i],
                'index': i,
                't_statistic': t_stat,
                'p_value': p_value,
                'before_mean': np.mean(before),
                'after_mean': np.mean(after)
            })
            
    return change_points