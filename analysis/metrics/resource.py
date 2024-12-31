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
# File: analysis/metrics/resource.py
# Description:
#   Resource utilization metrics for distributed Transformer inference,
#   focusing on memory and compute usage tracking, cache allocation,
#   and load distribution insights across heterogeneous devices.
#
# ---------------------------------------------------------------------------

"""
Provides classes and methods to measure resource utilization metrics, such as
memory usage, compute utilization, and load balancing efficiency. These metrics
help evaluate whether the assignment of model components optimally leverages
the available device resources over time.
"""


from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict

def calculate_resource_metrics(data: Dict) -> Dict:
    """
    Calculate comprehensive resource utilization metrics.
    
    Args:
        data: Dictionary containing resource utilization data
        
    Returns:
        Dictionary containing calculated resource metrics
    """
    metrics = {}
    
    # Calculate memory metrics
    metrics['memory'] = calculate_memory_metrics(data)
    
    # Calculate CPU metrics
    metrics['cpu'] = calculate_cpu_metrics(data)
    
    # Calculate efficiency metrics
    metrics['efficiency'] = analyze_resource_efficiency(data)
    
    return metrics

def calculate_memory_metrics(data: Dict) -> Dict:
    """
    Calculate memory-related metrics for each device.
    
    Args:
        data: Dictionary containing memory utilization data
        
    Returns:
        Dictionary containing memory metrics per device
    """
    memory_metrics = defaultdict(dict)
    
    for device_id, measurements in data['memory_measurements'].items():
        memory_usage = np.array(measurements['usage'])
        memory_capacity = measurements['capacity']
        
        # Basic utilization metrics
        memory_metrics[device_id].update({
            'mean_utilization': np.mean(memory_usage) / memory_capacity * 100,
            'peak_utilization': np.max(memory_usage) / memory_capacity * 100,
            'min_utilization': np.min(memory_usage) / memory_capacity * 100,
            'std_utilization': np.std(memory_usage) / memory_capacity * 100
        })
        
        # Memory pressure analysis
        pressure_metrics = analyze_memory_pressure(
            memory_usage,
            memory_capacity
        )
        memory_metrics[device_id]['pressure'] = pressure_metrics
        
        # Cache efficiency metrics
        if 'cache_size' in measurements:
            cache_metrics = analyze_cache_efficiency(
                measurements['cache_size'],
                measurements['cache_hits'],
                measurements['cache_misses']
            )
            memory_metrics[device_id]['cache'] = cache_metrics
            
    return dict(memory_metrics)

def calculate_cpu_metrics(data: Dict) -> Dict:
    """
    Calculate CPU-related metrics for each device.
    
    Args:
        data: Dictionary containing CPU utilization data
        
    Returns:
        Dictionary containing CPU metrics per device
    """
    cpu_metrics = defaultdict(dict)
    
    for device_id, measurements in data['cpu_measurements'].items():
        cpu_usage = np.array(measurements['usage'])
        cpu_capacity = measurements['capacity']
        
        # Basic utilization metrics
        cpu_metrics[device_id].update({
            'mean_utilization': np.mean(cpu_usage) / cpu_capacity * 100,
            'peak_utilization': np.max(cpu_usage) / cpu_capacity * 100,
            'min_utilization': np.min(cpu_usage) / cpu_capacity * 100,
            'std_utilization': np.std(cpu_usage) / cpu_capacity * 100
        })
        
        # Load distribution analysis
        load_metrics = analyze_cpu_load(cpu_usage, cpu_capacity)
        cpu_metrics[device_id]['load'] = load_metrics
        
        # Computational efficiency
        if 'flops_executed' in measurements:
            efficiency_metrics = analyze_compute_efficiency(
                measurements['flops_executed'],
                measurements['time_taken']
            )
            cpu_metrics[device_id]['efficiency'] = efficiency_metrics
            
    return dict(cpu_metrics)

def analyze_resource_efficiency(data: Dict) -> Dict:
    """
    Analyze overall resource efficiency across devices.
    
    Args:
        data: Dictionary containing resource utilization data
        
    Returns:
        Dictionary containing efficiency metrics
    """
    efficiency_metrics = {}
    
    # Calculate load balancing metrics
    efficiency_metrics['load_balancing'] = calculate_load_balancing_metrics(
        data['memory_measurements'],
        data['cpu_measurements']
    )
    
    # Calculate resource utilization efficiency
    efficiency_metrics['utilization'] = calculate_utilization_efficiency(
        data['memory_measurements'],
        data['cpu_measurements']
    )
    
    # Calculate scaling efficiency if available
    if 'scaling_measurements' in data:
        efficiency_metrics['scaling'] = analyze_scaling_efficiency(
            data['scaling_measurements']
        )
        
    return efficiency_metrics

def analyze_memory_pressure(
    memory_usage: np.ndarray,
    memory_capacity: float
) -> Dict:
    """
    Analyze memory pressure and potential bottlenecks.
    
    Args:
        memory_usage: Array of memory usage measurements
        memory_capacity: Total memory capacity
        
    Returns:
        Dictionary containing memory pressure metrics
    """
    # Calculate pressure metrics
    utilization = memory_usage / memory_capacity
    
    pressure_metrics = {
        'pressure_score': calculate_pressure_score(utilization),
        'high_pressure_periods': detect_high_pressure_periods(utilization),
        'pressure_trend': analyze_pressure_trend(utilization)
    }
    
    # Add warning thresholds
    pressure_metrics['warnings'] = {
        'high_utilization': np.mean(utilization > 0.9),
        'utilization_spikes': detect_spikes(utilization),
        'pressure_duration': calculate_pressure_duration(utilization)
    }
    
    return pressure_metrics

def analyze_cache_efficiency(
    cache_sizes: np.ndarray,
    cache_hits: np.ndarray,
    cache_misses: np.ndarray
) -> Dict:
    """
    Analyze cache efficiency metrics.
    
    Args:
        cache_sizes: Array of cache sizes over time
        cache_hits: Array of cache hit counts
        cache_misses: Array of cache miss counts
        
    Returns:
        Dictionary containing cache efficiency metrics
    """
    total_accesses = cache_hits + cache_misses
    
    return {
        'hit_rate': np.mean(cache_hits / total_accesses),
        'miss_rate': np.mean(cache_misses / total_accesses),
        'size_efficiency': np.mean(cache_hits / cache_sizes),
        'size_trend': analyze_size_trend(cache_sizes),
        'hit_rate_trend': analyze_hit_rate_trend(cache_hits / total_accesses)
    }

def analyze_cpu_load(
    cpu_usage: np.ndarray,
    cpu_capacity: float
) -> Dict:
    """
    Analyze CPU load distribution and patterns.
    
    Args:
        cpu_usage: Array of CPU usage measurements
        cpu_capacity: Total CPU capacity
        
    Returns:
        Dictionary containing CPU load metrics
    """
    utilization = cpu_usage / cpu_capacity
    
    return {
        'load_distribution': analyze_load_distribution(utilization),
        'load_stability': calculate_load_stability(utilization),
        'peak_patterns': analyze_peak_patterns(utilization)
    }

def analyze_compute_efficiency(
    flops_executed: np.ndarray,
    time_taken: np.ndarray
) -> Dict:
    """
    Analyze computational efficiency metrics.
    
    Args:
        flops_executed: Array of FLOPS executed
        time_taken: Array of execution times
        
    Returns:
        Dictionary containing compute efficiency metrics
    """
    flops_per_second = flops_executed / time_taken
    
    return {
        'mean_flops': np.mean(flops_per_second),
        'peak_flops': np.max(flops_per_second),
        'efficiency_trend': analyze_efficiency_trend(flops_per_second),
        'execution_stability': calculate_execution_stability(flops_per_second)
    }

def calculate_load_balancing_metrics(
    memory_data: Dict,
    cpu_data: Dict
) -> Dict:
    """Calculate load balancing metrics across devices"""
    memory_util = [np.mean(d['usage']) / d['capacity'] 
                  for d in memory_data.values()]
    cpu_util = [np.mean(d['usage']) / d['capacity'] 
                for d in cpu_data.values()]
    
    return {
        'memory_balance_score': calculate_balance_score(memory_util),
        'cpu_balance_score': calculate_balance_score(cpu_util),
        'overall_balance': calculate_overall_balance(memory_util, cpu_util)
    }

def calculate_utilization_efficiency(
    memory_data: Dict,
    cpu_data: Dict
) -> Dict:
    """Calculate resource utilization efficiency metrics"""
    memory_util = [np.mean(d['usage']) / d['capacity'] 
                  for d in memory_data.values()]
    cpu_util = [np.mean(d['usage']) / d['capacity'] 
                for d in cpu_data.values()]
    
    return {
        'memory_efficiency': calculate_efficiency_score(memory_util),
        'cpu_efficiency': calculate_efficiency_score(cpu_util),
        'combined_efficiency': calculate_combined_efficiency(
            memory_util,
            cpu_util
        )
    }

# Helper functions
def calculate_pressure_score(utilization: np.ndarray) -> float:
    """Calculate memory pressure score"""
    # Weight higher utilizations more heavily
    weighted_util = np.power(utilization, 2)
    return np.mean(weighted_util)

def detect_high_pressure_periods(utilization: np.ndarray) -> List[Dict]:
    """Detect periods of high memory pressure"""
    threshold = 0.9
    high_pressure = utilization > threshold
    
    # Find continuous periods
    changes = np.diff(high_pressure.astype(int))
    start_indices = np.where(changes == 1)[0] + 1
    end_indices = np.where(changes == -1)[0] + 1
    
    periods = []
    for start, end in zip(start_indices, end_indices):
        periods.append({
            'start_index': start,
            'end_index': end,
            'duration': end - start,
            'mean_utilization': np.mean(utilization[start:end])
        })
    
    return periods

def analyze_pressure_trend(utilization: np.ndarray) -> Dict:
    """Analyze trend in memory pressure"""
    times = np.arange(len(utilization))
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        times,
        utilization
    )
    
    return {
        'slope': slope,
        'intercept': intercept,
        'correlation': r_value,
        'significance': p_value
    }

def calculate_balance_score(utilizations: List[float]) -> float:
    """Calculate load balance score (0-1, higher is better)"""
    if not utilizations:
        return 0.0
    
    return 1.0 - (np.std(utilizations) / np.mean(utilizations))

def calculate_efficiency_score(utilizations: List[float]) -> float:
    """Calculate resource efficiency score (0-1, higher is better)"""
    if not utilizations:
        return 0.0
    
    # Penalize both under and over-utilization
    return 1.0 - np.mean(np.abs(np.array(utilizations) - 0.75))

def calculate_combined_efficiency(
    memory_util: List[float],
    cpu_util: List[float]
) -> float:
    """Calculate combined resource efficiency score"""
    memory_score = calculate_efficiency_score(memory_util)
    cpu_score = calculate_efficiency_score(cpu_util)
    
    # Weight memory and CPU equally
    return (memory_score + cpu_score) / 2

def analyze_scaling_efficiency(measurements: Dict) -> Dict:
    """
    Analyze how resource efficiency scales with workload.
    
    Args:
        measurements: Dictionary containing scaling measurements
        
    Returns:
        Dictionary containing scaling analysis
    """
    workload_sizes = measurements['workload_sizes']
    resource_usage = measurements['resource_usage']
    
    # Calculate scaling factor
    slope, intercept = np.polyfit(np.log(workload_sizes), np.log(resource_usage), 1)
    
    return {
        'scaling_exponent': slope,
        'scaling_constant': np.exp(intercept),
        'efficiency_ratio': calculate_efficiency_ratio(workload_sizes, resource_usage),
        'scaling_linearity': calculate_scaling_linearity(workload_sizes, resource_usage)
    }

def detect_spikes(utilization: np.ndarray) -> List[Dict]:
    """
    Detect significant spikes in resource utilization.
    
    Args:
        utilization: Array of utilization values
        
    Returns:
        List of detected spikes
    """
    # Calculate baseline and standard deviation
    baseline = np.median(utilization)
    mad = np.median(np.abs(utilization - baseline))
    threshold = baseline + 3 * mad
    
    # Find spikes
    spikes = []
    spike_indices = np.where(utilization > threshold)[0]
    
    for idx in spike_indices:
        spikes.append({
            'index': idx,
            'value': utilization[idx],
            'magnitude': (utilization[idx] - baseline) / mad,
            'duration': calculate_spike_duration(utilization, idx, threshold)
        })
        
    return spikes

def calculate_pressure_duration(utilization: np.ndarray) -> Dict:
    """
    Calculate duration of high pressure periods.
    
    Args:
        utilization: Array of utilization values
        
    Returns:
        Dictionary containing pressure duration metrics
    """
    threshold = 0.9  # 90% utilization threshold
    high_pressure = utilization > threshold
    
    # Find continuous periods
    pressure_periods = []
    start_idx = None
    
    for i, under_pressure in enumerate(high_pressure):
        if under_pressure and start_idx is None:
            start_idx = i
        elif not under_pressure and start_idx is not None:
            pressure_periods.append({
                'start': start_idx,
                'end': i,
                'duration': i - start_idx
            })
            start_idx = None
            
    return {
        'total_duration': sum(p['duration'] for p in pressure_periods),
        'average_duration': np.mean([p['duration'] for p in pressure_periods]) if pressure_periods else 0,
        'max_duration': max([p['duration'] for p in pressure_periods]) if pressure_periods else 0,
        'pressure_periods': pressure_periods
    }

def analyze_size_trend(cache_sizes: np.ndarray) -> Dict:
    """
    Analyze trend in cache size evolution.
    
    Args:
        cache_sizes: Array of cache sizes over time
        
    Returns:
        Dictionary containing trend analysis
    """
    # Calculate trend
    times = np.arange(len(cache_sizes))
    slope, intercept = np.polyfit(times, cache_sizes, 1)
    
    # Calculate growth rate
    growth_rates = np.diff(cache_sizes) / cache_sizes[:-1]
    
    return {
        'trend_slope': slope,
        'trend_intercept': intercept,
        'average_growth_rate': np.mean(growth_rates),
        'growth_stability': np.std(growth_rates)
    }

def analyze_hit_rate_trend(hit_rates: np.ndarray) -> Dict:
    """
    Analyze trend in cache hit rates.
    
    Args:
        hit_rates: Array of cache hit rates over time
        
    Returns:
        Dictionary containing hit rate analysis
    """
    # Calculate moving average and standard deviation
    window = min(10, len(hit_rates))
    rolling_mean = np.convolve(hit_rates, np.ones(window)/window, mode='valid')
    rolling_std = np.array([np.std(hit_rates[max(0, i-window):i+1]) 
                           for i in range(len(hit_rates))])
    
    return {
        'trend': analyze_monotonicity(hit_rates),
        'stability': 1 - (np.mean(rolling_std) / np.mean(hit_rates)),
        'convergence': calculate_convergence(hit_rates)
    }

def analyze_load_distribution(utilization: np.ndarray) -> Dict:
    """
    Analyze distribution of load values.
    
    Args:
        utilization: Array of utilization values
        
    Returns:
        Dictionary containing load distribution analysis
    """
    return {
        'mean_load': np.mean(utilization),
        'load_std': np.std(utilization),
        'load_skewness': stats.skew(utilization),
        'load_kurtosis': stats.kurtosis(utilization),
        'percentiles': {
            '50': np.percentile(utilization, 50),
            '90': np.percentile(utilization, 90),
            '95': np.percentile(utilization, 95),
            '99': np.percentile(utilization, 99)
        }
    }

def calculate_load_stability(utilization: np.ndarray) -> float:
    """
    Calculate stability of load over time.
    
    Args:
        utilization: Array of utilization values
        
    Returns:
        Stability score between 0 and 1
    """
    if len(utilization) < 2:
        return 1.0
        
    # Calculate variability measures
    std = np.std(utilization)
    mean = np.mean(utilization)
    cv = std / mean if mean > 0 else float('inf')
    
    # Transform to [0,1] range where 1 is most stable
    return 1.0 / (1.0 + cv)

def analyze_peak_patterns(utilization: np.ndarray) -> Dict:
    """
    Analyze patterns in peak resource utilization.
    
    Args:
        utilization: Array of utilization values
        
    Returns:
        Dictionary containing peak pattern analysis
    """
    # Define peak threshold
    threshold = np.mean(utilization) + 2 * np.std(utilization)
    peaks = utilization > threshold
    
    # Find peak periods
    peak_periods = []
    start_idx = None
    
    for i, is_peak in enumerate(peaks):
        if is_peak and start_idx is None:
            start_idx = i
        elif not is_peak and start_idx is not None:
            peak_periods.append({
                'start': start_idx,
                'end': i,
                'duration': i - start_idx,
                'max_value': np.max(utilization[start_idx:i])
            })
            start_idx = None
            
    return {
        'num_peaks': len(peak_periods),
        'average_peak_duration': np.mean([p['duration'] for p in peak_periods]) if peak_periods else 0,
        'peak_frequency': len(peak_periods) / len(utilization),
        'peak_periods': peak_periods
    }

def analyze_efficiency_trend(flops_per_second: np.ndarray) -> Dict:
    """
    Analyze trend in computational efficiency.
    
    Args:
        flops_per_second: Array of FLOPS measurements
        
    Returns:
        Dictionary containing efficiency trend analysis
    """
    # Calculate trend
    times = np.arange(len(flops_per_second))
    slope, intercept = np.polyfit(times, flops_per_second, 1)
    
    return {
        'trend_slope': slope,
        'trend_intercept': intercept,
        'efficiency_stability': calculate_stability(flops_per_second),
        'trend_significance': calculate_trend_significance(times, flops_per_second)
    }

def calculate_execution_stability(flops_per_second: np.ndarray) -> float:
    """
    Calculate stability of execution performance.
    
    Args:
        flops_per_second: Array of FLOPS measurements
        
    Returns:
        Stability score between 0 and 1
    """
    if len(flops_per_second) < 2:
        return 1.0
        
    # Calculate coefficient of variation
    cv = np.std(flops_per_second) / np.mean(flops_per_second)
    
    # Transform to [0,1] range where 1 is most stable
    return 1.0 / (1.0 + cv)

def calculate_overall_balance(
    memory_util: List[float],
    cpu_util: List[float]
) -> float:
    """
    Calculate overall resource balance score.
    
    Args:
        memory_util: List of memory utilization values
        cpu_util: List of CPU utilization values
        
    Returns:
        Balance score between 0 and 1
    """
    if not memory_util or not cpu_util:
        return 0.0
        
    # Calculate individual balance scores
    memory_balance = calculate_balance_score(memory_util)
    cpu_balance = calculate_balance_score(cpu_util)
    
    # Calculate correlation between memory and CPU utilization
    correlation = np.corrcoef(memory_util, cpu_util)[0,1]
    
    # Combine scores with correlation factor
    return (memory_balance + cpu_balance) * (1 + abs(correlation)) / 4

# Helper functions
def calculate_efficiency_ratio(
    workload_sizes: np.ndarray,
    resource_usage: np.ndarray
) -> float:
    """Calculate efficiency as ratio of workload to resource usage"""
    return np.mean(workload_sizes / resource_usage)

def calculate_scaling_linearity(
    workload_sizes: np.ndarray,
    resource_usage: np.ndarray
) -> float:
    """Calculate linearity of scaling relationship"""
    log_sizes = np.log(workload_sizes)
    log_usage = np.log(resource_usage)
    
    # Calculate R-squared of log-log relationship
    slope, intercept = np.polyfit(log_sizes, log_usage, 1)
    fitted = slope * log_sizes + intercept
    r_squared = 1 - np.sum((log_usage - fitted)**2) / np.sum((log_usage - np.mean(log_usage))**2)
    
    return r_squared

def calculate_spike_duration(
    utilization: np.ndarray,
    spike_idx: int,
    threshold: float
) -> int:
    """Calculate duration of a utilization spike"""
    duration = 1
    idx = spike_idx + 1
    
    while idx < len(utilization) and utilization[idx] > threshold:
        duration += 1
        idx += 1
        
    return duration

def analyze_monotonicity(values: np.ndarray) -> str:
    """Analyze if trend is monotonically increasing or decreasing"""
    diffs = np.diff(values)
    if np.all(diffs >= 0):
        return 'increasing'
    elif np.all(diffs <= 0):
        return 'decreasing'
    else:
        return 'fluctuating'

def calculate_convergence(values: np.ndarray) -> float:
    """Calculate convergence rate of values"""
    if len(values) < 2:
        return 1.0
        
    diffs = np.abs(np.diff(values))
    convergence_rate = np.mean(diffs[1:] < diffs[:-1])
    return convergence_rate

def calculate_stability(values: np.ndarray) -> float:
    """Calculate stability score based on variations"""
    if len(values) < 2:
        return 1.0
        
    variations = np.abs(np.diff(values) / values[:-1])
    return 1.0 / (1.0 + np.mean(variations))

def calculate_trend_significance(
    times: np.ndarray,
    values: np.ndarray
) -> float:
    """Calculate statistical significance of trend"""
    slope, intercept = np.polyfit(times, values, 1)
    fitted = slope * times + intercept
    residuals = values - fitted
    
    # Calculate R-squared
    ss_total = np.sum((values - np.mean(values))**2)
    ss_residual = np.sum(residuals**2)
    
    return 1 - (ss_residual / ss_total)