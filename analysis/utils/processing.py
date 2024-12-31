# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Dimitrios Kafetzis
#
# This file is part of the Transformer Inference Simulator project.
# Licensed under the MIT License; you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#   https://opensource.org/licenses/MIT
#
# Author:  Dimitrios Kafetzis (dimitrioskafetzis@gmail.com)
# File:    analysis/utils/processing.py
# Description:
#   Data processing and transformation utilities for metrics and results
#   in distributed transformer inference experiments. Includes cleaning,
#   normalization, aggregation, and other advanced operations.
#
# ---------------------------------------------------------------------------

"""
Provides functions to process raw experimental or metrics data into
cleaned, aggregated, and normalized forms, ensuring consistency and
facilitating in-depth analysis of distributed transformer inference
performance, resource usage, and communication behavior.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats

def process_raw_data(raw_data: Dict) -> Dict:
    """
    Process raw experimental data.
    
    Args:
        raw_data: Dictionary containing raw experimental data
        
    Returns:
        Dictionary containing processed data
    """
    processed_data = {}
    
    # Clean and normalize metrics
    processed_data['metrics'] = clean_metrics_data(raw_data['metrics'])
    
    # Aggregate results
    processed_data['aggregated'] = aggregate_results(raw_data)
    
    # Normalize metrics for comparison
    processed_data['normalized'] = normalize_metrics(processed_data['metrics'])
    
    return processed_data

def clean_metrics_data(metrics_data: Dict) -> Dict:
    """
    Clean and validate metrics data.
    
    Args:
        metrics_data: Dictionary containing raw metrics
        
    Returns:
        Dictionary containing cleaned metrics
    """
    cleaned_data = defaultdict(dict)
    
    for metric_type, measurements in metrics_data.items():
        # Remove invalid values
        valid_data = remove_invalid_values(measurements)
        
        # Remove outliers
        cleaned = remove_outliers(valid_data)
        
        # Fill missing values if necessary
        filled_data = fill_missing_values(cleaned)
        
        cleaned_data[metric_type] = filled_data
        
    return dict(cleaned_data)

def aggregate_results(raw_data: Dict) -> Dict:
    """
    Aggregate results across experiments.
    
    Args:
        raw_data: Dictionary containing raw experimental data
        
    Returns:
        Dictionary containing aggregated results
    """
    aggregated = defaultdict(lambda: defaultdict(list))
    
    # Aggregate metrics by category and algorithm
    for scenario, data in raw_data.items():
        for algo, metrics in data['algorithm_results'].items():
            for metric_type, values in metrics.items():
                aggregated[metric_type][algo].extend(values)
                
    # Calculate summary statistics
    summary = {}
    for metric_type, algo_data in aggregated.items():
        summary[metric_type] = {
            algo: calculate_summary_stats(values)
            for algo, values in algo_data.items()
        }
        
    return dict(summary)

def normalize_metrics(metrics_data: Dict) -> Dict:
    """
    Normalize metrics for fair comparison.
    
    Args:
        metrics_data: Dictionary containing metrics data
        
    Returns:
        Dictionary containing normalized metrics
    """
    normalized = defaultdict(dict)
    
    for metric_type, measurements in metrics_data.items():
        if isinstance(measurements, dict):
            # Handle nested dictionary structure
            for algo, values in measurements.items():
                if isinstance(values, (list, np.ndarray)):
                    normalized[metric_type][algo] = normalize_values(values)
                else:
                    normalized[metric_type][algo] = values
        elif isinstance(measurements, (list, np.ndarray)):
            # Handle direct list of values
            normalized[metric_type] = normalize_values(measurements)
            
    return dict(normalized)

def remove_invalid_values(data: Union[Dict, List, np.ndarray]) -> Union[Dict, List, np.ndarray]:
    """
    Remove invalid values from data.
    
    Args:
        data: Input data structure
        
    Returns:
        Data structure with invalid values removed
    """
    if isinstance(data, dict):
        return {
            k: remove_invalid_values(v)
            for k, v in data.items()
        }
    elif isinstance(data, list):
        return [
            x for x in data
            if isinstance(x, (int, float)) and not np.isnan(x) and not np.isinf(x)
        ]
    elif isinstance(data, np.ndarray):
        return data[~np.isnan(data) & ~np.isinf(data)]
    else:
        return data

def remove_outliers(
    data: Union[List, np.ndarray],
    threshold: float = 3.0
) -> np.ndarray:
    """
    Remove outliers using z-score method.
    
    Args:
        data: Input data
        threshold: Z-score threshold for outlier detection
        
    Returns:
        Array with outliers removed
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
        
    z_scores = np.abs(stats.zscore(data))
    return data[z_scores < threshold]

def fill_missing_values(
    data: Union[Dict, List, np.ndarray]
) -> Union[Dict, List, np.ndarray]:
    """
    Fill missing values in data.
    
    Args:
        data: Input data structure
        
    Returns:
        Data structure with missing values filled
    """
    if isinstance(data, dict):
        return {
            k: fill_missing_values(v)
            for k, v in data.items()
        }
    elif isinstance(data, (list, np.ndarray)):
        data_array = np.array(data)
        # Fill with median for numeric data
        if np.issubdtype(data_array.dtype, np.number):
            median = np.nanmedian(data_array)
            data_array[np.isnan(data_array)] = median
        return data_array
    else:
        return data

def normalize_values(
    values: Union[List, np.ndarray],
    method: str = 'minmax'
) -> np.ndarray:
    """
    Normalize values using specified method.
    
    Args:
        values: Input values to normalize
        method: Normalization method ('minmax' or 'zscore')
        
    Returns:
        Normalized values
    """
    if not isinstance(values, np.ndarray):
        values = np.array(values)
        
    if method == 'minmax':
        min_val = np.min(values)
        max_val = np.max(values)
        if max_val == min_val:
            return np.zeros_like(values)
        return (values - min_val) / (max_val - min_val)
        
    elif method == 'zscore':
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return np.zeros_like(values)
        return (values - mean) / std
        
    else:
        raise ValueError("Normalization method must be 'minmax' or 'zscore'")

def calculate_summary_stats(values: Union[List, np.ndarray]) -> Dict:
    """
    Calculate summary statistics for values.
    
    Args:
        values: Input values
        
    Returns:
        Dictionary containing summary statistics
    """
    values = np.array(values)
    
    return {
        'mean': np.mean(values),
        'median': np.median(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'percentiles': {
            '25': np.percentile(values, 25),
            '50': np.percentile(values, 50),
            '75': np.percentile(values, 75),
            '95': np.percentile(values, 95),
            '99': np.percentile(values, 99)
        },
        'distribution': {
            'skewness': stats.skew(values),
            'kurtosis': stats.kurtosis(values)
        }
    }

def process_time_series(time_series_data: Dict) -> Dict:
    """
    Process time series data for analysis.
    
    Args:
        time_series_data: Dictionary containing time series data
        
    Returns:
        Dictionary containing processed time series
    """
    processed = defaultdict(dict)
    
    for metric, series in time_series_data.items():
        # Remove trends
        detrended = remove_trend(series)
        
        # Smooth series
        smoothed = smooth_series(detrended)
        
        # Calculate rolling statistics
        rolling_stats = calculate_rolling_stats(smoothed)
        
        processed[metric] = {
            'detrended': detrended,
            'smoothed': smoothed,
            'rolling_stats': rolling_stats
        }
        
    return dict(processed)

def remove_trend(
    series: np.ndarray,
    method: str = 'linear'
) -> np.ndarray:
    """
    Remove trend from time series.
    
    Args:
        series: Input time series
        method: Detrending method ('linear' or 'polynomial')
        
    Returns:
        Detrended series
    """
    x = np.arange(len(series))
    
    if method == 'linear':
        # Linear detrending
        slope, intercept = np.polyfit(x, series, 1)
        trend = slope * x + intercept
        return series - trend
        
    elif method == 'polynomial':
        # Polynomial detrending
        coeffs = np.polyfit(x, series, 2)
        trend = np.polyval(coeffs, x)
        return series - trend
        
    else:
        raise ValueError("Method must be 'linear' or 'polynomial'")

def smooth_series(
    series: np.ndarray,
    window: int = 5,
    method: str = 'moving_average'
) -> np.ndarray:
    """
    Smooth time series data.
    
    Args:
        series: Input time series
        window: Smoothing window size
        method: Smoothing method ('moving_average' or 'exponential')
        
    Returns:
        Smoothed series
    """
    if method == 'moving_average':
        return np.convolve(series, np.ones(window)/window, mode='valid')
        
    elif method == 'exponential':
        alpha = 2.0 / (window + 1)
        smoothed = np.zeros_like(series)
        smoothed[0] = series[0]
        for i in range(1, len(series)):
            smoothed[i] = alpha * series[i] + (1 - alpha) * smoothed[i-1]
        return smoothed
        
    else:
        raise ValueError("Method must be 'moving_average' or 'exponential'")

def calculate_rolling_stats(
    series: np.ndarray,
    window: int = 10
) -> Dict:
    """
    Calculate rolling statistics for time series.
    
    Args:
        series: Input time series
        window: Window size for rolling calculations
        
    Returns:
        Dictionary containing rolling statistics
    """
    rolling_mean = np.array([
        np.mean(series[max(0, i-window):i+1])
        for i in range(len(series))
    ])
    
    rolling_std = np.array([
        np.std(series[max(0, i-window):i+1])
        for i in range(len(series))
    ])
    
    return {
        'mean': rolling_mean,
        'std': rolling_std,
        'upper_bound': rolling_mean + 2*rolling_std,
        'lower_bound': rolling_mean - 2*rolling_std
    }

def process_categorical_data(
    categorical_data: Dict
) -> Dict:
    """
    Process categorical data from experiments.
    
    Args:
        categorical_data: Dictionary containing categorical data
        
    Returns:
        Dictionary containing processed categorical data
    """
    processed = defaultdict(dict)
    
    for category, data in categorical_data.items():
        # Calculate frequencies
        frequencies = calculate_frequencies(data)
        
        # Calculate proportions
        proportions = {
            k: v/len(data) for k, v in frequencies.items()
        }
        
        # Calculate entropy
        entropy = stats.entropy(list(proportions.values()))
        
        processed[category] = {
            'frequencies': frequencies,
            'proportions': proportions,
            'entropy': entropy,
            'mode': max(frequencies.items(), key=lambda x: x[1])[0]
        }
        
    return dict(processed)

def calculate_frequencies(
    data: List
) -> Dict:
    """
    Calculate frequencies of categorical values.
    
    Args:
        data: List of categorical values
        
    Returns:
        Dictionary containing value frequencies
    """
    frequencies = defaultdict(int)
    for value in data:
        frequencies[value] += 1
    return dict(frequencies)