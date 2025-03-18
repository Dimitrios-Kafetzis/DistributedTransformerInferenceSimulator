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
# File: analysis/metrics/statistics.py
# Description:
#   Statistical methods and significance tests for evaluating distributed Transformer 
#   inference experiments and comparing different algorithmic approaches.
#
# ---------------------------------------------------------------------------

"""
Provides statistical functions and significance tests such as ANOVA, Kruskal-Wallis,
Tukey's HSD, and confidence interval calculations. These utilities are used to analyze
and validate results from the simulation and compare metrics across various distribution
algorithms and settings.

Each function is designed to help users understand whether differences between experimental
results are statistically significant and to provide confidence intervals for the measured metrics.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def calculate_statistics(data: Dict) -> Dict:
    """
    Calculate comprehensive statistical metrics for experimental results.
    
    This function aggregates various statistical analyses (significance tests,
    confidence interval calculations, and distribution analysis) into one summary
    dictionary. It is used to assess the performance differences between different
    Transformer inference algorithms.
    
    Args:
        data: Dictionary containing experimental measurements. It must include keys like
              'algorithm_results', 'metrics', and 'complex_metrics' that hold the experimental data.
        
    Returns:
        Dictionary containing statistical analysis results for significance tests, 
        confidence intervals, and distribution analysis.
    """
    stats_results = {}
    
    # Perform significance tests to compare different algorithmic approaches.
    stats_results['significance'] = perform_significance_tests(data)
    
    # Calculate confidence intervals for the measured metrics.
    stats_results['confidence_intervals'] = calculate_confidence_intervals(data)
    
    # Analyze the statistical distributions of the results.
    stats_results['distributions'] = analyze_distributions(data)
    
    return stats_results

def perform_significance_tests(data: Dict) -> Dict:
    """
    Perform statistical significance tests between algorithms.
    
    This function compares the results from different algorithms using:
      - Pairwise t-tests: Tests whether the means of two independent samples differ significantly.
      - ANOVA (Analysis of Variance): Determines whether there are any statistically significant differences
        between the means of three or more independent (unrelated) groups.
      - Kruskal-Wallis H-test: A non-parametric method for testing whether samples originate from the same distribution.
    
    These tests help determine if the differences observed in experimental results are likely due
    to chance or reflect true differences in performance.
    
    Args:
        data: Dictionary containing experimental measurements, including a key 'algorithm_results'
              which is a dictionary mapping algorithm names to their result arrays.
        
    Returns:
        Dictionary containing significance test results for pairwise comparisons, ANOVA, and Kruskal-Wallis tests.
    """
    significance_results = {}
    
    # Perform pairwise statistical tests between each pair of algorithms.
    significance_results['pairwise'] = perform_pairwise_tests(
        data['algorithm_results']
    )
    
    # Perform ANOVA tests to determine if at least one algorithm's mean result is different.
    significance_results['anova'] = perform_anova_tests(
        data['algorithm_results']
    )
    
    # Perform Kruskal-Wallis tests for cases where data does not follow a normal distribution.
    significance_results['kruskal'] = perform_kruskal_tests(
        data['algorithm_results']
    )
    
    return significance_results

def calculate_confidence_intervals(data: Dict) -> Dict:
    """
    Calculate confidence intervals for various metrics.
    
    Confidence intervals provide a range within which the true metric value is likely to fall.
    This function calculates CIs for simple metrics using t-distribution and also computes
    bootstrap confidence intervals for complex metrics.
    
    Args:
        data: Dictionary containing experimental measurements. Expected keys include 'metrics'
              (a dictionary mapping metric names to algorithm measurement arrays) and 
              'complex_metrics' for more advanced statistics.
        
    Returns:
        Dictionary containing confidence intervals for each metric and bootstrap intervals
        for complex metrics.
    """
    ci_results = {}
    
    # Calculate confidence intervals for each simple metric.
    for metric_name, metric_data in data['metrics'].items():
        ci_results[metric_name] = {
            algo: calculate_metric_ci(measurements)
            for algo, measurements in metric_data.items()
        }
        
    # Calculate bootstrap confidence intervals for complex metrics.
    ci_results['bootstrap'] = calculate_bootstrap_intervals(
        data['complex_metrics']
    )
    
    return ci_results

def analyze_distributions(data: Dict) -> Dict:
    """
    Analyze statistical distributions of experimental results.
    
    This function checks the normality of the data and extracts key distribution characteristics
    (mean, median, standard deviation, skewness, kurtosis, percentiles) for the results of each algorithm.
    Such analysis is essential to choose the appropriate significance tests (e.g., parametric vs. non-parametric).
    
    Args:
        data: Dictionary containing experimental measurements, specifically 'algorithm_results'.
        
    Returns:
        Dictionary containing results from normality tests and distribution characteristics.
    """
    distribution_results = {}
    
    # Test whether the results for each algorithm follow a normal distribution.
    distribution_results['normality'] = test_normality(
        data['algorithm_results']
    )
    
    # Analyze key characteristics of the distributions.
    distribution_results['characteristics'] = analyze_distribution_characteristics(
        data['algorithm_results']
    )
    
    return distribution_results

def perform_pairwise_tests(algorithm_results: Dict) -> Dict:
    """
    Perform pairwise statistical tests between algorithms.
    
    For each unique pair of algorithms, this function performs a t-test to compare the means.
    It also calculates Cohen's d effect size to quantify the difference. The result indicates
    whether the differences between the two algorithms are statistically significant.
    
    Args:
        algorithm_results: Dictionary with result arrays for each algorithm.
        
    Returns:
        Dictionary containing t-statistics, p-values, effect sizes, and significance (p < 0.05)
        for each algorithm pair.
    """
    pairwise_results = {}
    
    # List all algorithm names
    algorithms = list(algorithm_results.keys())
    n_algorithms = len(algorithms)
    
    # Compare each pair only once
    for i in range(n_algorithms):
        for j in range(i + 1, n_algorithms):
            algo1, algo2 = algorithms[i], algorithms[j]
            
            # Perform independent t-test between the two algorithm result sets.
            t_stat, p_value = stats.ttest_ind(
                algorithm_results[algo1],
                algorithm_results[algo2]
            )
            
            # Calculate effect size (Cohen's d) to understand the magnitude of difference.
            effect_size = calculate_cohens_d(
                algorithm_results[algo1],
                algorithm_results[algo2]
            )
            
            pairwise_results[f"{algo1}_vs_{algo2}"] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'effect_size': effect_size,
                'significant': p_value < 0.05
            }
            
    return pairwise_results

def perform_anova_tests(algorithm_results: Dict) -> Dict:
    """
    Perform ANOVA tests across algorithms.
    
    Analysis of Variance (ANOVA) is used to determine whether there are statistically significant 
    differences between the means of three or more independent groups. This function first performs 
    a one-way ANOVA test and then uses Tukey's Honest Significant Difference (HSD) test to perform
    pairwise comparisons between group means.
    
    Args:
        algorithm_results: Dictionary with result arrays for each algorithm.
        
    Returns:
        Dictionary containing the ANOVA F-statistic, p-value, a flag for significance (p < 0.05),
        and the results from Tukey's HSD test including statistics, p-values, and rejection flags.
    """
    # Prepare groups for the ANOVA test.
    groups = [results for results in algorithm_results.values()]
    
    # One-way ANOVA to test if any group mean is significantly different.
    f_stat, p_value = stats.f_oneway(*groups)
    
    # Prepare data for Tukey's HSD for multiple comparisons.
    data = []
    labels = []
    for algo, results in algorithm_results.items():
        data.extend(results)
        labels.extend([algo] * len(results))
        
    tukey = pairwise_tukeyhsd(data, labels)
    
    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'tukey_results': {
            'statistics': tukey.statistic,
            'p_values': tukey.pvalues,
            'significant': tukey.reject
        }
    }

def perform_kruskal_tests(algorithm_results: Dict) -> Dict:
    """
    Perform Kruskal-Wallis H-tests for non-normal distributions.
    
    The Kruskal-Wallis H-test is a non-parametric method used when the assumption of normality
    is violated. It tests whether samples originate from the same distribution. This function
    is useful when experimental results do not follow a normal distribution.
    
    Args:
        algorithm_results: Dictionary with result arrays for each algorithm.
        
    Returns:
        Dictionary containing the Kruskal-Wallis H statistic, p-value, and a significance flag.
    """
    # Prepare groups for the test.
    groups = [results for results in algorithm_results.values()]
    
    # Perform the non-parametric Kruskal-Wallis H-test.
    h_stat, p_value = stats.kruskal(*groups)
    
    return {
        'h_statistic': h_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

def calculate_metric_ci(
    measurements: np.ndarray,
    confidence_level: float = 0.95
) -> Dict:
    """
    Calculate confidence interval for a metric using the t-distribution.
    
    Confidence intervals provide a range in which the true mean of the measurements is likely
    to fall. This function calculates the mean, standard error, and the lower and upper bounds
    of the confidence interval.
    
    Args:
        measurements: Array of metric measurements.
        confidence_level: Confidence level for the interval (default: 0.95).
        
    Returns:
        Dictionary with the mean, standard error, lower and upper bounds of the confidence interval,
        and the confidence level.
    """
    mean = np.mean(measurements)
    std_err = stats.sem(measurements)
    
    ci = stats.t.interval(
        confidence_level,
        len(measurements) - 1,
        loc=mean,
        scale=std_err
    )
    
    return {
        'mean': mean,
        'std_error': std_err,
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        'confidence_level': confidence_level
    }

def calculate_bootstrap_intervals(
    metric_data: Dict,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95
) -> Dict:
    """
    Calculate bootstrap confidence intervals for complex metrics.
    
    Bootstrap methods repeatedly resample the data with replacement to estimate the
    sampling distribution. This function computes the mean for each bootstrap sample
    and then derives the confidence interval from the percentile distribution of these means.
    
    Args:
        metric_data: Dictionary containing metric data (each key maps to an array of measurements).
        n_bootstrap: Number of bootstrap samples (default: 10,000).
        confidence_level: Confidence level for the interval.
        
    Returns:
        Dictionary containing, for each metric, the mean and the lower and upper bounds of the
        bootstrap confidence interval.
    """
    bootstrap_results = {}
    
    for metric_name, values in metric_data.items():
        # Generate bootstrap samples
        bootstrap_samples = np.random.choice(
            values,
            size=(n_bootstrap, len(values)),
            replace=True
        )
        
        # Compute the mean for each bootstrap sample
        bootstrap_stats = np.mean(bootstrap_samples, axis=1)
        
        # Calculate the confidence interval based on the percentiles
        ci_lower = np.percentile(
            bootstrap_stats,
            (1 - confidence_level) / 2 * 100
        )
        ci_upper = np.percentile(
            bootstrap_stats,
            (1 + confidence_level) / 2 * 100
        )
        
        bootstrap_results[metric_name] = {
            'mean': np.mean(values),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'confidence_level': confidence_level
        }
        
    return bootstrap_results

def test_normality(algorithm_results: Dict) -> Dict:
    """
    Test for normality of distributions for each algorithm.
    
    This function uses the Shapiro-Wilk test, which is a widely used method for testing the 
    null hypothesis that the data was drawn from a normal distribution. In addition, it computes
    the correlation coefficient of the Q-Q plot (quantile-quantile plot) as a supplementary measure.
    
    Args:
        algorithm_results: Dictionary with result arrays for each algorithm.
        
    Returns:
        Dictionary containing the Shapiro-Wilk test statistic, p-value, Q-Q plot correlation,
        and a boolean flag indicating if the data is considered normally distributed (p >= 0.05).
    """
    normality_results = {}
    
    for algo, results in algorithm_results.items():
        # Perform the Shapiro-Wilk test for normality.
        w_stat, p_value = stats.shapiro(results)
        
        # Calculate the correlation coefficient for the Q-Q plot.
        qq_correlation = calculate_qq_correlation(results)
        
        normality_results[algo] = {
            'shapiro_statistic': w_stat,
            'shapiro_p_value': p_value,
            'qq_correlation': qq_correlation,
            'is_normal': p_value >= 0.05  # Conventionally, p >= 0.05 means normality cannot be rejected.
        }
        
    return normality_results

def analyze_distribution_characteristics(algorithm_results: Dict) -> Dict:
    """
    Analyze key characteristics of the result distributions.
    
    This function calculates central tendency and dispersion metrics, including mean, median,
    standard deviation, skewness (asymmetry), kurtosis (tailedness), and selected percentiles.
    These statistics help in understanding the overall shape and variability of the data,
    which is important when selecting the appropriate statistical tests.
    
    Args:
        algorithm_results: Dictionary with result arrays for each algorithm.
        
    Returns:
        Dictionary with calculated statistics for each algorithm.
    """
    characteristics = {}
    
    for algo, results in algorithm_results.items():
        characteristics[algo] = {
            'mean': np.mean(results),
            'median': np.median(results),
            'std': np.std(results),
            'skewness': stats.skew(results),
            'kurtosis': stats.kurtosis(results),
            'percentiles': {
                '25': np.percentile(results, 25),
                '50': np.percentile(results, 50),
                '75': np.percentile(results, 75),
                '95': np.percentile(results, 95)
            }
        }
        
    return characteristics

# Helper functions
def calculate_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size for two groups.
    
    Cohen's d is a measure of the difference between two means in terms of standard deviation,
    and it quantifies the effect size. A larger absolute value indicates a larger difference.
    
    Args:
        group1: First group of measurements.
        group2: Second group of measurements.
        
    Returns:
        The Cohen's d value.
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Compute the pooled standard deviation.
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def calculate_qq_correlation(data: np.ndarray) -> float:
    """
    Calculate the correlation coefficient for a Q-Q plot.
    
    A Q-Q (quantile-quantile) plot compares the quantiles of the data against the quantiles of a 
    theoretical normal distribution. The Pearson correlation coefficient between the sorted data
    and the theoretical quantiles serves as an indicator of normality.
    
    Args:
        data: Array of measurements.
        
    Returns:
        The Pearson correlation coefficient between the data and the theoretical normal quantiles.
    """
    sorted_data = np.sort(data)
    theoretical_quantiles = stats.norm.ppf(
        np.linspace(0.5/len(data), 1 - 0.5/len(data), len(data))
    )
    
    return stats.pearsonr(sorted_data, theoretical_quantiles)[0]
