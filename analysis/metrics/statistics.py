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
    
    Args:
        data: Dictionary containing experimental measurements
        
    Returns:
        Dictionary containing statistical analysis results
    """
    stats_results = {}
    
    # Perform significance tests
    stats_results['significance'] = perform_significance_tests(data)
    
    # Calculate confidence intervals
    stats_results['confidence_intervals'] = calculate_confidence_intervals(data)
    
    # Analyze distributions
    stats_results['distributions'] = analyze_distributions(data)
    
    return stats_results

def perform_significance_tests(data: Dict) -> Dict:
    """
    Perform statistical significance tests between algorithms.
    
    Args:
        data: Dictionary containing experimental measurements
        
    Returns:
        Dictionary containing significance test results
    """
    significance_results = {}
    
    # Perform pairwise tests between algorithms
    significance_results['pairwise'] = perform_pairwise_tests(
        data['algorithm_results']
    )
    
    # Perform ANOVA tests
    significance_results['anova'] = perform_anova_tests(
        data['algorithm_results']
    )
    
    # Perform Kruskal-Wallis tests for non-normal distributions
    significance_results['kruskal'] = perform_kruskal_tests(
        data['algorithm_results']
    )
    
    return significance_results

def calculate_confidence_intervals(data: Dict) -> Dict:
    """
    Calculate confidence intervals for various metrics.
    
    Args:
        data: Dictionary containing experimental measurements
        
    Returns:
        Dictionary containing confidence intervals
    """
    ci_results = {}
    
    # Calculate CIs for each metric
    for metric_name, metric_data in data['metrics'].items():
        ci_results[metric_name] = {
            algo: calculate_metric_ci(measurements)
            for algo, measurements in metric_data.items()
        }
        
    # Calculate bootstrap CIs for complex metrics
    ci_results['bootstrap'] = calculate_bootstrap_intervals(
        data['complex_metrics']
    )
    
    return ci_results

def analyze_distributions(data: Dict) -> Dict:
    """
    Analyze statistical distributions of experimental results.
    
    Args:
        data: Dictionary containing experimental measurements
        
    Returns:
        Dictionary containing distribution analysis
    """
    distribution_results = {}
    
    # Test for normality
    distribution_results['normality'] = test_normality(
        data['algorithm_results']
    )
    
    # Analyze distribution characteristics
    distribution_results['characteristics'] = analyze_distribution_characteristics(
        data['algorithm_results']
    )
    
    return distribution_results

def perform_pairwise_tests(algorithm_results: Dict) -> Dict:
    """
    Perform pairwise statistical tests between algorithms.
    
    Args:
        algorithm_results: Dictionary with results for each algorithm
        
    Returns:
        Dictionary containing pairwise test results
    """
    pairwise_results = {}
    
    # Get all algorithm pairs
    algorithms = list(algorithm_results.keys())
    n_algorithms = len(algorithms)
    
    for i in range(n_algorithms):
        for j in range(i + 1, n_algorithms):
            algo1, algo2 = algorithms[i], algorithms[j]
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(
                algorithm_results[algo1],
                algorithm_results[algo2]
            )
            
            # Calculate effect size (Cohen's d)
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
    
    Args:
        algorithm_results: Dictionary with results for each algorithm
        
    Returns:
        Dictionary containing ANOVA results
    """
    # Prepare data for ANOVA
    groups = [results for results in algorithm_results.values()]
    
    # Perform one-way ANOVA
    f_stat, p_value = stats.f_oneway(*groups)
    
    # Perform Tukey's HSD test for multiple comparisons
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
    
    Args:
        algorithm_results: Dictionary with results for each algorithm
        
    Returns:
        Dictionary containing Kruskal-Wallis test results
    """
    # Prepare data for Kruskal-Wallis test
    groups = [results for results in algorithm_results.values()]
    
    # Perform Kruskal-Wallis H-test
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
    Calculate confidence interval for a metric.
    
    Args:
        measurements: Array of metric measurements
        confidence_level: Confidence level (default: 0.95)
        
    Returns:
        Dictionary containing confidence interval information
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
    
    Args:
        metric_data: Dictionary containing metric data
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level
        
    Returns:
        Dictionary containing bootstrap intervals
    """
    bootstrap_results = {}
    
    for metric_name, values in metric_data.items():
        # Perform bootstrap sampling
        bootstrap_samples = np.random.choice(
            values,
            size=(n_bootstrap, len(values)),
            replace=True
        )
        
        # Calculate statistic for each bootstrap sample
        bootstrap_stats = np.mean(bootstrap_samples, axis=1)
        
        # Calculate confidence intervals
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
    Test normality of distributions for each algorithm.
    
    Args:
        algorithm_results: Dictionary with results for each algorithm
        
    Returns:
        Dictionary containing normality test results
    """
    normality_results = {}
    
    for algo, results in algorithm_results.items():
        # Shapiro-Wilk test
        w_stat, p_value = stats.shapiro(results)
        
        # Q-Q plot correlation
        qq_correlation = calculate_qq_correlation(results)
        
        normality_results[algo] = {
            'shapiro_statistic': w_stat,
            'shapiro_p_value': p_value,
            'qq_correlation': qq_correlation,
            'is_normal': p_value >= 0.05
        }
        
    return normality_results

def analyze_distribution_characteristics(algorithm_results: Dict) -> Dict:
    """
    Analyze characteristics of result distributions.
    
    Args:
        algorithm_results: Dictionary with results for each algorithm
        
    Returns:
        Dictionary containing distribution characteristics
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
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def calculate_qq_correlation(data: np.ndarray) -> float:
    """Calculate correlation coefficient for Q-Q plot"""
    sorted_data = np.sort(data)
    theoretical_quantiles = stats.norm.ppf(
        np.linspace(0.5/len(data), 1 - 0.5/len(data), len(data))
    )
    
    return stats.pearsonr(sorted_data, theoretical_quantiles)[0]