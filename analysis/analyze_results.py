from pathlib import Path
from typing import Dict, Optional
import json
from datetime import datetime

from .utils.data_loading import load_experiment_results, load_metrics_data
from .utils.processing import process_raw_data
from .metrics import (
    calculate_performance_metrics,
    calculate_resource_metrics,
    calculate_network_metrics,
    calculate_statistics
)
from .plotting import (
    create_performance_plots,
    create_resource_plots,
    create_network_plots,
    create_comparison_plots
)

def analyze_all_results(
    results_dir: Path,
    output_dir: Path,
    save_intermediate: bool = True
) -> Dict:
    """
    Analyze all experimental results and generate visualizations.
    
    Args:
        results_dir: Directory containing experiment results
        output_dir: Directory to save analysis results
        save_intermediate: Whether to save intermediate analysis results
        
    Returns:
        Dictionary containing all analysis results
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and process raw data
    raw_results = load_experiment_results(results_dir)
    processed_data = process_raw_data(raw_results)
    
    # Calculate metrics
    metrics = {
        'performance': calculate_performance_metrics(processed_data),
        'resources': calculate_resource_metrics(processed_data),
        'network': calculate_network_metrics(processed_data),
        'statistics': calculate_statistics(processed_data)
    }
    
    # Generate plots
    plots = {
        'performance': create_performance_plots(
            metrics['performance'],
            output_dir / 'plots/performance'
        ),
        'resources': create_resource_plots(
            metrics['resources'],
            output_dir / 'plots/resources'
        ),
        'network': create_network_plots(
            metrics['network'],
            output_dir / 'plots/network'
        ),
        'comparison': create_comparison_plots(
            metrics,
            output_dir / 'plots/comparison'
        )
    }
    
    # Compile analysis results
    analysis_results = {
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics,
        'plots': {
            category: [str(p) for p in plot_paths]
            for category, plot_paths in plots.items()
        },
        'summary': _create_summary(metrics)
    }
    
    # Save results
    if save_intermediate:
        _save_analysis_results(analysis_results, output_dir)
        
    return analysis_results

def analyze_specific_scenario(
    results_dir: Path,
    scenario_name: str,
    output_dir: Path,
    save_intermediate: bool = True
) -> Dict:
    """
    Analyze results for a specific scenario.
    
    Args:
        results_dir: Directory containing experiment results
        scenario_name: Name of scenario to analyze
        output_dir: Directory to save analysis results
        save_intermediate: Whether to save intermediate results
        
    Returns:
        Dictionary containing scenario analysis results
    """
    # Load scenario-specific data
    scenario_dir = results_dir / scenario_name
    scenario_data = load_metrics_data(scenario_dir)
    
    # Calculate metrics
    metrics = {
        'performance': calculate_performance_metrics(scenario_data),
        'resources': calculate_resource_metrics(scenario_data),
        'network': calculate_network_metrics(scenario_data),
        'statistics': calculate_statistics(scenario_data)
    }
    
    # Generate plots
    scenario_plots_dir = output_dir / 'plots' / scenario_name
    plots = {
        'performance': create_performance_plots(
            metrics['performance'],
            scenario_plots_dir / 'performance'
        ),
        'resources': create_resource_plots(
            metrics['resources'],
            scenario_plots_dir / 'resources'
        ),
        'network': create_network_plots(
            metrics['network'],
            scenario_plots_dir / 'network'
        )
    }
    
    # Compile results
    scenario_results = {
        'timestamp': datetime.now().isoformat(),
        'scenario': scenario_name,
        'metrics': metrics,
        'plots': {
            category: [str(p) for p in plot_paths]
            for category, plot_paths in plots.items()
        },
        'summary': _create_summary(metrics)
    }
    
    # Save results
    if save_intermediate:
        _save_scenario_results(scenario_results, output_dir, scenario_name)
        
    return scenario_results

def _create_summary(metrics: Dict) -> Dict:
    """Create summary of analysis results"""
    return {
        'performance': {
            'average_latency': metrics['performance'].get('average_latency'),
            'latency_std': metrics['performance'].get('latency_std'),
            'throughput': metrics['performance'].get('throughput')
        },
        'resources': {
            'peak_memory': metrics['resources'].get('peak_memory'),
            'average_cpu': metrics['resources'].get('average_cpu'),
            'resource_efficiency': metrics['resources'].get('efficiency')
        },
        'network': {
            'total_data_transferred': metrics['network'].get('total_data_transferred'),
            'average_bandwidth': metrics['network'].get('average_bandwidth'),
            'communication_overhead': metrics['network'].get('overhead')
        },
        'statistical_significance': metrics['statistics'].get('significance_tests')
    }

def _save_analysis_results(results: Dict, output_dir: Path) -> None:
    """Save analysis results to files"""
    # Save main results
    with open(output_dir / 'analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    # Save summary separately
    with open(output_dir / 'analysis_summary.json', 'w') as f:
        json.dump(results['summary'], f, indent=2)
        
def _save_scenario_results(
    results: Dict,
    output_dir: Path,
    scenario_name: str
) -> None:
    """Save scenario-specific results"""
    scenario_dir = output_dir / scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    with open(scenario_dir / 'analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    # Save summary
    with open(scenario_dir / 'summary.json', 'w') as f:
        json.dump(results['summary'], f, indent=2)