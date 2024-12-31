#!/usr/bin/env python3

import os
from pathlib import Path
from datetime import datetime
import json
import argparse

from src.utils import SimulationLogger, load_config
from experiments.scenarios import (
    # Main scenarios
    EdgeClusterBasicScenario,
    EdgeClusterScalabilityScenario,
    EdgeClusterFailureScenario,
    
    DistributedEdgeBasicScenario,
    DistributedEdgeCommunicationScenario,
    DistributedEdgeHeterogeneityScenario,
    
    HybridCloudBasicScenario,
    HybridCloudTierBalancingScenario,
    HybridCloudLatencyScenario,
    
    # Baseline scenarios
    run_all_baselines,
    analyze_baseline_results
)

def setup_directories(base_dir: str = "results") -> dict:
    """Create necessary directories for results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dirs = {
        'base': Path(base_dir) / timestamp,
        'edge_cluster': Path(base_dir) / timestamp / 'edge_cluster',
        'distributed_edge': Path(base_dir) / timestamp / 'distributed_edge',
        'hybrid_cloud': Path(base_dir) / timestamp / 'hybrid_cloud',
        'baselines': Path(base_dir) / timestamp / 'baselines'
    }
    
    # Create all directories
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        
    return dirs

def run_edge_cluster_experiments(output_dir: Path, logger: SimulationLogger) -> dict:
    """Run all edge cluster experiments"""
    logger.log_event("experiment", "Starting edge cluster experiments")
    
    config_path = "experiments/configs/edge_cluster.yaml"
    results = {}
    
    scenarios = [
        ('basic', EdgeClusterBasicScenario),
        ('scalability', EdgeClusterScalabilityScenario),
        ('failure', EdgeClusterFailureScenario)
    ]
    
    for name, scenario_class in scenarios:
        scenario_dir = output_dir / name
        scenario_dir.mkdir(exist_ok=True)
        
        logger.log_event("scenario", f"Running edge cluster {name} scenario")
        
        scenario = scenario_class(
            config_path=config_path,
            output_dir=scenario_dir,
            logger=logger
        )
        results[name] = scenario.execute()
        
    return results

def run_distributed_edge_experiments(output_dir: Path, logger: SimulationLogger) -> dict:
    """Run all distributed edge experiments"""
    logger.log_event("experiment", "Starting distributed edge experiments")
    
    config_path = "experiments/configs/distributed_edge.yaml"
    results = {}
    
    scenarios = [
        ('basic', DistributedEdgeBasicScenario),
        ('communication', DistributedEdgeCommunicationScenario),
        ('heterogeneity', DistributedEdgeHeterogeneityScenario)
    ]
    
    for name, scenario_class in scenarios:
        scenario_dir = output_dir / name
        scenario_dir.mkdir(exist_ok=True)
        
        logger.log_event("scenario", f"Running distributed edge {name} scenario")
        
        scenario = scenario_class(
            config_path=config_path,
            output_dir=scenario_dir,
            logger=logger
        )
        results[name] = scenario.execute()
        
    return results

def run_hybrid_cloud_experiments(output_dir: Path, logger: SimulationLogger) -> dict:
    """Run all hybrid cloud experiments"""
    logger.log_event("experiment", "Starting hybrid cloud experiments")
    
    config_path = "experiments/configs/hybrid_cloud_edge.yaml"
    results = {}
    
    scenarios = [
        ('basic', HybridCloudBasicScenario),
        ('tier_balancing', HybridCloudTierBalancingScenario),
        ('latency', HybridCloudLatencyScenario)
    ]
    
    for name, scenario_class in scenarios:
        scenario_dir = output_dir / name
        scenario_dir.mkdir(exist_ok=True)
        
        logger.log_event("scenario", f"Running hybrid cloud {name} scenario")
        
        scenario = scenario_class(
            config_path=config_path,
            output_dir=scenario_dir,
            logger=logger
        )
        results[name] = scenario.execute()
        
    return results

def run_baseline_comparisons(output_dir: Path, logger: SimulationLogger) -> dict:
    """Run baseline comparison experiments"""
    logger.log_event("experiment", "Starting baseline comparisons")
    
    # Run baselines for each configuration
    configs = [
        ('edge_cluster', "experiments/configs/edge_cluster.yaml"),
        ('distributed_edge', "experiments/configs/distributed_edge.yaml"),
        ('hybrid_cloud', "experiments/configs/hybrid_cloud_edge.yaml")
    ]
    
    results = {}
    for config_name, config_path in configs:
        config_dir = output_dir / config_name
        config_dir.mkdir(exist_ok=True)
        
        logger.log_event("baseline", f"Running baselines for {config_name}")
        
        # Run all baselines for this configuration
        baseline_results = run_all_baselines(
            config_path=config_path,
            output_dir=config_dir,
            logger=logger
        )
        
        # Analyze results
        analysis = analyze_baseline_results(
            results=baseline_results,
            output_dir=config_dir,
            logger=logger
        )
        
        results[config_name] = {
            'baseline_results': baseline_results,
            'analysis': analysis
        }
        
    return results

def save_experiment_results(results: dict, output_dir: Path):
    """Save all experiment results"""
    # Save main results
    with open(output_dir / 'experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'num_experiments': len(results),
        'success_rate': sum(1 for r in results.values() if r['success']) / len(results),
        'configurations_tested': list(results.keys())
    }
    
    with open(output_dir / 'experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Run transformer inference experiments')
    parser.add_argument('--output-dir', default='results',
                       help='Base directory for results')
    parser.add_argument('--configs-dir', default='experiments/configs',
                       help='Directory containing configuration files')
    args = parser.parse_args()
    
    # Setup directories
    dirs = setup_directories(args.output_dir)
    
    # Setup logging
    logger = SimulationLogger(
        name="transformer_inference_experiments",
        log_dir=dirs['base'] / 'logs'
    )
    
    try:
        # Run all experiments
        results = {
            'edge_cluster': run_edge_cluster_experiments(
                dirs['edge_cluster'], logger
            ),
            'distributed_edge': run_distributed_edge_experiments(
                dirs['distributed_edge'], logger
            ),
            'hybrid_cloud': run_hybrid_cloud_experiments(
                dirs['hybrid_cloud'], logger
            ),
            'baselines': run_baseline_comparisons(
                dirs['baselines'], logger
            )
        }
        
        # Save results
        save_experiment_results(results, dirs['base'])
        
        logger.log_event("complete", "All experiments completed successfully")
        
    except Exception as e:
        logger.log_error("experiment_error", f"Error running experiments: {str(e)}")
        raise
        
    finally:
        logger.cleanup()

if __name__ == "__main__":
    main()