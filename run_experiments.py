#!/usr/bin/env python3
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
# File: run_experiments.py
# Description:
#   This file provides the main entry-point for running various experiments
#   in the Transformer Inference Simulator. It parses command-line arguments,
#   loads configurations, executes specified scenarios, and saves results.
#
# ---------------------------------------------------------------------------

"""
Additional docstring if needed:

Example usage:
    ./run_experiments.py --output-dir results/
    # or specify a particular scenario:
    ./run_experiments.py --output-dir results/ --scenario toy
"""

import os
from pathlib import Path
from datetime import datetime
import json
import argparse

from src.utils import SimulationLogger, LogLevel, load_config
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
from experiments.scenarios.common import ScenarioResult

# Import toy scenarios (original "ToyScenario" and new "ToyComparisonScenario")
try:
    from experiments.scenarios.toy_scenarios import ToyScenario
    from experiments.scenarios.toy_comparison_scenarios import ToyComparisonScenario
    TOY_AVAILABLE = True
except ImportError:
    # If you haven't created toy_scenarios.py or the new class yet, this will fail
    TOY_AVAILABLE = False


def setup_directories(base_dir: str = "results") -> dict:
    """Create necessary directories for results, timestamped to avoid overwrites."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dirs = {
        'base': Path(base_dir) / timestamp,
        'toy': Path(base_dir) / timestamp / 'toy',
        'toy_comparison': Path(base_dir) / timestamp / 'toy_comparison',
        'edge_cluster': Path(base_dir) / timestamp / 'edge_cluster',
        'distributed_edge': Path(base_dir) / timestamp / 'distributed_edge',
        'hybrid_cloud': Path(base_dir) / timestamp / 'hybrid_cloud',
        'baselines': Path(base_dir) / timestamp / 'baselines'
    }
    
    # Create all directories
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        
    return dirs


def scenario_result_to_dict(scenario_result: ScenarioResult) -> dict:
    """
    Convert a ScenarioResult object to a JSON-serializable dictionary.
    This avoids the 'Object of type ScenarioResult is not JSON serializable' error.
    """
    return {
        'scenario_name': scenario_result.scenario_name,
        'start_time': scenario_result.start_time.isoformat(),
        'end_time': scenario_result.end_time.isoformat(),
        'metrics': scenario_result.metrics,  # presumably a dict
        'success': scenario_result.success,
        'error': scenario_result.error
    }


def run_toy_experiments(output_dir: Path, logger: SimulationLogger) -> dict:
    """
    Run the 'ToyScenario' (a small toy example scenario) if available.
    This scenario just does random usage, no baseline comparison.
    """
    results = {}
    if not TOY_AVAILABLE:
        logger.log_event("experiment", "ToyScenario not available - skipping")
        return results
    
    logger.log_event("experiment", "Starting toy scenario experiments")

    # Example config path for your toy scenario
    config_path = "experiments/configs/toy_example.yaml"

    scenario_dir = output_dir
    scenario_dir.mkdir(exist_ok=True)

    logger.log_event("scenario", "Running ToyScenario (random usage)")

    # 1) ToyScenario (purely random usage)
    scenario_config = load_config(config_path)
    scenario = ToyScenario(
        config=scenario_config,
        output_dir=scenario_dir,
        logger=logger
    )
    
    sr = scenario.execute()  # ScenarioResult
    results["toy_random"] = scenario_result_to_dict(sr)
    return results


def run_toy_comparison_experiments(output_dir: Path, logger: SimulationLogger) -> dict:
    """
    Run the 'ToyComparisonScenario' which compares Baseline vs Resource-Aware in a toy environment.
    """
    results = {}
    if not TOY_AVAILABLE:
        logger.log_event("experiment", "ToyComparisonScenario not available - skipping")
        return results
    
    logger.log_event("experiment", "Starting toy comparison scenario experiments")

    # We can reuse the same toy_example.yaml or create a toy_comparison.yaml
    config_path = "experiments/configs/toy_comparison.yaml"

    scenario_dir = output_dir
    scenario_dir.mkdir(exist_ok=True)

    logger.log_event("scenario", "Running ToyComparisonScenario (baselines vs. resource-aware)")

    # 2) ToyComparisonScenario
    scenario_config = load_config(config_path)
    scenario = ToyComparisonScenario(
        config=scenario_config,
        output_dir=scenario_dir,
        logger=logger
    )
    
    sr = scenario.execute()  # ScenarioResult
    results["toy_comparison"] = scenario_result_to_dict(sr)
    return results


def run_edge_cluster_experiments(output_dir: Path, logger: SimulationLogger) -> dict:
    """Run all edge cluster experiments (basic, scalability, failure)."""
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

        scenario_config = load_config(config_path)
        scenario = scenario_class(
            config=scenario_config,
            output_dir=scenario_dir,
            logger=logger
        )
        
        sr = scenario.execute()  # ScenarioResult
        results[name] = scenario_result_to_dict(sr)
        
    return results


def run_distributed_edge_experiments(output_dir: Path, logger: SimulationLogger) -> dict:
    """Run all distributed edge experiments (basic, communication, heterogeneity)."""
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

        scenario_config = load_config(config_path)
        scenario = scenario_class(
            config=scenario_config,
            output_dir=scenario_dir,
            logger=logger
        )
        
        sr = scenario.execute()
        results[name] = scenario_result_to_dict(sr)
        
    return results


def run_hybrid_cloud_experiments(output_dir: Path, logger: SimulationLogger) -> dict:
    """Run all hybrid cloud experiments (basic, tier-balancing, latency)."""
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

        scenario_config = load_config(config_path)
        scenario = scenario_class(
            config=scenario_config,
            output_dir=scenario_dir,
            logger=logger
        )
        
        sr = scenario.execute()
        results[name] = scenario_result_to_dict(sr)
        
    return results


def run_baseline_comparisons(output_dir: Path, logger: SimulationLogger) -> dict:
    """
    Run baseline comparison experiments across multiple configurations:
    edge_cluster, distributed_edge, hybrid_cloud.
    Then analyze results for each config in place.
    """
    logger.log_event("experiment", "Starting baseline comparisons")
    
    configs = [
        ('edge_cluster', "experiments/configs/edge_cluster.yaml"),
        ('distributed_edge', "experiments/configs/distributed_edge.yaml"),
        ('hybrid_cloud', "experiments/configs/hybrid_cloud_edge.yaml")
    ]
    
    results = {}
    for config_name, config_path in configs:
        config_dir = output_dir / config_name
        config_dir.mkdir(parents=True, exist_ok=True)
        
        logger.log_event("baseline", f"Running baselines for {config_name}")
        
        # baseline_results -> dict of {ScenarioName: ScenarioResult}
        baseline_results_dict = run_all_baselines(
            config_path=config_path,
            output_dir=config_dir,
            logger=logger
        )
        
        # Convert each scenario result to dict
        for scenario_name, sr in baseline_results_dict.items():
            baseline_results_dict[scenario_name] = scenario_result_to_dict(sr)
        
        # Analyze
        analysis = analyze_baseline_results(
            results=baseline_results_dict,
            output_dir=config_dir,
            logger=logger
        )
        
        results[config_name] = {
            'baseline_results': baseline_results_dict,
            'analysis': analysis
        }
        
    return results


def build_experiment_summary(full_results: dict) -> dict:
    """
    Build a summary that aggregates success counts across all scenarios.
    'full_results' is the final dictionary with structure:
      {
        'toy': {...},
        'toy_comparison': {...},
        'edge_cluster': {...},
        'distributed_edge': {...},
        'hybrid_cloud': {...},
        'baselines': {...}
      }
    We'll parse each scenario's 'success' entry to compute overall success counts.
    """
    total_scenarios = 0
    total_successes = 0

    def parse_scenario_dict(sd: dict) -> bool:
        """
        Each scenario dict is like:
           {
             'scenario_name': 'EdgeClusterBasicScenario',
             'start_time': '...',
             'end_time': '...',
             'metrics': {...},
             'success': True/False,
             'error': ...
           }
        Return True if success, else False.
        """
        return bool(sd.get('success', False))

    # 1) toy
    if 'toy' in full_results and isinstance(full_results['toy'], dict):
        for scn_name, scn_res in full_results['toy'].items():
            total_scenarios += 1
            if parse_scenario_dict(scn_res):
                total_successes += 1

    # 2) toy_comparison
    if 'toy_comparison' in full_results and isinstance(full_results['toy_comparison'], dict):
        for scn_name, scn_res in full_results['toy_comparison'].items():
            total_scenarios += 1
            if parse_scenario_dict(scn_res):
                total_successes += 1

    # 3) edge_cluster, distributed_edge, hybrid_cloud
    for top_key in ['edge_cluster', 'distributed_edge', 'hybrid_cloud']:
        if top_key in full_results and isinstance(full_results[top_key], dict):
            scenario_dict = full_results[top_key]
            for scn, scn_res in scenario_dict.items():
                total_scenarios += 1
                if parse_scenario_dict(scn_res):
                    total_successes += 1

    # 4) baselines
    if 'baselines' in full_results and isinstance(full_results['baselines'], dict):
        for config_name, baseline_dict in full_results['baselines'].items():
            baseline_results = baseline_dict.get('baseline_results', {})
            for scn_name, scn_res in baseline_results.items():
                total_scenarios += 1
                if parse_scenario_dict(scn_res):
                    total_successes += 1

    if total_scenarios == 0:
        success_rate = 0.0
    else:
        success_rate = float(total_successes) / float(total_scenarios)

    return {
        'timestamp': datetime.now().isoformat(),
        'num_experiments': total_scenarios,
        'total_success': total_successes,
        'success_rate': success_rate
    }


def save_experiment_results(full_results: dict, output_dir: Path):
    """Save all experiment results and a summary to JSON files."""
    # 1) Dump the entire nested results structure
    with open(output_dir / 'experiment_results.json', 'w') as f:
        json.dump(full_results, f, indent=2)

    # 2) Build summary
    summary = build_experiment_summary(full_results)

    # 3) Save summary
    with open(output_dir / 'experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Run transformer inference experiments')
    parser.add_argument('--output-dir', default='results',
                        help='Base directory for results')
    parser.add_argument('--configs-dir', default='experiments/configs',
                        help='Directory containing configuration files (not heavily used now)')
    parser.add_argument('--scenario',
                        choices=[
                            'toy',              # Only runs the random usage "ToyScenario"
                            'toy_comparison',   # Only runs the baseline vs. RA "ToyComparisonScenario"
                            'edge_cluster',
                            'distributed_edge',
                            'hybrid_cloud',
                            'baselines',
                            'all'
                        ],
                        default='all',
                        help="Which scenario to run (default: all).")
    args = parser.parse_args()
    
    # Setup directories
    dirs = setup_directories(args.output_dir)
    
    # Setup logging
    logger = SimulationLogger(
        name="transformer_inference_experiments",
        log_dir=dirs['base'] / 'logs',
        level=LogLevel.DEBUG,
        console_output=True,
        file_output=True
    )
    
    try:
        results = {}
        scenario_choice = args.scenario
        
        # If user wants "toy" or "all"
        if scenario_choice in ['toy', 'all']:
            results['toy'] = run_toy_experiments(dirs['toy'], logger)

        # If user wants "toy_comparison" or "all"
        if scenario_choice in ['toy_comparison', 'all']:
            results['toy_comparison'] = run_toy_comparison_experiments(dirs['toy_comparison'], logger)

        # If user wants "edge_cluster" or "all"
        if scenario_choice in ['edge_cluster', 'all']:
            results['edge_cluster'] = run_edge_cluster_experiments(dirs['edge_cluster'], logger)

        # If user wants "distributed_edge" or "all"
        if scenario_choice in ['distributed_edge', 'all']:
            results['distributed_edge'] = run_distributed_edge_experiments(dirs['distributed_edge'], logger)

        # If user wants "hybrid_cloud" or "all"
        if scenario_choice in ['hybrid_cloud', 'all']:
            results['hybrid_cloud'] = run_hybrid_cloud_experiments(dirs['hybrid_cloud'], logger)
        
        # If user wants "baselines" or "all"
        if scenario_choice in ['baselines', 'all']:
            results['baselines'] = run_baseline_comparisons(dirs['baselines'], logger)

        # Save all results
        save_experiment_results(results, dirs['base'])
        
        logger.log_event("complete", "Requested experiments completed successfully")
        
    except Exception as e:
        logger.log_error("experiment_error", f"Error running experiments: {str(e)}")
        raise
        
    finally:
        logger.cleanup()


if __name__ == "__main__":
    main()
