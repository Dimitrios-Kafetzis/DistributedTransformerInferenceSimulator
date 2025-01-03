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
    
This script orchestrates scenario execution by loading specified configuration files,
setting up the environment, and logging final results for multiple scenario types
(e.g., edge cluster, distributed edge, hybrid cloud, and baseline comparisons).
"""

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
from experiments.scenarios.common import ScenarioResult

def setup_directories(base_dir: str = "results") -> dict:
    """Create necessary directories for results, timestamped to avoid overwrites."""
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

def scenario_result_to_dict(scenario_result: ScenarioResult) -> dict:
    """
    Convert a ScenarioResult object to a JSON-serializable dictionary.
    This avoids the 'Object of type ScenarioResult is not JSON serializable' error.
    """
    return {
        'scenario_name': scenario_result.scenario_name,
        'start_time': scenario_result.start_time.isoformat(),
        'end_time': scenario_result.end_time.isoformat(),
        'metrics': scenario_result.metrics,       # metrics is presumably a dict -> OK
        'success': scenario_result.success,
        'error': scenario_result.error
    }

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

        # Load config for each scenario
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
    
    # Each config in the list -> run all baselines
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
        
        # baseline_results -> dict of {ScenarioName: ScenarioResult}
        baseline_results_dict = run_all_baselines(
            config_path=config_path,
            output_dir=config_dir,
            logger=logger
        )
        
        # Convert each scenario result to dict for JSON
        for scenario_name, sr in baseline_results_dict.items():
            baseline_results_dict[scenario_name] = scenario_result_to_dict(sr)
        
        # Analyze results (the function expects old format but we can pass the
        # dictionary-of-ScenarioResult-dicts if needed).
        # If 'analyze_baseline_results' is expecting ScenarioResult objects,
        # we can adapt it or revert to the old approach. Let's adapt it:
        analysis = analyze_baseline_results(
            results=baseline_results_dict,  # dictionary but each scenario is now a dict
            output_dir=config_dir,
            logger=logger
        )
        
        results[config_name] = {
            'baseline_results': baseline_results_dict,  # dict of scenario result dicts
            'analysis': analysis
        }
        
    return results

def build_experiment_summary(full_results: dict) -> dict:
    """
    Build a summary that aggregates success counts across all scenarios.
    'full_results' is the final dictionary with structure:
      {
        'edge_cluster': {
          'basic': {...}, 'scalability': {...}, 'failure': {...}
        },
        'distributed_edge': {...},
        'hybrid_cloud': {...},
        'baselines': {
          'edge_cluster': {
             'baseline_results': {
                'GreedyBaselineScenario': {...}, ...
             },
             'analysis': {...}
          },
          ...
        }
      }
    We'll parse each scenario's 'success' entry to compute overall success counts.
    """
    total_scenarios = 0
    total_successes = 0

    # helper function
    def parse_scenario_dict(sd: dict):
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
        Return success as bool or None if absent
        """
        if 'success' in sd and isinstance(sd['success'], bool):
            return sd['success']
        return False

    # 1) edge_cluster, distributed_edge, hybrid_cloud
    for top_key in ['edge_cluster', 'distributed_edge', 'hybrid_cloud']:
        if top_key not in full_results:
            continue
        # each is a dict of scenario_name -> scenario_result_dict
        scenario_dict = full_results[top_key]
        for scn, scn_res in scenario_dict.items():
            # scn_res is like {'scenario_name':..., 'success':..., ...}
            total_scenarios += 1
            s = parse_scenario_dict(scn_res)
            if s:
                total_successes += 1

    # 2) baselines
    if 'baselines' in full_results:
        # this is a dict of config_name -> { 'baseline_results': {...}, 'analysis': {...} }
        for config_name, baseline_dict in full_results['baselines'].items():
            if 'baseline_results' not in baseline_dict:
                continue
            scenario_dict = baseline_dict['baseline_results']  # e.g. 'GreedyBaselineScenario': {...}
            for scn, scn_res in scenario_dict.items():
                total_scenarios += 1
                s = parse_scenario_dict(scn_res)
                if s:
                    total_successes += 1

    if total_scenarios == 0:
        success_rate = 0.0
    else:
        success_rate = float(total_successes) / float(total_scenarios)

    summary = {
        'timestamp': datetime.now().isoformat(),
        'num_experiments': total_scenarios,
        'total_success': total_successes,
        'success_rate': success_rate
    }
    return summary

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
