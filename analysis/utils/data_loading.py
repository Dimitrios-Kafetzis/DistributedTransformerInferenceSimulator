from pathlib import Path
from typing import Dict, List, Optional, Union
import json
import yaml
import pandas as pd
import numpy as np
from collections import defaultdict

def load_experimental_data(
    results_dir: Union[str, Path],
    experiment_name: Optional[str] = None
) -> Dict:
    """
    Load experimental results from directory.
    
    Args:
        results_dir: Directory containing experiment results
        experiment_name: Optional specific experiment to load
        
    Returns:
        Dictionary containing loaded experimental data
    """
    results_dir = Path(results_dir)
    
    # Load main results file
    results_file = results_dir / 'experiment_results.json'
    if not results_file.exists():
        raise FileNotFoundError(f"No results file found at {results_file}")
        
    with open(results_file, 'r') as f:
        results = json.load(f)
        
    # Load specific experiment if requested
    if experiment_name:
        if experiment_name not in results:
            raise ValueError(f"Experiment {experiment_name} not found in results")
        return results[experiment_name]
        
    return results

def load_metrics_data(metrics_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Load metrics data from JSONL files.
    
    Args:
        metrics_dir: Directory containing metrics files
        
    Returns:
        DataFrame containing all metrics data
    """
    metrics_dir = Path(metrics_dir)
    
    # Find all metrics files
    metric_files = list(metrics_dir.glob('*.jsonl'))
    if not metric_files:
        raise FileNotFoundError(f"No metric files found in {metrics_dir}")
        
    # Load and combine all metrics
    dfs = []
    for file in metric_files:
        df = pd.read_json(file, lines=True)
        df['source_file'] = file.stem
        dfs.append(df)
        
    return pd.concat(dfs, ignore_index=True)

def load_configuration(
    config_path: Union[str, Path]
) -> Dict:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
    with open(config_path, 'r') as f:
        if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
            return yaml.safe_load(f)
        elif config_path.suffix == '.json':
            return json.load(f)
        else:
            raise ValueError("Configuration file must be YAML or JSON")

def ensure_directory(directory: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if necessary.
    
    Args:
        directory: Directory path to ensure
        
    Returns:
        Path object for the directory
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

def load_scenario_results(
    results_dir: Union[str, Path],
    scenario_type: str
) -> Dict:
    """
    Load results for a specific scenario type.
    
    Args:
        results_dir: Base results directory
        scenario_type: Type of scenario to load
        
    Returns:
        Dictionary containing scenario results
    """
    results_dir = Path(results_dir)
    scenario_dir = results_dir / scenario_type
    
    if not scenario_dir.exists():
        raise FileNotFoundError(f"No results found for scenario {scenario_type}")
        
    # Load metrics data
    metrics = load_metrics_data(scenario_dir / 'metrics')
    
    # Load scenario configuration
    config = load_configuration(scenario_dir / 'config.yaml')
    
    return {
        'metrics': metrics,
        'config': config,
        'type': scenario_type
    }

def load_baseline_comparison(results_dir: Union[str, Path]) -> Dict:
    """
    Load baseline comparison results.
    
    Args:
        results_dir: Base results directory
        
    Returns:
        Dictionary containing baseline comparison results
    """
    results_dir = Path(results_dir)
    baselines_dir = results_dir / 'baselines'
    
    if not baselines_dir.exists():
        raise FileNotFoundError("No baseline comparison results found")
        
    # Load results for each baseline
    baseline_results = {}
    for baseline_dir in baselines_dir.iterdir():
        if baseline_dir.is_dir():
            baseline_results[baseline_dir.name] = {
                'metrics': load_metrics_data(baseline_dir / 'metrics'),
                'config': load_configuration(baseline_dir / 'config.yaml')
            }
            
    return baseline_results

def load_algorithm_results(results_dir: Union[str, Path]) -> Dict:
    """
    Load results for different algorithms.
    
    Args:
        results_dir: Base results directory
        
    Returns:
        Dictionary containing algorithm results
    """
    # Load all experiment results
    all_results = load_experimental_data(results_dir)
    
    # Extract algorithm-specific results
    algorithm_results = defaultdict(list)
    
    for scenario in all_results.values():
        for algo, results in scenario['algorithm_results'].items():
            algorithm_results[algo].extend(results['metrics'])
            
    return dict(algorithm_results)

def save_processed_results(
    results: Dict,
    output_dir: Union[str, Path],
    format: str = 'json'
) -> None:
    """
    Save processed results to file.
    
    Args:
        results: Results to save
        output_dir: Directory to save results
        format: Output format ('json' or 'csv')
    """
    output_dir = ensure_directory(output_dir)
    
    if format == 'json':
        with open(output_dir / 'processed_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
    elif format == 'csv':
        # Convert results to DataFrame
        df = pd.DataFrame.from_dict(results, orient='index')
        df.to_csv(output_dir / 'processed_results.csv')
        
    else:
        raise ValueError("Format must be 'json' or 'csv'")