import pytest
import tempfile
from pathlib import Path
import yaml
import json
from copy import deepcopy

from src.environment import WorkloadType
from src.utils.config import (
    NetworkConfig,
    ResourceConfig,
    WorkloadConfig,
    AlgorithmConfig,
    ExperimentConfig,
    SimulationConfig,
    create_default_config,
    load_config,
    save_config,
    validate_config,
    merge_configs
)

@pytest.fixture
def network_config():
    """Create a network configuration"""
    return NetworkConfig(
        topology_type="edge_cluster",
        num_devices=8,
        min_bandwidth=1.0,
        max_bandwidth=10.0,
        edge_probability=0.3,
        seed=42
    )

@pytest.fixture
def resource_config():
    """Create a resource configuration"""
    return ResourceConfig(
        memory_mu=2.0,
        memory_sigma=0.5,
        memory_min=2.0,
        memory_max=16.0,
        compute_mu=5.0,
        compute_sigma=0.5,
        compute_min=10.0,
        compute_max=100.0,
        seed=42
    )

@pytest.fixture
def workload_config():
    """Create a workload configuration"""
    return WorkloadConfig(
        model_type=WorkloadType.SMALL,
        initial_sequence_lengths=[128, 256],
        generation_steps=[32, 64],
        precision_bytes=4,
        seed=42
    )

@pytest.fixture
def algorithm_config():
    """Create an algorithm configuration"""
    return AlgorithmConfig(
        migration_threshold=0.9,
        backtrack_limit=100,
        cache_placement_strategy="colocated",
        enable_dynamic_adjustment=True
    )

@pytest.fixture
def experiment_config():
    """Create an experiment configuration"""
    return ExperimentConfig(
        name="test_experiment",
        description="Test experiment configuration",
        num_runs=10,
        checkpoint_interval=5,
        time_limit=3600,
        metrics_output_dir="results/test",
        save_intermediate=True
    )

@pytest.fixture
def simulation_config(
    network_config,
    resource_config,
    workload_config,
    algorithm_config,
    experiment_config
):
    """Create a complete simulation configuration"""
    return SimulationConfig(
        network=network_config,
        resources=resource_config,
        workload=workload_config,
        algorithm=algorithm_config,
        experiment=experiment_config
    )

class TestNetworkConfig:
    def test_initialization(self, network_config):
        """Test network configuration initialization"""
        assert network_config.topology_type == "edge_cluster"
        assert network_config.num_devices == 8
        assert network_config.min_bandwidth == 1.0
        assert network_config.max_bandwidth == 10.0
        assert network_config.edge_probability == 0.3
        assert network_config.seed == 42

    def test_validation(self):
        """Test network configuration validation"""
        # Invalid device count
        with pytest.raises(ValueError):
            NetworkConfig(
                topology_type="edge_cluster",
                num_devices=0,
                min_bandwidth=1.0,
                max_bandwidth=10.0
            )

        # Invalid bandwidth
        with pytest.raises(ValueError):
            NetworkConfig(
                topology_type="edge_cluster",
                num_devices=8,
                min_bandwidth=10.0,  # min > max
                max_bandwidth=1.0
            )

        # Invalid probability
        with pytest.raises(ValueError):
            NetworkConfig(
                topology_type="edge_cluster",
                num_devices=8,
                min_bandwidth=1.0,
                max_bandwidth=10.0,
                edge_probability=2.0
            )

class TestResourceConfig:
    def test_initialization(self, resource_config):
        """Test resource configuration initialization"""
        assert resource_config.memory_mu == 2.0
        assert resource_config.memory_sigma == 0.5
        assert resource_config.memory_min == 2.0
        assert resource_config.memory_max == 16.0
        assert resource_config.compute_mu == 5.0
        assert resource_config.compute_sigma == 0.5
        assert resource_config.compute_min == 10.0
        assert resource_config.compute_max == 100.0
        assert resource_config.seed == 42

    def test_validation(self):
        """Test resource configuration validation"""
        # Invalid memory range
        with pytest.raises(ValueError):
            ResourceConfig(
                memory_mu=2.0,
                memory_sigma=0.5,
                memory_min=16.0,  # min > max
                memory_max=2.0,
                compute_mu=5.0,
                compute_sigma=0.5,
                compute_min=10.0,
                compute_max=100.0
            )

        # Invalid compute range
        with pytest.raises(ValueError):
            ResourceConfig(
                memory_mu=2.0,
                memory_sigma=0.5,
                memory_min=2.0,
                memory_max=16.0,
                compute_mu=5.0,
                compute_sigma=0.5,
                compute_min=100.0,  # min > max
                compute_max=10.0
            )

class TestWorkloadConfig:
    def test_initialization(self, workload_config):
        """Test workload configuration initialization"""
        assert workload_config.model_type == WorkloadType.SMALL
        assert workload_config.initial_sequence_lengths == [128, 256]
        assert workload_config.generation_steps == [32, 64]
        assert workload_config.precision_bytes == 4
        assert workload_config.seed == 42

    def test_validation(self):
        """Test workload configuration validation"""
        # Invalid sequence lengths
        with pytest.raises(ValueError):
            WorkloadConfig(
                model_type=WorkloadType.SMALL,
                initial_sequence_lengths=[0],
                generation_steps=[32]
            )

        # Invalid generation steps
        with pytest.raises(ValueError):
            WorkloadConfig(
                model_type=WorkloadType.SMALL,
                initial_sequence_lengths=[128],
                generation_steps=[0]
            )

def test_config_serialization(simulation_config, tmp_path):
    """Test configuration serialization to YAML and JSON"""
    # Test YAML serialization
    yaml_path = tmp_path / "config.yaml"
    save_config(simulation_config, yaml_path)
    loaded_yaml = load_config(yaml_path)
    assert loaded_yaml.to_dict() == simulation_config.to_dict()

    # Test JSON serialization
    json_path = tmp_path / "config.json"
    save_config(simulation_config, json_path)
    loaded_json = load_config(json_path)
    assert loaded_json.to_dict() == simulation_config.to_dict()

def test_config_validation(simulation_config):
    """Test configuration validation"""
    assert validate_config(simulation_config)

    # Test invalid network config
    invalid_config = deepcopy(simulation_config)
    invalid_config.network.num_devices = 0
    assert not validate_config(invalid_config)

    # Test invalid resource config
    invalid_config = deepcopy(simulation_config)
    invalid_config.resources.memory_min = -1
    assert not validate_config(invalid_config)

def test_config_merging():
    """Test configuration merging"""
    base_config = create_default_config()
    override = {
        'network': {'num_devices': 16},
        'experiment': {'num_runs': 20}
    }

    merged = merge_configs(base_config, override)
    assert merged.network.num_devices == 16
    assert merged.experiment.num_runs == 20
    # Other values should remain unchanged
    assert merged.network.topology_type == base_config.network.topology_type

def test_default_config():
    """Test default configuration creation"""
    config = create_default_config()
    assert isinstance(config, SimulationConfig)
    assert validate_config(config)

def test_config_file_handling():
    """Test configuration file handling"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test missing file
        with pytest.raises(FileNotFoundError):
            load_config(Path(temp_dir) / "nonexistent.yaml")

        # Test invalid file extension
        with pytest.raises(ValueError):
            load_config(Path(temp_dir) / "invalid.txt")

        # Test invalid YAML content
        invalid_yaml = Path(temp_dir) / "invalid.yaml"
        invalid_yaml.write_text("invalid: ]: yaml")
        with pytest.raises(yaml.YAMLError):
            load_config(invalid_yaml)

        # Test invalid JSON content
        invalid_json = Path(temp_dir) / "invalid.json"
        invalid_json.write_text("invalid json")
        with pytest.raises(json.JSONDecodeError):
            load_config(invalid_json)