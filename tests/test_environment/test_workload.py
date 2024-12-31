import pytest
import numpy as np
from src.core import TransformerConfig, Transformer
from src.environment.workload import (
    WorkloadType,
    SequenceConfig,
    TransformerWorkload,
    WorkloadGenerator,
    validate_workload
)

@pytest.fixture
def sequence_config():
    """Create basic sequence configuration"""
    return SequenceConfig(
        initial_length=128,
        num_steps=32,
        precision_bytes=4
    )

@pytest.fixture
def workload_generator():
    """Create workload generator"""
    return WorkloadGenerator(seed=42)

class TestWorkloadType:
    def test_workload_types(self):
        """Test workload type enumeration"""
        assert len(WorkloadType) == 3
        assert WorkloadType.SMALL in WorkloadType
        assert WorkloadType.MEDIUM in WorkloadType
        assert WorkloadType.LARGE in WorkloadType
        
    def test_model_configurations(self, workload_generator):
        """Test model configurations for each type"""
        configs = workload_generator.model_configs
        
        # Check SMALL model config
        assert configs[WorkloadType.SMALL].num_heads == 8
        assert configs[WorkloadType.SMALL].embedding_dim == 512
        
        # Check MEDIUM model config
        assert configs[WorkloadType.MEDIUM].num_heads == 16
        assert configs[WorkloadType.MEDIUM].embedding_dim == 1024
        
        # Check LARGE model config
        assert configs[WorkloadType.LARGE].num_heads == 32
        assert configs[WorkloadType.LARGE].embedding_dim == 2048

class TestSequenceConfig:
    def test_config_initialization(self, sequence_config):
        """Test sequence configuration initialization"""
        assert sequence_config.initial_length == 128
        assert sequence_config.num_steps == 32
        assert sequence_config.precision_bytes == 4

    def test_invalid_config(self):
        """Test invalid sequence configuration"""
        # Invalid initial length
        with pytest.raises(ValueError):
            SequenceConfig(initial_length=0, num_steps=32)
            
        # Invalid number of steps
        with pytest.raises(ValueError):
            SequenceConfig(initial_length=128, num_steps=0)
            
        # Invalid precision
        with pytest.raises(ValueError):
            SequenceConfig(initial_length=128, num_steps=32, precision_bytes=0)

class TestWorkloadGenerator:
    def test_generator_initialization(self, workload_generator):
        """Test workload generator initialization"""
        assert workload_generator.model_configs is not None
        assert len(workload_generator.sequence_lengths) > 0
        assert len(workload_generator.generation_steps) > 0

    def test_workload_generation(self, workload_generator, sequence_config):
        """Test single workload generation"""
        workload = workload_generator.generate_workload(
            WorkloadType.SMALL,
            sequence_config
        )
        
        assert isinstance(workload, TransformerWorkload)
        assert isinstance(workload.transformer, Transformer)
        assert workload.workload_type == WorkloadType.SMALL
        assert workload.sequence_config == sequence_config

    def test_all_combinations(self, workload_generator):
        """Test generation of all workload combinations"""
        workloads = workload_generator.generate_all_combinations()
        
        # Calculate expected number of combinations
        expected_count = (len(WorkloadType) * 
                        len(workload_generator.sequence_lengths) * 
                        len(workload_generator.generation_steps))
        
        assert len(workloads) == expected_count
        
        # Check uniqueness of configurations
        configs = [(w.workload_type, 
                   w.sequence_config.initial_length,
                   w.sequence_config.num_steps) for w in workloads]
        assert len(set(configs)) == expected_count

    def test_evaluation_suite(self, workload_generator):
        """Test evaluation suite generation"""
        suite = workload_generator.generate_evaluation_suite()
        
        # Check suite structure
        assert 'edge_cluster' in suite
        assert 'distributed_edge' in suite
        assert 'hybrid_cloud_edge' in suite
        
        # Check scenario-specific workloads
        assert all(w.workload_type == WorkloadType.SMALL 
                  for w in suite['edge_cluster'])
        
        assert all(w.workload_type in [WorkloadType.SMALL, WorkloadType.MEDIUM]
                  for w in suite['distributed_edge'])
        
        assert any(w.workload_type == WorkloadType.LARGE 
                  for w in suite['hybrid_cloud_edge'])

class TestTransformerWorkload:
    def test_workload_initialization(self, workload_generator, sequence_config):
        """Test transformer workload initialization"""
        workload = workload_generator.generate_workload(
            WorkloadType.SMALL,
            sequence_config
        )
        
        assert workload.transformer.config.num_heads == 8
        assert workload.transformer.config.embedding_dim == 512
        assert workload.transformer.current_sequence_length == sequence_config.initial_length
        assert len(workload.description) > 0

    def test_memory_requirements(self, workload_generator):
        """Test memory requirement calculations"""
        workload = workload_generator.generate_workload(
            WorkloadType.SMALL,
            SequenceConfig(initial_length=128, num_steps=32)
        )
        
        memory_reqs = workload_generator.calculate_memory_requirements(workload)
        
        assert memory_reqs['initial_memory_gb'] > 0
        assert memory_reqs['memory_per_step_gb'] > 0
        assert memory_reqs['total_memory_gb'] > memory_reqs['initial_memory_gb']

    def test_compute_requirements(self, workload_generator):
        """Test compute requirement calculations"""
        workload = workload_generator.generate_workload(
            WorkloadType.SMALL,
            SequenceConfig(initial_length=128, num_steps=32)
        )
        
        compute_reqs = workload_generator.calculate_compute_requirements(workload)
        
        assert compute_reqs['initial_flops'] > 0
        assert compute_reqs['flops_per_step'] > 0
        assert compute_reqs['total_flops'] > compute_reqs['initial_flops']

def test_workload_validation():
    """Test workload validation function"""
    # Create valid workload
    transformer = Transformer(
        TransformerConfig(
            num_heads=8,
            embedding_dim=512,
            initial_sequence_length=128
        )
    )
    
    valid_workload = TransformerWorkload(
        transformer=transformer,
        sequence_config=SequenceConfig(initial_length=128, num_steps=32),
        workload_type=WorkloadType.SMALL,
        description="Test workload"
    )
    assert validate_workload(valid_workload)
    
    # Test invalid configurations
    invalid_transformer = Transformer(
        TransformerConfig(
            num_heads=7,  # Invalid: not matching SMALL config
            embedding_dim=512,
            initial_sequence_length=128
        )
    )
    
    invalid_workload = TransformerWorkload(
        transformer=invalid_transformer,
        sequence_config=SequenceConfig(initial_length=128, num_steps=32),
        workload_type=WorkloadType.SMALL,
        description="Invalid workload"
    )
    assert not validate_workload(invalid_workload)