import pytest
import numpy as np
from typing import Dict

from src.core import (
    Device,
    Network,
    Transformer,
    TransformerConfig,
    TransformerComponent
)
from src.algorithms import (
    ResourceAwareDistributor,
    ScoringFunction,
    AssignmentResult
)

@pytest.fixture
def transformer_config():
    """Create transformer configuration"""
    return TransformerConfig(
        num_heads=4,
        embedding_dim=256,
        initial_sequence_length=64,
        precision_bytes=4
    )

@pytest.fixture
def devices():
    """Create test devices"""
    return {
        "device_0": Device(
            device_id="device_0",
            memory_capacity=8.0,  # 8 GB
            compute_capacity=100.0e9,  # 100 GFLOPS
            is_source=True
        ),
        "device_1": Device(
            device_id="device_1",
            memory_capacity=4.0,
            compute_capacity=50.0e9
        ),
        "device_2": Device(
            device_id="device_2",
            memory_capacity=4.0,
            compute_capacity=50.0e9
        )
    }

@pytest.fixture
def network(devices):
    """Create test network"""
    network = Network()
    for device_id in devices:
        network.add_device(device_id)
    network.add_link("device_0", "device_1", bandwidth=10.0)
    network.add_link("device_1", "device_2", bandwidth=5.0)
    return network

@pytest.fixture
def transformer(transformer_config):
    """Create transformer instance"""
    return Transformer(transformer_config)

@pytest.fixture
def distributor(transformer, network, devices):
    """Create resource-aware distributor"""
    return ResourceAwareDistributor(
        transformer=transformer,
        network=network,
        devices=devices
    )

class TestResourceAwareDistributor:
    def test_distributor_initialization(self, distributor):
        """Test distributor initialization"""
        assert distributor.transformer is not None
        assert distributor.network is not None
        assert distributor.devices is not None
        assert isinstance(distributor.scoring, ScoringFunction)

    def test_initial_assignment(self, distributor):
        """Test initial component assignment"""
        result = distributor.compute_assignment(
            generation_step=0,
            previous_assignments=None,
            previous_cache=None
        )
        
        assert isinstance(result, AssignmentResult)
        assert result.is_feasible
        assert len(result.component_assignments) > 0
        # All components should be assigned
        assert len(result.component_assignments) == len(
            distributor.transformer.get_all_components()
        )

    def test_cache_assignment(self, distributor):
        """Test cache assignment along with components"""
        result = distributor.compute_assignment(0)
        
        # Verify cache assignments
        assert len(result.cache_assignments) > 0
        # Should have cache assignment for each attention head
        assert len(result.cache_assignments) == distributor.transformer.config.num_heads

    def test_sequence_progression(self, distributor):
        """Test assignments as sequence length grows"""
        # Initial assignment
        result1 = distributor.compute_assignment(0)
        assert result1.is_feasible
        
        # Next step
        result2 = distributor.compute_assignment(
            1,
            previous_assignments=result1.component_assignments,
            previous_cache=result1.cache_assignments
        )
        assert result2.is_feasible
        
        # Resource usage should increase
        assert (sum(result2.resource_usage[dev]['memory_used'] 
                   for dev in result2.resource_usage) >
                sum(result1.resource_usage[dev]['memory_used'] 
                    for dev in result1.resource_usage))

    def test_resource_contention_resolution(self, distributor):
        """Test handling of resource contention"""
        # Artificially reduce device capacity
        original_capacity = distributor.devices["device_0"].memory.capacity
        distributor.devices["device_0"].memory.capacity = 1.0  # Very limited memory
        
        result = distributor.compute_assignment(0)
        
        # Should still find feasible assignment
        assert result.is_feasible
        # Components should be distributed across other devices
        assigned_to_device_0 = sum(
            1 for dev in result.component_assignments.values()
            if dev == "device_0"
        )
        assert assigned_to_device_0 < len(result.component_assignments)
        
        # Restore capacity
        distributor.devices["device_0"].memory.capacity = original_capacity

    def test_communication_optimization(self, distributor):
        """Test optimization of communication costs"""
        result = distributor.compute_assignment(0)
        
        # For components with dependencies, check if they're placed efficiently
        for comp_id, device_id in result.component_assignments.items():
            if comp_id == "projection":
                # Projection layer should be close to attention heads
                head_devices = {
                    result.component_assignments[f"head_{i}"]
                    for i in range(distributor.transformer.config.num_heads)
                }
                # Should minimize number of unique devices
                assert len(head_devices) <= 2

    def test_load_balancing(self, distributor):
        """Test load balancing across devices"""
        result = distributor.compute_assignment(0)
        
        # Calculate resource utilization per device
        device_utils = {
            device_id: result.resource_usage[device_id]['compute_utilization']
            for device_id in distributor.devices
        }
        
        # Check if utilization is reasonably balanced
        util_values = list(device_utils.values())
        assert max(util_values) - min(util_values) < 50  # Less than 50% difference

    def test_assignment_stability(self, distributor):
        """Test stability of assignments across steps"""
        # Get initial assignment
        result1 = distributor.compute_assignment(0)
        
        # Get next step assignment
        result2 = distributor.compute_assignment(
            1,
            previous_assignments=result1.component_assignments,
            previous_cache=result1.cache_assignments
        )
        
        # Count changes in assignment
        changes = sum(
            1 for comp_id, dev_id in result1.component_assignments.items()
            if result2.component_assignments[comp_id] != dev_id
        )
        
        # Should minimize unnecessary changes
        assert changes < len(result1.component_assignments) // 2

    def test_cache_colocating(self, distributor):
        """Test co-location of computation and cache"""
        result = distributor.compute_assignment(0)
        
        # Check if caches are co-located with their components
        for comp_id, cache_device in result.cache_assignments.items():
            compute_device = result.component_assignments[comp_id]
            assert cache_device == compute_device

    def test_scoring_function(self, distributor):
        """Test the scoring function"""
        # Get a component and device
        component = distributor.transformer.get_component("head_0")
        device = distributor.devices["device_0"]
        
        score = distributor.scoring.compute(
            component=component,
            device=device,
            network=distributor.network,
            transformer=distributor.transformer,
            current_assignments={},
            cache_assignments={},
            generation_step=0
        )
        
        assert isinstance(score, float)
        assert score >= 0

    def test_error_handling(self, distributor):
        """Test handling of invalid configurations"""
        # Set impossible resource requirements
        for device in distributor.devices.values():
            device.memory.capacity = 1e-4  # 0.0001 GB = 0.1 MB. Too small for any component
            
        result = distributor.compute_assignment(0)
        assert not result.is_feasible
        
        # Verify error information
        assert result.error is not None

class TestAssignmentResult:
    def test_result_initialization(self):
        """Test assignment result initialization"""
        result = AssignmentResult(
            component_assignments={"comp_1": "device_0"},
            cache_assignments={"comp_1": "device_0"},
            estimated_latency=1.0,
            resource_usage={"device_0": {"memory_used": 1.0}},
            is_feasible=True
        )
        
        assert result.is_feasible
        assert result.estimated_latency == 1.0
        assert "comp_1" in result.component_assignments
        assert "comp_1" in result.cache_assignments

    def test_infeasible_result(self):
        """Test creation of infeasible result"""
        result = AssignmentResult(
            component_assignments={},
            cache_assignments={},
            estimated_latency=float('inf'),
            resource_usage={},
            is_feasible=False,
            error="Resource constraints violated"
        )
        
        assert not result.is_feasible
        assert result.error == "Resource constraints violated"
        assert result.estimated_latency == float('inf')

def test_end_to_end_distribution(distributor):
    """Test end-to-end distribution process"""
    # We'll track resource usage at each step here instead of storing component->device.
    usage_history = []
    
    # Run for several steps
    for step in range(5):
        result = distributor.compute_assignment(
            step,
            # If your design requires using previous assignments/caches, you can keep them:
            # previous_assignments=assignments[-1] if assignments else None,
            # previous_cache=cache_history[-1] if cache_history else None
        )
        
        # The assignment must be feasible
        assert result.is_feasible
        
        # Save the entire device usage snapshot for this step
        usage_history.append(result.resource_usage)
        
        # Compare memory usage across steps
        if step > 0:
            # sum memory used by *all* devices at (step-1)
            prev_usage = sum(
                usage_dict['memory_used']
                for usage_dict in usage_history[step - 1].values()
            )
            # sum memory used by *all* devices at current step
            curr_usage = sum(
                usage_dict['memory_used']
                for usage_dict in usage_history[step].values()
            )
            # Ensure usage is not decreasing
            assert curr_usage >= prev_usage
