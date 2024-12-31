import pytest
from src.core.device import Device, ResourceState

@pytest.fixture
def basic_device():
    """Create a basic device for testing"""
    return Device(
        device_id="test_device",
        memory_capacity=8.0,  # 8 GB
        compute_capacity=100.0,  # 100 GFLOPS
        is_source=False
    )

@pytest.fixture
def source_device():
    """Create a source device for testing"""
    return Device(
        device_id="source_device",
        memory_capacity=16.0,
        compute_capacity=200.0,
        is_source=True
    )

class TestDevice:
    def test_device_initialization(self, basic_device):
        """Test proper device initialization"""
        assert basic_device.device_id == "test_device"
        assert basic_device.memory.capacity == 8.0
        assert basic_device.compute.capacity == 100.0
        assert not basic_device.is_source
        assert basic_device.memory.used == 0.0
        assert basic_device.compute.used == 0.0

    def test_source_device_initialization(self, source_device):
        """Test source device initialization"""
        assert source_device.is_source
        assert source_device.memory.capacity == 16.0
        assert source_device.compute.capacity == 200.0

    def test_resource_allocation(self, basic_device):
        """Test resource allocation functionality"""
        # Try valid allocation
        success = basic_device.allocate_resources(
            component_id="test_component",
            memory_req=2.0,
            compute_req=50.0
        )
        assert success
        assert basic_device.memory.used == 2.0
        assert basic_device.compute.used == 50.0

        # Try allocation exceeding capacity
        success = basic_device.allocate_resources(
            component_id="large_component",
            memory_req=10.0,  # Exceeds remaining capacity
            compute_req=10.0
        )
        assert not success
        assert basic_device.memory.used == 2.0  # Should remain unchanged
        assert basic_device.compute.used == 50.0

    def test_resource_deallocation(self, basic_device):
        """Test resource deallocation"""
        # First allocate some resources
        basic_device.allocate_resources(
            component_id="test_component",
            memory_req=2.0,
            compute_req=50.0
        )

        # Now deallocate
        basic_device.deallocate_resources("test_component")
        assert basic_device.memory.used == 0.0
        assert basic_device.compute.used == 0.0

    def test_cache_allocation(self, basic_device):
        """Test cache allocation functionality"""
        success = basic_device.allocate_resources(
            component_id="cache_component",
            memory_req=1.0,
            compute_req=10.0,
            cache_size=3.0
        )
        assert success
        assert basic_device.memory.used == 4.0  # 1.0 memory + 3.0 cache
        assert basic_device.compute.used == 10.0
        assert "cache_component" in basic_device.cache_assignments

    def test_resource_state_tracking(self, basic_device):
        """Test resource state tracking"""
        basic_device.allocate_resources(
            component_id="test_component",
            memory_req=4.0,
            compute_req=50.0
        )

        state = basic_device.get_resource_state()
        assert state['memory_used'] == 4.0
        assert state['memory_capacity'] == 8.0
        assert state['compute_used'] == 50.0
        assert state['compute_capacity'] == 100.0
        assert state['memory_utilization'] == 50.0  # 4.0/8.0 * 100
        assert state['compute_utilization'] == 50.0  # 50.0/100.0 * 100

    def test_multiple_allocations(self, basic_device):
        """Test multiple resource allocations"""
        # Allocate first component
        success1 = basic_device.allocate_resources(
            component_id="component1",
            memory_req=2.0,
            compute_req=30.0
        )
        assert success1

        # Allocate second component
        success2 = basic_device.allocate_resources(
            component_id="component2",
            memory_req=3.0,
            compute_req=40.0
        )
        assert success2

        # Verify total usage
        assert basic_device.memory.used == 5.0
        assert basic_device.compute.used == 70.0

        # Try third allocation that should fail
        success3 = basic_device.allocate_resources(
            component_id="component3",
            memory_req=4.0,  # Would exceed capacity
            compute_req=20.0
        )
        assert not success3

class TestResourceState:
    def test_resource_state_initialization(self):
        """Test ResourceState initialization"""
        state = ResourceState(capacity=10.0)
        assert state.capacity == 10.0
        assert state.used == 0.0

    def test_available_resource_calculation(self):
        """Test available resource calculation"""
        state = ResourceState(capacity=10.0)
        state.used = 4.0
        assert state.available == 6.0

    def test_utilization_calculation(self):
        """Test utilization percentage calculation"""
        state = ResourceState(capacity=10.0)
        state.used = 5.0
        assert state.utilization == 50.0  # 5.0/10.0 * 100

    def test_zero_capacity_handling(self):
        """Test handling of zero capacity"""
        state = ResourceState(capacity=0.0)
        assert state.utilization == 100.0  # Should return 100% for zero capacity