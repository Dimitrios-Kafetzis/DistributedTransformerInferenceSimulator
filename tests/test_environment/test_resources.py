import pytest
import numpy as np
from src.core import Device
from src.environment.resources import (
    DeviceCapabilities,
    LogNormalDistribution,
    ResourceDistributor,
    BandwidthManager,
    create_devices,
    validate_resource_distribution
)

@pytest.fixture
def basic_resource_distributor():
    """Create a basic resource distributor"""
    return ResourceDistributor(
        num_devices=8,
        memory_distribution=LogNormalDistribution(
            mu=2.0,
            sigma=0.5,
            min_value=2.0,
            max_value=16.0
        ),
        compute_distribution=LogNormalDistribution(
            mu=5.0,
            sigma=0.5,
            min_value=10.0,
            max_value=100.0
        ),
        seed=42
    )

@pytest.fixture
def bandwidth_manager():
    """Create a bandwidth manager"""
    return BandwidthManager(
        cloud_bandwidth=(20.0, 40.0),    # 20-40 Gbps
        regional_bandwidth=(5.0, 10.0),   # 5-10 Gbps
        edge_bandwidth=(0.1, 1.0)         # 100Mbps-1Gbps
    )

class TestLogNormalDistribution:
    def test_distribution_initialization(self):
        """Test initialization of log-normal distribution"""
        dist = LogNormalDistribution(
            mu=2.0,
            sigma=0.5,
            min_value=1.0,
            max_value=10.0
        )
        assert dist.mu == 2.0
        assert dist.sigma == 0.5
        assert dist.min_value == 1.0
        assert dist.max_value == 10.0

    def test_sampling(self):
        """Test sampling from distribution"""
        dist = LogNormalDistribution(
            mu=2.0,
            sigma=0.5,
            min_value=1.0,
            max_value=10.0
        )
        samples = dist.sample(size=1000)
        
        # Check sample size
        assert len(samples) == 1000
        
        # Check value bounds
        assert np.all(samples >= 1.0)
        assert np.all(samples <= 10.0)
        
        # Check distribution shape
        assert np.mean(np.log(samples)) > 0  # Log-normal property

class TestResourceDistributor:
    def test_distributor_initialization(self, basic_resource_distributor):
        """Test resource distributor initialization"""
        assert basic_resource_distributor.num_devices == 8
        assert isinstance(basic_resource_distributor.memory_dist, LogNormalDistribution)
        assert isinstance(basic_resource_distributor.compute_dist, LogNormalDistribution)

    def test_capability_generation(self, basic_resource_distributor):
        """Test generation of device capabilities"""
        capabilities = basic_resource_distributor.generate_capabilities()
        
        # Check number of devices
        assert len(capabilities) == 8
        
        # Check device properties
        for device_id, caps in capabilities.items():
            assert isinstance(caps, DeviceCapabilities)
            assert 2.0 <= caps.memory_capacity <= 16.0
            assert 10.0 <= caps.compute_capacity <= 100.0
            assert caps.device_tier in ['cloud', 'regional', 'edge']
            
        # Check source device designation
        assert any(caps.is_source for caps in capabilities.values())

    def test_tier_distribution(self, basic_resource_distributor):
        """Test distribution of devices across tiers"""
        capabilities = basic_resource_distributor.generate_capabilities()
        
        # Count devices per tier
        tier_counts = {
            'cloud': 0,
            'regional': 0,
            'edge': 0
        }
        for caps in capabilities.values():
            tier_counts[caps.device_tier] += 1
            
        # Check tier proportions
        assert tier_counts['cloud'] >= 1
        assert tier_counts['regional'] >= 1
        assert tier_counts['edge'] >= 1
        assert sum(tier_counts.values()) == 8

    def test_resource_scaling(self, basic_resource_distributor):
        """Test resource scaling across tiers"""
        capabilities = basic_resource_distributor.generate_capabilities()
        
        # Get resources by tier
        cloud_resources = [caps for caps in capabilities.values() 
                         if caps.device_tier == 'cloud']
        edge_resources = [caps for caps in capabilities.values() 
                        if caps.device_tier == 'edge']
        
        # Cloud should have more resources than edge
        assert (np.mean([c.memory_capacity for c in cloud_resources]) >
                np.mean([c.memory_capacity for c in edge_resources]))
        assert (np.mean([c.compute_capacity for c in cloud_resources]) >
                np.mean([c.compute_capacity for c in edge_resources]))

class TestBandwidthManager:
    def test_manager_initialization(self, bandwidth_manager):
        """Test bandwidth manager initialization"""
        assert bandwidth_manager.bandwidth_ranges['cloud'] == (20.0, 40.0)
        assert bandwidth_manager.bandwidth_ranges['regional'] == (5.0, 10.0)
        assert bandwidth_manager.bandwidth_ranges['edge'] == (0.1, 1.0)

    def test_bandwidth_assignment(self, bandwidth_manager):
        """Test bandwidth assignment between tiers"""
        # Test intra-tier bandwidth
        cloud_bw = bandwidth_manager.get_bandwidth('cloud', 'cloud')
        assert 20.0 <= cloud_bw <= 40.0
        
        edge_bw = bandwidth_manager.get_bandwidth('edge', 'edge')
        assert 0.1 <= edge_bw <= 1.0
        
        # Test inter-tier bandwidth (should use lower tier's range)
        cloud_edge_bw = bandwidth_manager.get_bandwidth('cloud', 'edge')
        assert 0.1 <= cloud_edge_bw <= 1.0
        
        regional_edge_bw = bandwidth_manager.get_bandwidth('regional', 'edge')
        assert 0.1 <= regional_edge_bw <= 1.0

def test_device_creation():
    """Test device creation from capabilities"""
    capabilities = {
        'device_0': DeviceCapabilities(
            memory_capacity=8.0,
            compute_capacity=100.0,
            is_source=True,
            device_tier='cloud'
        ),
        'device_1': DeviceCapabilities(
            memory_capacity=4.0,
            compute_capacity=50.0,
            is_source=False,
            device_tier='edge'
        )
    }
    
    devices = create_devices(capabilities)
    
    # Check device creation
    assert len(devices) == 2
    assert isinstance(devices['device_0'], Device)
    assert isinstance(devices['device_1'], Device)
    
    # Check properties transfer
    assert devices['device_0'].memory.capacity == 8.0
    assert devices['device_0'].compute.capacity == 100.0
    assert devices['device_0'].is_source
    
    assert devices['device_1'].memory.capacity == 4.0
    assert devices['device_1'].compute.capacity == 50.0
    assert not devices['device_1'].is_source

def test_resource_distribution_validation():
    """Test validation of resource distribution"""
    # Valid distribution
    valid_capabilities = {
        'device_0': DeviceCapabilities(
            memory_capacity=8.0,
            compute_capacity=100.0,
            is_source=True,
            device_tier='cloud'
        ),
        'device_1': DeviceCapabilities(
            memory_capacity=4.0,
            compute_capacity=50.0,
            is_source=False,
            device_tier='edge'
        )
    }
    assert validate_resource_distribution(valid_capabilities)
    
    # Invalid distribution (no source device)
    invalid_capabilities = {
        'device_0': DeviceCapabilities(
            memory_capacity=8.0,
            compute_capacity=100.0,
            is_source=False,
            device_tier='cloud'
        ),
        'device_1': DeviceCapabilities(
            memory_capacity=4.0,
            compute_capacity=50.0,
            is_source=False,
            device_tier='edge'
        )
    }
    assert not validate_resource_distribution(invalid_capabilities)
    
    # Invalid distribution (negative resources)
    invalid_capabilities = {
        'device_0': DeviceCapabilities(
            memory_capacity=-8.0,
            compute_capacity=100.0,
            is_source=True,
            device_tier='cloud'
        )
    }
    assert not validate_resource_distribution(invalid_capabilities)