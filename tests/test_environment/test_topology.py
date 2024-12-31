import pytest
import networkx as nx
import numpy as np
from src.core import Network
from src.environment.topology import (
    TopologyConfig,
    NetworkTopologyGenerator,
    EdgeClusterTopology,
    DistributedEdgeTopology,
    HybridCloudEdgeTopology,
    create_topology,
    validate_topology
)

@pytest.fixture
def basic_config():
    """Create basic topology configuration"""
    return TopologyConfig(
        num_devices=8,
        min_bandwidth=1.0,
        max_bandwidth=10.0,
        edge_probability=0.3,
        seed=42
    )

@pytest.fixture
def edge_config():
    """Create edge cluster configuration"""
    return TopologyConfig(
        num_devices=8,
        min_bandwidth=1.0,
        max_bandwidth=10.0,
        edge_probability=0.3,
        seed=42
    )

@pytest.fixture
def distributed_config():
    """Create distributed edge configuration"""
    return TopologyConfig(
        num_devices=16,
        min_bandwidth=0.1,
        max_bandwidth=1.0,
        edge_probability=0.3,
        seed=42
    )

@pytest.fixture
def hybrid_config():
    """Create hybrid cloud-edge configuration"""
    return TopologyConfig(
        num_devices=24,
        min_bandwidth=0.1,
        max_bandwidth=40.0,
        edge_probability=0.3,
        seed=42
    )

class TestNetworkTopologyGenerator:
    def test_base_generator_initialization(self, basic_config):
        """Test base topology generator initialization"""
        generator = NetworkTopologyGenerator(basic_config)
        assert generator.config == basic_config
        
    def test_base_generator_abstract_method(self, basic_config):
        """Test that base generator's generate method is abstract"""
        generator = NetworkTopologyGenerator(basic_config)
        with pytest.raises(NotImplementedError):
            generator.generate()
            
    def test_bandwidth_assignment(self, basic_config):
        """Test bandwidth assignment helper method"""
        generator = NetworkTopologyGenerator(basic_config)
        network = Network()
        network.add_device("device_0")
        network.add_device("device_1")
        network.topology.add_edge("device_0", "device_1")
        
        # Inject test method to access protected method
        def assign_test_bandwidth(self, network):
            self._assign_bandwidths(network)
            
        NetworkTopologyGenerator.assign_test_bandwidth = assign_test_bandwidth
        generator.assign_test_bandwidth(network)
        
        # Check bandwidth assignment
        link = network.links[("device_0", "device_1")]
        assert basic_config.min_bandwidth <= link.bandwidth <= basic_config.max_bandwidth

class TestEdgeClusterTopology:
    def test_topology_generation(self, edge_config):
        """Test edge cluster topology generation"""
        generator = EdgeClusterTopology(edge_config)
        network = generator.generate()
        
        # Check basic properties
        assert isinstance(network, Network)
        assert len(network.topology.nodes()) == 8
        assert network.topology.is_connected()
        
        # Check hierarchical structure
        # One high-capacity node connected to mid-tier nodes
        core_node = 'device_0'
        assert network.topology.degree[core_node] >= 2
        
        # Check bandwidth assignment
        for _, _, data in network.topology.edges(data=True):
            assert edge_config.min_bandwidth <= data['bandwidth'] <= edge_config.max_bandwidth

    def test_core_edge_separation(self, edge_config):
        """Test core and edge node separation"""
        generator = EdgeClusterTopology(edge_config)
        network = generator.generate()
        
        # Core nodes should have higher bandwidth connections
        core_edges = list(network.topology.edges('device_0', data=True))
        edge_edges = list(network.topology.edges('device_7', data=True))
        
        avg_core_bw = np.mean([e[2]['bandwidth'] for e in core_edges])
        avg_edge_bw = np.mean([e[2]['bandwidth'] for e in edge_edges])
        
        assert avg_core_bw > avg_edge_bw

    def test_device_connectivity(self, edge_config):
        """Test device connectivity patterns"""
        generator = EdgeClusterTopology(edge_config)
        network = generator.generate()
        
        # All devices should be reachable from core
        core_node = 'device_0'
        for node in network.topology.nodes():
            path = nx.shortest_path(network.topology, core_node, node)
            assert path is not None
            assert len(path) <= 3  # Maximum 3 hops from core

class TestDistributedEdgeTopology:
    def test_topology_generation(self, distributed_config):
        """Test distributed edge topology generation"""
        generator = DistributedEdgeTopology(distributed_config)
        network = generator.generate()
        
        # Check basic properties
        assert isinstance(network, Network)
        assert len(network.topology.nodes()) == 16
        assert network.topology.is_connected()
        
        # Check mesh-like properties
        avg_degree = np.mean(list(dict(network.topology.degree()).values()))
        assert avg_degree >= 2  # Each node should connect to at least 2 others
        
        # Check geographic clustering
        clusters = self._identify_clusters(network.topology)
        assert len(clusters) >= 2  # Should have multiple clusters

    def test_bandwidth_distribution(self, distributed_config):
        """Test bandwidth distribution in mesh network"""
        generator = DistributedEdgeTopology(distributed_config)
        network = generator.generate()
        
        bandwidths = [data['bandwidth'] 
                     for _, _, data in network.topology.edges(data=True)]
        
        # Check bandwidth range
        assert all(distributed_config.min_bandwidth <= bw <= distributed_config.max_bandwidth 
                  for bw in bandwidths)
        
        # Check variability
        assert len(set(bandwidths)) > 1  # Should have varying bandwidths

    def test_cluster_connectivity(self, distributed_config):
        """Test connectivity between clusters"""
        generator = DistributedEdgeTopology(distributed_config)
        network = generator.generate()
        
        # Identify clusters
        clusters = self._identify_clusters(network.topology)
        
        # Verify inter-cluster connectivity
        for c1 in clusters:
            for c2 in clusters:
                if c1 != c2:
                    # Should have at least one edge between clusters
                    cross_edges = [
                        (u, v) for u, v in network.topology.edges()
                        if (u in c1 and v in c2) or (u in c2 and v in c1)
                    ]
                    assert len(cross_edges) > 0
    
    @staticmethod
    def _identify_clusters(topology):
        """Helper to identify network clusters"""
        return list(nx.community.greedy_modularity_communities(topology))

class TestHybridCloudEdgeTopology:
    def test_topology_generation(self, hybrid_config):
        """Test hybrid cloud-edge topology generation"""
        generator = HybridCloudEdgeTopology(hybrid_config)
        network = generator.generate()
        
        # Check basic properties
        assert isinstance(network, Network)
        assert len(network.topology.nodes()) == 24
        assert network.topology.is_connected()
        
        # Check tier separation
        cloud_nodes = [f"device_{i}" for i in range(4)]
        regional_nodes = [f"device_{i}" for i in range(4, 12)]
        edge_nodes = [f"device_{i}" for i in range(12, 24)]
        
        # Check node degree (cloud nodes should be well-connected)
        cloud_degrees = [network.topology.degree(node) for node in cloud_nodes]
        assert all(d >= 2 for d in cloud_degrees)

    def test_tier_connectivity(self, hybrid_config):
        """Test connectivity between tiers"""
        generator = HybridCloudEdgeTopology(hybrid_config)
        network = generator.generate()
        
        cloud_nodes = set(f"device_{i}" for i in range(4))
        regional_nodes = set(f"device_{i}" for i in range(4, 12))
        edge_nodes = set(f"device_{i}" for i in range(12, 24))
        
        # Check regional to cloud connectivity
        for regional in regional_nodes:
            paths_to_cloud = [
                nx.shortest_path(network.topology, regional, cloud)
                for cloud in cloud_nodes
            ]
            assert any(len(path) == 2 for path in paths_to_cloud)  # Direct connection
            
        # Check edge to regional connectivity
        for edge in edge_nodes:
            paths_to_regional = [
                nx.shortest_path(network.topology, edge, regional)
                for regional in regional_nodes
            ]
            assert any(len(path) == 2 for path in paths_to_regional)

    def test_bandwidth_hierarchy(self, hybrid_config):
        """Test bandwidth hierarchy between tiers"""
        generator = HybridCloudEdgeTopology(hybrid_config)
        network = generator.generate()
        
        def get_tier_bandwidth(u, v):
            if int(u.split('_')[1]) < 4 and int(v.split('_')[1]) < 4:
                return 'cloud'
            elif int(u.split('_')[1]) >= 12 or int(v.split('_')[1]) >= 12:
                return 'edge'
            return 'regional'
            
        bandwidths = {
            'cloud': [],
            'regional': [],
            'edge': []
        }
        
        for (u, v), link in network.links.items():
            tier = get_tier_bandwidth(u, v)
            bandwidths[tier].append(link.bandwidth)
            
        # Verify bandwidth hierarchy
        assert min(bandwidths['cloud']) > max(bandwidths['regional'])
        assert min(bandwidths['regional']) > max(bandwidths['edge'])

def test_topology_creation():
    """Test topology creation factory function"""
    configs = {
        'edge_cluster': TopologyConfig(num_devices=8, min_bandwidth=1.0, max_bandwidth=10.0),
        'distributed_edge': TopologyConfig(num_devices=16, min_bandwidth=0.1, max_bandwidth=1.0),
        'hybrid_cloud_edge': TopologyConfig(num_devices=24, min_bandwidth=0.1, max_bandwidth=40.0)
    }
    
    for topology_type, config in configs.items():
        network = create_topology(topology_type, config)
        assert isinstance(network, Network)
        assert network.topology.is_connected()
        assert len(network.topology.nodes()) == config.num_devices
        
    # Test invalid topology type
    with pytest.raises(ValueError):
        create_topology('invalid_type', configs['edge_cluster'])

def test_topology_validation():
    """Test topology validation function"""
    # Valid topology
    config = TopologyConfig(num_devices=8, min_bandwidth=1.0, max_bandwidth=10.0)
    valid_network = create_topology('edge_cluster', config)
    assert validate_topology(valid_network)
    
    # Invalid cases
    invalid_cases = [
        # Disconnected topology
        (lambda: Network(), "Disconnected topology"),
        
        # Missing bandwidth
        (lambda: _create_network_without_bandwidth(), "Missing bandwidth"),
        
        # Empty topology
        (lambda: Network(), "Empty topology")
    ]
    
    for create_invalid, case_name in invalid_cases:
        invalid_network = create_invalid()
        assert not validate_topology(invalid_network), f"Failed case: {case_name}"

def _create_network_without_bandwidth():
    """Helper to create invalid network without bandwidth"""
    network = Network()
    network.add_device("device_0")
    network.add_device("device_1")
    network.topology.add_edge("device_0", "device_1")
    return network