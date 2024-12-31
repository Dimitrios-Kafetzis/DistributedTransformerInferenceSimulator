import pytest
import networkx as nx
from src.core.network import Network, Link

@pytest.fixture
def basic_network():
    """Create a basic network for testing"""
    network = Network()
    network.add_device("device_0")
    network.add_device("device_1")
    network.add_device("device_2")
    return network

@pytest.fixture
def connected_network():
    """Create a connected network with links for testing"""
    network = Network()
    
    # Add devices
    for i in range(3):
        network.add_device(f"device_{i}")
        
    # Add links
    network.add_link("device_0", "device_1", bandwidth=10.0)  # 10 Gbps
    network.add_link("device_1", "device_2", bandwidth=5.0)   # 5 Gbps
    return network

class TestNetwork:
    def test_network_initialization(self):
        """Test network initialization"""
        network = Network(directed=True)
        assert isinstance(network.topology, nx.DiGraph)
        assert len(network.links) == 0
        assert network.total_data_transferred == 0.0
        assert network.peak_bandwidth_usage == 0.0

    def test_device_addition(self, basic_network):
        """Test adding devices to network"""
        assert len(basic_network.topology.nodes()) == 3
        assert "device_0" in basic_network.topology
        assert "device_1" in basic_network.topology
        assert "device_2" in basic_network.topology

    def test_link_addition(self, connected_network):
        """Test adding links to network"""
        assert len(connected_network.links) == 2
        assert ("device_0", "device_1") in connected_network.links
        assert ("device_1", "device_2") in connected_network.links
        
        # Verify bandwidths
        assert connected_network.links[("device_0", "device_1")].bandwidth == 10.0
        assert connected_network.links[("device_1", "device_2")].bandwidth == 5.0

    def test_path_finding(self, connected_network):
        """Test finding paths between devices"""
        path = connected_network.get_path("device_0", "device_2")
        assert path == ["device_0", "device_1", "device_2"]
        
        # Test non-existent path
        with pytest.raises(ValueError):
            connected_network.get_path("device_0", "non_existent")

    def test_transfer_time_calculation(self, connected_network):
        """Test calculation of transfer times"""
        # Calculate transfer time for 1GB over 10Gbps link
        time = connected_network.calculate_transfer_time(
            "device_0",
            "device_1",
            1.0  # 1 GB
        )
        assert time == pytest.approx(0.1)  # 1 GB / 10 Gbps = 0.1 seconds
        
        # Calculate transfer time over multi-hop path
        time = connected_network.calculate_transfer_time(
            "device_0",
            "device_2",
            1.0  # 1 GB
        )
        assert time == pytest.approx(0.2)  # Limited by 5 Gbps link

    def test_transfer_management(self, connected_network):
        """Test management of data transfers"""
        # Start a transfer
        transfer = connected_network.start_transfer(
            "device_0",
            "device_1",
            data_size=2.0,  # 2 GB
            start_time=0.0
        )
        
        assert len(connected_network.active_transfers) == 1
        assert connected_network.total_data_transferred == 2.0
        
        # Complete the transfer
        connected_network.complete_transfer(transfer)
        assert len(connected_network.active_transfers) == 0

    def test_bandwidth_tracking(self, connected_network):
        """Test bandwidth usage tracking"""
        # Start multiple transfers
        transfer1 = connected_network.start_transfer(
            "device_0",
            "device_1",
            data_size=5.0,
            start_time=0.0
        )
        
        transfer2 = connected_network.start_transfer(
            "device_1",
            "device_2",
            data_size=2.0,
            start_time=0.0
        )
        
        # Check peak bandwidth usage
        assert connected_network.peak_bandwidth_usage > 0.0
        
        # Complete transfers
        connected_network.complete_transfer(transfer1)
        connected_network.complete_transfer(transfer2)
        
        # Check total data transferred
        assert connected_network.total_data_transferred == 7.0

    def test_network_state(self, connected_network):
        """Test network state reporting"""
        # Add some transfers
        connected_network.start_transfer(
            "device_0",
            "device_1",
            data_size=1.0,
            start_time=0.0
        )
        
        state = connected_network.get_network_state()
        assert state['total_data_transferred'] == 1.0
        assert state['peak_bandwidth_usage'] >= 0.0
        assert state['active_transfers'] == 1
        assert state['topology_stats']['nodes'] == 3
        assert state['topology_stats']['edges'] == 2

    def test_bidirectional_links(self):
        """Test creation and usage of bidirectional links"""
        network = Network()
        network.add_device("A")
        network.add_device("B")
        
        # Add bidirectional link
        network.add_link("A", "B", bandwidth=10.0, bidirectional=True)
        
        assert ("A", "B") in network.links
        assert ("B", "A") in network.links
        assert network.links[("A", "B")].bandwidth == network.links[("B", "A")].bandwidth

class TestLink:
    def test_link_initialization(self):
        """Test Link class initialization"""
        link = Link(bandwidth=10.0)
        assert link.bandwidth == 10.0
        assert link.used_bandwidth == 0.0
        assert link.available_bandwidth == 10.0

    def test_bandwidth_usage(self):
        """Test bandwidth usage calculations"""
        link = Link(bandwidth=10.0)
        assert link.available_bandwidth == 10.0
        
        # Use some bandwidth
        link.used_bandwidth = 4.0
        assert link.available_bandwidth == 6.0
        
        # Try to use more than available
        link.used_bandwidth = 12.0
        assert link.available_bandwidth == 0.0

    def test_zero_bandwidth_handling(self):
        """Test handling of zero bandwidth links"""
        link = Link(bandwidth=0.0)
        assert link.available_bandwidth == 0.0
        
        # Any usage should still show zero available
        link.used_bandwidth = 1.0
        assert link.available_bandwidth == 0.0