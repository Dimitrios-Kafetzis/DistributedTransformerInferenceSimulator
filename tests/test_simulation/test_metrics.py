import pytest
import numpy as np
from datetime import datetime
from pathlib import Path
import tempfile
import json

from src.core import Event, EventType, Device, Network
from src.simulation.metrics import (
    MetricsCollector,
    PerformanceMetrics,
    ResourceMetrics,
    CommunicationMetrics
)

@pytest.fixture
def metrics_collector():
    """Create a basic metrics collector"""
    return MetricsCollector()

@pytest.fixture
def devices():
    """Create test devices"""
    return {
        "device_0": Device(
            device_id="device_0",
            memory_capacity=8.0,
            compute_capacity=100.0,
            is_source=True
        ),
        "device_1": Device(
            device_id="device_1",
            memory_capacity=4.0,
            compute_capacity=50.0
        )
    }

@pytest.fixture
def network(devices):
    """Create test network"""
    network = Network()
    for device_id in devices:
        network.add_device(device_id)
    network.add_link("device_0", "device_1", bandwidth=10.0)
    return network

class TestMetricsCollector:
    def test_collector_initialization(self, metrics_collector):
        """Test metrics collector initialization"""
        assert isinstance(metrics_collector.performance, PerformanceMetrics)
        assert isinstance(metrics_collector.resources, ResourceMetrics)
        assert isinstance(metrics_collector.communication, CommunicationMetrics)
        assert metrics_collector.step_start_time is None
        assert metrics_collector.current_step == 0

    def test_step_timing(self, metrics_collector):
        """Test step timing measurement"""
        # Start step
        metrics_collector.start_step(1)
        assert metrics_collector.step_start_time is not None
        assert metrics_collector.current_step == 1
        
        # End step
        metrics_collector.end_step()
        assert metrics_collector.step_start_time is None
        assert len(metrics_collector.performance.step_latencies) == 1

    def test_event_recording(self, metrics_collector):
        """Test event recording"""
        # Record compute event
        compute_event = Event(
            time=1.0,
            event_type=EventType.COMPUTE_START,
            component_id="test_component",
            source_device="device_0"
        )
        metrics_collector.record_event(compute_event)
        
        # Record transfer event
        transfer_event = Event(
            time=2.0,
            event_type=EventType.TRANSFER_START,
            component_id="test_component",
            source_device="device_0",
            target_device="device_1",
            data_size=1.0
        )
        metrics_collector.record_event(transfer_event)
        
        # Verify records
        assert len(metrics_collector.active_computations) > 0
        assert len(metrics_collector.active_transfers) > 0

    def test_resource_state_recording(self, metrics_collector, devices, network):
        """Test resource state recording"""
        metrics_collector.record_resource_state(devices, network)
        
        # Verify device metrics
        assert len(metrics_collector.resources.memory_utilization) > 0
        assert len(metrics_collector.resources.compute_utilization) > 0
        
        # Verify network metrics
        assert metrics_collector.communication.total_data_transferred >= 0

    def test_migration_recording(self, metrics_collector):
        """Test migration event recording"""
        metrics_collector.record_migration(
            component_id="test_component",
            source_device="device_0",
            target_device="device_1",
            cost=1.5
        )
        
        assert metrics_collector.communication.total_migrations == 1
        assert len(metrics_collector.communication.migration_costs) == 1
        assert metrics_collector.communication.migration_costs[0] == 1.5

    def test_metrics_summary(self, metrics_collector, devices, network):
        """Test metrics summary generation"""
        # Record some metrics
        metrics_collector.start_step(1)
        metrics_collector.record_resource_state(devices, network)
        metrics_collector.end_step()
        
        # Generate summary
        summary = metrics_collector.get_summary()
        
        assert 'performance' in summary
        assert 'resources' in summary
        assert 'communication' in summary

    def test_metrics_file_saving(self, metrics_collector):
        """Test saving metrics to file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_metrics.json"
            metrics_collector.save_to_file(file_path)
            
            assert file_path.exists()
            
            # Verify file content
            with open(file_path, 'r') as f:
                saved_metrics = json.load(f)
                assert isinstance(saved_metrics, dict)

class TestPerformanceMetrics:
    def test_latency_calculations(self):
        """Test latency metric calculations"""
        metrics = PerformanceMetrics()
        
        # Add some latencies
        metrics.step_latencies = [1.0, 2.0, 1.5, 1.8]
        
        assert metrics.average_latency == pytest.approx(1.575)
        assert metrics.latency_std == pytest.approx(0.4349, rel=1e-3)

    def test_computation_ratio(self):
        """Test computation ratio calculation"""
        metrics = PerformanceMetrics()
        metrics.total_time = 10.0
        metrics.computation_times = [2.0, 3.0, 1.0]
        
        assert metrics.computation_ratio == 0.6  # 6/10

    def test_communication_ratio(self):
        """Test communication ratio calculation"""
        metrics = PerformanceMetrics()
        metrics.total_time = 10.0
        metrics.communication_times = [1.0, 2.0, 1.0]
        
        assert metrics.communication_ratio == 0.4  # 4/10

class TestResourceMetrics:
    def test_utilization_tracking(self):
        """Test resource utilization tracking"""
        metrics = ResourceMetrics()
        
        # Add utilization data
        metrics.memory_utilization["device_0"] = [0.5, 0.6, 0.7]
        metrics.compute_utilization["device_0"] = [0.4, 0.5, 0.6]
        
        stats = metrics.get_utilization_stats()
        assert "device_0" in stats
        assert stats["device_0"]["avg_memory"] == pytest.approx(0.6)
        assert stats["device_0"]["avg_compute"] == pytest.approx(0.5)

    def test_average_utilization(self):
        """Test average utilization calculation"""
        metrics = ResourceMetrics()
        metrics.memory_utilization["device_0"] = [0.5, 0.6, 0.7]
        
        util = metrics.get_average_utilization("device_0")
        assert util["memory"] == pytest.approx(0.6)

class TestCommunicationMetrics:
    def test_migration_metrics(self):
        """Test migration metrics calculation"""
        metrics = CommunicationMetrics()
        metrics.migration_costs = [1.0, 2.0, 1.5]
        metrics.total_migrations = 3
        
        assert metrics.average_migration_cost == pytest.approx(1.5)

    def test_link_statistics(self):
        """Test link statistics calculation"""
        metrics = CommunicationMetrics()
        
        # Add bandwidth utilization data
        metrics.bandwidth_utilization["link_0"] = [0.5, 0.6, 0.7]
        metrics.transfer_counts["link_0"] = 3
        
        stats = metrics.get_link_stats()
        assert "link_0" in stats
        assert stats["link_0"]["avg_utilization"] == pytest.approx(0.6)
        assert stats["link_0"]["transfer_count"] == 3

    def test_transfer_tracking(self):
        """Test transfer tracking"""
        metrics = CommunicationMetrics()
        metrics.total_data_transferred = 10.0
        metrics.transfer_counts["link_0"] = 5
        
        assert metrics.total_data_transferred == 10.0
        assert metrics.transfer_counts["link_0"] == 5

def test_end_to_end_metrics(metrics_collector, devices, network):
    """Test end-to-end metrics collection"""
    # Simulate a complete step
    metrics_collector.start_step(1)
    
    # Record compute event
    compute_start = Event(
        time=1.0,
        event_type=EventType.COMPUTE_START,
        component_id="comp_1",
        source_device="device_0"
    )
    metrics_collector.record_event(compute_start)
    
    # Record compute completion
    compute_end = Event(
        time=2.0,
        event_type=EventType.COMPUTE_COMPLETE,
        component_id="comp_1",
        source_device="device_0"
    )
    metrics_collector.record_event(compute_end)
    
    # Record transfer
    transfer_start = Event(
        time=2.0,
        event_type=EventType.TRANSFER_START,
        component_id="comp_1",
        source_device="device_0",
        target_device="device_1",
        data_size=1.0
    )
    metrics_collector.record_event(transfer_start)
    
    # Record resource state
    metrics_collector.record_resource_state(devices, network)
    
    # End step
    metrics_collector.end_step()
    
    # Get summary and verify
    summary = metrics_collector.get_summary()
    assert summary['performance']['total_time'] > 0
    assert len(summary['resources']['utilization']) > 0
    assert summary['communication']['total_data_transferred'] >= 0