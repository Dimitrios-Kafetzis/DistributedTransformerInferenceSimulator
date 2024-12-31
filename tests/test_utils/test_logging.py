import pytest
import tempfile
import json
import logging
from pathlib import Path
import time
from datetime import datetime

from src.utils.logging import (
    LogLevel,
    SimulationLogger,
    NullLogger,
    setup_logging
)

@pytest.fixture
def temp_log_dir():
    """Create temporary directory for logs"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def simulation_logger(temp_log_dir):
    """Create simulation logger"""
    return SimulationLogger(
        name="test_simulation",
        log_dir=temp_log_dir,
        level=LogLevel.DEBUG,
        console_output=True,
        file_output=True
    )

class TestLogLevel:
    def test_log_level_values(self):
        """Test log level enumeration"""
        assert LogLevel.DEBUG.value == logging.DEBUG
        assert LogLevel.INFO.value == logging.INFO
        assert LogLevel.WARNING.value == logging.WARNING
        assert LogLevel.ERROR.value == logging.ERROR
        assert LogLevel.CRITICAL.value == logging.CRITICAL

class TestSimulationLogger:
    def test_logger_initialization(self, simulation_logger, temp_log_dir):
        """Test logger initialization"""
        assert simulation_logger.name == "test_simulation"
        assert simulation_logger.log_dir == temp_log_dir
        assert simulation_logger.level == LogLevel.DEBUG
        
        # Check log files created
        log_files = list(temp_log_dir.glob("*.log"))
        assert len(log_files) == 1
        metrics_file = temp_log_dir / "test_simulation_metrics.jsonl"
        assert metrics_file.exists()

    def test_event_logging(self, simulation_logger):
        """Test event logging functionality"""
        # Log different types of events
        events = [
            ("computation", "Component computation started"),
            ("transfer", "Data transfer completed"),
            ("migration", "Component migrated between devices")
        ]
        
        for event_type, message in events:
            simulation_logger.log_event(
                event_type=event_type,
                message=message,
                level=LogLevel.INFO
            )
            
        # Let metrics writer process events
        time.sleep(0.1)
        
        # Check log file contents
        log_file = list(simulation_logger.log_dir.glob("*.log"))[0]
        log_contents = log_file.read_text()
        
        for _, message in events:
            assert message in log_contents

    def test_metrics_logging(self, simulation_logger):
        """Test metrics logging functionality"""
        # Log different metrics
        metrics = {
            'latency': 100.0,
            'throughput': 1000.0,
            'memory_usage': {
                'device_0': 0.8,
                'device_1': 0.6
            }
        }
        
        simulation_logger.log_metrics(metrics)
        
        # Let metrics writer process the metrics
        time.sleep(0.1)
        
        # Check metrics file contents
        metrics_file = simulation_logger.log_dir / f"{simulation_logger.name}_metrics.jsonl"
        with open(metrics_file, 'r') as f:
            logged_metrics = json.loads(f.readline())
            assert 'timestamp' in logged_metrics
            assert logged_metrics['metrics'] == metrics

    def test_resource_state_logging(self, simulation_logger):
        """Test resource state logging"""
        simulation_logger.log_resource_state(
            step=1,
            device_id="device_0",
            memory_used=4.0,
            memory_total=8.0,
            compute_used=50.0,
            compute_total=100.0
        )
        
        time.sleep(0.1)
        
        # Verify metrics
        metrics_file = simulation_logger.log_dir / f"{simulation_logger.name}_metrics.jsonl"
        with open(metrics_file, 'r') as f:
            logged_data = json.loads(f.readline())
            resource_state = logged_data['metrics']['resource_state']
            
            assert resource_state['memory']['used'] == 4.0
            assert resource_state['memory']['total'] == 8.0
            assert resource_state['memory']['utilization'] == 0.5
            
            assert resource_state['compute']['used'] == 50.0
            assert resource_state['compute']['total'] == 100.0
            assert resource_state['compute']['utilization'] == 0.5

    def test_component_assignment_logging(self, simulation_logger):
        """Test component assignment logging"""
        simulation_logger.log_component_assignment(
            step=1,
            component_id="head_0",
            device_id="device_1",
            assignment_type="initial"
        )
        
        time.sleep(0.1)
        
        # Check log contents
        log_file = list(simulation_logger.log_dir.glob("*.log"))[0]
        log_contents = log_file.read_text()
        assert "Migrating head_0 from device_0 to device_1" in log_contents
        
        # Check metrics file for migration data
        metrics_file = simulation_logger.log_dir / f"{simulation_logger.name}_metrics.jsonl"
        with open(metrics_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                if 'migration' in data:
                    assert data['migration']['component_id'] == "head_0"
                    assert data['migration']['source'] == "device_0"
                    assert data['migration']['target'] == "device_1"
                    assert data['migration']['cost'] == 0.5
                    break

    def test_communication_logging(self, simulation_logger):
        """Test communication event logging"""
        simulation_logger.log_communication(
            step=1,
            source_component="head_0",
            target_component="projection",
            data_size=2.0,
            transfer_time=0.5
        )
        
        time.sleep(0.1)
        
        # Check metrics file
        metrics_file = simulation_logger.log_dir / f"{simulation_logger.name}_metrics.jsonl"
        with open(metrics_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                if 'communication' in data['metrics']:
                    comm = data['metrics']['communication']
                    assert comm['source'] == "head_0"
                    assert comm['target'] == "projection"
                    assert comm['data_size'] == 2.0
                    assert comm['transfer_time'] == 0.5
                    break

    def test_error_logging(self, simulation_logger):
        """Test error logging functionality"""
        error_type = "simulation_error"
        error_message = "Simulation failed due to resource exhaustion"
        
        simulation_logger.log_error(
            error_type=error_type,
            message=error_message,
            device_id="device_0"
        )
        
        time.sleep(0.1)
        
        # Check log file
        log_file = list(simulation_logger.log_dir.glob("*.log"))[0]
        log_contents = log_file.read_text()
        assert error_message in log_contents
        assert error_type in log_contents
        assert "ERROR" in log_contents

    def test_warning_logging(self, simulation_logger):
        """Test warning logging functionality"""
        warning_type = "resource_warning"
        warning_message = "High memory utilization detected"
        
        simulation_logger.log_warning(
            warning_type=warning_type,
            message=warning_message,
            device_id="device_0",
            utilization=0.95
        )
        
        time.sleep(0.1)
        
        # Check log file
        log_file = list(simulation_logger.log_dir.glob("*.log"))[0]
        log_contents = log_file.read_text()
        assert warning_message in log_contents
        assert warning_type in log_contents
        assert "WARNING" in log_contents

    def test_performance_logging(self, simulation_logger):
        """Test performance metrics logging"""
        metrics = {
            'latency': 100.0,
            'throughput': 1000.0,
            'gpu_utilization': 0.8
        }
        
        simulation_logger.log_performance(
            step=1,
            metrics=metrics
        )
        
        time.sleep(0.1)
        
        # Check metrics file
        metrics_file = simulation_logger.log_dir / f"{simulation_logger.name}_metrics.jsonl"
        with open(metrics_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                if 'performance' in data['metrics']:
                    perf = data['metrics']['performance']
                    assert perf == metrics
                    break

    def test_logger_cleanup(self, simulation_logger):
        """Test logger cleanup"""
        # Log some metrics
        simulation_logger.log_metrics({'test': 1.0})
        
        # Cleanup
        simulation_logger.cleanup()
        
        # Try logging after cleanup
        simulation_logger.log_metrics({'test': 2.0})
        
        time.sleep(0.1)
        
        # Check metrics file
        metrics_file = simulation_logger.log_dir / f"{simulation_logger.name}_metrics.jsonl"
        with open(metrics_file, 'r') as f:
            lines = f.readlines()
            # Should only contain the first metric
            assert len(lines) == 1
            assert json.loads(lines[0])['metrics']['test'] == 1.0

class TestNullLogger:
    def test_null_logger(self):
        """Test null logger functionality"""
        logger = NullLogger()
        
        # All these should do nothing
        logger.log_event("test", "message")
        logger.log_metrics({'test': 1.0})
        logger.log_error("error", "message")
        logger.cleanup()

def test_logging_setup():
    """Test logging setup utility function"""
    with tempfile.TemporaryDirectory() as temp_dir:
        logger = setup_logging(
            experiment_name="test_experiment",
            log_dir=temp_dir,
            console_level=LogLevel.INFO,
            file_level=LogLevel.DEBUG
        )
        
        assert isinstance(logger, SimulationLogger)
        assert logger.name == "test_experiment"
        assert logger.level == LogLevel.DEBUG
        
        # Test logging
        logger.log_event("test", "Test message")
        
        time.sleep(0.1)
        
        # Check files created
        log_dir = Path(temp_dir)
        assert any(log_dir.glob("*.log"))
        assert any(log_dir.glob("*_metrics.jsonl"))