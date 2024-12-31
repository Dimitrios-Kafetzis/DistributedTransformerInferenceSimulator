import pytest
from pathlib import Path
import tempfile
from datetime import datetime

from src.core import (
    Device,
    Network,
    Transformer,
    TransformerConfig,
    EventQueue
)
from src.simulation.engine import (
    SimulationEngine,
    SimulationState,
    SimulationConfig
)

@pytest.fixture
def basic_config():
    """Create a basic simulation configuration"""
    return SimulationConfig(
        max_steps=10,
        time_limit=100.0,
        checkpoint_interval=5,
        enable_logging=True,
        random_seed=42
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
            compute_capacity=100.0,  # 100 GFLOPS
            is_source=True
        ),
        "device_1": Device(
            device_id="device_1",
            memory_capacity=4.0,
            compute_capacity=50.0
        ),
        "device_2": Device(
            device_id="device_2",
            memory_capacity=4.0,
            compute_capacity=50.0
        )
    }

@pytest.fixture
def network(devices):
    """Create test network"""
    network = Network()
    
    # Add devices
    for device_id in devices:
        network.add_device(device_id)
        
    # Add connections
    network.add_link("device_0", "device_1", bandwidth=10.0)
    network.add_link("device_1", "device_2", bandwidth=5.0)
    
    return network

@pytest.fixture
def simulation_engine(basic_config, transformer_config, devices, network):
    """Create simulation engine with basic setup"""
    transformer = Transformer(transformer_config)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        return SimulationEngine(
            transformer=transformer,
            network=network,
            devices=devices,
            config=basic_config
        )

class TestSimulationEngine:
    def test_engine_initialization(self, simulation_engine):
        """Test simulation engine initialization"""
        assert simulation_engine.state.current_step == 0
        assert simulation_engine.state.current_time == 0.0
        assert not simulation_engine.state.is_running
        assert isinstance(simulation_engine.event_queue, EventQueue)

    def test_initialization_phase(self, simulation_engine):
        """Test simulation initialization phase"""
        simulation_engine.initialize()
        
        # Check state after initialization
        assert not simulation_engine.state.is_running
        assert len(simulation_engine.state.active_components) == 0
        assert len(simulation_engine.state.pending_transfers) == 0
        
        # Check event queue has initial generation step
        first_event = simulation_engine.event_queue.get_next_event()
        assert first_event is not None
        assert first_event.event_type.GENERATION_STEP

    def test_single_step(self, simulation_engine):
        """Test single simulation step execution"""
        simulation_engine.initialize()
        simulation_engine.state.is_running = True
        
        # Execute one step
        result = simulation_engine.step()
        assert result  # Should return True to continue
        assert simulation_engine.state.current_step == 1

    def test_complete_run(self, simulation_engine):
        """Test complete simulation run"""
        simulation_engine.run()
        
        assert not simulation_engine.state.is_running
        assert simulation_engine.state.current_step == simulation_engine.config.max_steps
        assert simulation_engine.state.current_time <= simulation_engine.config.time_limit

    def test_time_limit_enforcement(self, simulation_engine):
        """Test enforcement of time limit"""
        # Set very short time limit
        simulation_engine.config.time_limit = 0.1
        simulation_engine.run()
        
        assert not simulation_engine.state.is_running
        assert simulation_engine.state.current_time <= 0.1

    def test_component_tracking(self, simulation_engine):
        """Test tracking of active components"""
        simulation_engine.initialize()
        simulation_engine.state.is_running = True
        
        # Execute steps until component activation
        while len(simulation_engine.state.active_components) == 0:
            if not simulation_engine.step():
                break
                
        # Verify component tracking
        assert len(simulation_engine.state.active_components) > 0

    def test_transfer_tracking(self, simulation_engine):
        """Test tracking of data transfers"""
        simulation_engine.initialize()
        simulation_engine.state.is_running = True
        
        # Run until transfer occurs
        transfer_detected = False
        for _ in range(10):  # Limit iterations
            if len(simulation_engine.state.pending_transfers) > 0:
                transfer_detected = True
                break
            if not simulation_engine.step():
                break
                
        assert transfer_detected

    def test_checkpoint_interval(self, simulation_engine):
        """Test checkpoint interval functionality"""
        # Set up logging
        simulation_engine.logger = None  # Mock logger for testing
        log_messages = []
        
        # Monkey patch logging
        def mock_log():
            log_messages.append("checkpoint-called")
        simulation_engine._log_checkpoint = mock_log
        
        # Run simulation
        simulation_engine.run()
        
        # Check checkpoint messages
        assert len(log_messages) >= len(range(0, simulation_engine.config.max_steps, 
                                            simulation_engine.config.checkpoint_interval))

    def test_error_handling(self, simulation_engine):
        """Test error handling during simulation"""
        def failing_step(_event):
            raise RuntimeError("Test error")
            
        # Inject failing function
        simulation_engine._handle_generation_step = failing_step
        
        # Run should handle error gracefully
        with pytest.raises(RuntimeError):
            simulation_engine.run()
            
        assert not simulation_engine.state.is_running

    def test_resource_constraint_enforcement(self, simulation_engine):
        """Test enforcement of resource constraints"""
        simulation_engine.initialize()
        simulation_engine.state.is_running = True
        
        # Monitor resource usage
        max_memory_usage = 0.0
        max_compute_usage = 0.0
        
        # Run simulation and track maximum resource usage
        while simulation_engine.step():
            for device in simulation_engine.devices.values():
                max_memory_usage = max(max_memory_usage, device.memory.used)
                max_compute_usage = max(max_compute_usage, device.compute.used)
                
                # Verify constraints aren't violated
                assert device.memory.used <= device.memory.capacity
                assert device.compute.used <= device.compute.capacity

    def test_deterministic_execution(self, devices, network):
        """Test deterministic execution with same seed"""
        def run_simulation(seed, devices, network):
            config = SimulationConfig(
                max_steps=10,
                time_limit=100.0,
                random_seed=seed
            )
            transformer_config = TransformerConfig(
                num_heads=4,
                embedding_dim=256,
                initial_sequence_length=64
            )
            transformer = Transformer(transformer_config)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                engine = SimulationEngine(
                    transformer=transformer,
                    network=network,
                    devices=devices,
                    config=config
                )
                engine.run()
                return engine.state.current_time
                
        # Run twice with same seed
        time1 = run_simulation(42, devices, network)
        time2 = run_simulation(42, devices, network)
        
        assert time1 == time2

class TestSimulationState:
    def test_state_initialization(self):
        """Test simulation state initialization"""
        state = SimulationState()
        assert state.current_step == 0
        assert state.current_time == 0.0
        assert not state.is_running
        assert len(state.active_components) == 0
        assert len(state.pending_transfers) == 0

    def test_state_updates(self):
        """Test simulation state updates"""
        state = SimulationState()
        
        # Update step and time
        state.current_step = 5
        state.current_time = 10.0
        assert state.current_step == 5
        assert state.current_time == 10.0
        
        # Add active component
        state.active_components.add("component_1")
        assert "component_1" in state.active_components
        
        # Add pending transfer
        state.pending_transfers["transfer_1"] = {"time": 15.0}
        assert "transfer_1" in state.pending_transfers

class TestSimulationConfig:
    def test_config_validation(self):
        """Test configuration validation"""
        # Valid config
        valid_config = SimulationConfig(
            max_steps=10,
            time_limit=100.0
        )
        assert valid_config.max_steps > 0
        assert valid_config.time_limit > 0
        
        # Invalid steps
        with pytest.raises(ValueError):
            SimulationConfig(max_steps=-1, time_limit=100.0)
            
        # Invalid time limit
        with pytest.raises(ValueError):
            SimulationConfig(max_steps=10, time_limit=-1.0)

    def test_checkpoint_interval_validation(self):
        """Test checkpoint interval validation"""
        config = SimulationConfig(
            max_steps=10,
            time_limit=100.0,
            checkpoint_interval=5
        )
        assert config.checkpoint_interval <= config.max_steps
        
        with pytest.raises(ValueError):
            SimulationConfig(
                max_steps=10,
                time_limit=100.0,
                checkpoint_interval=20  # Greater than max_steps
            )