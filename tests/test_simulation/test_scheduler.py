import pytest
from datetime import datetime
from typing import Dict, List

from src.core import (
    Device,
    Network,
    Transformer,
    TransformerConfig,
    EventQueue,
    Event,
    EventType
)
from src.simulation.scheduler import (
    EventScheduler,
    ExecutionPlan,
    ComponentState
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

@pytest.fixture
def transformer(transformer_config):
    """Create transformer instance"""
    return Transformer(transformer_config)

@pytest.fixture
def event_queue():
    """Create event queue"""
    return EventQueue()

@pytest.fixture
def scheduler(transformer, network, devices, event_queue):
    """Create scheduler instance"""
    return EventScheduler(
        transformer=transformer,
        network=network,
        devices=devices,
        event_queue=event_queue
    )

@pytest.fixture
def basic_plan(transformer, devices):
    """Create a basic execution plan"""
    device_assignments = {
        "head_0": "device_0",
        "head_1": "device_0",
        "head_2": "device_1",
        "head_3": "device_1",
        "projection": "device_1",
        "ffn": "device_0"
    }
    
    plan = ExecutionPlan(
        step=0,
        sequence_length=64,
        dependencies=create_dependencies(transformer),
        device_assignments=device_assignments
    )
    
    # Ensure each component is initialized as PENDING
    for comp_id in plan.dependencies.keys():
        plan.component_states[comp_id] = ComponentState.PENDING
    
    return plan

def create_dependencies(transformer: Transformer) -> Dict[str, set]:
    """Helper function to create dependency graph"""
    dependencies = {}
    
    # Attention heads have no dependencies
    for head in transformer.attention_heads:
        dependencies[head.component_id] = set()
        
    # Projection depends on all heads
    dependencies["projection"] = {
        head.component_id for head in transformer.attention_heads
    }
    
    # FFN depends on projection
    dependencies["ffn"] = {"projection"}
    
    return dependencies

class TestEventScheduler:
    def test_scheduler_initialization(self, scheduler):
        """Test scheduler initialization"""
        assert scheduler.transformer is not None
        assert scheduler.network is not None
        assert scheduler.devices is not None
        assert scheduler.event_queue is not None
        assert scheduler.current_plan is None
        assert isinstance(scheduler.active_transfers, dict)

    def test_execution_plan_creation(self, scheduler, devices):
        """Test creation of execution plan"""
        device_assignments = {
            comp.component_id: "device_0"
            for comp in scheduler.transformer.get_all_components()
        }
        
        plan = scheduler.create_execution_plan(
            step=0,
            device_assignments=device_assignments
        )
        
        assert plan.step == 0
        assert plan.sequence_length == scheduler.transformer.current_sequence_length
        assert isinstance(plan.dependencies, dict)
        assert isinstance(plan.device_assignments, dict)

    def test_dependency_tracking(self, scheduler, basic_plan):
        """Test dependency tracking in execution plan"""
        # Verify projection layer dependencies
        assert "projection" in basic_plan.dependencies
        assert len(basic_plan.dependencies["projection"]) == 4  # All attention heads
        
        # Verify FFN dependencies
        assert "ffn" in basic_plan.dependencies
        assert "projection" in basic_plan.dependencies["ffn"]

    def test_execution_scheduling(self, scheduler, basic_plan):
        """Test scheduling of component execution"""
        scheduler.schedule_execution(basic_plan)
        
        # Verify events were scheduled
        events = []
        while True:
            event = scheduler.event_queue.get_next_event()
            if event is None:
                break
            events.append(event)
            
        assert len(events) > 0  # Should have scheduled some events
        
        # Verify event ordering
        for i in range(1, len(events)):
            assert events[i].time >= events[i-1].time

    def test_ready_components_identification(self, scheduler, basic_plan):
        """Test identification of ready-to-execute components"""
        scheduler.current_plan = basic_plan
        
        # Initially, all attention heads should be ready
        ready = scheduler._get_ready_components()
        assert len(ready) == 4  # All attention heads
        assert all(comp_id.startswith("head_") for comp_id in ready)
        
        # Mark some heads as completed
        basic_plan.component_states["head_0"] = ComponentState.COMPLETED
        basic_plan.component_states["head_1"] = ComponentState.COMPLETED
        
        ready = scheduler._get_ready_components()
        assert len(ready) == 2  # Remaining heads

    def test_component_scheduling(self, scheduler, basic_plan):
        """Test scheduling of individual components"""
        scheduler.current_plan = basic_plan
        
        # Schedule an attention head
        scheduler._schedule_component("head_0")
        
        events = []
        while True:
            event = scheduler.event_queue.get_next_event()
            if event is None:
                break
            events.append(event)
            
        assert len(events) >= 2  # Should have start and complete events
        assert events[0].event_type == EventType.COMPUTE_START
        assert events[1].event_type == EventType.COMPUTE_COMPLETE

    def test_transfer_scheduling(self, scheduler, basic_plan):
        """Test scheduling of data transfers"""
        scheduler.current_plan = basic_plan
        
        # Schedule component that requires transfer
        scheduler._schedule_component("projection")  # Requires head outputs
        
        events = []
        while True:
            event = scheduler.event_queue.get_next_event()
            if event is None:
                break
            events.append(event)
            
        # Should have transfer events
        transfer_events = [e for e in events 
                         if e.event_type in (EventType.TRANSFER_START,
                                           EventType.TRANSFER_COMPLETE)]
        assert len(transfer_events) > 0

    def test_completion_handling(self, scheduler, basic_plan):
        """Test handling of component completion"""
        scheduler.current_plan = basic_plan
        
        # Create completion event
        event = Event(
            time=1.0,
            event_type=EventType.COMPUTE_COMPLETE,
            component_id="head_0"
        )
        
        scheduler.handle_event_completion(event)
        
        assert (scheduler.current_plan.component_states["head_0"] == 
                ComponentState.COMPLETED)

    def test_resource_constraints(self, scheduler, basic_plan):
        """Test enforcement of resource constraints"""
        scheduler.current_plan = basic_plan
        
        # Try to schedule too many components on one device
        device = scheduler.devices["device_0"]
        original_capacity = device.compute.capacity
        device.compute.capacity = 1.0  # Very low capacity
        
        # Should handle resource constraints gracefully
        scheduler.schedule_execution(basic_plan)
        
        # Restore capacity
        device.compute.capacity = original_capacity

    def test_dynamic_scheduling(self, scheduler, basic_plan):
        """Test dynamic scheduling adjustments"""
        scheduler.schedule_execution(basic_plan)
        
        # Simulate completion of some components
        event = Event(
            time=1.0,
            event_type=EventType.COMPUTE_COMPLETE,
            component_id="head_0"
        )
        scheduler.handle_event_completion(event)
        
        # Verify new components are scheduled
        events = []
        while True:
            event = scheduler.event_queue.get_next_event()
            if event is None:
                break
            events.append(event)
            
        assert len(events) > 0

class TestExecutionPlan:
    def test_plan_initialization(self, basic_plan):
        """Test execution plan initialization"""
        assert basic_plan.step == 0
        assert basic_plan.sequence_length == 64
        assert isinstance(basic_plan.dependencies, dict)
        assert isinstance(basic_plan.device_assignments, dict)
        assert isinstance(basic_plan.component_states, dict)
        assert isinstance(basic_plan.completion_times, dict)

    def test_component_state_tracking(self, basic_plan):
        """Test component state tracking"""
        # Initially all components should be PENDING
        for state in basic_plan.component_states.values():
            assert state == ComponentState.PENDING
            
        # Update states
        basic_plan.component_states["head_0"] = ComponentState.EXECUTING
        assert basic_plan.component_states["head_0"] == ComponentState.EXECUTING
        
        basic_plan.component_states["head_0"] = ComponentState.COMPLETED
        assert basic_plan.component_states["head_0"] == ComponentState.COMPLETED

    def test_completion_time_tracking(self, basic_plan):
        """Test completion time tracking"""
        basic_plan.completion_times["head_0"] = 1.0
        basic_plan.completion_times["head_1"] = 2.0
        
        assert basic_plan.completion_times["head_0"] == 1.0
        assert basic_plan.completion_times["head_1"] == 2.0
        assert basic_plan.estimated_completion_time == 2.0

class TestComponentState:
    def test_state_transitions(self):
        """Test component state transitions"""
        # Test valid transitions
        states = [
            ComponentState.PENDING,
            ComponentState.READY,
            ComponentState.EXECUTING,
            ComponentState.TRANSFERRING,
            ComponentState.COMPLETED
        ]
        
        # Verify all states are unique
        assert len(set(states)) == len(states)
        
        # Test string representation
        for state in states:
            assert isinstance(state.value, str)
            assert len(state.value) > 0