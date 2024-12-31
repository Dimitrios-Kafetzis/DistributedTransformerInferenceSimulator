import pytest
from datetime import datetime
from src.core.event import Event, EventType, EventQueue

@pytest.fixture
def basic_event():
    """Create a basic computation event"""
    return Event(
        time=1.0,
        event_type=EventType.COMPUTE_START,
        component_id="test_component",
        source_device="device_0"
    )

@pytest.fixture
def transfer_event():
    """Create a data transfer event"""
    return Event(
        time=2.0,
        event_type=EventType.TRANSFER_START,
        component_id="test_component",
        source_device="device_0",
        target_device="device_1",
        data_size=1.0
    )

@pytest.fixture
def empty_queue():
    """Create an empty event queue"""
    return EventQueue()

@pytest.fixture
def populated_queue(basic_event, transfer_event):
    """Create an event queue with some events"""
    queue = EventQueue()
    queue.schedule_event(basic_event)
    queue.schedule_event(transfer_event)
    return queue

class TestEvent:
    def test_event_initialization(self, basic_event):
        """Test basic event initialization"""
        assert basic_event.time == 1.0
        assert basic_event.event_type == EventType.COMPUTE_START
        assert basic_event.component_id == "test_component"
        assert basic_event.source_device == "device_0"

    def test_transfer_event_validation(self):
        """Test transfer event validation"""
        # Transfer events require source and target devices
        with pytest.raises(AssertionError):
            Event(
                time=1.0,
                event_type=EventType.TRANSFER_START,
                component_id="test_component"
            )

    def test_event_metadata(self):
        """Test event metadata handling"""
        event = Event(
            time=1.0,
            event_type=EventType.COMPUTE_START,
            component_id="test_component",
            metadata={"flops": 1000, "memory": 2.0}
        )
        assert event.metadata["flops"] == 1000
        assert event.metadata["memory"] == 2.0

    def test_event_ordering(self, basic_event, transfer_event):
        """Test event ordering based on time"""
        assert basic_event < transfer_event  # time 1.0 < 2.0
        
        # Events with same time should be ordered by creation
        same_time_event = Event(
            time=1.0,
            event_type=EventType.COMPUTE_COMPLETE,
            component_id="test_component"
        )
        assert basic_event < same_time_event

class TestEventQueue:
    def test_queue_initialization(self, empty_queue):
        """Test event queue initialization"""
        assert empty_queue.current_time == 0.0
        assert empty_queue.queue.empty()

    def test_event_scheduling(self, empty_queue, basic_event):
        """Test event scheduling"""
        empty_queue.schedule_event(basic_event)
        assert not empty_queue.queue.empty()
        assert empty_queue.peek_next_time() == 1.0

    def test_past_event_scheduling(self, populated_queue):
        """Test scheduling events in the past"""
        populated_queue.current_time = 2.0
        past_event = Event(
            time=1.0,
            event_type=EventType.COMPUTE_START,
            component_id="test_component"
        )
        with pytest.raises(AssertionError):
            populated_queue.schedule_event(past_event)

    def test_event_processing(self, populated_queue):
        """Test event retrieval and processing"""
        # Get first event
        event = populated_queue.get_next_event()
        assert event.time == 1.0
        assert populated_queue.current_time == 1.0
        
        # Get second event
        event = populated_queue.get_next_event()
        assert event.time == 2.0
        assert populated_queue.current_time == 2.0
        
        # Queue should be empty now
        assert populated_queue.get_next_event() is None

    def test_computation_scheduling(self, empty_queue):
        """Test scheduling computation events"""
        empty_queue.schedule_computation(
            component_id="test_component",
            device_id="device_0",
            computation_time=1.5,
            metadata={"flops": 1000}
        )
        
        # Should create start and complete events
        start_event = empty_queue.get_next_event()
        assert start_event.event_type == EventType.COMPUTE_START
        assert start_event.time == 0.0
        
        complete_event = empty_queue.get_next_event()
        assert complete_event.event_type == EventType.COMPUTE_COMPLETE
        assert complete_event.time == 1.5

    def test_transfer_scheduling(self, empty_queue):
        """Test scheduling transfer events"""
        empty_queue.schedule_transfer(
            component_id="test_component",
            source_device="device_0",
            target_device="device_1",
            data_size=2.0,
            transfer_time=3.0
        )
        
        # Should create start and complete events
        start_event = empty_queue.get_next_event()
        assert start_event.event_type == EventType.TRANSFER_START
        assert start_event.time == 0.0
        assert start_event.data_size == 2.0
        
        complete_event = empty_queue.get_next_event()
        assert complete_event.event_type == EventType.TRANSFER_COMPLETE
        assert complete_event.time == 3.0

    def test_generation_step_scheduling(self, empty_queue):
        """Test scheduling generation step events"""
        empty_queue.schedule_generation_step(
            step_number=1,
            metadata={"sequence_length": 128}
        )
        
        event = empty_queue.get_next_event()
        assert event.event_type == EventType.GENERATION_STEP
        assert event.component_id == "step_1"
        assert event.metadata["sequence_length"] == 128

    def test_cache_update_scheduling(self, empty_queue):
        """Test scheduling cache update events"""
        empty_queue.schedule_cache_update(
            head_id="head_0",
            device_id="device_0",
            new_cache_size=1.5,
            metadata={"generation_step": 10}
        )
        
        event = empty_queue.get_next_event()
        assert event.event_type == EventType.CACHE_UPDATE
        assert event.component_id == "head_0"
        assert event.source_device == "device_0"
        assert event.data_size == 1.5

    def test_mixed_event_ordering(self, empty_queue):
        """Test ordering of different event types"""
        # Schedule events in non-chronological order
        empty_queue.schedule_transfer(
            component_id="comp_1",
            source_device="dev_0",
            target_device="dev_1",
            data_size=1.0,
            transfer_time=3.0
        )
        
        empty_queue.schedule_computation(
            component_id="comp_2",
            device_id="dev_0",
            computation_time=1.0
        )
        
        empty_queue.schedule_generation_step(step_number=1)
        
        # Events should come out in chronological order
        event1 = empty_queue.get_next_event()
        event2 = empty_queue.get_next_event()
        event3 = empty_queue.get_next_event()
        event4 = empty_queue.get_next_event()
        
        assert event1.event_type == EventType.GENERATION_STEP  # t=0
        assert event2.event_type == EventType.TRANSFER_START   # t=0
        assert event3.event_type == EventType.COMPUTE_START    # t=0
        assert event4.event_type == EventType.COMPUTE_COMPLETE # t=1.0

    def test_peek_functionality(self, populated_queue):
        """Test peeking at next event time"""
        assert populated_queue.peek_next_time() == 1.0
        
        # Peek shouldn't change queue state
        assert populated_queue.peek_next_time() == 1.0
        
        # After getting an event, peek should show next event
        populated_queue.get_next_event()
        assert populated_queue.peek_next_time() == 2.0