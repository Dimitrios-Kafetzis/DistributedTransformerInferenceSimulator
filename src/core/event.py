# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Dimitrios Kafetzis
#
# This file is part of the Transformer Inference Simulator project.
# Licensed under the MIT License; you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#   https://opensource.org/licenses/MIT
#
# Author:  Dimitrios Kafetzis (dimitrioskafetzis@gmail.com)
# File:    src/core/event.py
# Description:
#   Provides the Event data structure, enumeration of event types, and
#   the EventQueue for scheduling and processing simulation events.
#
# ---------------------------------------------------------------------------

"""
Defines the Event class with multiple event types (compute, transfer, generation step, etc.)
and the EventQueue that manages simulation events in chronological order, enabling
discrete-event simulation for distributed transformer inference.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional
from queue import PriorityQueue

# We use a module-level or class-level counter to assign a unique creation index
_event_creation_counter = 0

class EventType(Enum):
    """Types of events in the simulation"""
    COMPUTE_START = auto()
    COMPUTE_COMPLETE = auto()
    TRANSFER_START = auto()
    TRANSFER_COMPLETE = auto()
    GENERATION_STEP = auto()
    CACHE_UPDATE = auto()
    RESOURCE_UPDATE = auto()

@dataclass
class Event:
    """Represents a discrete event in the simulation"""
    
    # We do NOT use order=True on the dataclass
    time: float
    event_type: EventType
    component_id: str
    source_device: Optional[str] = None
    target_device: Optional[str] = None
    data_size: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    # Tie-breaker for events with the same time
    sort_index: int = field(init=False, repr=False)

    def __post_init__(self):
        # If this event_type requires certain fields, do asserts:
        if self.event_type in [EventType.TRANSFER_START, EventType.TRANSFER_COMPLETE]:
            assert self.source_device is not None, "Transfer events require source_device"
            assert self.target_device is not None, "Transfer events require target_device"
            assert self.data_size is not None, "Transfer events require data_size"
        
        global _event_creation_counter
        _event_creation_counter += 1
        self.sort_index = _event_creation_counter

    def __lt__(self, other: "Event") -> bool:
        """
        Compare events primarily by 'time'. If 'time' is equal,
        we force GENERATION_STEP events to come before others.
        If both are (or neither is) GENERATION_STEP, then we fall
        back to the creation order (sort_index).
        """
        if self.time != other.time:
            return self.time < other.time

        # If times are the same, prefer GENERATION_STEP first
        if self.event_type == EventType.GENERATION_STEP and other.event_type != EventType.GENERATION_STEP:
            return True
        if other.event_type == EventType.GENERATION_STEP and self.event_type != EventType.GENERATION_STEP:
            return False

        # Otherwise, fall back to creation order
        return self.sort_index < other.sort_index

class EventQueue:
    """Manages simulation events in chronological order"""
    
    def __init__(self):
        self.queue = PriorityQueue()
        self.current_time = 0.0

    def schedule_event(self, event: Event) -> None:
        """Add an event to the queue."""
        # Ensure events are not scheduled in the past
        assert event.time >= self.current_time, (
            f"Cannot schedule event in the past. "
            f"Current time: {self.current_time}, Event time: {event.time}"
        )
        self.queue.put(event)

    def get_next_event(self) -> Optional[Event]:
        """Retrieve the next event from the queue."""
        if self.queue.empty():
            return None
        event = self.queue.get()
        self.current_time = event.time
        return event

    def peek_next_time(self) -> Optional[float]:
        """Look at the time of the next event without removing it."""
        if self.queue.empty():
            return None
        # PriorityQueue internal list is self.queue.queue
        return self.queue.queue[0].time

    def schedule_computation(
        self,
        component_id: str,
        device_id: str,
        computation_time: float,
        metadata: Optional[Dict[str, Any]] = None,
        start_time: float = None
    ) -> None:
        """
        Schedule the start and completion of a computation.
        
        If `start_time` is provided, we use it for the COMPUTE_START event time,
        else we default to `self.current_time`. Then completion is start_time + computation_time.
        """
        # If no explicit start_time is given, default to the queue's current_time
        if start_time is None:
            start_time = self.current_time

        complete_time = start_time + computation_time

        # COMPILE START event
        self.schedule_event(Event(
            time=start_time,
            event_type=EventType.COMPUTE_START,
            component_id=component_id,
            source_device=device_id,
            metadata=metadata
        ))
        
        # COMPUTE COMPLETE event
        self.schedule_event(Event(
            time=complete_time,
            event_type=EventType.COMPUTE_COMPLETE,
            component_id=component_id,
            source_device=device_id,
            metadata=metadata
        ))

    def schedule_transfer(
        self,
        component_id: str,
        source_device: str,
        target_device: str,
        data_size: float,
        transfer_time: float,
        start_time: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None        
    ):
        """
        Schedule the start and completion of a data transfer.
        
        If `start_time` is provided, we use that for the TRANSFER_START event time,
        else we default to `self.current_time`. Then the complete event is at start_time + transfer_time.
        """
        if start_time is None:
            start_time = self.current_time

        complete_time = start_time + transfer_time

        # TRANSFER START event
        self.schedule_event(Event(
            time=start_time,
            event_type=EventType.TRANSFER_START,
            component_id=component_id,
            source_device=source_device,
            target_device=target_device,
            data_size=data_size,
            metadata=metadata
        ))
        
        # TRANSFER COMPLETE event
        self.schedule_event(Event(
            time=complete_time,
            event_type=EventType.TRANSFER_COMPLETE,
            component_id=component_id,
            source_device=source_device,
            target_device=target_device,
            data_size=data_size,
            metadata=metadata
        ))

    def schedule_generation_step(
        self,
        step_number: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        component_id: Optional[str] = None,
        override_time: Optional[float] = None
    ) -> None:
        """
        - If 'step_number' is given, we create e.g. "step_<step_number>".
        - If 'component_id' is given, use that directly.
        - If 'override_time' is given, schedule at that time.
          Otherwise schedule at 'self.current_time'.
        - If the test code calls schedule_generation_step(step_number=1, metadata=...),
          we handle that.
        - If the engine calls schedule_generation_step(component_id=..., override_time=...),
          we handle that too.
        """
    
        # Decide on the final component_id
        if step_number is not None:
            # test-style usage
            final_comp_id = f"step_{step_number}"
        elif component_id is not None:
            # engine-style usage
            final_comp_id = component_id
        else:
            # fallback
            final_comp_id = "step_1"
    
        # Decide on the final time
        if override_time is not None:
            final_time = override_time
        else:
            final_time = self.current_time
    
        # Create and schedule
        evt = Event(
            time=final_time,
            event_type=EventType.GENERATION_STEP,
            component_id=final_comp_id,
            metadata=metadata
        )
        self.schedule_event(evt)

    def schedule_cache_update(
        self,
        head_id: str,
        device_id: str,
        new_cache_size: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Schedule a K/V cache update."""
        self.schedule_event(Event(
            time=self.current_time,
            event_type=EventType.CACHE_UPDATE,
            component_id=head_id,
            source_device=device_id,
            data_size=new_cache_size,
            metadata=metadata
        ))
