"""
Core components for the transformer inference simulation.
This module provides the fundamental building blocks including device management,
network topology, transformer components, and event handling.
"""

from .device import Device, ResourceState
from .network import Network, Link
from .transformer import (
    TransformerConfig,
    TransformerComponent,
    AttentionHead,
    ProjectionLayer,
    FeedForwardNetwork,
    Transformer
)
from .event import (
    EventType,
    Event,
    EventQueue
)

# Version information
__version__ = '0.1.0'

# Define what should be available when using "from core import *"
__all__ = [
    # Device management
    'Device',
    'ResourceState',
    
    # Network components
    'Network',
    'Link',
    
    # Transformer components
    'TransformerConfig',
    'TransformerComponent',
    'AttentionHead',
    'ProjectionLayer',
    'FeedForwardNetwork',
    'Transformer',
    
    # Event system
    'EventType',
    'Event',
    'EventQueue',
]

# Module level doc strings for key components
Device.__doc__ = "Represents a compute device with specific resource capabilities."
Network.__doc__ = "Manages network topology and communication between devices."
Transformer.__doc__ = "Main transformer class that manages all components for inference."
EventQueue.__doc__ = "Manages the discrete event simulation timeline."