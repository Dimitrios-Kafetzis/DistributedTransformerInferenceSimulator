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
# File:    src/core/__init__.py
# Description:
#   Initializes the core module, providing fundamental classes for devices,
#   networks, transformer components, and event handling in the simulator.
#
# ---------------------------------------------------------------------------

"""
Initializes the core subpackage, exposing essential classes like Device, Network,
Transformer, and EventQueue that form the backbone of the simulator's functionality.
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