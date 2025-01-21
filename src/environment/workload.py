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
# File:    src/environment/workload.py
# Description:
#   Implements the WorkloadGenerator, which creates transformer workloads of
#   different sizes and sequence lengths, as well as data structures for
#   representing workload types in simulation.
#
# ---------------------------------------------------------------------------

"""
Contains classes and functions for generating Transformer-based inference
workloads, including different model sizes (SMALL, MEDIUM, LARGE), etc.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional
import numpy as np
import copy

# We do a local import of Transformer/TransformerConfig only inside the method,
# to avoid top-level import from core.

class WorkloadType(Enum):
    """Types of predefined transformer workloads."""
    SMALL = auto()
    MEDIUM = auto()
    LARGE = auto()
    EXTRA_LARGE = auto()

@dataclass
class SequenceConfig:
    """Configuration for sequence generation."""
    initial_length: int
    num_steps: int
    precision_bytes: int = 4  # float32 by default

    def __post_init__(self):
        if self.initial_length <= 0:
            raise ValueError("initial_length must be > 0")
        if self.num_steps <= 0:
            raise ValueError("num_steps must be > 0")
        if self.precision_bytes <= 0:
            raise ValueError("precision_bytes must be > 0")

@dataclass
class TransformerWorkload:
    """
    Represents a complete transformer workload configuration.
    """
    transformer: "object"   # We'll store an instance of core.Transformer (opaque from here).
    sequence_config: SequenceConfig
    workload_type: WorkloadType
    description: str

class WorkloadGenerator:
    """
    Generates transformer workloads for evaluation scenarios.
    """
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        
        # We'll do a local import to define self.model_configs (because it references TransformerConfig).
        from ..core import TransformerConfig
        
        self.model_configs = {
            WorkloadType.SMALL: TransformerConfig(
                num_heads=8, embedding_dim=512, initial_sequence_length=0, precision_bytes=4
            ),
            WorkloadType.MEDIUM: TransformerConfig(
                num_heads=16, embedding_dim=1024, initial_sequence_length=0, precision_bytes=4
            ),
            WorkloadType.LARGE: TransformerConfig(
                num_heads=32, embedding_dim=2048, initial_sequence_length=0, precision_bytes=4
            ),
            WorkloadType.EXTRA_LARGE: TransformerConfig(
                num_heads=64, embedding_dim=4096, initial_sequence_length=0, precision_bytes=4
            )
        }
        
        self.sequence_lengths = [128, 256, 512]
        self.generation_steps = [32, 64, 128]

    def generate_workload(
        self,
        workload_type: WorkloadType,
        sequence_config: Optional[SequenceConfig] = None
    ) -> TransformerWorkload:
        """
        Create a TransformerWorkload, with the chosen type & sequence config.
        """
        if sequence_config is None:
            sequence_config = self._generate_random_sequence_config()
        
        from ..core import TransformerConfig, Transformer
        
        # Copy the base config so we can adjust it
        model_config = copy.deepcopy(self.model_configs[workload_type])
        model_config.initial_sequence_length = sequence_config.initial_length
        model_config.precision_bytes = sequence_config.precision_bytes
        
        transformer_obj = Transformer(model_config)
        
        description = (
            f"{workload_type.name} Transformer ({model_config.num_heads} heads, "
            f"D={model_config.embedding_dim}) - Initial length={sequence_config.initial_length}, "
            f"Steps={sequence_config.num_steps}"
        )
        return TransformerWorkload(
            transformer=transformer_obj,
            sequence_config=sequence_config,
            workload_type=workload_type,
            description=description
        )

    def generate_all_combinations(self) -> List[TransformerWorkload]:
        """Generate a list of workloads for all combos of type, length, steps."""
        workloads = []
        for wtype in WorkloadType:
            for l in self.sequence_lengths:
                for s in self.generation_steps:
                    seq_cfg = SequenceConfig(initial_length=l, num_steps=s)
                    wk = self.generate_workload(wtype, seq_cfg)
                    workloads.append(wk)
        return workloads
        
    def generate_evaluation_suite(self) -> Dict[str, List[TransformerWorkload]]:
        """
        Example suite grouping workloads by some scenario name.
        """
        suite = {
            'edge_cluster': [],
            'distributed_edge': [],
            'hybrid_cloud_edge': []
        }
        
        # Example: edge_cluster => only SMALL
        from ..core import TransformerConfig, Transformer  # local import again
        suite['edge_cluster'].extend([
            self.generate_workload(
                WorkloadType.SMALL,
                SequenceConfig(initial_length=l, num_steps=s)
            )
            for l in [128, 256] for s in [32, 64]
        ])
        
        # distributed => small + medium
        suite['distributed_edge'].extend([
            self.generate_workload(
                wtype,
                SequenceConfig(initial_length=l, num_steps=s)
            )
            for wtype in [WorkloadType.SMALL, WorkloadType.MEDIUM]
            for l in [128, 256, 512]
            for s in [32, 64, 128]
        ])
        
        # hybrid => all
        suite['hybrid_cloud_edge'].extend([
            self.generate_workload(
                wtype,
                SequenceConfig(initial_length=l, num_steps=s)
            )
            for wtype in WorkloadType
            for l in self.sequence_lengths
            for s in self.generation_steps
        ])
        
        return suite

    def _generate_random_sequence_config(self) -> SequenceConfig:
        init_l = int(np.random.choice(self.sequence_lengths))
        num_s = int(np.random.choice(self.generation_steps))
        return SequenceConfig(initial_length=init_l, num_steps=num_s)

def validate_workload(workload: TransformerWorkload) -> bool:
    """
    Return False if invalid. Avoid top-level reference to Transformer classes.
    """
    seq_cfg = workload.sequence_config
    if seq_cfg.initial_length <= 0 or seq_cfg.num_steps <= 0 or seq_cfg.precision_bytes <= 0:
        return False
    
    # We'll do a local import and check if the transformer's config matches standard expectations
    from ..core import Transformer, TransformerConfig
    if not isinstance(workload.transformer, Transformer):
        return False
    
    conf = workload.transformer.config
    if conf.num_heads <= 0 or conf.embedding_dim <= 0:
        return False
    
    # Must divide evenly
    if (conf.embedding_dim % conf.num_heads) != 0:
        return False
    
    # Quick standard check
    standard_cfg = {
        WorkloadType.SMALL: (8, 512),
        WorkloadType.MEDIUM: (16, 1024),
        WorkloadType.LARGE: (32, 2048),
        WorkloadType.EXTRA_LARGE: (64, 4096),
    }
    (exp_heads, exp_dim) = standard_cfg[workload.workload_type]
    if conf.num_heads != exp_heads or conf.embedding_dim != exp_dim:
        return False
    
    return True
