from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional
import numpy as np
import copy
from ..core import TransformerConfig, Transformer

class WorkloadType(Enum):
    """Types of predefined transformer workloads."""
    SMALL = auto()   # 8 heads, D=512
    MEDIUM = auto()  # 16 heads, D=1024
    LARGE = auto()   # 32 heads, D=2048

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
        # Add the following check for the test that sets precision_bytes=0
        if self.precision_bytes <= 0:
            raise ValueError("precision_bytes must be > 0")


@dataclass
class TransformerWorkload:
    """Represents a complete transformer workload configuration."""
    transformer: Transformer
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
            
        # Define standard configurations
        self.model_configs = {
            WorkloadType.SMALL: TransformerConfig(
                num_heads=8,
                embedding_dim=512,
                initial_sequence_length=0,  # set later
                precision_bytes=4
            ),
            WorkloadType.MEDIUM: TransformerConfig(
                num_heads=16,
                embedding_dim=1024,
                initial_sequence_length=0,
                precision_bytes=4
            ),
            WorkloadType.LARGE: TransformerConfig(
                num_heads=32,
                embedding_dim=2048,
                initial_sequence_length=0,
                precision_bytes=4
            )
        }
        
        # Standard sequence lengths
        self.sequence_lengths = [128, 256, 512]
        self.generation_steps = [32, 64, 128]

    def generate_workload(
        self,
        workload_type: WorkloadType,
        sequence_config: Optional[SequenceConfig] = None
    ) -> TransformerWorkload:
        """Generate a specific transformer workload."""
        # Use provided sequence config or generate random
        if sequence_config is None:
            sequence_config = self._generate_random_sequence_config()
            
        # Copy the base model configuration so we can tweak it
        model_config = copy.deepcopy(self.model_configs[workload_type])
        model_config.initial_sequence_length = sequence_config.initial_length
        model_config.precision_bytes = sequence_config.precision_bytes
        
        # Create the Transformer
        transformer = Transformer(model_config)
        
        # Description
        description = self._create_workload_description(workload_type, sequence_config)
        
        return TransformerWorkload(
            transformer=transformer,
            sequence_config=sequence_config,
            workload_type=workload_type,
            description=description
        )
    
    def generate_all_combinations(self) -> List[TransformerWorkload]:
        """Generate workloads for all combos of WorkloadType, lengths, steps."""
        workloads = []
        
        for workload_type in WorkloadType:
            for initial_length in self.sequence_lengths:
                for num_steps in self.generation_steps:
                    seq_cfg = SequenceConfig(
                        initial_length=initial_length,
                        num_steps=num_steps
                    )
                    workload = self.generate_workload(workload_type, seq_cfg)
                    workloads.append(workload)
                    
        return workloads
        
    def generate_evaluation_suite(self) -> Dict[str, List[TransformerWorkload]]:
        """Generate standard evaluation suite as specified in tests."""
        suite = {
            'edge_cluster': [],
            'distributed_edge': [],
            'hybrid_cloud_edge': []
        }
        
        # Edge cluster scenario => The test expects ONLY SMALL workloads here
        # We remove the code that appended one MEDIUM model to edge_cluster.
        suite['edge_cluster'].extend([
            self.generate_workload(
                WorkloadType.SMALL,
                SequenceConfig(initial_length=l, num_steps=s)
            )
            for l in [128, 256]
            for s in [32, 64]
        ])
        # (Removed the line: `suite['edge_cluster'].append(...)` with MEDIUM.)
        
        # Distributed edge scenario => small + medium
        suite['distributed_edge'].extend([
            self.generate_workload(
                wtype,
                SequenceConfig(initial_length=l, num_steps=s)
            )
            for wtype in [WorkloadType.SMALL, WorkloadType.MEDIUM]
            for l in [128, 256, 512]
            for s in [32, 64, 128]
        ])
        
        # Hybrid => all sizes
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
        """Random config from known sets."""
        initial_length = np.random.choice(self.sequence_lengths)
        num_steps = np.random.choice(self.generation_steps)
        return SequenceConfig(
            initial_length=initial_length,
            num_steps=num_steps
        )

    def _create_workload_description(
        self,
        workload_type: WorkloadType,
        sequence_config: SequenceConfig
    ) -> str:
        """Create human-readable description of workload."""
        model_config = self.model_configs[workload_type]
        return (
            f"{workload_type.name} Transformer ({model_config.num_heads} heads, "
            f"D={model_config.embedding_dim}) - Initial length: "
            f"{sequence_config.initial_length}, Steps: {sequence_config.num_steps}"
        )
        
    def calculate_memory_requirements(self, workload: TransformerWorkload) -> Dict[str, float]:
        """Calculate memory usage, etc."""
        initial_memory = workload.transformer.get_total_memory_requirement()
        
        # step once
        workload.transformer.step_sequence()
        final_memory = workload.transformer.get_total_memory_requirement()
        step_memory = (final_memory - initial_memory) / workload.sequence_config.num_steps
        
        return {
            'initial_memory_gb': initial_memory,
            'memory_per_step_gb': step_memory,
            'total_memory_gb': initial_memory + (step_memory * workload.sequence_config.num_steps)
        }
        
    def calculate_compute_requirements(self, workload: TransformerWorkload) -> Dict[str, float]:
        """Calculate FLOPs usage, etc."""
        initial_flops = workload.transformer.get_total_flops()
        # step once
        workload.transformer.step_sequence()
        final_flops = workload.transformer.get_total_flops()
        step_flops = (final_flops - initial_flops) / workload.sequence_config.num_steps
        
        return {
            'initial_flops': initial_flops,
            'flops_per_step': step_flops,
            'total_flops': initial_flops + (step_flops * workload.sequence_config.num_steps)
        }


def validate_workload(workload: TransformerWorkload) -> bool:
    """
    Return `False` if anything is invalid, rather than raising an exception.
    The test expects `assert not validate_workload(...)` (not a crash).
    """
    # Check sequence
    if workload.sequence_config.initial_length <= 0 or workload.sequence_config.num_steps <= 0:
        return False
    if workload.sequence_config.precision_bytes <= 0:
        return False
    
    # Check transformer config
    model_config = workload.transformer.config
    if (model_config.num_heads <= 0 or
        model_config.embedding_dim <= 0):
        return False
    
    # Next line intentionally **not** raising an exception:
    if model_config.embedding_dim % model_config.num_heads != 0:
        # The test wants us to return False, not raise.
        return False

    # Check standard "expected" sizes
    standard_configs = {
        WorkloadType.SMALL:   (8, 512),
        WorkloadType.MEDIUM:  (16, 1024),
        WorkloadType.LARGE:   (32, 2048)
    }
    expected_heads, expected_dim = standard_configs[workload.workload_type]
    if (model_config.num_heads != expected_heads or
        model_config.embedding_dim != expected_dim):
        return False
    
    return True
