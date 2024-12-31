from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

@dataclass
class TransformerConfig:
    """Configuration for transformer model parameters"""
    num_heads: int
    embedding_dim: int
    initial_sequence_length: int
    precision_bytes: int = 4  # e.g., 4 for float32
    
    @property
    def head_dim(self) -> int:
        """Dimension per attention head"""
        return self.embedding_dim // self.num_heads

    def __post_init__(self):
        if self.num_heads <= 0 or self.embedding_dim <= 0:
            raise ValueError("Invalid config: heads and dim must be > 0")

class TransformerComponent:
    """Base class for transformer components"""
    def __init__(self, config: TransformerConfig, component_id: str):
        self.config = config
        self.component_id = component_id
        
    def compute_memory_requirements(self, sequence_length: int) -> float:
        """Calculate memory requirements in GB"""
        raise NotImplementedError
        
    def compute_flops(self, sequence_length: int) -> float:
        """Calculate computational requirements in FLOPS"""
        raise NotImplementedError

class AttentionHead(TransformerComponent):
    """Represents a single attention head in the transformer"""
    
    def __init__(self, config: TransformerConfig, head_idx: int):
        super().__init__(config, f"head_{head_idx}")
        self.head_idx = head_idx
        self.cache_size = 0.0  # Current K/V cache size in GB
        
    def compute_memory_requirements(self, sequence_length: int) -> float:
        """Calculate memory requirements including QKV matrices and weights"""
        # QKV matrices memory (eq. 1 from paper)
        qkv_memory = (3 * sequence_length * self.config.head_dim * 
                     self.config.precision_bytes)
        
        # QKV weights memory (eq. 2 from paper)
        weight_memory = 3 * self.config.embedding_dim * self.config.head_dim * \
                       self.config.precision_bytes
        
        # Convert to GB
        return (qkv_memory + weight_memory) / (1024**3)
        
    def compute_cache_memory(self, generation_step: int) -> float:
        """Calculate K/V cache memory requirement at current generation step"""
        # Cache memory (eq. 3 from paper)
        cache_size = generation_step * self.config.embedding_dim * \
                    self.config.precision_bytes
        
        # Convert to GB
        return cache_size / (1024**3)
        
    def compute_flops(self, sequence_length: int) -> float:
        """Calculate FLOPs for QKV transformations and attention computation"""
        # QKV transformation FLOPs (eq. 7 from paper)
        qkv_flops = 3 * sequence_length * self.config.embedding_dim * \
                    self.config.head_dim
        
        # Attention score FLOPs (eq. 8 from paper)
        attn_flops = sequence_length * sequence_length * self.config.head_dim
        
        return qkv_flops + attn_flops

class ProjectionLayer(TransformerComponent):
    """Represents the projection layer after attention heads"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__(config, "projection")
        
    def compute_memory_requirements(self, sequence_length: int) -> float:
        """Calculate memory requirements for projection"""
        # Projection memory (eq. 5 from paper)
        memory = sequence_length * self.config.embedding_dim * \
                self.config.precision_bytes
        
        # Convert to GB
        return memory / (1024**3)
        
    def compute_flops(self, sequence_length: int) -> float:
        """Calculate FLOPs for projection computation"""
        # Projection FLOPs (eq. 10 from paper)
        return sequence_length * self.config.embedding_dim * self.config.embedding_dim

class FeedForwardNetwork(TransformerComponent):
    """Represents the feed-forward network"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__(config, "ffn")
        
    def compute_memory_requirements(self, sequence_length: int) -> float:
        """Calculate memory requirements for FFN"""
        # FFN memory (eq. 6 from paper)
        memory = 4 * sequence_length * self.config.embedding_dim * \
                self.config.precision_bytes
        
        # Convert to GB
        return memory / (1024**3)
        
    def compute_flops(self, sequence_length: int) -> float:
        """Calculate FLOPs for FFN computation"""
        # FFN FLOPs (eq. 11 from paper)
        return 8 * sequence_length * self.config.embedding_dim * self.config.embedding_dim

class Transformer:
    """Main transformer class that manages all components"""
    
    def __init__(self, config: TransformerConfig):
        self.config = config
        
        # Initialize components
        self.attention_heads = [
            AttentionHead(config, i) 
            for i in range(config.num_heads)
        ]
        self.projection = ProjectionLayer(config)
        self.ffn = FeedForwardNetwork(config)
        
        # Track current sequence length
        self.current_sequence_length = config.initial_sequence_length

        self.components = self.attention_heads + [self.projection, self.ffn]
        
    def get_component(self, component_id: str) -> TransformerComponent:
        """Retrieve component by ID"""
        if component_id.startswith("head_"):
            head_idx = int(component_id.split("_")[1])
            return self.attention_heads[head_idx]
        elif component_id == "projection":
            return self.projection
        elif component_id == "ffn":
            return self.ffn
        else:
            raise ValueError(f"Unknown component ID: {component_id}")
            
    def get_all_components(self) -> List[TransformerComponent]:
        """Return list of all components"""
        return self.attention_heads + [self.projection, self.ffn]
        
    def step_sequence(self) -> None:
        """Increment sequence length for next generation step"""
        self.current_sequence_length += 1
        
    def get_total_memory_requirement(self) -> float:
        """Calculate total memory requirement across all components"""
        total = 0.0
        for component in self.get_all_components():
            total += component.compute_memory_requirements(self.current_sequence_length)
            if isinstance(component, AttentionHead):
                total += component.compute_cache_memory(
                    self.current_sequence_length - self.config.initial_sequence_length
                )
        return total
        
    def get_total_flops(self) -> float:
        """Calculate total FLOPs across all components"""
        return sum(
            component.compute_flops(self.current_sequence_length)
            for component in self.get_all_components()
        )