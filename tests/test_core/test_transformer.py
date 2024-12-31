import pytest
import numpy as np
from src.core.transformer import (
    TransformerConfig,
    TransformerComponent,
    AttentionHead,
    ProjectionLayer,
    FeedForwardNetwork,
    Transformer
)

@pytest.fixture
def basic_config():
    """Create a basic transformer configuration"""
    return TransformerConfig(
        num_heads=8,
        embedding_dim=512,
        initial_sequence_length=128,
        precision_bytes=4
    )

@pytest.fixture
def medium_config():
    """Create a medium-sized transformer configuration"""
    return TransformerConfig(
        num_heads=16,
        embedding_dim=1024,
        initial_sequence_length=256,
        precision_bytes=4
    )

@pytest.fixture
def basic_transformer(basic_config):
    """Create a basic transformer instance"""
    return Transformer(basic_config)

class TestTransformerConfig:
    def test_config_initialization(self, basic_config):
        """Test transformer configuration initialization"""
        assert basic_config.num_heads == 8
        assert basic_config.embedding_dim == 512
        assert basic_config.initial_sequence_length == 128
        assert basic_config.precision_bytes == 4

    def test_head_dimension_calculation(self, basic_config):
        """Test head dimension calculation"""
        assert basic_config.head_dim == 64  # 512/8
        
    def test_invalid_dimensions(self):
        """Test handling of invalid dimensions by verifying validate_workload() returns False."""
    
        from src.environment.workload import validate_workload, SequenceConfig, TransformerWorkload, WorkloadType
        from src.core import TransformerConfig, Transformer
    
        # Create a config with embedding_dim NOT divisible by num_heads
        config = TransformerConfig(
            num_heads=7,
            embedding_dim=512,
            initial_sequence_length=128,  # This is allowed
            precision_bytes=4
        )
    
        # Build a Transformer from that config
        transformer = Transformer(config)
    
        # Build a minimal SequenceConfig
        seq_cfg = SequenceConfig(initial_length=64, num_steps=16)
    
        # Build a workload that references this transformer
        invalid_workload = TransformerWorkload(
            transformer=transformer,
            sequence_config=seq_cfg,
            workload_type=WorkloadType.SMALL,
            description="Invalid dimension test"
        )
    
        # Now check that validate_workload() returns False
        assert not validate_workload(invalid_workload), (
            "Expected validate_workload() to return False when embedding_dim isn't divisible by num_heads."
        )

class TestAttentionHead:
    def test_head_initialization(self, basic_config):
        """Test attention head initialization"""
        head = AttentionHead(basic_config, head_idx=0)
        assert head.head_idx == 0
        assert head.cache_size == 0.0
        assert head.component_id == "head_0"

    def test_memory_requirements(self, basic_config):
        """Test memory requirement calculations"""
        head = AttentionHead(basic_config, head_idx=0)
        
        # Calculate expected memory for sequence length 128
        mem_req = head.compute_memory_requirements(128)
        
        # QKV matrices: 3 * seq_len * head_dim * bytes
        expected_qkv = (3 * 128 * basic_config.head_dim * 4) / (1024**3)
        # QKV weights: 3 * emb_dim * head_dim * bytes
        expected_weights = (3 * basic_config.embedding_dim * basic_config.head_dim * 4) / (1024**3)
        
        assert mem_req == pytest.approx(expected_qkv + expected_weights)

    def test_cache_memory(self, basic_config):
        """Test K/V cache memory calculations"""
        head = AttentionHead(basic_config, head_idx=0)
        
        # Test cache size for 10 generated tokens
        cache_size = head.compute_cache_memory(10)
        
        # Cache size: num_tokens * embedding_dim * bytes
        expected_size = (10 * basic_config.embedding_dim * 4) / (1024**3)
        assert cache_size == pytest.approx(expected_size)

    def test_compute_flops(self, basic_config):
        """Test FLOPS calculation"""
        head = AttentionHead(basic_config, head_idx=0)
        
        flops = head.compute_flops(128)
        
        # QKV transformations: 3 * seq_len * emb_dim * head_dim
        qkv_flops = 3 * 128 * basic_config.embedding_dim * basic_config.head_dim
        # Attention computation: seq_len * seq_len * head_dim
        attn_flops = 128 * 128 * basic_config.head_dim
        
        assert flops == pytest.approx(qkv_flops + attn_flops)

class TestProjectionLayer:
    def test_projection_initialization(self, basic_config):
        """Test projection layer initialization"""
        proj = ProjectionLayer(basic_config)
        assert proj.component_id == "projection"

    def test_memory_requirements(self, basic_config):
        """Test memory requirement calculations"""
        proj = ProjectionLayer(basic_config)
        
        mem_req = proj.compute_memory_requirements(128)
        
        # Memory: seq_len * embedding_dim * bytes
        expected_mem = (128 * basic_config.embedding_dim * 4) / (1024**3)
        assert mem_req == pytest.approx(expected_mem)

    def test_compute_flops(self, basic_config):
        """Test FLOPS calculation"""
        proj = ProjectionLayer(basic_config)
        
        flops = proj.compute_flops(128)
        
        # FLOPS: seq_len * embedding_dim * embedding_dim
        expected_flops = 128 * basic_config.embedding_dim * basic_config.embedding_dim
        assert flops == pytest.approx(expected_flops)

class TestFeedForwardNetwork:
    def test_ffn_initialization(self, basic_config):
        """Test FFN initialization"""
        ffn = FeedForwardNetwork(basic_config)
        assert ffn.component_id == "ffn"

    def test_memory_requirements(self, basic_config):
        """Test memory requirement calculations"""
        ffn = FeedForwardNetwork(basic_config)
        
        mem_req = ffn.compute_memory_requirements(128)
        
        # Memory: 4 * seq_len * embedding_dim * bytes
        expected_mem = (4 * 128 * basic_config.embedding_dim * 4) / (1024**3)
        assert mem_req == pytest.approx(expected_mem)

    def test_compute_flops(self, basic_config):
        """Test FLOPS calculation"""
        ffn = FeedForwardNetwork(basic_config)
        
        flops = ffn.compute_flops(128)
        
        # FLOPS: 8 * seq_len * embedding_dim * embedding_dim
        expected_flops = 8 * 128 * basic_config.embedding_dim * basic_config.embedding_dim
        assert flops == pytest.approx(expected_flops)

class TestTransformer:
    def test_transformer_initialization(self, basic_transformer):
        """Test transformer initialization"""
        assert len(basic_transformer.attention_heads) == 8
        assert basic_transformer.current_sequence_length == 128

    def test_sequence_stepping(self, basic_transformer):
        """Test sequence length progression"""
        initial_length = basic_transformer.current_sequence_length
        basic_transformer.step_sequence()
        assert basic_transformer.current_sequence_length == initial_length + 1

    def test_component_retrieval(self, basic_transformer):
        """Test component retrieval functionality"""
        # Test attention head retrieval
        head = basic_transformer.get_component("head_0")
        assert isinstance(head, AttentionHead)
        
        # Test projection retrieval
        proj = basic_transformer.get_component("projection")
        assert isinstance(proj, ProjectionLayer)
        
        # Test FFN retrieval
        ffn = basic_transformer.get_component("ffn")
        assert isinstance(ffn, FeedForwardNetwork)
        
        # Test invalid component
        with pytest.raises(ValueError):
            basic_transformer.get_component("invalid")

    def test_total_memory_requirement(self, basic_transformer):
        """Test total memory requirement calculation"""
        total_mem = basic_transformer.get_total_memory_requirement()
        
        # Calculate expected memory manually
        expected_mem = 0.0
        
        # Add attention head memory
        for head in basic_transformer.attention_heads:
            expected_mem += head.compute_memory_requirements(
                basic_transformer.current_sequence_length
            )
            
        # Add projection memory
        expected_mem += basic_transformer.projection.compute_memory_requirements(
            basic_transformer.current_sequence_length
        )
        
        # Add FFN memory
        expected_mem += basic_transformer.ffn.compute_memory_requirements(
            basic_transformer.current_sequence_length
        )
        
        assert total_mem == pytest.approx(expected_mem)

    def test_sequence_length_impact(self, basic_transformer):
        """Test impact of sequence length on resource requirements"""
        initial_mem = basic_transformer.get_total_memory_requirement()
        initial_flops = basic_transformer.get_total_flops()
        
        # Generate 10 tokens
        for _ in range(10):
            basic_transformer.step_sequence()
            
        final_mem = basic_transformer.get_total_memory_requirement()
        final_flops = basic_transformer.get_total_flops()
        
        # Memory and FLOPS should increase
        assert final_mem > initial_mem
        assert final_flops > initial_flops

    def test_different_sizes(self, basic_config, medium_config):
        """Test transformers of different sizes"""
        small_transformer = Transformer(basic_config)
        medium_transformer = Transformer(medium_config)
        
        # Medium should require more resources
        assert (medium_transformer.get_total_memory_requirement() > 
                small_transformer.get_total_memory_requirement())
        assert (medium_transformer.get_total_flops() > 
                small_transformer.get_total_flops())

    def test_all_components(self, basic_transformer):
        """Test getting all components"""
        components = basic_transformer.get_all_components()
        
        # Should have num_heads + 2 components (heads + projection + FFN)
        assert len(components) == basic_transformer.config.num_heads + 2
        
        # Check component types
        assert all(isinstance(c, (AttentionHead, ProjectionLayer, FeedForwardNetwork))
                  for c in components)