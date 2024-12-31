"""
Experiment scenarios for evaluating transformer inference distribution.
Provides structured test scenarios for different network configurations
and baseline comparisons.
"""

from .common import (
    BaseScenario,
    ScenarioResult,
    run_scenario,
    load_scenario_config
)

from .edge_cluster_scenarios import (
    EdgeClusterBasicScenario,
    EdgeClusterScalabilityScenario,
    EdgeClusterFailureScenario
)

from .distributed_edge_scenarios import (
    DistributedEdgeBasicScenario,
    DistributedEdgeCommunicationScenario,
    DistributedEdgeHeterogeneityScenario
)

from .hybrid_cloud_scenarios import (
    HybridCloudBasicScenario,
    HybridCloudTierBalancingScenario,
    HybridCloudLatencyScenario
)

from .baseline_scenarios import (
    GreedyBaselineScenario,
    RoundRobinBaselineScenario,
    StaticBaselineScenario,
    DynamicMigrationBaselineScenario
)

# Version information
__version__ = '0.1.0'

# Define public interface
__all__ = [
    # Base components
    'BaseScenario',
    'ScenarioResult',
    'run_scenario',
    'load_scenario_config',
    
    # Edge cluster scenarios
    'EdgeClusterBasicScenario',
    'EdgeClusterScalabilityScenario',
    'EdgeClusterFailureScenario',
    
    # Distributed edge scenarios
    'DistributedEdgeBasicScenario',
    'DistributedEdgeCommunicationScenario',
    'DistributedEdgeHeterogeneityScenario',
    
    # Hybrid cloud scenarios
    'HybridCloudBasicScenario',
    'HybridCloudTierBalancingScenario',
    'HybridCloudLatencyScenario',
    
    # Baseline scenarios
    'GreedyBaselineScenario',
    'RoundRobinBaselineScenario',
    'StaticBaselineScenario',
    'DynamicMigrationBaselineScenario'
]