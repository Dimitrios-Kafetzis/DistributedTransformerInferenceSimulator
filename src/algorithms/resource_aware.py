from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from ..core import Device, Network, Transformer, TransformerComponent
from .utils import ResourceRequirements, CommunicationCost, validate_assignment

@dataclass
class ScoringFunction:
    """Implementation of the scoring function S(i,j,t) from the paper"""
    
    @staticmethod
    def compute(
        component: TransformerComponent,
        device: Device,
        network: Network,
        transformer: Transformer,
        current_assignments: Dict[str, str],
        cache_assignments: Dict[str, str],
        generation_step: int
    ) -> float:
        """
        Compute the scoring function S(i,j,t) as defined in Section IV
        Returns infinity for infeasible assignments
        """
        # Check if device is source node (privacy constraint)
        if device.is_source and component.component_id not in ["input", "position"]:
            return float('inf')
            
        # Compute base scoring function B(i,j,t)
        compute_ratio = _compute_ratio(component, device, transformer)
        memory_ratio = _memory_ratio(
            component, device, transformer, generation_step
        )
        comm_ratio = _communication_ratio(
            component, device, network, transformer,
            current_assignments, cache_assignments
        )
        
        # Return maximum of all ratios (as per paper)
        return max(compute_ratio, memory_ratio, comm_ratio)

def _compute_ratio(
    component: TransformerComponent,
    device: Device,
    transformer: Transformer
) -> float:
    """Calculate computation ratio (eq. 7-8 from paper)"""
    flops = component.compute_flops(transformer.current_sequence_length)
    return flops / device.compute.capacity

def _memory_ratio(
    component: TransformerComponent,
    device: Device,
    transformer: Transformer,
    generation_step: int
) -> float:
    """Calculate memory ratio including cache requirements"""
    memory_req = component.compute_memory_requirements(
        transformer.current_sequence_length
    )
    
    # Add cache requirements if applicable
    if hasattr(component, 'compute_cache_memory'):
        memory_req += component.compute_cache_memory(generation_step)
        
    return memory_req / device.memory.capacity

def _communication_ratio(
    component: TransformerComponent,
    device: Device,
    network: Network,
    transformer: Transformer,
    current_assignments: Dict[str, str],
    cache_assignments: Dict[str, str]
) -> float:
    """Calculate communication ratio based on dependencies"""
    total_comm_cost = 0.0
    
    # Get component dependencies
    deps = _get_dependencies(component.component_id, transformer)
    
    # Calculate communication costs for each dependency
    for dep_id in deps:
        if dep_id in current_assignments:
            source_device = current_assignments[dep_id]
            if source_device != device.device_id:
                data_size = _estimate_transfer_size(
                    dep_id,
                    component.component_id,
                    transformer
                )
                transfer_time = network.calculate_transfer_time(
                    source_device,
                    device.device_id,
                    data_size
                )
                total_comm_cost += transfer_time
                
    return total_comm_cost

@dataclass
class AssignmentResult:
    """Results from the distribution algorithm"""
    component_assignments: Dict[str, str]  # component_id -> device_id
    cache_assignments: Dict[str, str]      # head_id -> device_id
    estimated_latency: float
    resource_usage: Dict[str, Dict[str, float]]
    is_feasible: bool
    error: Optional[str] = None  # Add error field with default None

class ResourceAwareDistributor:
    """
    Implementation of the resource-aware distribution algorithm
    from Section IV of the paper
    """
    
    def __init__(
        self,
        transformer: Transformer,
        network: Network,
        devices: Dict[str, Device]
    ):
        self.transformer = transformer
        self.network = network
        self.devices = devices
        self.scoring = ScoringFunction()
        
    def compute_assignment(
        self,
        generation_step: int,
        previous_assignments: Optional[Dict[str, str]] = None,
        previous_cache: Optional[Dict[str, str]] = None
    ) -> AssignmentResult:
        """Compute optimal component assignments for current generation step"""
        # Initialize assignments
        assignments = {} if previous_assignments is None else previous_assignments.copy()
        cache_assignments = {} if previous_cache is None else previous_cache.copy()
        
        # Reset device states - handle cache components separately
        for device in self.devices.values():
            # First deallocate regular components
            for comp_id in list(device.assigned_components.keys()):
                if not comp_id.endswith('_cache'):
                    device.deallocate_resources(comp_id)
            # Then deallocate cache
            for comp_id in list(device.assigned_components.keys()):
                if comp_id.endswith('_cache'):
                    device.deallocate_resources(comp_id)
                    
        # Get sorted list of components by resource demand
        components = self._sort_by_resource_demand()
        
        if not components:
            return AssignmentResult(
                component_assignments={},
                cache_assignments={},
                estimated_latency=float('inf'),
                resource_usage=self._get_resource_usage(),
                is_feasible=False,
                error="No components to assign"
            )
                
        # Try to assign each component
        for component in components:
            memory_req = component.compute_memory_requirements(
                self.transformer.current_sequence_length
            )
            compute_req = component.compute_flops(
                self.transformer.current_sequence_length
            )
            
            # Calculate cache requirements separately
            cache_req = None
            if hasattr(component, 'compute_cache_memory'):
                cache_req = component.compute_cache_memory(generation_step)
                
            # Find best device assignment
            best_device = self._find_best_device(
                component,
                assignments,
                cache_assignments,
                generation_step
            )
                
            if best_device is not None:
                # First try to allocate main resources
                main_success = best_device.allocate_resources(
                    component.component_id,
                    memory_req,
                    compute_req
                )
                
                # Then try to allocate cache if needed
                cache_success = True
                if main_success and cache_req is not None:
                    cache_success = best_device.allocate_resources(
                        f"{component.component_id}_cache",
                        cache_req,
                        0.0  # Cache doesn't need compute resources
                    )
                    
                if main_success and cache_success:
                    # Record assignments
                    assignments[component.component_id] = best_device.device_id
                    if cache_req is not None:
                        cache_assignments[component.component_id] = best_device.device_id
                else:
                    # Deallocate on failure
                    if main_success:
                        best_device.deallocate_resources(component.component_id)
                    if cache_success and cache_req is not None:
                        best_device.deallocate_resources(f"{component.component_id}_cache")
                        
                    # Try to resolve resource contention
                    success = self._resolve_resource_contention(
                        component,
                        assignments,
                        cache_assignments,
                        generation_step
                    )
                    if not success:
                        return AssignmentResult(
                            component_assignments=assignments,
                            cache_assignments=cache_assignments,
                            estimated_latency=float('inf'),
                            resource_usage=self._get_resource_usage(),
                            is_feasible=False,
                            error=f"Failed to allocate resources for {component.component_id}"
                        )
            else:
                # No suitable device found
                return AssignmentResult(
                    component_assignments=assignments,
                    cache_assignments=cache_assignments,
                    estimated_latency=float('inf'),
                    resource_usage=self._get_resource_usage(),
                    is_feasible=False,
                    error=f"No suitable device found for {component.component_id}"
                )    

        # Add utilization metrics to resource usage
        resource_usage = self._get_resource_usage()
        for dev_id, usage in resource_usage.items():
            device = self.devices[dev_id]
            usage['compute_utilization'] = device.compute.used / device.compute.capacity
            usage['memory_utilization'] = device.memory.used / device.memory.capacity    

        # Validate final assignment
        is_feasible = validate_assignment(
            assignments,
            cache_assignments,
            self.transformer,
            self.devices,
            self.network,
            generation_step
        )
        
        # Calculate estimated latency
        latency = self._estimate_latency(
            assignments, 
            cache_assignments,
            generation_step
        ) if is_feasible else float('inf')
        
        return AssignmentResult(
            component_assignments=assignments,
            cache_assignments=cache_assignments,
            estimated_latency=latency,
            resource_usage=resource_usage,
            is_feasible=is_feasible
        )

    def _sort_by_resource_demand(self) -> List[TransformerComponent]:
        """Sort components by their resource requirements"""
        components = self.transformer.get_all_components()
        
        # Calculate total resource demand for each component
        demands = []
        for comp in components:
            memory = comp.compute_memory_requirements(
                self.transformer.current_sequence_length
            )
            flops = comp.compute_flops(
                self.transformer.current_sequence_length
            )
            
            # Add cache requirements if applicable
            if hasattr(comp, 'compute_cache_memory'):
                memory += comp.compute_cache_memory(0)  # Initial step
                
            # Normalize demands to [0,1] range
            max_memory = max(dev.memory.capacity for dev in self.devices.values())
            max_compute = max(dev.compute.capacity for dev in self.devices.values())
            
            memory_norm = memory / max_memory if max_memory > 0 else 0
            flops_norm = flops / max_compute if max_compute > 0 else 0
            
            demands.append(memory_norm + flops_norm)
            
        # Sort components by demand (descending)
        return [comp for _, comp in sorted(
            zip(demands, components),
            key=lambda x: x[0],
            reverse=True
        )]

    def _find_best_device(
        self,
        component: TransformerComponent,
        current_assignments: Dict[str, str],
        cache_assignments: Dict[str, str],
        generation_step: int
    ) -> Optional[Device]:
        """
        Find the best device for a component based on scoring function,
        printing debug info about memory/compute requirements and scoring.
        """
        best_score = float('inf')
        best_device = None
        
        # Calculate base resource requirements
        memory_req = component.compute_memory_requirements(
            self.transformer.current_sequence_length
        )
        compute_req = component.compute_flops(
            self.transformer.current_sequence_length
        )
        
        # Add cache requirement if applicable
        if hasattr(component, 'compute_cache_memory'):
            cache_req = component.compute_cache_memory(generation_step)
            memory_req += cache_req
        
        # DEBUG: Print the final memory/compute requirements for this component
        print(
            f"[DEBUG] Searching device for component='{component.component_id}', "
            f"memory_req={memory_req:.6f} GB, compute_req={compute_req:.6f}"
        )
        
        for device in self.devices.values():
            mem_avail = device.memory.available
            comp_avail = device.compute.available
            
            # DEBUG: Print available resources for each device
            print(
                f"  [DEBUG] Checking device='{device.device_id}' with "
                f"mem_avail={mem_avail:.6f} GB, comp_avail={comp_avail:.6f}, "
                f"is_source={device.is_source}"
            )
            
            # Skip if device can’t accommodate
            if not device.can_accommodate(memory_req, compute_req):
                print(f"    [DEBUG] => cannot accommodate (mem_req={memory_req:.6f} / {mem_avail:.6f}, "
                      f"compute_req={compute_req:.6f} / {comp_avail:.6f})")
                continue
            
            score = self.scoring.compute(
                component,
                device,
                self.network,
                self.transformer,
                current_assignments,
                cache_assignments,
                generation_step
            )
            
            # DEBUG: Show scoring function result
            print(f"    [DEBUG] => device='{device.device_id}', score={score:.6f}")
            
            # Update best device if score is better
            if score < best_score:
                best_score = score
                best_device = device
        
        # DEBUG: Show final pick (if any)
        if best_device:
            print(
                f"[DEBUG] Best device for component='{component.component_id}' => "
                f"'{best_device.device_id}' with final score={best_score:.6f}"
            )
        else:
            print(
                f"[DEBUG] No suitable device found for component='{component.component_id}' "
                f"with memory_req={memory_req:.6f}, compute_req={compute_req:.6f}"
            )
        
        return best_device

        
    def _resolve_resource_contention(
        self,
        component: TransformerComponent,
        assignments: Dict[str, str],
        cache_assignments: Dict[str, str],
        generation_step: int
    ) -> bool:
        """
        Implement resource contention resolution as described in Section IV.B
        Returns True if resolution successful, False otherwise
        """
        # Find most constrained resource
        resource_type = self._identify_constraining_resource(
            component,
            assignments,
            cache_assignments,
            generation_step
        )
        
        # Try to free up resources through reassignment
        components_to_reassign = self._select_components_for_reassignment(
            resource_type,
            assignments,
            cache_assignments,
            generation_step
        )
        
        # Attempt reassignment
        success = self._attempt_reassignment(
            components_to_reassign,
            assignments,
            cache_assignments,
            generation_step
        )
        
        return success
        
    def _identify_constraining_resource(
        self,
        component: TransformerComponent,
        assignments: Dict[str, str],
        cache_assignments: Dict[str, str],
        generation_step: int
    ) -> str:
        """Identify the most constraining resource type"""
        compute_usage = {dev_id: 0.0 for dev_id in self.devices}
        memory_usage = {dev_id: 0.0 for dev_id in self.devices}
        
        # Calculate current resource usage
        for comp_id, dev_id in assignments.items():
            comp = self.transformer.get_component(comp_id)
            compute_usage[dev_id] += comp.compute_flops(
                self.transformer.current_sequence_length
            )
            memory_usage[dev_id] += comp.compute_memory_requirements(
                self.transformer.current_sequence_length
            )
            
        # Add cache usage
        for head_id, dev_id in cache_assignments.items():
            head = self.transformer.get_component(head_id)
            if hasattr(head, 'compute_cache_memory'):
                memory_usage[dev_id] += head.compute_cache_memory(generation_step)
                
        # Calculate usage ratios
        compute_ratio = max(
            usage / self.devices[dev_id].compute.capacity
            for dev_id, usage in compute_usage.items()
        )
        memory_ratio = max(
            usage / self.devices[dev_id].memory.capacity
            for dev_id, usage in memory_usage.items()
        )
        
        return 'compute' if compute_ratio > memory_ratio else 'memory'
        
    def _get_resource_usage(self) -> Dict[str, Dict[str, float]]:
        """Calculate current resource usage for all devices"""
        usage = {}
        for dev_id, device in self.devices.items():
            usage[dev_id] = {
                'memory_used': device.memory.used,
                'memory_capacity': device.memory.capacity,
                'compute_used': device.compute.used,
                'compute_capacity': device.compute.capacity,
                'compute_utilization': (
                    device.compute.used / device.compute.capacity
                    if device.compute.capacity > 0 else 0.0
                ),
                'memory_utilization': (
                    device.memory.used / device.memory.capacity
                    if device.memory.capacity > 0 else 0.0
                ),
            }
        return usage
        
    def _estimate_latency(
        self,
        assignments: Dict[str, str],
        cache_assignments: Dict[str, str],
        generation_step: int
    ) -> float:
        """Estimate end-to-end latency for current assignments"""
        # Calculate computation time for each device
        compute_times = {dev_id: 0.0 for dev_id in self.devices}
        for comp_id, dev_id in assignments.items():
            component = self.transformer.get_component(comp_id)
            flops = component.compute_flops(self.transformer.current_sequence_length)
            compute_times[dev_id] += flops / self.devices[dev_id].compute.capacity
            
        # Calculate communication time between dependent components
        comm_time = 0.0
        for comp_id, dev_id in assignments.items():
            component = self.transformer.get_component(comp_id)
            deps = _get_dependencies(comp_id, self.transformer)
            
            for dep_id in deps:
                if dep_id in assignments and assignments[dep_id] != dev_id:
                    data_size = _estimate_transfer_size(
                        dep_id,
                        comp_id,
                        self.transformer
                    )
                    comm_time += self.network.calculate_transfer_time(
                        assignments[dep_id],
                        dev_id,
                        data_size
                    )
                    
        # Total latency is max computation time plus communication time
        max_compute_time = max(compute_times.values())
        return max_compute_time + comm_time
        
    def _check_feasibility(
        self,
        component: TransformerComponent,
        device: Device,
        current_assignments: Dict[str, str],
        cache_assignments: Dict[str, str],
        generation_step: int
    ) -> bool:
        """Check if assigning component to device is feasible"""
        # Calculate required resources
        memory_req = component.compute_memory_requirements(
            self.transformer.current_sequence_length
        )
        compute_req = component.compute_flops(
            self.transformer.current_sequence_length
        )
        
        # Add cache requirement if applicable
        if hasattr(component, 'compute_cache_memory'):
            memory_req += component.compute_cache_memory(generation_step)
            
        # Check if device has sufficient available resources
        return device.can_accommodate(memory_req, compute_req)
        
    def _select_components_for_reassignment(
        self,
        resource_type: str,
        assignments: Dict[str, str],
        cache_assignments: Dict[str, str],
        generation_step: int
    ) -> List[str]:
        """Select components to reassign to resolve resource contention"""
        components_to_reassign = []
        
        # Calculate resource usage per component
        usage_per_component = {}
        for comp_id, dev_id in assignments.items():
            component = self.transformer.get_component(comp_id)
            if resource_type == 'compute':
                usage = component.compute_flops(
                    self.transformer.current_sequence_length
                )
            else:  # memory
                usage = component.compute_memory_requirements(
                    self.transformer.current_sequence_length
                )
                if comp_id in cache_assignments:
                    usage += component.compute_cache_memory(generation_step)
                    
            usage_per_component[comp_id] = usage
            
        # Sort components by resource usage (descending)
        sorted_components = sorted(
            usage_per_component.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Select minimal set of components that would free enough resources
        total_freed = 0.0
        target_reduction = self._calculate_required_reduction(
            resource_type,
            assignments,
            cache_assignments,
            generation_step
        )
        
        for comp_id, usage in sorted_components:
            components_to_reassign.append(comp_id)
            total_freed += usage
            if total_freed >= target_reduction:
                break
                
        return components_to_reassign
        
    def _attempt_reassignment(
        self,
        components: List[str],
        assignments: Dict[str, str],
        cache_assignments: Dict[str, str],
        generation_step: int
    ) -> bool:
        """Attempt to reassign selected components"""
        # Store original assignments in case we need to rollback
        original_assignments = assignments.copy()
        original_cache = cache_assignments.copy()
        
        # Remove selected components from current assignments
        for comp_id in components:
            if comp_id in assignments:
                del assignments[comp_id]
            if comp_id in cache_assignments:
                del cache_assignments[comp_id]
                
        # Try to assign components to new devices
        success = True
        for comp_id in components:
            component = self.transformer.get_component(comp_id)
            best_device = self._find_best_device(
                component,
                assignments,
                cache_assignments,
                generation_step
            )
            
            if best_device is None:
                # Reassignment failed, restore original assignments
                assignments.update(original_assignments)
                cache_assignments.update(original_cache)
                success = False
                break
            else:
                # Assign to new device
                assignments[comp_id] = best_device.device_id
                if hasattr(component, 'compute_cache_memory'):
                    cache_assignments[comp_id] = best_device.device_id
                    
        return success
        
    def _calculate_required_reduction(
        self,
        resource_type: str,
        assignments: Dict[str, str],
        cache_assignments: Dict[str, str],
        generation_step: int
    ) -> float:
        """Calculate how much resource usage needs to be reduced"""
        current_usage = {dev_id: 0.0 for dev_id in self.devices}
        capacities = {}
        
        for dev_id, device in self.devices.items():
            if resource_type == 'compute':
                capacities[dev_id] = device.compute.capacity
            else:  # memory
                capacities[dev_id] = device.memory.capacity
                
        # Calculate current usage
        for comp_id, dev_id in assignments.items():
            component = self.transformer.get_component(comp_id)
            if resource_type == 'compute':
                usage = component.compute_flops(
                    self.transformer.current_sequence_length
                )
            else:  # memory
                usage = component.compute_memory_requirements(
                    self.transformer.current_sequence_length
                )
                if comp_id in cache_assignments:
                    usage += component.compute_cache_memory(generation_step)
                    
            current_usage[dev_id] += usage
            
        # Find maximum oversubscription
        max_oversubscription = max(
            max(0.0, usage - capacities[dev_id])
            for dev_id, usage in current_usage.items()
        )
        
        return max_oversubscription * 1.1  # Add 10% margin

def _get_dependencies(
    component_id: str,
    transformer: Transformer
) -> Set[str]:
    """Get dependencies for a component"""
    dependencies = set()
    
    if component_id == "projection":
        # Projection depends on all attention heads
        dependencies.update(
            head.component_id for head in transformer.attention_heads
        )
    elif component_id == "ffn":
        # FFN depends on projection
        dependencies.add("projection")
    elif component_id.startswith("head_"):
        # Attention heads depend on their previous cache state
        pass  # Cache dependencies are handled separately
        
    return dependencies

def _estimate_transfer_size(
    source_id: str,
    target_id: str,
    transformer: Transformer
) -> float:
    """Estimate size of data transfer between components in GB"""
    if source_id.startswith("head_") and target_id == "projection":
        # Transfer attention head output
        return (transformer.current_sequence_length * 
                transformer.config.head_dim * 
                transformer.config.precision_bytes) / (1024**3)
    elif source_id == "projection" and target_id == "ffn":
        # Transfer projection output
        return (transformer.current_sequence_length * 
                transformer.config.embedding_dim * 
                transformer.config.precision_bytes) / (1024**3)
    return 0.0