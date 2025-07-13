//! Topology management for DAA coordination

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use std::collections::HashMap;
use ruv_swarm_daa::TopologyType;

/// Configuration for coordination topology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyConfig {
    pub topology_type: CoordinationTopology,
    pub max_agents: usize,
    pub auto_scaling: bool,
    pub fault_tolerance: bool,
    pub optimization_strategy: OptimizationStrategy,
}

/// Coordination topology types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationTopology {
    /// Mesh topology - all agents can communicate with each other
    Mesh {
        enable_clustering: bool,
        cluster_size: usize,
    },
    
    /// Hierarchical topology - tree structure with parent-child relationships
    Hierarchical {
        max_depth: usize,
        branching_factor: usize,
        coordinator_redundancy: bool,
    },
    
    /// Ring topology - sequential processing in circular connections
    Ring {
        enable_bidirectional: bool,
        failure_recovery: RingRecoveryStrategy,
    },
    
    /// Star topology - central coordinator with peripheral agents
    Star {
        central_coordinator: Uuid,
        enable_backup_coordinator: bool,
        load_balancing: LoadBalancingStrategy,
    },
    
    /// Hybrid topology - combination of multiple topologies
    Hybrid {
        primary_topology: Box<CoordinationTopology>,
        secondary_topology: Box<CoordinationTopology>,
        switching_criteria: SwitchingCriteria,
    },
    
    /// Custom topology defined by user
    Custom {
        name: String,
        configuration: HashMap<String, serde_json::Value>,
    },
}

/// Strategies for topology optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    /// Optimize for minimum latency
    MinimizeLatency,
    
    /// Optimize for maximum throughput
    MaximizeThroughput,
    
    /// Optimize for fault tolerance
    MaximizeFaultTolerance,
    
    /// Balance between different factors
    Balanced {
        latency_weight: f32,
        throughput_weight: f32,
        fault_tolerance_weight: f32,
    },
    
    /// Adaptive optimization based on current conditions
    Adaptive {
        learning_rate: f32,
        adaptation_threshold: f32,
    },
}

/// Recovery strategies for ring topology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RingRecoveryStrategy {
    /// Skip failed agents and continue
    Skip,
    
    /// Wait for failed agents to recover
    Wait { timeout_seconds: u32 },
    
    /// Replace failed agents with new ones
    Replace,
    
    /// Switch to alternative topology
    SwitchTopology(Box<CoordinationTopology>),
}

/// Load balancing strategies for star topology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastConnections,
    WeightedRoundRobin { weights: HashMap<Uuid, f32> },
    ResourceBased,
    PerformanceBased,
}

/// Criteria for switching between topologies in hybrid mode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwitchingCriteria {
    pub agent_count_threshold: usize,
    pub load_threshold: f32,
    pub error_rate_threshold: f32,
    pub latency_threshold_ms: u32,
    pub switch_cooldown_seconds: u32,
}

impl Default for TopologyConfig {
    fn default() -> Self {
        Self {
            topology_type: CoordinationTopology::Hierarchical {
                max_depth: 3,
                branching_factor: 4,
                coordinator_redundancy: true,
            },
            max_agents: 16,
            auto_scaling: true,
            fault_tolerance: true,
            optimization_strategy: OptimizationStrategy::Balanced {
                latency_weight: 0.3,
                throughput_weight: 0.4,
                fault_tolerance_weight: 0.3,
            },
        }
    }
}

impl CoordinationTopology {
    /// Convert to ruv-swarm-daa TopologyType
    pub fn to_daa_topology(&self) -> TopologyType {
        match self {
            CoordinationTopology::Mesh { .. } => TopologyType::Mesh,
            CoordinationTopology::Hierarchical { .. } => TopologyType::Hierarchical,
            CoordinationTopology::Ring { .. } => TopologyType::Ring,
            CoordinationTopology::Star { .. } => TopologyType::Star,
            CoordinationTopology::Hybrid { primary_topology, .. } => {
                primary_topology.to_daa_topology()
            },
            CoordinationTopology::Custom { .. } => TopologyType::Hierarchical, // Default fallback
        }
    }
    
    /// Get recommended agent count for this topology
    pub fn recommended_agent_count(&self) -> usize {
        match self {
            CoordinationTopology::Mesh { .. } => 8, // Mesh scales well up to ~8 agents
            CoordinationTopology::Hierarchical { branching_factor, max_depth, .. } => {
                // Calculate total capacity of hierarchical tree
                let mut total = 1; // Root coordinator
                let mut level_size = 1;
                for _ in 0..*max_depth {
                    level_size *= branching_factor;
                    total += level_size;
                }
                total.min(64) // Cap at reasonable limit
            },
            CoordinationTopology::Ring { .. } => 12, // Ring works well with moderate agent counts
            CoordinationTopology::Star { .. } => 16, // Star can handle more agents with central coordinator
            CoordinationTopology::Hybrid { primary_topology, .. } => {
                primary_topology.recommended_agent_count()
            },
            CoordinationTopology::Custom { .. } => 8, // Conservative default
        }
    }
    
    /// Check if topology supports the given agent count efficiently
    pub fn supports_agent_count(&self, agent_count: usize) -> bool {
        let recommended = self.recommended_agent_count();
        agent_count <= recommended * 2 // Allow up to 2x recommended
    }
    
    /// Get fault tolerance characteristics
    pub fn fault_tolerance_level(&self) -> FaultToleranceLevel {
        match self {
            CoordinationTopology::Mesh { .. } => FaultToleranceLevel::High,
            CoordinationTopology::Hierarchical { coordinator_redundancy: true, .. } => FaultToleranceLevel::Medium,
            CoordinationTopology::Hierarchical { coordinator_redundancy: false, .. } => FaultToleranceLevel::Low,
            CoordinationTopology::Ring { failure_recovery, .. } => {
                match failure_recovery {
                    RingRecoveryStrategy::Replace => FaultToleranceLevel::Medium,
                    RingRecoveryStrategy::SwitchTopology(_) => FaultToleranceLevel::High,
                    _ => FaultToleranceLevel::Low,
                }
            },
            CoordinationTopology::Star { enable_backup_coordinator: true, .. } => FaultToleranceLevel::Medium,
            CoordinationTopology::Star { enable_backup_coordinator: false, .. } => FaultToleranceLevel::Low,
            CoordinationTopology::Hybrid { .. } => FaultToleranceLevel::High,
            CoordinationTopology::Custom { .. } => FaultToleranceLevel::Unknown,
        }
    }
    
    /// Get expected communication overhead
    pub fn communication_overhead(&self, agent_count: usize) -> CommunicationOverhead {
        match self {
            CoordinationTopology::Mesh { .. } => {
                // O(n²) communication complexity
                if agent_count <= 4 {
                    CommunicationOverhead::Low
                } else if agent_count <= 8 {
                    CommunicationOverhead::Medium
                } else {
                    CommunicationOverhead::High
                }
            },
            CoordinationTopology::Hierarchical { .. } => {
                // O(log n) communication complexity
                CommunicationOverhead::Low
            },
            CoordinationTopology::Ring { .. } => {
                // O(n) communication complexity
                if agent_count <= 8 {
                    CommunicationOverhead::Low
                } else {
                    CommunicationOverhead::Medium
                }
            },
            CoordinationTopology::Star { .. } => {
                // O(1) for agents, O(n) for coordinator
                if agent_count <= 16 {
                    CommunicationOverhead::Low
                } else {
                    CommunicationOverhead::Medium
                }
            },
            CoordinationTopology::Hybrid { .. } => CommunicationOverhead::Medium,
            CoordinationTopology::Custom { .. } => CommunicationOverhead::Unknown,
        }
    }
}

/// Fault tolerance levels
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FaultToleranceLevel {
    Low,
    Medium,
    High,
    Unknown,
}

/// Communication overhead levels
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommunicationOverhead {
    Low,
    Medium,
    High,
    Unknown,
}

impl OptimizationStrategy {
    /// Select optimal topology based on strategy and constraints
    pub fn select_topology(&self, constraints: &TopologyConstraints) -> CoordinationTopology {
        match self {
            OptimizationStrategy::MinimizeLatency => {
                if constraints.agent_count <= 8 {
                    CoordinationTopology::Mesh { 
                        enable_clustering: false,
                        cluster_size: constraints.agent_count,
                    }
                } else {
                    CoordinationTopology::Star { 
                        central_coordinator: Uuid::new_v4(),
                        enable_backup_coordinator: true,
                        load_balancing: LoadBalancingStrategy::LeastConnections,
                    }
                }
            },
            OptimizationStrategy::MaximizeThroughput => {
                CoordinationTopology::Hierarchical {
                    max_depth: 2,
                    branching_factor: (constraints.agent_count as f32).sqrt().ceil() as usize,
                    coordinator_redundancy: false,
                }
            },
            OptimizationStrategy::MaximizeFaultTolerance => {
                if constraints.agent_count <= 6 {
                    CoordinationTopology::Mesh { 
                        enable_clustering: true,
                        cluster_size: 3,
                    }
                } else {
                    CoordinationTopology::Hybrid {
                        primary_topology: Box::new(CoordinationTopology::Hierarchical {
                            max_depth: 2,
                            branching_factor: 3,
                            coordinator_redundancy: true,
                        }),
                        secondary_topology: Box::new(CoordinationTopology::Mesh {
                            enable_clustering: true,
                            cluster_size: 4,
                        }),
                        switching_criteria: SwitchingCriteria::default(),
                    }
                }
            },
            OptimizationStrategy::Balanced { .. } => {
                CoordinationTopology::Hierarchical {
                    max_depth: 3,
                    branching_factor: 4,
                    coordinator_redundancy: true,
                }
            },
            OptimizationStrategy::Adaptive { .. } => {
                // Start with hierarchical and adapt based on performance
                CoordinationTopology::Hierarchical {
                    max_depth: 2,
                    branching_factor: (constraints.agent_count as f32 / 2.0).ceil() as usize,
                    coordinator_redundancy: true,
                }
            },
        }
    }
}

/// Constraints for topology selection
#[derive(Debug, Clone)]
pub struct TopologyConstraints {
    pub agent_count: usize,
    pub max_latency_ms: u32,
    pub min_throughput: f32,
    pub fault_tolerance_required: bool,
    pub resource_constraints: ResourceConstraints,
}

/// Resource constraints for topology
#[derive(Debug, Clone)]
pub struct ResourceConstraints {
    pub max_memory_mb: u32,
    pub max_cpu_percent: f32,
    pub max_network_bandwidth_mbps: f32,
}

impl Default for SwitchingCriteria {
    fn default() -> Self {
        Self {
            agent_count_threshold: 10,
            load_threshold: 0.8,
            error_rate_threshold: 0.05,
            latency_threshold_ms: 1000,
            switch_cooldown_seconds: 300,
        }
    }
}

impl Default for TopologyConstraints {
    fn default() -> Self {
        Self {
            agent_count: 8,
            max_latency_ms: 1000,
            min_throughput: 10.0,
            fault_tolerance_required: true,
            resource_constraints: ResourceConstraints {
                max_memory_mb: 1024,
                max_cpu_percent: 80.0,
                max_network_bandwidth_mbps: 100.0,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_topology_agent_count_support() {
        let mesh = CoordinationTopology::Mesh { 
            enable_clustering: false, 
            cluster_size: 8 
        };
        assert!(mesh.supports_agent_count(8));
        assert!(mesh.supports_agent_count(16)); // 2x recommended
        assert!(!mesh.supports_agent_count(20)); // > 2x recommended
        
        let hierarchical = CoordinationTopology::Hierarchical {
            max_depth: 2,
            branching_factor: 3,
            coordinator_redundancy: true,
        };
        assert!(hierarchical.supports_agent_count(10)); // 1 + 3 + 9 = 13 capacity
    }
    
    #[test]
    fn test_fault_tolerance_levels() {
        let mesh = CoordinationTopology::Mesh { 
            enable_clustering: true, 
            cluster_size: 4 
        };
        assert_eq!(mesh.fault_tolerance_level(), FaultToleranceLevel::High);
        
        let star_no_backup = CoordinationTopology::Star {
            central_coordinator: Uuid::new_v4(),
            enable_backup_coordinator: false,
            load_balancing: LoadBalancingStrategy::RoundRobin,
        };
        assert_eq!(star_no_backup.fault_tolerance_level(), FaultToleranceLevel::Low);
    }
    
    #[test]
    fn test_optimization_strategy_selection() {
        let constraints = TopologyConstraints {
            agent_count: 6,
            ..Default::default()
        };
        
        let strategy = OptimizationStrategy::MinimizeLatency;
        let topology = strategy.select_topology(&constraints);
        
        match topology {
            CoordinationTopology::Mesh { .. } => {}, // Expected for small agent count
            _ => panic!("Expected mesh topology for latency optimization with small agent count"),
        }
    }
}