/// Topology Builder for DAA Coordination
/// Builds and manages different network topologies for agent coordination

use super::*;
use async_trait::async_trait;
use crate::Topology;
use std::collections::{HashMap, HashSet};
use uuid::Uuid;
use serde::{Deserialize, Serialize};

/// Topology builder for creating different network configurations
pub struct TopologyBuilder {
    nodes: HashSet<Uuid>,
    connections: HashMap<Uuid, HashSet<Uuid>>,
    node_types: HashMap<Uuid, NodeType>,
    node_metadata: HashMap<Uuid, NodeMetadata>,
}

/// Node type in the topology
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum NodeType {
    Controller,
    Extractor,
    Validator,
    Enhancer,
    Formatter,
    Hub,
    Leaf,
}

/// Node metadata for topology management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMetadata {
    pub node_id: Uuid,
    pub node_type: NodeType,
    pub capabilities: Vec<String>,
    pub load_capacity: f64,
    pub current_load: f64,
    pub reliability_score: f64,
    pub response_time: f64,
}

/// Topology configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyConfig {
    pub topology_type: TopologyType,
    pub max_connections_per_node: usize,
    pub redundancy_factor: f64,
    pub load_balancing_enabled: bool,
    pub fault_tolerance_enabled: bool,
}

impl TopologyBuilder {
    /// Create new topology builder
    pub fn new() -> Self {
        Self {
            nodes: HashSet::new(),
            connections: HashMap::new(),
            node_types: HashMap::new(),
            node_metadata: HashMap::new(),
        }
    }
    
    /// Add node to topology
    pub fn add_node(&mut self, node_id: Uuid, node_type: NodeType) -> &mut Self {
        self.nodes.insert(node_id);
        self.node_types.insert(node_id, node_type.clone());
        self.connections.insert(node_id, HashSet::new());
        
        // Create default metadata
        let metadata = NodeMetadata {
            node_id,
            node_type,
            capabilities: vec![],
            load_capacity: 1.0,
            current_load: 0.0,
            reliability_score: 1.0,
            response_time: 0.0,
        };
        self.node_metadata.insert(node_id, metadata);
        
        self
    }
    
    /// Add connection between nodes
    pub fn add_connection(&mut self, from: Uuid, to: Uuid) -> &mut Self {
        if self.nodes.contains(&from) && self.nodes.contains(&to) {
            self.connections.entry(from).or_insert_with(HashSet::new).insert(to);
            self.connections.entry(to).or_insert_with(HashSet::new).insert(from);
        }
        self
    }
    
    /// Build star topology
    pub fn build_star_topology(&mut self, center_node: Uuid) -> Result<(), Box<dyn std::error::Error>> {
        if !self.nodes.contains(&center_node) {
            return Err("Center node not found in topology".into());
        }
        
        // Connect center node to all other nodes
        let nodes_to_connect: Vec<Uuid> = self.nodes.iter().copied().collect();
        for node_id in nodes_to_connect {
            if node_id != center_node {
                self.add_connection(center_node, node_id);
            }
        }
        
        Ok(())
    }
    
    /// Build mesh topology
    pub fn build_mesh_topology(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let nodes: Vec<Uuid> = self.nodes.iter().cloned().collect();
        
        // Connect every node to every other node
        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                self.add_connection(nodes[i], nodes[j]);
            }
        }
        
        Ok(())
    }
    
    /// Build ring topology
    pub fn build_ring_topology(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let nodes: Vec<Uuid> = self.nodes.iter().cloned().collect();
        
        if nodes.len() < 3 {
            return Err("Ring topology requires at least 3 nodes".into());
        }
        
        // Connect each node to the next in the ring
        for i in 0..nodes.len() {
            let next_index = (i + 1) % nodes.len();
            self.add_connection(nodes[i], nodes[next_index]);
        }
        
        Ok(())
    }
    
    /// Build pipeline topology
    pub fn build_pipeline_topology(&mut self, stage_order: Vec<NodeType>) -> Result<(), Box<dyn std::error::Error>> {
        let mut stage_nodes: HashMap<NodeType, Vec<Uuid>> = HashMap::new();
        
        // Group nodes by type
        for (&node_id, node_type) in &self.node_types {
            stage_nodes.entry(node_type.clone()).or_insert_with(Vec::new).push(node_id);
        }
        
        // Connect stages in order
        for i in 0..(stage_order.len() - 1) {
            let current_stage = &stage_order[i];
            let next_stage = &stage_order[i + 1];
            
            if let (Some(current_nodes), Some(next_nodes)) = (stage_nodes.get(current_stage), stage_nodes.get(next_stage)) {
                // Connect all nodes in current stage to all nodes in next stage
                for &current_node in current_nodes {
                    for &next_node in next_nodes {
                        self.add_connection(current_node, next_node);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Build hybrid topology
    pub fn build_hybrid_topology(&mut self, config: &TopologyConfig) -> Result<(), Box<dyn std::error::Error>> {
        // Create a hybrid topology that combines different patterns
        
        // First, create a star topology for controllers
        let controllers: Vec<Uuid> = self.node_types.iter()
            .filter(|(_, node_type)| matches!(node_type, NodeType::Controller))
            .map(|(&id, _)| id)
            .collect();
        
        if let Some(&primary_controller) = controllers.first() {
            // Connect primary controller to all other controllers
            for &controller in &controllers[1..] {
                self.add_connection(primary_controller, controller);
            }
            
            // Connect controllers to their respective agent types
            for &controller in &controllers {
                self.connect_controller_to_agents(controller)?;
            }
        }
        
        // Add redundant connections for fault tolerance
        if config.fault_tolerance_enabled {
            self.add_redundant_connections(config.redundancy_factor)?;
        }
        
        Ok(())
    }
    
    /// Connect controller to appropriate agent types
    fn connect_controller_to_agents(&mut self, controller: Uuid) -> Result<(), Box<dyn std::error::Error>> {
        let agent_types = vec![
            NodeType::Extractor,
            NodeType::Validator,
            NodeType::Enhancer,
            NodeType::Formatter,
        ];
        
        for node_type in agent_types {
            let agents: Vec<Uuid> = self.node_types.iter()
                .filter(|(_, nt)| std::mem::discriminant(*nt) == std::mem::discriminant(&node_type))
                .map(|(&id, _)| id)
                .collect();
            
            // Connect controller to all agents of this type
            for &agent in &agents {
                self.add_connection(controller, agent);
            }
        }
        
        Ok(())
    }
    
    /// Add redundant connections for fault tolerance
    fn add_redundant_connections(&mut self, redundancy_factor: f64) -> Result<(), Box<dyn std::error::Error>> {
        let nodes: Vec<Uuid> = self.nodes.iter().cloned().collect();
        let target_connections = (nodes.len() as f64 * redundancy_factor) as usize;
        
        // Add random connections to increase redundancy
        for _ in 0..target_connections {
            if let (Some(&node1), Some(&node2)) = (nodes.get(0), nodes.get(1)) {
                if node1 != node2 {
                    self.add_connection(node1, node2);
                }
            }
        }
        
        Ok(())
    }
    
    /// Build topology based on configuration
    pub fn build_topology(&mut self, config: &TopologyConfig) -> Result<Box<dyn Topology>, Box<dyn std::error::Error>> {
        match config.topology_type {
            TopologyType::Star => {
                if let Some(&center) = self.nodes.iter().next() {
                    self.build_star_topology(center)?;
                }
            }
            TopologyType::Mesh => {
                self.build_mesh_topology()?;
            }
            TopologyType::Pipeline => {
                let stage_order = vec![
                    NodeType::Controller,
                    NodeType::Extractor,
                    NodeType::Enhancer,
                    NodeType::Validator,
                    NodeType::Formatter,
                ];
                self.build_pipeline_topology(stage_order)?;
            }
            TopologyType::Hybrid => {
                self.build_hybrid_topology(config)?;
            }
        }
        
        // Create topology instance
        let topology = BuiltTopology {
            topology_type: config.topology_type.clone(),
            nodes: self.nodes.clone(),
            connections: self.connections.clone(),
            node_metadata: self.node_metadata.clone(),
        };
        
        Ok(Box::new(topology))
    }
    
    /// Build tree topology
    fn build_tree_topology(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let nodes: Vec<Uuid> = self.nodes.iter().cloned().collect();
        
        if nodes.is_empty() {
            return Ok(());
        }
        
        // Create a simple tree structure
        let root = nodes[0];
        let mut parent_nodes = vec![root];
        
        // Connect remaining nodes as children
        for &node in &nodes[1..] {
            if let Some(&parent) = parent_nodes.first() {
                self.add_connection(parent, node);
                parent_nodes.push(node);
                
                // Limit children per parent
                if parent_nodes.len() > 3 {
                    parent_nodes.remove(0);
                }
            }
        }
        
        Ok(())
    }
    
    /// Get topology statistics
    pub fn get_topology_stats(&self) -> TopologyStats {
        let total_connections: usize = self.connections.values().map(|conns| conns.len()).sum();
        let avg_connections = if self.nodes.is_empty() {
            0.0
        } else {
            total_connections as f64 / self.nodes.len() as f64
        };
        
        TopologyStats {
            total_nodes: self.nodes.len(),
            total_connections: total_connections / 2, // Undirected graph
            average_connections_per_node: avg_connections,
            max_connections: self.connections.values().map(|conns| conns.len()).max().unwrap_or(0),
            min_connections: self.connections.values().map(|conns| conns.len()).min().unwrap_or(0),
        }
    }
}

/// Built topology implementation
pub struct BuiltTopology {
    topology_type: TopologyType,
    nodes: HashSet<Uuid>,
    connections: HashMap<Uuid, HashSet<Uuid>>,
    node_metadata: HashMap<Uuid, NodeMetadata>,
}

#[async_trait]
impl Topology for BuiltTopology {
    fn topology_type(&self) -> crate::topology_traits::TopologyType {
        // Convert from our TopologyType to the trait's TopologyType
        match self.topology_type {
            super::TopologyType::Star => crate::topology_traits::TopologyType::Star,
            super::TopologyType::Mesh => crate::topology_traits::TopologyType::Mesh,
            super::TopologyType::Pipeline => crate::topology_traits::TopologyType::Custom("Pipeline".to_string()),
            super::TopologyType::Hybrid => crate::topology_traits::TopologyType::Custom("Hybrid".to_string()),
        }
    }
    
    async fn add_node(&mut self, node_id: Uuid) -> anyhow::Result<()> {
        self.nodes.insert(node_id);
        self.connections.insert(node_id, HashSet::new());
        Ok(())
    }
    
    async fn remove_node(&mut self, node_id: Uuid) -> anyhow::Result<()> {
        self.nodes.remove(&node_id);
        self.connections.remove(&node_id);
        
        // Remove connections to this node from other nodes
        for connections in self.connections.values_mut() {
            connections.remove(&node_id);
        }
        
        Ok(())
    }
    
    fn get_connections(&self, node_id: Uuid) -> Vec<Uuid> {
        self.connections.get(&node_id).map(|conns| conns.iter().cloned().collect()).unwrap_or_default()
    }
    
    fn connection_count(&self) -> usize {
        self.connections.values().map(|conns| conns.len()).sum::<usize>() / 2
    }
    
    fn are_connected(&self, node1: Uuid, node2: Uuid) -> bool {
        self.connections.get(&node1).map(|conns| conns.contains(&node2)).unwrap_or(false)
    }
}

/// Topology statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyStats {
    pub total_nodes: usize,
    pub total_connections: usize,
    pub average_connections_per_node: f64,
    pub max_connections: usize,
    pub min_connections: usize,
}

impl Default for TopologyBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for TopologyConfig {
    fn default() -> Self {
        Self {
            topology_type: TopologyType::Mesh,
            max_connections_per_node: 10,
            redundancy_factor: 0.2,
            load_balancing_enabled: true,
            fault_tolerance_enabled: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_topology_builder_star() {
        let mut builder = TopologyBuilder::new();
        
        let center = Uuid::new_v4();
        let node1 = Uuid::new_v4();
        let node2 = Uuid::new_v4();
        
        builder.add_node(center, NodeType::Controller);
        builder.add_node(node1, NodeType::Extractor);
        builder.add_node(node2, NodeType::Validator);
        
        builder.build_star_topology(center).unwrap();
        
        let stats = builder.get_topology_stats();
        assert_eq!(stats.total_nodes, 3);
        assert_eq!(stats.total_connections, 2);
    }
    
    #[test]
    fn test_topology_builder_mesh() {
        let mut builder = TopologyBuilder::new();
        
        let node1 = Uuid::new_v4();
        let node2 = Uuid::new_v4();
        let node3 = Uuid::new_v4();
        
        builder.add_node(node1, NodeType::Controller);
        builder.add_node(node2, NodeType::Extractor);
        builder.add_node(node3, NodeType::Validator);
        
        builder.build_mesh_topology().unwrap();
        
        let stats = builder.get_topology_stats();
        assert_eq!(stats.total_nodes, 3);
        assert_eq!(stats.total_connections, 3);
    }
}