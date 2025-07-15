//! Network topology implementations

use anyhow::Result;
use async_trait::async_trait;
use uuid::Uuid;
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet};

/// Topology type enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopologyType {
    Mesh,
    Star,
    Ring,
    Tree,
    Custom(String),
}

/// Network topology trait
#[async_trait]
pub trait Topology: Send + Sync {
    /// Get topology type
    fn topology_type(&self) -> TopologyType;
    
    /// Add a node to the topology
    async fn add_node(&mut self, node_id: Uuid) -> Result<()>;
    
    /// Remove a node from the topology
    async fn remove_node(&mut self, node_id: Uuid) -> Result<()>;
    
    /// Get connections for a node
    fn get_connections(&self, node_id: Uuid) -> Vec<Uuid>;
    
    /// Get total connection count
    fn connection_count(&self) -> usize;
    
    /// Check if two nodes are connected
    fn are_connected(&self, node1: Uuid, node2: Uuid) -> bool;
}

/// Mesh topology implementation
#[derive(Debug)]
pub struct MeshTopology {
    nodes: HashSet<Uuid>,
}

impl MeshTopology {
    pub fn new() -> Self {
        Self {
            nodes: HashSet::new(),
        }
    }
}

#[async_trait]
impl Topology for MeshTopology {
    fn topology_type(&self) -> TopologyType {
        TopologyType::Mesh
    }
    
    async fn add_node(&mut self, node_id: Uuid) -> Result<()> {
        self.nodes.insert(node_id);
        Ok(())
    }
    
    async fn remove_node(&mut self, node_id: Uuid) -> Result<()> {
        self.nodes.remove(&node_id);
        Ok(())
    }
    
    fn get_connections(&self, node_id: Uuid) -> Vec<Uuid> {
        if self.nodes.contains(&node_id) {
            self.nodes.iter().filter(|&&id| id != node_id).copied().collect()
        } else {
            Vec::new()
        }
    }
    
    fn connection_count(&self) -> usize {
        let n = self.nodes.len();
        if n > 1 { n * (n - 1) } else { 0 }
    }
    
    fn are_connected(&self, node1: Uuid, node2: Uuid) -> bool {
        self.nodes.contains(&node1) && self.nodes.contains(&node2) && node1 != node2
    }
}

/// Star topology implementation
#[derive(Debug)]
pub struct StarTopology {
    center: Option<Uuid>,
    nodes: HashSet<Uuid>,
}

impl StarTopology {
    pub fn new() -> Self {
        Self {
            center: None,
            nodes: HashSet::new(),
        }
    }
}

#[async_trait]
impl Topology for StarTopology {
    fn topology_type(&self) -> TopologyType {
        TopologyType::Star
    }
    
    async fn add_node(&mut self, node_id: Uuid) -> Result<()> {
        if self.center.is_none() {
            self.center = Some(node_id);
        }
        self.nodes.insert(node_id);
        Ok(())
    }
    
    async fn remove_node(&mut self, node_id: Uuid) -> Result<()> {
        self.nodes.remove(&node_id);
        if self.center == Some(node_id) {
            self.center = self.nodes.iter().next().copied();
        }
        Ok(())
    }
    
    fn get_connections(&self, node_id: Uuid) -> Vec<Uuid> {
        if let Some(center) = self.center {
            if node_id == center {
                self.nodes.iter().filter(|&&id| id != center).copied().collect()
            } else if self.nodes.contains(&node_id) {
                vec![center]
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        }
    }
    
    fn connection_count(&self) -> usize {
        if self.nodes.len() > 1 { (self.nodes.len() - 1) * 2 } else { 0 }
    }
    
    fn are_connected(&self, node1: Uuid, node2: Uuid) -> bool {
        if let Some(center) = self.center {
            (node1 == center || node2 == center) && 
            self.nodes.contains(&node1) && 
            self.nodes.contains(&node2) && 
            node1 != node2
        } else {
            false
        }
    }
}

/// Create topology based on type
pub fn create_topology(topology_type: TopologyType) -> Box<dyn Topology> {
    match topology_type {
        TopologyType::Mesh => Box::new(MeshTopology::new()),
        TopologyType::Star => Box::new(StarTopology::new()),
        _ => Box::new(MeshTopology::new()), // Default to mesh
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_mesh_topology() {
        let mut topology = MeshTopology::new();
        
        let node1 = Uuid::new_v4();
        let node2 = Uuid::new_v4();
        
        topology.add_node(node1).await.unwrap();
        topology.add_node(node2).await.unwrap();
        
        assert!(topology.are_connected(node1, node2));
        assert!(topology.are_connected(node2, node1));
        assert_eq!(topology.connection_count(), 2);
        
        let connections1 = topology.get_connections(node1);
        assert_eq!(connections1.len(), 1);
        assert!(connections1.contains(&node2));
    }
    
    #[tokio::test]
    async fn test_star_topology() {
        let mut topology = StarTopology::new();
        
        let center = Uuid::new_v4();
        let node1 = Uuid::new_v4();
        let node2 = Uuid::new_v4();
        
        topology.add_node(center).await.unwrap();
        topology.add_node(node1).await.unwrap();
        topology.add_node(node2).await.unwrap();
        
        assert!(topology.are_connected(center, node1));
        assert!(topology.are_connected(center, node2));
        assert!(!topology.are_connected(node1, node2));
        
        let center_connections = topology.get_connections(center);
        assert_eq!(center_connections.len(), 2);
        
        let node1_connections = topology.get_connections(node1);
        assert_eq!(node1_connections.len(), 1);
        assert!(node1_connections.contains(&center));
    }
}