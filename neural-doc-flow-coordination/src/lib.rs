//! Neural Document Flow Coordination
//! 
//! Dynamic Autonomous Agent coordination for neural document flow system.
//! This crate provides DAA coordination capabilities and agent management.

use neural_doc_flow_core::Result;
use async_trait::async_trait;
use std::collections::HashMap;
use uuid::Uuid;
use serde::{Serialize, Deserialize};

pub mod agent;
pub mod topology;
pub mod messaging;
pub mod consensus;
pub mod state;

pub use agent::{Agent, AgentType, AgentState};
pub use topology::{Topology, TopologyType};
pub use messaging::{Message, MessageBus};
pub use consensus::{ConsensusEngine, ConsensusResult};
pub use state::{CoordinationState, SharedState};

/// DAA Coordination Manager
#[derive(Debug)]
pub struct CoordinationManager {
    agents: HashMap<Uuid, Box<dyn Agent>>,
    topology: Box<dyn Topology>,
    message_bus: MessageBus,
    consensus_engine: ConsensusEngine,
    shared_state: SharedState,
}

impl CoordinationManager {
    /// Create a new coordination manager
    pub fn new(topology_type: TopologyType) -> Self {
        Self {
            agents: HashMap::new(),
            topology: topology::create_topology(topology_type),
            message_bus: MessageBus::new(),
            consensus_engine: ConsensusEngine::new(),
            shared_state: SharedState::new(),
        }
    }
    
    /// Add an agent to the coordination system
    pub async fn add_agent(&mut self, agent: Box<dyn Agent>) -> Result<Uuid> {
        let agent_id = agent.id();
        self.topology.add_node(agent_id).await?;
        self.agents.insert(agent_id, agent);
        Ok(agent_id)
    }
    
    /// Remove an agent from the coordination system
    pub async fn remove_agent(&mut self, agent_id: Uuid) -> Result<()> {
        self.topology.remove_node(agent_id).await?;
        self.agents.remove(&agent_id);
        Ok(())
    }
    
    /// Send a message to a specific agent
    pub async fn send_message(&self, from: Uuid, to: Uuid, message: Message) -> Result<()> {
        self.message_bus.send(from, to, message).await
    }
    
    /// Broadcast a message to all agents
    pub async fn broadcast_message(&self, from: Uuid, message: Message) -> Result<()> {
        for agent_id in self.agents.keys() {
            if *agent_id != from {
                self.message_bus.send(from, *agent_id, message.clone()).await?;
            }
        }
        Ok(())
    }
    
    /// Get coordination status
    pub fn status(&self) -> CoordinationStatus {
        CoordinationStatus {
            agent_count: self.agents.len(),
            topology_type: self.topology.topology_type(),
            active_connections: self.topology.connection_count(),
            message_queue_size: self.message_bus.queue_size(),
        }
    }
    
    /// Start coordination loop
    pub async fn start(&mut self) -> Result<()> {
        // Initialize all agents
        for agent in self.agents.values_mut() {
            agent.start().await?;
        }
        
        // Start message processing
        self.message_bus.start().await?;
        
        Ok(())
    }
    
    /// Stop coordination
    pub async fn stop(&mut self) -> Result<()> {
        // Stop all agents
        for agent in self.agents.values_mut() {
            agent.stop().await?;
        }
        
        // Stop message processing
        self.message_bus.stop().await?;
        
        Ok(())
    }
}

/// Coordination status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationStatus {
    pub agent_count: usize,
    pub topology_type: TopologyType,
    pub active_connections: usize,
    pub message_queue_size: usize,
}

/// Trait for coordination-aware components
#[async_trait]
pub trait Coordinated: Send + Sync {
    /// Initialize coordination
    async fn init_coordination(&mut self, manager: &CoordinationManager) -> Result<()>;
    
    /// Handle coordination messages
    async fn handle_coordination_message(&mut self, message: Message) -> Result<()>;
    
    /// Get coordination status
    fn coordination_status(&self) -> CoordinationStatus;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::MockAgent;
    
    #[tokio::test]
    async fn test_coordination_manager() {
        let mut manager = CoordinationManager::new(TopologyType::Mesh);
        
        let agent = Box::new(MockAgent::new("test-agent"));
        let agent_id = manager.add_agent(agent).await.unwrap();
        
        let status = manager.status();
        assert_eq!(status.agent_count, 1);
        assert_eq!(status.topology_type, TopologyType::Mesh);
        
        manager.remove_agent(agent_id).await.unwrap();
        let status = manager.status();
        assert_eq!(status.agent_count, 0);
    }
}