//! Agent definitions and implementations

use neural_doc_flow_core::Result;
use async_trait::async_trait;
use uuid::Uuid;
use serde::{Serialize, Deserialize};

/// Agent type enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentType {
    Coordinator,
    Processor,
    Source,
    Output,
    Monitor,
    Custom(String),
}

/// Agent state enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentState {
    Initializing,
    Running,
    Paused,
    Stopping,
    Stopped,
    Error(String),
}

/// Core agent trait
#[async_trait]
pub trait Agent: Send + Sync {
    /// Get agent ID
    fn id(&self) -> Uuid;
    
    /// Get agent name
    fn name(&self) -> &str;
    
    /// Get agent type
    fn agent_type(&self) -> AgentType;
    
    /// Get current state
    fn state(&self) -> AgentState;
    
    /// Start the agent
    async fn start(&mut self) -> Result<()>;
    
    /// Stop the agent
    async fn stop(&mut self) -> Result<()>;
    
    /// Pause the agent
    async fn pause(&mut self) -> Result<()>;
    
    /// Resume the agent
    async fn resume(&mut self) -> Result<()>;
    
    /// Execute agent logic
    async fn execute(&mut self) -> Result<()>;
    
    /// Handle incoming message
    async fn handle_message(&mut self, message: crate::Message) -> Result<()>;
}

/// Mock agent for testing
#[derive(Debug)]
pub struct MockAgent {
    id: Uuid,
    name: String,
    agent_type: AgentType,
    state: AgentState,
}

impl MockAgent {
    pub fn new(name: &str) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.to_string(),
            agent_type: AgentType::Custom("mock".to_string()),
            state: AgentState::Initializing,
        }
    }
}

#[async_trait]
impl Agent for MockAgent {
    fn id(&self) -> Uuid {
        self.id
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn agent_type(&self) -> AgentType {
        self.agent_type.clone()
    }
    
    fn state(&self) -> AgentState {
        self.state.clone()
    }
    
    async fn start(&mut self) -> Result<()> {
        self.state = AgentState::Running;
        Ok(())
    }
    
    async fn stop(&mut self) -> Result<()> {
        self.state = AgentState::Stopped;
        Ok(())
    }
    
    async fn pause(&mut self) -> Result<()> {
        self.state = AgentState::Paused;
        Ok(())
    }
    
    async fn resume(&mut self) -> Result<()> {
        self.state = AgentState::Running;
        Ok(())
    }
    
    async fn execute(&mut self) -> Result<()> {
        // Mock execution
        Ok(())
    }
    
    async fn handle_message(&mut self, _message: crate::Message) -> Result<()> {
        // Mock message handling
        Ok(())
    }
}