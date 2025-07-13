use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};
use uuid::Uuid;
use anyhow::Result;

use crate::{
    messaging::{Message, MessagePriority},
    resources::ResourceRequirement,
    AgentCapability,
};

/// Base agent trait for all DAA agents
#[async_trait]
pub trait Agent: Send + Sync {
    /// Get the unique agent ID
    fn id(&self) -> Uuid;
    
    /// Get the agent type identifier
    fn agent_type(&self) -> &str;
    
    /// Get agent capabilities
    fn capabilities(&self) -> Vec<AgentCapability>;
    
    /// Initialize the agent
    async fn initialize(&mut self) -> Result<()>;
    
    /// Process an incoming message
    async fn process_message(&self, message: Message) -> Result<()>;
    
    /// Get current agent status
    async fn status(&self) -> AgentStatus;
    
    /// Shutdown the agent gracefully
    async fn shutdown(&mut self) -> Result<()>;
    
    /// Get resource requirements
    fn resource_requirements(&self) -> Vec<ResourceRequirement>;
    
    /// Handle coordination message
    async fn handle_coordination(&self, message: CoordinationMessage) -> Result<()>;
}

/// Agent status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStatus {
    pub id: Uuid,
    pub agent_type: String,
    pub state: AgentState,
    pub capabilities: Vec<AgentCapability>,
    pub current_tasks: usize,
    pub completed_tasks: usize,
    pub error_count: usize,
    pub last_heartbeat: chrono::DateTime<chrono::Utc>,
}

/// Agent state enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentState {
    Initializing,
    Ready,
    Processing,
    Paused,
    Failed,
    ShuttingDown,
}

/// Coordination message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationMessage {
    TaskAssignment {
        task_id: Uuid,
        priority: MessagePriority,
        deadline: Option<chrono::DateTime<chrono::Utc>>,
    },
    StatusRequest,
    ResourceAllocation {
        resources: Vec<ResourceRequirement>,
    },
    Heartbeat,
    Shutdown,
}

/// Base agent implementation with common functionality
pub struct BaseAgent {
    pub id: Uuid,
    pub agent_type: String,
    pub state: Arc<RwLock<AgentState>>,
    pub capabilities: Vec<AgentCapability>,
    pub message_sender: broadcast::Sender<Message>,
    pub task_counter: Arc<RwLock<TaskCounter>>,
}

#[derive(Default)]
pub struct TaskCounter {
    pub current: usize,
    pub completed: usize,
    pub errors: usize,
}

impl BaseAgent {
    pub fn new(
        agent_type: String,
        capabilities: Vec<AgentCapability>,
        message_sender: broadcast::Sender<Message>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            agent_type,
            state: Arc::new(RwLock::new(AgentState::Initializing)),
            capabilities,
            message_sender,
            task_counter: Arc::new(RwLock::new(TaskCounter::default())),
        }
    }
    
    pub async fn set_state(&self, state: AgentState) {
        *self.state.write().await = state;
    }
    
    pub async fn increment_task(&self) {
        let mut counter = self.task_counter.write().await;
        counter.current += 1;
    }
    
    pub async fn complete_task(&self) {
        let mut counter = self.task_counter.write().await;
        if counter.current > 0 {
            counter.current -= 1;
            counter.completed += 1;
        }
    }
    
    pub async fn record_error(&self) {
        let mut counter = self.task_counter.write().await;
        counter.errors += 1;
    }
}

/// Macro to implement common agent methods
#[macro_export]
macro_rules! impl_agent_base {
    ($agent_type:ty) => {
        impl $agent_type {
            pub fn id(&self) -> Uuid {
                self.base.id
            }
            
            pub fn agent_type(&self) -> &str {
                &self.base.agent_type
            }
            
            pub fn capabilities(&self) -> Vec<AgentCapability> {
                self.base.capabilities.clone()
            }
            
            pub async fn status(&self) -> AgentStatus {
                let state = *self.base.state.read().await;
                let counter = self.base.task_counter.read().await;
                
                AgentStatus {
                    id: self.base.id,
                    agent_type: self.base.agent_type.clone(),
                    state,
                    capabilities: self.base.capabilities.clone(),
                    current_tasks: counter.current,
                    completed_tasks: counter.completed,
                    error_count: counter.errors,
                    last_heartbeat: chrono::Utc::now(),
                }
            }
        }
    };
}