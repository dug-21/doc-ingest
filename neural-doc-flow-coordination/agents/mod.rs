/// DAA Agent Framework for Neural Document Processing
/// High-performance distributed agent coordination with neural processing capabilities

pub mod controller;
pub mod extractor;
pub mod validator;
pub mod enhancer;
pub mod formatter;
pub mod base;
pub mod consensus_coordinator;
pub mod pipeline_connector;

use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;
use serde::{Deserialize, Serialize};
use async_trait::async_trait;

/// DAA Agent Types for Neural Document Processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentType {
    Controller,
    Extractor,
    Validator,
    Enhancer,
    Formatter,
}

/// Agent State for coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentState {
    Initializing,
    Ready,
    Processing,
    Waiting,
    Completed,
    Error(String),
}

/// DAA Agent Capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCapabilities {
    pub neural_processing: bool,
    pub text_enhancement: bool,
    pub layout_analysis: bool,
    pub quality_assessment: bool,
    pub coordination: bool,
    pub fault_tolerance: bool,
}

/// Base DAA Agent Trait
#[async_trait]
pub trait DaaAgent: Send + Sync {
    fn id(&self) -> Uuid;
    fn agent_type(&self) -> AgentType;
    fn state(&self) -> AgentState;
    fn capabilities(&self) -> AgentCapabilities;
    
    async fn initialize(&mut self) -> Result<(), Box<dyn std::error::Error>>;
    async fn process(&mut self, input: Vec<u8>) -> Result<Vec<u8>, Box<dyn std::error::Error>>;
    async fn coordinate(&mut self, message: CoordinationMessage) -> Result<(), Box<dyn std::error::Error>>;
    async fn shutdown(&mut self) -> Result<(), Box<dyn std::error::Error>>;
}

/// Coordination Message for inter-agent communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationMessage {
    pub id: Uuid,
    pub from: Uuid,
    pub to: Option<Uuid>, // None for broadcast
    pub message_type: MessageType,
    pub payload: Vec<u8>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub priority: u8, // 0-255, 255 = highest
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageType {
    Task,
    Result,
    Status,
    Error,
    Coordination,
    Heartbeat,
    Shutdown,
}

/// Task result structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    pub task_id: Uuid,
    pub document_id: String,
    pub result_type: TaskType,
    pub data: Vec<u8>,
    pub confidence: f64,
    pub processing_time: f64,
    pub agent_id: Uuid,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Task execution state for tracking progress
#[derive(Debug, Clone)]
pub struct TaskExecutionState {
    pub task_id: Uuid,
    pub document_id: String,
    pub assigned_agents: Vec<Uuid>,
    pub completed_stages: Vec<TaskType>,
    pub current_stage: TaskType,
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub deadline: Option<chrono::DateTime<chrono::Utc>>,
}

/// Task result aggregator for collecting results from agents
#[derive(Debug, Clone)]
pub struct TaskResultAggregator {
    pub pending_tasks: std::collections::HashMap<Uuid, TaskExecutionState>,
    pub completed_results: std::collections::HashMap<Uuid, TaskResult>,
    pub failed_tasks: std::collections::HashMap<Uuid, String>,
}

impl Default for TaskResultAggregator {
    fn default() -> Self {
        Self {
            pending_tasks: std::collections::HashMap::new(),
            completed_results: std::collections::HashMap::new(),
            failed_tasks: std::collections::HashMap::new(),
        }
    }
}

/// Global coordination statistics
#[derive(Debug, Clone)]
pub struct CoordinationStats {
    pub tasks_completed: u64,
    pub tasks_failed: u64,
    pub average_processing_time: f64,
    pub accuracy_score: f64,
    pub throughput: f64,
    pub agent_utilization: std::collections::HashMap<Uuid, f64>,
}

/// Agent Registry for DAA coordination
pub struct AgentRegistry {
    agents: Arc<RwLock<std::collections::HashMap<Uuid, Box<dyn DaaAgent>>>>,
    message_bus: Arc<RwLock<Vec<CoordinationMessage>>>,
    topology: super::topologies::TopologyType,
    task_queue: Arc<RwLock<Vec<ProcessingTask>>>,
    result_aggregator: Arc<RwLock<TaskResultAggregator>>,
    coordination_stats: Arc<RwLock<CoordinationStats>>,
}

impl AgentRegistry {
    pub fn new(topology: super::topologies::TopologyType) -> Self {
        Self {
            agents: Arc::new(RwLock::new(std::collections::HashMap::new())),
            message_bus: Arc::new(RwLock::new(Vec::new())),
            topology,
            task_queue: Arc::new(RwLock::new(Vec::new())),
            result_aggregator: Arc::new(RwLock::new(TaskResultAggregator::default())),
            coordination_stats: Arc::new(RwLock::new(CoordinationStats {
                tasks_completed: 0,
                tasks_failed: 0,
                average_processing_time: 0.0,
                accuracy_score: 0.0,
                throughput: 0.0,
                agent_utilization: std::collections::HashMap::new(),
            })),
        }
    }
    
    pub async fn register_agent(&self, agent: Box<dyn DaaAgent>) -> Result<Uuid, Box<dyn std::error::Error>> {
        let id = agent.id();
        let mut agents = self.agents.write().await;
        agents.insert(id, agent);
        Ok(id)
    }
    
    pub async fn send_message(&self, message: CoordinationMessage) -> Result<(), Box<dyn std::error::Error>> {
        let mut bus = self.message_bus.write().await;
        bus.push(message);
        Ok(())
    }
    
    pub async fn process_messages(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut bus = self.message_bus.write().await;
        let messages = bus.drain(..).collect::<Vec<_>>();
        drop(bus);
        
        for message in messages {
            if let Some(target_id) = message.to {
                // Direct message
                let agents = self.agents.read().await;
                if agents.contains_key(&target_id) {
                    drop(agents);
                    // Get mutable access to specific agent
                    let mut agents = self.agents.write().await;
                    if let Some(agent) = agents.get_mut(&target_id) {
                        if let Err(e) = agent.coordinate(message.clone()).await {
                            eprintln!("Failed to coordinate message with agent {}: {}", target_id, e);
                        }
                    }
                }
            } else {
                // Broadcast message
                let mut agents = self.agents.write().await;
                for (_, agent) in agents.iter_mut() {
                    if let Err(e) = agent.coordinate(message.clone()).await {
                        eprintln!("Failed to broadcast message to agent: {}", e);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Distribute task across agents based on topology and capabilities
    pub async fn distribute_task(&self, task: ProcessingTask) -> Result<(), Box<dyn std::error::Error>> {
        let mut task_queue = self.task_queue.write().await;
        let mut aggregator = self.result_aggregator.write().await;
        
        // Add to pending tasks
        let execution_state = TaskExecutionState {
            task_id: task.id,
            document_id: task.document_id.clone(),
            assigned_agents: Vec::new(),
            completed_stages: Vec::new(),
            current_stage: task.task_type.clone(),
            started_at: chrono::Utc::now(),
            deadline: task.deadline,
        };
        
        aggregator.pending_tasks.insert(task.id, execution_state);
        task_queue.push(task.clone());
        
        // Find suitable agents based on task type and capabilities
        let target_agents = self.find_suitable_agents(&task.task_type).await?;
        
        if target_agents.is_empty() {
            aggregator.failed_tasks.insert(task.id, "No suitable agents available".to_string());
            return Err("No suitable agents available for task".into());
        }
        
        // Distribute task to selected agents
        for agent_id in target_agents {
            let task_message = CoordinationMessage {
                id: Uuid::new_v4(),
                from: Uuid::new_v4(), // Registry ID
                to: Some(agent_id),
                message_type: MessageType::Task,
                payload: serde_json::to_vec(&task)?,
                timestamp: chrono::Utc::now(),
                priority: task.priority,
            };
            
            self.send_message(task_message).await?;
        }
        
        Ok(())
    }
    
    /// Find suitable agents for a task type
    async fn find_suitable_agents(&self, task_type: &TaskType) -> Result<Vec<Uuid>, Box<dyn std::error::Error>> {
        let agents = self.agents.read().await;
        let mut suitable_agents = Vec::new();
        
        for (agent_id, agent) in agents.iter() {
            let agent_type = agent.agent_type();
            let capabilities = agent.capabilities();
            
            let is_suitable = match task_type {
                TaskType::DocumentExtraction => matches!(agent_type, AgentType::Extractor) && capabilities.text_enhancement,
                TaskType::TextEnhancement => matches!(agent_type, AgentType::Enhancer) && capabilities.neural_processing,
                TaskType::LayoutAnalysis => matches!(agent_type, AgentType::Enhancer) && capabilities.layout_analysis,
                TaskType::QualityAssessment => matches!(agent_type, AgentType::Validator) && capabilities.quality_assessment,
                TaskType::Formatting => matches!(agent_type, AgentType::Formatter),
                TaskType::Validation => matches!(agent_type, AgentType::Validator),
            };
            
            if is_suitable {
                suitable_agents.push(*agent_id);
            }
        }
        
        Ok(suitable_agents)
    }
    
    /// Aggregate results from multiple agents
    pub async fn aggregate_results(&self, task_id: Uuid) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let aggregator = self.result_aggregator.read().await;
        
        if let Some(result) = aggregator.completed_results.get(&task_id) {
            Ok(result.data.clone())
        } else {
            Err("Task results not available".into())
        }
    }
    
    /// Handle task result from an agent
    pub async fn handle_task_result(&self, result: TaskResult) -> Result<(), Box<dyn std::error::Error>> {
        let mut aggregator = self.result_aggregator.write().await;
        let mut stats = self.coordination_stats.write().await;
        
        // Store result
        aggregator.completed_results.insert(result.task_id, result.clone());
        
        // Update execution state
        if let Some(execution_state) = aggregator.pending_tasks.get_mut(&result.task_id) {
            execution_state.completed_stages.push(result.result_type.clone());
            
            // Check if task is complete
            if self.is_task_complete(&execution_state).await {
                // Task is complete, update stats
                stats.tasks_completed += 1;
                stats.accuracy_score = (stats.accuracy_score + result.confidence) / 2.0;
                stats.average_processing_time = (stats.average_processing_time + result.processing_time) / 2.0;
                
                // Update agent utilization
                let current_util = stats.agent_utilization.get(&result.agent_id).unwrap_or(&0.0);
                stats.agent_utilization.insert(result.agent_id, current_util + result.processing_time);
                
                // Remove from pending
                aggregator.pending_tasks.remove(&result.task_id);
            }
        }
        
        Ok(())
    }
    
    /// Check if a task is complete based on execution state
    async fn is_task_complete(&self, execution_state: &TaskExecutionState) -> bool {
        // Simple completion check - in real implementation, this would be more sophisticated
        !execution_state.completed_stages.is_empty()
    }
    
    /// Get coordination statistics
    pub async fn get_coordination_stats(&self) -> CoordinationStats {
        self.coordination_stats.read().await.clone()
    }
}

/// Auto-spawn agent based on task requirements
pub async fn auto_spawn_agent(
    agent_type: AgentType,
    capabilities: AgentCapabilities,
    registry: &AgentRegistry,
) -> Result<Uuid, Box<dyn std::error::Error + Send + Sync>> {
    let agent: Box<dyn DaaAgent> = match agent_type {
        AgentType::Controller => Box::new(controller::ControllerAgent::new(capabilities)),
        AgentType::Extractor => Box::new(extractor::ExtractorAgent::new(capabilities)),
        AgentType::Validator => Box::new(validator::ValidatorAgent::new(capabilities)),
        AgentType::Enhancer => Box::new(enhancer::EnhancerAgent::new(capabilities)),
        AgentType::Formatter => Box::new(formatter::FormatterAgent::new(capabilities)),
    };
    
    registry.register_agent(agent).await
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { 
            Box::new(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to register agent: {}", e)))
        })
}