//! # DAA Coordination Module
//! 
//! This module provides distributed autonomous agent coordination for neural document processing.
//! It replaces claude-flow with pure Rust DAA coordination as specified in iteration5.
//! 
//! ## Key Features
//! 
//! - **DAA Agent Coordination**: Distributed processing with multiple topologies
//! - **Neural Integration**: Seamless ruv-FANN neural network coordination
//! - **Fault Tolerance**: Self-healing and consensus mechanisms
//! - **Async Runtime**: Full tokio integration for high-performance processing
//! - **Memory Persistence**: Cross-session coordination state management
//! 
//! ## Coordination Patterns
//! 
//! ### Multi-Document Batch Processing
//! ```rust
//! use neural_doc_flow_core::coordination::{DocumentCoordinator, BatchConfig};
//! 
//! let coordinator = DocumentCoordinator::new().await?;
//! let batch_config = BatchConfig::new()
//!     .with_parallel_extraction()
//!     .with_neural_enhancement()
//!     .with_result_aggregation();
//!     
//! let results = coordinator.process_batch(documents, batch_config).await?;
//! ```
//! 
//! ### Parallel Source Extraction
//! ```rust
//! // Spawn specialized extraction agents
//! coordinator.spawn_extraction_agents(&[
//!     ExtractionAgent::pdf_specialist(),
//!     ExtractionAgent::table_specialist(), 
//!     ExtractionAgent::text_specialist(),
//! ]).await?;
//! ```

pub mod agent;
pub mod task;
pub mod topology;
pub mod consensus;
pub mod memory;

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use tokio::sync::mpsc;
use uuid::Uuid;
use serde::{Deserialize, Serialize};
use async_trait::async_trait;
use tracing::{info, debug, warn, error};

// DAA imports
use daa::{Agent, Coordinator as DaaCoordinator, Topology, Message};
use ruv_swarm_daa::{
    StandardDAAAgent, 
    CognitivePattern, 
    NeuralCoordinator,
    SwarmConfig,
    AgentConfig,
    TopologyType
};

// Re-exports
pub use agent::{DocumentAgent, ExtractionAgent, ValidationAgent};
pub use task::{DocumentTask, TaskStatus, TaskResult};
pub use topology::{CoordinationTopology, TopologyConfig};
pub use consensus::{ConsensusResult, ValidationConsensus};
pub use memory::{CoordinationMemory, MemoryKey, MemoryValue};

use crate::types::{Document, ProcessingResult, ProcessingError, ErrorType};
use crate::error::{CoreError, Result};

/// Configuration for DAA coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationConfig {
    /// Topology type for agent coordination
    pub topology: TopologyType,
    /// Maximum number of agents to spawn
    pub max_agents: usize,
    /// Neural coordination settings
    pub neural_config: Option<NeuralCoordinationConfig>,
    /// Memory persistence settings
    pub memory_config: MemoryConfig,
    /// Fault tolerance configuration
    pub fault_tolerance: FaultToleranceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralCoordinationConfig {
    /// Enable neural enhancement coordination
    pub enable_neural_coordination: bool,
    /// Learning rate for neural models
    pub learning_rate: f32,
    /// Shared model state between agents
    pub shared_model_state: bool,
    /// SIMD acceleration for neural processing
    pub enable_simd: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Enable persistent memory across sessions
    pub persistent: bool,
    /// Memory storage path
    pub storage_path: Option<String>,
    /// Memory compression
    pub compression: bool,
    /// TTL for memory entries (seconds)
    pub ttl_seconds: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultToleranceConfig {
    /// Enable Byzantine fault tolerance
    pub byzantine_tolerance: bool,
    /// Auto-recovery for failed agents
    pub auto_recovery: bool,
    /// Maximum retry attempts
    pub max_retries: usize,
    /// Timeout for agent operations (milliseconds)
    pub operation_timeout_ms: u64,
}

impl Default for CoordinationConfig {
    fn default() -> Self {
        Self {
            topology: TopologyType::Hierarchical,
            max_agents: 8,
            neural_config: Some(NeuralCoordinationConfig {
                enable_neural_coordination: true,
                learning_rate: 0.001,
                shared_model_state: true,
                enable_simd: true,
            }),
            memory_config: MemoryConfig {
                persistent: true,
                storage_path: Some(".neural-doc-flow/memory".to_string()),
                compression: true,
                ttl_seconds: Some(3600), // 1 hour
            },
            fault_tolerance: FaultToleranceConfig {
                byzantine_tolerance: true,
                auto_recovery: true,
                max_retries: 3,
                operation_timeout_ms: 30000, // 30 seconds
            },
        }
    }
}

/// Main coordinator for document processing with DAA
#[derive(Debug)]
pub struct DocumentCoordinator {
    /// Unique coordinator ID
    pub id: Uuid,
    /// DAA coordinator instance
    daa_coordinator: Arc<RwLock<DaaCoordinator<DocumentTask>>>,
    /// Active agents
    agents: Arc<RwLock<HashMap<Uuid, Box<dyn DocumentAgent>>>>,
    /// Coordination configuration
    config: CoordinationConfig,
    /// Message channel for inter-agent communication
    message_tx: mpsc::UnboundedSender<CoordinationMessage>,
    /// Message receiver
    message_rx: Arc<RwLock<Option<mpsc::UnboundedReceiver<CoordinationMessage>>>>,
    /// Neural coordinator for ruv-FANN integration
    neural_coordinator: Option<Arc<NeuralCoordinator>>,
    /// Persistent memory for coordination state
    memory: Arc<CoordinationMemory>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationMessage {
    pub id: Uuid,
    pub sender: Uuid,
    pub recipient: Option<Uuid>, // None for broadcast
    pub task_id: Option<Uuid>,
    pub message_type: MessageType,
    pub payload: serde_json::Value,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageType {
    TaskAssignment,
    TaskComplete,
    TaskFailed,
    StatusUpdate,
    ConsensusRequest,
    ConsensusResponse,
    NeuralModelUpdate,
    MemorySync,
    AgentSpawned,
    AgentTerminated,
}

impl DocumentCoordinator {
    /// Create a new document coordinator with default configuration
    pub async fn new() -> Result<Self> {
        Self::with_config(CoordinationConfig::default()).await
    }
    
    /// Create a new document coordinator with custom configuration
    pub async fn with_config(config: CoordinationConfig) -> Result<Self> {
        info!("Initializing DocumentCoordinator with DAA");
        debug!("Configuration: {:?}", config);
        
        let id = Uuid::new_v4();
        
        // Initialize DAA coordinator
        let swarm_config = SwarmConfig {
            topology: config.topology,
            max_agents: config.max_agents,
            ..Default::default()
        };
        
        let daa_coordinator = DaaCoordinator::new(swarm_config)
            .map_err(|e| CoreError::CoordinationError(format!("Failed to initialize DAA coordinator: {}", e)))?;
        
        // Create message channel for inter-agent communication
        let (message_tx, message_rx) = mpsc::unbounded_channel();
        
        // Initialize neural coordinator if enabled
        let neural_coordinator = if config.neural_config.as_ref()
            .map(|nc| nc.enable_neural_coordination)
            .unwrap_or(false) 
        {
            Some(Arc::new(NeuralCoordinator::new()))
        } else {
            None
        };
        
        // Initialize persistent memory
        let memory = Arc::new(CoordinationMemory::new(&config.memory_config).await?);
        
        let coordinator = Self {
            id,
            daa_coordinator: Arc::new(RwLock::new(daa_coordinator)),
            agents: Arc::new(RwLock::new(HashMap::new())),
            config,
            message_tx,
            message_rx: Arc::new(RwLock::new(Some(message_rx))),
            neural_coordinator,
            memory,
        };
        
        info!("DocumentCoordinator {} initialized successfully", id);
        Ok(coordinator)
    }
    
    /// Spawn a new agent for document processing
    pub async fn spawn_agent(&self, agent_type: AgentType) -> Result<Uuid> {
        info!("Spawning {} agent", agent_type);
        
        let agent_id = Uuid::new_v4();
        
        // Create agent configuration based on type
        let agent_config = match agent_type {
            AgentType::Coordinator => AgentConfig {
                role: "coordinator".to_string(),
                cognitive_pattern: CognitivePattern::Systems,
                neural_config: None,
            },
            AgentType::Extractor => AgentConfig {
                role: "extractor".to_string(),
                cognitive_pattern: CognitivePattern::Convergent,
                neural_config: self.config.neural_config.clone(),
            },
            AgentType::Validator => AgentConfig {
                role: "validator".to_string(),
                cognitive_pattern: CognitivePattern::Critical,
                neural_config: None,
            },
            AgentType::Formatter => AgentConfig {
                role: "formatter".to_string(),
                cognitive_pattern: CognitivePattern::Lateral,
                neural_config: None,
            },
        };
        
        // Create the agent through DAA
        let daa_agent = StandardDAAAgent::builder()
            .with_id(agent_id)
            .with_cognitive_pattern(agent_config.cognitive_pattern)
            .with_neural_coordinator(self.neural_coordinator.clone())
            .build()
            .await
            .map_err(|e| CoreError::AgentError(format!("Failed to create DAA agent: {}", e)))?;
        
        // Wrap in document agent
        let doc_agent: Box<dyn DocumentAgent> = match agent_type {
            AgentType::Coordinator => Box::new(agent::CoordinatorAgent::new(agent_id, daa_agent)),
            AgentType::Extractor => Box::new(agent::ExtractionAgent::new(agent_id, daa_agent)),
            AgentType::Validator => Box::new(agent::ValidationAgent::new(agent_id, daa_agent)),
            AgentType::Formatter => Box::new(agent::FormatterAgent::new(agent_id, daa_agent)),
        };
        
        // Register agent
        {
            let mut agents = self.agents.write();
            agents.insert(agent_id, doc_agent);
        }
        
        // Store agent spawn in memory
        self.memory.store(
            MemoryKey::AgentSpawn(agent_id),
            MemoryValue::AgentInfo {
                agent_type,
                spawned_at: chrono::Utc::now(),
                status: AgentStatus::Active,
            }
        ).await?;
        
        info!("Agent {} spawned successfully as {}", agent_id, agent_type);
        Ok(agent_id)
    }
    
    /// Process a batch of documents with coordinated agents
    pub async fn process_batch(&self, documents: Vec<Document>, config: BatchConfig) -> Result<Vec<ProcessingResult>> {
        info!("Processing batch of {} documents", documents.len());
        
        // Store batch processing start in memory
        let batch_id = Uuid::new_v4();
        self.memory.store(
            MemoryKey::BatchProcess(batch_id),
            MemoryValue::BatchInfo {
                document_count: documents.len(),
                started_at: chrono::Utc::now(),
                config: config.clone(),
            }
        ).await?;
        
        // Spawn necessary agents based on configuration
        let mut agent_ids = Vec::new();
        
        if config.parallel_extraction {
            for _ in 0..config.extraction_agents {
                let agent_id = self.spawn_agent(AgentType::Extractor).await?;
                agent_ids.push(agent_id);
            }
        }
        
        if config.neural_enhancement {
            // Neural enhancement requires specialized agents
            for _ in 0..config.neural_agents {
                let agent_id = self.spawn_agent(AgentType::Validator).await?;
                agent_ids.push(agent_id);
            }
        }
        
        if config.result_aggregation {
            let coordinator_id = self.spawn_agent(AgentType::Coordinator).await?;
            agent_ids.push(coordinator_id);
        }
        
        // Distribute documents among agents
        let mut tasks = Vec::new();
        for (i, document) in documents.into_iter().enumerate() {
            let agent_id = agent_ids[i % agent_ids.len()];
            let task = DocumentTask::ProcessDocument {
                document,
                options: config.processing_options.clone(),
            };
            tasks.push((agent_id, task));
        }
        
        // Execute tasks in parallel
        let mut results = Vec::new();
        let futures = tasks.into_iter().map(|(agent_id, task)| async move {
            let agents = self.agents.read();
            if let Some(agent) = agents.get(&agent_id) {
                agent.process_task(task).await
            } else {
                Err(CoreError::AgentError(format!("Agent {} not found", agent_id)))
            }
        });
        
        // Collect results
        let task_results = futures_util::future::try_join_all(futures).await?;
        
        for task_result in task_results {
            if let TaskResult::ProcessingComplete(result) = task_result {
                results.push(result);
            }
        }
        
        // Store batch completion in memory
        self.memory.store(
            MemoryKey::BatchComplete(batch_id),
            MemoryValue::BatchResult {
                completed_at: chrono::Utc::now(),
                results_count: results.len(),
                success: true,
            }
        ).await?;
        
        info!("Batch processing completed: {} results", results.len());
        Ok(results)
    }
    
    /// Get status of all active agents
    pub async fn get_agent_status(&self) -> HashMap<Uuid, AgentStatus> {
        let agents = self.agents.read();
        let mut status_map = HashMap::new();
        
        for (id, agent) in agents.iter() {
            status_map.insert(*id, agent.status().await);
        }
        
        status_map
    }
    
    /// Shutdown coordinator and all agents
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down DocumentCoordinator {}", self.id);
        
        // Shutdown all agents
        let agents = self.agents.write();
        for (id, agent) in agents.iter() {
            debug!("Shutting down agent {}", id);
            agent.shutdown().await.unwrap_or_else(|e| {
                warn!("Failed to shutdown agent {}: {}", id, e);
            });
        }
        
        // Clear agents
        drop(agents);
        self.agents.write().clear();
        
        info!("DocumentCoordinator {} shutdown complete", self.id);
        Ok(())
    }
}

/// Configuration for batch processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    pub parallel_extraction: bool,
    pub neural_enhancement: bool,
    pub result_aggregation: bool,
    pub extraction_agents: usize,
    pub neural_agents: usize,
    pub processing_options: ProcessingOptions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingOptions {
    pub enable_ocr: bool,
    pub extract_tables: bool,
    pub extract_images: bool,
    pub confidence_threshold: f32,
    pub timeout_seconds: u32,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            parallel_extraction: true,
            neural_enhancement: true,
            result_aggregation: true,
            extraction_agents: 3,
            neural_agents: 2,
            processing_options: ProcessingOptions {
                enable_ocr: true,
                extract_tables: true,
                extract_images: false,
                confidence_threshold: 0.8,
                timeout_seconds: 300,
            },
        }
    }
}

/// Types of agents that can be spawned
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AgentType {
    Coordinator,
    Extractor,
    Validator,
    Formatter,
}

impl std::fmt::Display for AgentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentType::Coordinator => write!(f, "coordinator"),
            AgentType::Extractor => write!(f, "extractor"),
            AgentType::Validator => write!(f, "validator"),
            AgentType::Formatter => write!(f, "formatter"),
        }
    }
}

/// Status of an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentStatus {
    Active,
    Busy,
    Idle,
    Failed,
    Shutdown,
}

// Import futures_util for async utilities
use futures_util;

// Add missing imports
use chrono;