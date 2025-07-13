use std::{
    sync::Arc,
    time::{Duration, Instant},
    collections::HashMap,
};
use tokio::sync::{RwLock, Mutex, broadcast};
use uuid::Uuid;
use anyhow::{Result, anyhow};
use serde::{Serialize, Deserialize};

use crate::{
    agents::base::{Agent, AgentState},
    messaging::{Message, MessagePriority},
};

/// System-wide fault tolerance coordinator
pub struct FaultToleranceCoordinator {
    health_monitors: Arc<RwLock<HashMap<Uuid, HealthMonitor>>>,
    recovery_strategies: Arc<RwLock<HashMap<String, Box<dyn RecoveryStrategy>>>>,
    failover_manager: Arc<FailoverManager>,
    checkpoint_manager: Arc<CheckpointManager>,
    event_sender: broadcast::Sender<FaultEvent>,
}

/// Health monitoring for agents
pub struct HealthMonitor {
    agent_id: Uuid,
    last_heartbeat: Arc<RwLock<Instant>>,
    health_status: Arc<RwLock<HealthStatus>>,
    failure_count: Arc<Mutex<u32>>,
    config: HealthCheckConfig,
}

#[derive(Clone)]
pub struct HealthCheckConfig {
    pub heartbeat_interval: Duration,
    pub heartbeat_timeout: Duration,
    pub max_failures: u32,
    pub recovery_delay: Duration,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            heartbeat_interval: Duration::from_secs(10),
            heartbeat_timeout: Duration::from_secs(30),
            max_failures: 3,
            recovery_delay: Duration::from_secs(60),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Failed,
    Recovering,
}

/// Recovery strategy trait
#[async_trait::async_trait]
pub trait RecoveryStrategy: Send + Sync {
    async fn recover(&self, agent_id: Uuid, failure_type: FailureType) -> Result<RecoveryAction>;
    fn strategy_name(&self) -> &str;
}

#[derive(Debug, Clone)]
pub enum FailureType {
    Heartbeat,
    Communication,
    Processing,
    Resource,
    Unknown,
}

#[derive(Debug, Clone)]
pub enum RecoveryAction {
    Restart,
    Failover(Uuid), // Target agent ID
    Retry,
    Isolate,
    None,
}

/// Failover management
pub struct FailoverManager {
    primary_agents: Arc<RwLock<HashMap<Uuid, Vec<Uuid>>>>, // Primary -> Backup agents
    active_failovers: Arc<RwLock<HashMap<Uuid, FailoverInfo>>>,
}

#[derive(Clone)]
struct FailoverInfo {
    failed_agent: Uuid,
    backup_agent: Uuid,
    started_at: Instant,
    state_transferred: bool,
}

/// Checkpoint management for state recovery
pub struct CheckpointManager {
    checkpoints: Arc<RwLock<HashMap<Uuid, Vec<Checkpoint>>>>,
    max_checkpoints_per_agent: usize,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub id: Uuid,
    pub agent_id: Uuid,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub state: serde_json::Value,
    pub metadata: CheckpointMetadata,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub tasks_completed: u64,
    pub messages_processed: u64,
    pub last_message_id: Option<Uuid>,
}

/// Fault events for monitoring
#[derive(Debug, Clone)]
pub enum FaultEvent {
    AgentFailure { agent_id: Uuid, failure_type: FailureType },
    AgentRecovered { agent_id: Uuid },
    FailoverInitiated { failed: Uuid, backup: Uuid },
    FailoverCompleted { failed: Uuid, backup: Uuid },
    CheckpointCreated { agent_id: Uuid },
    SystemDegraded { reason: String },
}

impl FaultToleranceCoordinator {
    pub fn new() -> (Self, broadcast::Receiver<FaultEvent>) {
        let (tx, rx) = broadcast::channel(1000);
        
        (Self {
            health_monitors: Arc::new(RwLock::new(HashMap::new())),
            recovery_strategies: Arc::new(RwLock::new(HashMap::new())),
            failover_manager: Arc::new(FailoverManager::new()),
            checkpoint_manager: Arc::new(CheckpointManager::new(10)),
            event_sender: tx,
        }, rx)
    }
    
    /// Register an agent for health monitoring
    pub async fn register_agent(&self, agent_id: Uuid, config: HealthCheckConfig) -> Result<()> {
        let monitor = HealthMonitor::new(agent_id, config);
        let mut monitors = self.health_monitors.write().await;
        monitors.insert(agent_id, monitor);
        Ok(())
    }
    
    /// Unregister an agent
    pub async fn unregister_agent(&self, agent_id: Uuid) -> Result<()> {
        let mut monitors = self.health_monitors.write().await;
        monitors.remove(&agent_id);
        Ok(())
    }
    
    /// Register a recovery strategy
    pub async fn register_recovery_strategy(&self, name: String, strategy: Box<dyn RecoveryStrategy>) {
        let mut strategies = self.recovery_strategies.write().await;
        strategies.insert(name, strategy);
    }
    
    /// Process heartbeat from agent
    pub async fn heartbeat(&self, agent_id: Uuid) -> Result<()> {
        let monitors = self.health_monitors.read().await;
        if let Some(monitor) = monitors.get(&agent_id) {
            monitor.update_heartbeat().await;
            
            // Check if agent is recovering
            let status = monitor.get_status().await;
            if status == HealthStatus::Recovering {
                monitor.set_status(HealthStatus::Healthy).await;
                let _ = self.event_sender.send(FaultEvent::AgentRecovered { agent_id });
            }
            
            Ok(())
        } else {
            Err(anyhow!("Agent not registered for monitoring"))
        }
    }
    
    /// Start health monitoring loop
    pub async fn start_monitoring(&self) {
        let coordinator = self.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(5));
            
            loop {
                interval.tick().await;
                coordinator.check_health().await;
            }
        });
    }
    
    async fn check_health(&self) {
        let monitors = self.health_monitors.read().await;
        
        for (agent_id, monitor) in monitors.iter() {
            if monitor.is_timeout().await {
                let mut failure_count = monitor.failure_count.lock().await;
                *failure_count += 1;
                
                if *failure_count >= monitor.config.max_failures {
                    monitor.set_status(HealthStatus::Failed).await;
                    drop(failure_count);
                    
                    // Trigger recovery
                    let _ = self.event_sender.send(FaultEvent::AgentFailure {
                        agent_id: *agent_id,
                        failure_type: FailureType::Heartbeat,
                    });
                    
                    self.handle_agent_failure(*agent_id, FailureType::Heartbeat).await;
                } else {
                    monitor.set_status(HealthStatus::Degraded).await;
                }
            }
        }
    }
    
    async fn handle_agent_failure(&self, agent_id: Uuid, failure_type: FailureType) {
        // Try recovery strategies
        let strategies = self.recovery_strategies.read().await;
        
        for (_, strategy) in strategies.iter() {
            match strategy.recover(agent_id, failure_type.clone()).await {
                Ok(action) => {
                    self.execute_recovery_action(agent_id, action).await;
                    break;
                }
                Err(e) => {
                    tracing::warn!("Recovery strategy {} failed: {}", strategy.strategy_name(), e);
                }
            }
        }
    }
    
    async fn execute_recovery_action(&self, agent_id: Uuid, action: RecoveryAction) {
        match action {
            RecoveryAction::Restart => {
                // Mark agent as recovering
                if let Some(monitor) = self.health_monitors.read().await.get(&agent_id) {
                    monitor.set_status(HealthStatus::Recovering).await;
                    monitor.reset_failure_count().await;
                }
            }
            RecoveryAction::Failover(backup_id) => {
                self.failover_manager.initiate_failover(agent_id, backup_id).await;
                let _ = self.event_sender.send(FaultEvent::FailoverInitiated {
                    failed: agent_id,
                    backup: backup_id,
                });
            }
            RecoveryAction::Retry => {
                // Reset failure count for retry
                if let Some(monitor) = self.health_monitors.read().await.get(&agent_id) {
                    monitor.reset_failure_count().await;
                }
            }
            RecoveryAction::Isolate => {
                // Mark agent as failed and don't attempt recovery
                if let Some(monitor) = self.health_monitors.read().await.get(&agent_id) {
                    monitor.set_status(HealthStatus::Failed).await;
                }
            }
            RecoveryAction::None => {}
        }
    }
    
    /// Create checkpoint for agent
    pub async fn create_checkpoint(&self, agent_id: Uuid, state: serde_json::Value, metadata: CheckpointMetadata) -> Result<()> {
        self.checkpoint_manager.create_checkpoint(agent_id, state, metadata).await?;
        let _ = self.event_sender.send(FaultEvent::CheckpointCreated { agent_id });
        Ok(())
    }
    
    /// Restore agent from checkpoint
    pub async fn restore_from_checkpoint(&self, agent_id: Uuid) -> Result<Checkpoint> {
        self.checkpoint_manager.get_latest_checkpoint(agent_id).await
    }
    
    /// Setup failover for agent
    pub async fn setup_failover(&self, primary_id: Uuid, backup_ids: Vec<Uuid>) -> Result<()> {
        self.failover_manager.setup_failover(primary_id, backup_ids).await
    }
}

impl HealthMonitor {
    fn new(agent_id: Uuid, config: HealthCheckConfig) -> Self {
        Self {
            agent_id,
            last_heartbeat: Arc::new(RwLock::new(Instant::now())),
            health_status: Arc::new(RwLock::new(HealthStatus::Healthy)),
            failure_count: Arc::new(Mutex::new(0)),
            config,
        }
    }
    
    async fn update_heartbeat(&self) {
        *self.last_heartbeat.write().await = Instant::now();
    }
    
    async fn is_timeout(&self) -> bool {
        let last = *self.last_heartbeat.read().await;
        Instant::now().duration_since(last) > self.config.heartbeat_timeout
    }
    
    async fn get_status(&self) -> HealthStatus {
        *self.health_status.read().await
    }
    
    async fn set_status(&self, status: HealthStatus) {
        *self.health_status.write().await = status;
    }
    
    async fn reset_failure_count(&self) {
        *self.failure_count.lock().await = 0;
    }
}

impl FailoverManager {
    fn new() -> Self {
        Self {
            primary_agents: Arc::new(RwLock::new(HashMap::new())),
            active_failovers: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    async fn setup_failover(&self, primary_id: Uuid, backup_ids: Vec<Uuid>) -> Result<()> {
        let mut primaries = self.primary_agents.write().await;
        primaries.insert(primary_id, backup_ids);
        Ok(())
    }
    
    async fn initiate_failover(&self, failed_id: Uuid, backup_id: Uuid) {
        let failover_info = FailoverInfo {
            failed_agent: failed_id,
            backup_agent: backup_id,
            started_at: Instant::now(),
            state_transferred: false,
        };
        
        let mut active = self.active_failovers.write().await;
        active.insert(failed_id, failover_info);
    }
    
    async fn complete_failover(&self, failed_id: Uuid) -> Result<()> {
        let mut active = self.active_failovers.write().await;
        active.remove(&failed_id)
            .ok_or_else(|| anyhow!("No active failover for agent"))?;
        Ok(())
    }
}

impl CheckpointManager {
    fn new(max_checkpoints_per_agent: usize) -> Self {
        Self {
            checkpoints: Arc::new(RwLock::new(HashMap::new())),
            max_checkpoints_per_agent,
        }
    }
    
    async fn create_checkpoint(&self, agent_id: Uuid, state: serde_json::Value, metadata: CheckpointMetadata) -> Result<()> {
        let checkpoint = Checkpoint {
            id: Uuid::new_v4(),
            agent_id,
            timestamp: chrono::Utc::now(),
            state,
            metadata,
        };
        
        let mut checkpoints = self.checkpoints.write().await;
        let agent_checkpoints = checkpoints.entry(agent_id).or_insert_with(Vec::new);
        
        // Maintain max checkpoints limit
        if agent_checkpoints.len() >= self.max_checkpoints_per_agent {
            agent_checkpoints.remove(0);
        }
        
        agent_checkpoints.push(checkpoint);
        Ok(())
    }
    
    async fn get_latest_checkpoint(&self, agent_id: Uuid) -> Result<Checkpoint> {
        let checkpoints = self.checkpoints.read().await;
        checkpoints.get(&agent_id)
            .and_then(|cps| cps.last())
            .cloned()
            .ok_or_else(|| anyhow!("No checkpoint found for agent"))
    }
}

impl Clone for FaultToleranceCoordinator {
    fn clone(&self) -> Self {
        Self {
            health_monitors: self.health_monitors.clone(),
            recovery_strategies: self.recovery_strategies.clone(),
            failover_manager: self.failover_manager.clone(),
            checkpoint_manager: self.checkpoint_manager.clone(),
            event_sender: self.event_sender.clone(),
        }
    }
}

/// Default recovery strategies
pub struct RestartRecoveryStrategy;

#[async_trait::async_trait]
impl RecoveryStrategy for RestartRecoveryStrategy {
    async fn recover(&self, _agent_id: Uuid, failure_type: FailureType) -> Result<RecoveryAction> {
        match failure_type {
            FailureType::Heartbeat | FailureType::Communication => Ok(RecoveryAction::Restart),
            _ => Ok(RecoveryAction::None),
        }
    }
    
    fn strategy_name(&self) -> &str {
        "restart_recovery"
    }
}