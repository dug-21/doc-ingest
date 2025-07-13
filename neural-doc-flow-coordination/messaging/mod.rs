/// DAA Message Passing System for Neural Document Processing
/// High-performance inter-agent communication with priority queuing and fault tolerance

use super::agents::{CoordinationMessage, MessageType};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{RwLock, Notify};
use uuid::Uuid;

pub mod priority_queue;
pub mod fault_tolerance;
pub mod routing;
pub mod protocols;

/// Message priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MessagePriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Core message structure for inter-agent communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: Uuid,
    pub from: Uuid,
    pub to: Uuid,
    pub priority: MessagePriority,
    pub content: serde_json::Value,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub correlation_id: Option<Uuid>,
}

/// Message Bus for DAA coordination
pub struct MessageBus {
    pub message_queues: Arc<RwLock<HashMap<Uuid, VecDeque<CoordinationMessage>>>>,
    pub broadcast_queue: Arc<RwLock<VecDeque<CoordinationMessage>>>,
    pub message_router: Arc<MessageRouter>,
    pub fault_tolerance: Arc<RwLock<FaultToleranceManager>>,
    pub performance_metrics: Arc<RwLock<MessageMetrics>>,
    pub notification: Arc<Notify>,
}

/// Message Router for intelligent message routing
pub struct MessageRouter {
    pub routing_table: Arc<RwLock<HashMap<Uuid, AgentRoute>>>,
    pub topology_type: super::topologies::TopologyType,
    pub load_balancer: LoadBalancer,
}

#[derive(Debug, Clone)]
pub struct AgentRoute {
    pub agent_id: Uuid,
    pub agent_type: super::agents::AgentType,
    pub capabilities: super::agents::AgentCapabilities,
    pub current_load: f32,
    pub response_time: f64,
    pub reliability_score: f64,
}

/// Load Balancer for message distribution
#[derive(Debug, Clone)]
pub struct LoadBalancer {
    pub strategy: LoadBalancingStrategy,
    pub agent_loads: HashMap<Uuid, f32>,
    pub performance_history: HashMap<Uuid, Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastLoaded,
    WeightedRoundRobin,
    PerformanceBased,
    CapabilityMatching,
}

/// Fault Tolerance Manager
pub struct FaultToleranceManager {
    pub failed_agents: HashMap<Uuid, FailureInfo>,
    pub message_retry_queue: VecDeque<RetryMessage>,
    pub circuit_breakers: HashMap<Uuid, CircuitBreaker>,
    pub backup_routes: HashMap<Uuid, Vec<Uuid>>,
}

#[derive(Debug, Clone)]
pub struct FailureInfo {
    pub agent_id: Uuid,
    pub failure_count: u32,
    pub last_failure: chrono::DateTime<chrono::Utc>,
    pub failure_type: FailureType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FailureType {
    Timeout,
    ProcessingError,
    NetworkError,
    CapacityOverload,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct RetryMessage {
    pub message: CoordinationMessage,
    pub retry_count: u32,
    pub max_retries: u32,
    pub backoff_delay: u64, // milliseconds
    pub next_retry: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    pub agent_id: Uuid,
    pub state: CircuitBreakerState,
    pub failure_threshold: u32,
    pub timeout_duration: u64,
    pub last_failure_time: Option<chrono::DateTime<chrono::Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CircuitBreakerState {
    Closed,  // Normal operation
    Open,    // Failing, no messages sent
    HalfOpen, // Testing if agent is back online
}

/// Message Performance Metrics
#[derive(Debug, Clone)]
pub struct MessageMetrics {
    pub messages_sent: u64,
    pub messages_delivered: u64,
    pub messages_failed: u64,
    pub average_delivery_time: f64,
    pub throughput: f64, // messages per second
    pub queue_sizes: HashMap<Uuid, usize>,
    pub agent_response_times: HashMap<Uuid, f64>,
}

impl Default for MessageMetrics {
    fn default() -> Self {
        Self {
            messages_sent: 0,
            messages_delivered: 0,
            messages_failed: 0,
            average_delivery_time: 0.0,
            throughput: 0.0,
            queue_sizes: HashMap::new(),
            agent_response_times: HashMap::new(),
        }
    }
}

impl MessageBus {
    /// Create new message bus
    pub fn new(topology_type: super::topologies::TopologyType) -> Self {
        Self {
            message_queues: Arc::new(RwLock::new(HashMap::new())),
            broadcast_queue: Arc::new(RwLock::new(VecDeque::new())),
            message_router: Arc::new(MessageRouter::new(topology_type)),
            fault_tolerance: Arc::new(RwLock::new(FaultToleranceManager::new())),
            performance_metrics: Arc::new(RwLock::new(MessageMetrics::default())),
            notification: Arc::new(Notify::new()),
        }
    }
    
    /// Send message to specific agent or broadcast
    pub async fn send_message(&self, message: CoordinationMessage) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let start_time = std::time::Instant::now();
        
        // Update metrics
        {
            let mut metrics = self.performance_metrics.write().await;
            metrics.messages_sent += 1;
        }
        
        // Check circuit breaker if targeting specific agent
        if let Some(target_id) = message.to {
            let ft_manager = self.fault_tolerance.read().await;
            if let Some(circuit_breaker) = ft_manager.circuit_breakers.get(&target_id) {
                if matches!(circuit_breaker.state, CircuitBreakerState::Open) {
                    return self.handle_circuit_breaker_open(message, target_id).await;
                }
            }
        }
        
        // Route message
        let route_result = if message.to.is_some() {
            self.route_direct_message(message).await
        } else {
            self.route_broadcast_message(message).await
        };
        
        // Update delivery metrics
        let delivery_time = start_time.elapsed().as_secs_f64();
        {
            let mut metrics = self.performance_metrics.write().await;
            if route_result.is_ok() {
                metrics.messages_delivered += 1;
                metrics.average_delivery_time = (metrics.average_delivery_time + delivery_time) / 2.0;
            } else {
                metrics.messages_failed += 1;
            }
        }
        
        // Notify waiting processors
        self.notification.notify_waiters();
        
        route_result
    }
    
    /// Route direct message to specific agent
    async fn route_direct_message(&self, message: CoordinationMessage) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let target_id = message.to.unwrap();
        
        // Find optimal route
        let route = self.message_router.find_route(message.from, target_id).await?;
        
        // Add to agent's message queue
        {
            let mut queues = self.message_queues.write().await;
            let queue = queues.entry(route.agent_id).or_insert_with(VecDeque::new);
            
            // Priority insertion based on message priority
            let insert_pos = queue.iter()
                .position(|m| m.priority < message.priority)
                .unwrap_or(queue.len());
            
            queue.insert(insert_pos, message);
        }
        
        // Update queue size metrics
        {
            let queues = self.message_queues.read().await;
            let mut metrics = self.performance_metrics.write().await;
            for (agent_id, queue) in queues.iter() {
                metrics.queue_sizes.insert(*agent_id, queue.len());
            }
        }
        
        Ok(())
    }
    
    /// Route broadcast message
    async fn route_broadcast_message(&self, message: CoordinationMessage) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let target_agents = self.message_router.get_broadcast_targets(message.from).await?;
        
        for target_id in target_agents {
            let mut targeted_message = message.clone();
            targeted_message.to = Some(target_id);
            
            // Add to each target's queue
            {
                let mut queues = self.message_queues.write().await;
                let queue = queues.entry(target_id).or_insert_with(VecDeque::new);
                
                let insert_pos = queue.iter()
                    .position(|m| m.priority < message.priority)
                    .unwrap_or(queue.len());
                
                queue.insert(insert_pos, targeted_message);
            }
        }
        
        Ok(())
    }
    
    /// Handle circuit breaker open state
    async fn handle_circuit_breaker_open(&self, message: CoordinationMessage, failed_agent_id: Uuid) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Try to find backup route
        let ft_manager = self.fault_tolerance.read().await;
        if let Some(backup_routes) = ft_manager.backup_routes.get(&failed_agent_id) {
            for &backup_id in backup_routes {
                if !ft_manager.circuit_breakers.get(&backup_id)
                    .map_or(false, |cb| matches!(cb.state, CircuitBreakerState::Open)) {
                    
                    let mut backup_message = message.clone();
                    backup_message.to = Some(backup_id);
                    
                    drop(ft_manager);
                    return self.route_direct_message(backup_message).await;
                }
            }
        }
        
        // No backup available, add to retry queue
        let retry_message = RetryMessage {
            message,
            retry_count: 0,
            max_retries: 3,
            backoff_delay: 1000, // 1 second
            next_retry: chrono::Utc::now() + chrono::Duration::milliseconds(1000),
        };
        
        drop(ft_manager);
        let mut ft_manager = self.fault_tolerance.write().await;
        ft_manager.message_retry_queue.push_back(retry_message);
        
        Err("Agent unavailable, message queued for retry".into())
    }
    
    /// Receive message for specific agent
    pub async fn receive_message(&self, agent_id: Uuid) -> Option<CoordinationMessage> {
        let mut queues = self.message_queues.write().await;
        if let Some(queue) = queues.get_mut(&agent_id) {
            queue.pop_front()
        } else {
            None
        }
    }
    
    /// Register agent with message bus
    pub async fn register_agent(&self, agent_id: Uuid, agent_type: super::agents::AgentType, capabilities: super::agents::AgentCapabilities) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Add to routing table
        let route = AgentRoute {
            agent_id,
            agent_type,
            capabilities,
            current_load: 0.0,
            response_time: 0.0,
            reliability_score: 1.0,
        };
        
        {
            let mut routing_table = self.message_router.routing_table.write().await;
            routing_table.insert(agent_id, route);
        }
        
        // Initialize message queue
        {
            let mut queues = self.message_queues.write().await;
            queues.insert(agent_id, VecDeque::new());
        }
        
        // Initialize circuit breaker
        {
            let mut ft_manager = self.fault_tolerance.write().await;
            ft_manager.circuit_breakers.insert(agent_id, CircuitBreaker {
                agent_id,
                state: CircuitBreakerState::Closed,
                failure_threshold: 5,
                timeout_duration: 30000, // 30 seconds
                last_failure_time: None,
            });
        }
        
        Ok(())
    }
    
    /// Unregister agent from message bus
    pub async fn unregister_agent(&self, agent_id: Uuid) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Remove from routing table
        {
            let mut routing_table = self.message_router.routing_table.write().await;
            routing_table.remove(&agent_id);
        }
        
        // Remove message queue
        {
            let mut queues = self.message_queues.write().await;
            queues.remove(&agent_id);
        }
        
        // Remove circuit breaker
        {
            let mut ft_manager = self.fault_tolerance.write().await;
            ft_manager.circuit_breakers.remove(&agent_id);
        }
        
        Ok(())
    }
    
    /// Process retry queue
    pub async fn process_retry_queue(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let current_time = chrono::Utc::now();
        let mut retry_messages = Vec::new();
        
        // Collect messages ready for retry
        {
            let mut ft_manager = self.fault_tolerance.write().await;
            while let Some(retry_msg) = ft_manager.message_retry_queue.pop_front() {
                if retry_msg.next_retry <= current_time {
                    retry_messages.push(retry_msg);
                } else {
                    ft_manager.message_retry_queue.push_front(retry_msg);
                    break;
                }
            }
        }
        
        // Retry messages
        for mut retry_msg in retry_messages {
            retry_msg.retry_count += 1;
            
            if retry_msg.retry_count <= retry_msg.max_retries {
                // Try to send again
                if self.send_message(retry_msg.message.clone()).await.is_err() {
                    // Failed again, re-queue with exponential backoff
                    retry_msg.backoff_delay *= 2;
                    retry_msg.next_retry = current_time + chrono::Duration::milliseconds(retry_msg.backoff_delay as i64);
                    
                    let mut ft_manager = self.fault_tolerance.write().await;
                    ft_manager.message_retry_queue.push_back(retry_msg);
                }
            } else {
                // Max retries exceeded, mark as permanent failure
                let mut metrics = self.performance_metrics.write().await;
                metrics.messages_failed += 1;
            }
        }
        
        Ok(())
    }
    
    /// Report agent failure
    pub async fn report_agent_failure(&self, agent_id: Uuid, failure_type: FailureType) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut ft_manager = self.fault_tolerance.write().await;
        
        // Update failure info
        {
            let failure_info = ft_manager.failed_agents.entry(agent_id).or_insert(FailureInfo {
                agent_id,
                failure_count: 0,
                last_failure: chrono::Utc::now(),
                failure_type: failure_type.clone(),
            });
            
            failure_info.failure_count += 1;
            failure_info.last_failure = chrono::Utc::now();
            failure_info.failure_type = failure_type;
        }
        
        // Update circuit breaker
        let failure_count = ft_manager.failed_agents.get(&agent_id).map(|info| info.failure_count).unwrap_or(0);
        if let Some(circuit_breaker) = ft_manager.circuit_breakers.get_mut(&agent_id) {
            circuit_breaker.last_failure_time = Some(chrono::Utc::now());
            
            if failure_count >= circuit_breaker.failure_threshold {
                circuit_breaker.state = CircuitBreakerState::Open;
            }
        }
        
        Ok(())
    }
    
    /// Get message bus metrics
    pub async fn get_metrics(&self) -> MessageMetrics {
        self.performance_metrics.read().await.clone()
    }
    
    /// Optimize message routing
    pub async fn optimize_routing(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.message_router.optimize_routes().await
    }
}

impl MessageRouter {
    /// Create new message router
    pub fn new(topology_type: super::topologies::TopologyType) -> Self {
        Self {
            routing_table: Arc::new(RwLock::new(HashMap::new())),
            topology_type,
            load_balancer: LoadBalancer {
                strategy: LoadBalancingStrategy::PerformanceBased,
                agent_loads: HashMap::new(),
                performance_history: HashMap::new(),
            },
        }
    }
    
    /// Find optimal route to target agent
    pub async fn find_route(&self, from: Uuid, to: Uuid) -> Result<AgentRoute, Box<dyn std::error::Error + Send + Sync>> {
        let routing_table = self.routing_table.read().await;
        
        if let Some(route) = routing_table.get(&to) {
            Ok(route.clone())
        } else {
            Err(format!("No route found to agent {}", to).into())
        }
    }
    
    /// Get broadcast targets based on topology
    pub async fn get_broadcast_targets(&self, from: Uuid) -> Result<Vec<Uuid>, Box<dyn std::error::Error + Send + Sync>> {
        let routing_table = self.routing_table.read().await;
        
        match self.topology_type {
            super::topologies::TopologyType::Star => {
                // In star topology, coordinator broadcasts to all spokes
                Ok(routing_table.keys().filter(|&&id| id != from).copied().collect())
            }
            super::topologies::TopologyType::Mesh => {
                // In mesh topology, broadcast to all connected agents
                Ok(routing_table.keys().filter(|&&id| id != from).copied().collect())
            }
            super::topologies::TopologyType::Pipeline => {
                // In pipeline topology, send to next stage only
                if let Some(next_agent) = self.find_next_pipeline_agent(from).await {
                    Ok(vec![next_agent])
                } else {
                    Ok(vec![])
                }
            }
            super::topologies::TopologyType::Hybrid => {
                // In hybrid topology, use intelligent routing
                Ok(routing_table.keys().filter(|&&id| id != from).copied().collect())
            }
        }
    }
    
    /// Find next agent in pipeline
    async fn find_next_pipeline_agent(&self, current: Uuid) -> Option<Uuid> {
        // Simplified pipeline logic - would be more sophisticated in real implementation
        let routing_table = self.routing_table.read().await;
        
        // Find next agent based on agent type sequence
        if let Some(current_route) = routing_table.get(&current) {
            match current_route.agent_type {
                super::agents::AgentType::Controller => {
                    // Controller -> Extractor
                    self.find_agent_by_type(&routing_table, super::agents::AgentType::Extractor)
                }
                super::agents::AgentType::Extractor => {
                    // Extractor -> Enhancer
                    self.find_agent_by_type(&routing_table, super::agents::AgentType::Enhancer)
                }
                super::agents::AgentType::Enhancer => {
                    // Enhancer -> Validator
                    self.find_agent_by_type(&routing_table, super::agents::AgentType::Validator)
                }
                super::agents::AgentType::Validator => {
                    // Validator -> Formatter
                    self.find_agent_by_type(&routing_table, super::agents::AgentType::Formatter)
                }
                super::agents::AgentType::Formatter => {
                    // End of pipeline
                    None
                }
            }
        } else {
            None
        }
    }
    
    /// Find agent by type
    fn find_agent_by_type(&self, routing_table: &HashMap<Uuid, AgentRoute>, agent_type: super::agents::AgentType) -> Option<Uuid> {
        routing_table.iter()
            .find(|(_, route)| std::mem::discriminant(&route.agent_type) == std::mem::discriminant(&agent_type))
            .map(|(id, _)| *id)
    }
    
    /// Optimize routing table
    pub async fn optimize_routes(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Update agent performance metrics and optimize routing
        // This would analyze performance history and adjust routing preferences
        Ok(())
    }
}

impl FaultToleranceManager {
    /// Create new fault tolerance manager
    pub fn new() -> Self {
        Self {
            failed_agents: HashMap::new(),
            message_retry_queue: VecDeque::new(),
            circuit_breakers: HashMap::new(),
            backup_routes: HashMap::new(),
        }
    }
    
    /// Add backup route for agent
    pub fn add_backup_route(&mut self, primary_agent: Uuid, backup_agent: Uuid) {
        self.backup_routes.entry(primary_agent).or_insert_with(Vec::new).push(backup_agent);
    }
    
    /// Check and update circuit breaker states
    pub fn update_circuit_breakers(&mut self) {
        let current_time = chrono::Utc::now();
        
        for circuit_breaker in self.circuit_breakers.values_mut() {
            if matches!(circuit_breaker.state, CircuitBreakerState::Open) {
                if let Some(last_failure) = circuit_breaker.last_failure_time {
                    let time_since_failure = current_time.signed_duration_since(last_failure);
                    if time_since_failure.num_milliseconds() >= circuit_breaker.timeout_duration as i64 {
                        circuit_breaker.state = CircuitBreakerState::HalfOpen;
                    }
                }
            }
        }
    }
}