use std::{
    collections::HashMap,
    sync::Arc,
};
use tokio::sync::{RwLock, broadcast};
use uuid::Uuid;
use anyhow::{Result, anyhow};
use regex::Regex;

use crate::{
    messaging::{Message, MessagePriority},
    AgentCapability,
};

/// Routing strategy for message distribution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingStrategy {
    Direct,          // Route to specific agent
    Broadcast,       // Route to all agents
    RoundRobin,      // Distribute evenly
    LoadBalanced,    // Route to least loaded
    CapabilityBased, // Route based on capabilities
    ContentBased,    // Route based on message content
}

/// Route definition
#[derive(Debug, Clone)]
pub struct Route {
    pub id: Uuid,
    pub name: String,
    pub pattern: RoutePattern,
    pub targets: Vec<Uuid>,
    pub strategy: RoutingStrategy,
    pub priority: i32,
    pub enabled: bool,
}

/// Pattern for matching messages
#[derive(Clone)]
pub enum RoutePattern {
    /// Match by message content field
    ContentField { field: String, pattern: Regex },
    /// Match by sender ID
    Sender(Uuid),
    /// Match by message priority
    Priority(MessagePriority),
    /// Match by custom predicate
    Custom(Arc<dyn MessagePredicate>),
    /// Always match
    All,
}

pub trait MessagePredicate: Send + Sync {
    fn matches(&self, message: &Message) -> bool;
}

impl std::fmt::Debug for RoutePattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RoutePattern::ContentField { field, pattern } => {
                f.debug_struct("ContentField")
                    .field("field", field)
                    .field("pattern", &pattern.as_str())
                    .finish()
            }
            RoutePattern::Sender(id) => f.debug_tuple("Sender").field(id).finish(),
            RoutePattern::Priority(p) => f.debug_tuple("Priority").field(p).finish(),
            RoutePattern::Custom(_) => f.debug_struct("Custom").finish(),
            RoutePattern::All => write!(f, "All"),
        }
    }
}

/// Router for message distribution
pub struct MessageRouter {
    routes: Arc<RwLock<Vec<Route>>>,
    agent_registry: Arc<RwLock<AgentRegistry>>,
    round_robin_state: Arc<RwLock<HashMap<Uuid, usize>>>,
    stats: Arc<RwLock<RoutingStats>>,
}

/// Registry of available agents
pub struct AgentRegistry {
    agents: HashMap<Uuid, AgentInfo>,
    capability_index: HashMap<AgentCapability, Vec<Uuid>>,
}

#[derive(Clone)]
pub struct AgentInfo {
    pub id: Uuid,
    pub capabilities: Vec<AgentCapability>,
    pub load: f64,
    pub available: bool,
}

#[derive(Default)]
pub struct RoutingStats {
    pub messages_routed: u64,
    pub routing_failures: u64,
    pub route_hits: HashMap<Uuid, u64>,
}

impl MessageRouter {
    pub fn new() -> Self {
        Self {
            routes: Arc::new(RwLock::new(Vec::new())),
            agent_registry: Arc::new(RwLock::new(AgentRegistry::new())),
            round_robin_state: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(RoutingStats::default())),
        }
    }
    
    /// Add a new route
    pub async fn add_route(&self, route: Route) -> Result<()> {
        let mut routes = self.routes.write().await;
        
        // Check for duplicate route names
        if routes.iter().any(|r| r.name == route.name) {
            return Err(anyhow!("Route with name '{}' already exists", route.name));
        }
        
        routes.push(route);
        routes.sort_by_key(|r| -r.priority); // Sort by priority (highest first)
        Ok(())
    }
    
    /// Remove a route by ID
    pub async fn remove_route(&self, route_id: Uuid) -> Result<()> {
        let mut routes = self.routes.write().await;
        let original_len = routes.len();
        routes.retain(|r| r.id != route_id);
        
        if routes.len() == original_len {
            Err(anyhow!("Route not found"))
        } else {
            Ok(())
        }
    }
    
    /// Register an agent
    pub async fn register_agent(&self, agent_info: AgentInfo) -> Result<()> {
        let mut registry = self.agent_registry.write().await;
        registry.register(agent_info);
        Ok(())
    }
    
    /// Unregister an agent
    pub async fn unregister_agent(&self, agent_id: Uuid) -> Result<()> {
        let mut registry = self.agent_registry.write().await;
        registry.unregister(agent_id);
        Ok(())
    }
    
    /// Update agent load
    pub async fn update_agent_load(&self, agent_id: Uuid, load: f64) -> Result<()> {
        let mut registry = self.agent_registry.write().await;
        registry.update_load(agent_id, load)
    }
    
    /// Route a message
    pub async fn route(&self, message: &Message) -> Result<Vec<Uuid>> {
        let routes = self.routes.read().await;
        let mut stats = self.stats.write().await;
        stats.messages_routed += 1;
        
        // Find matching routes
        for route in routes.iter() {
            if !route.enabled {
                continue;
            }
            
            if self.matches_pattern(&route.pattern, message).await {
                let targets = self.resolve_targets(route, message).await?;
                
                if !targets.is_empty() {
                    // Update stats
                    *stats.route_hits.entry(route.id).or_insert(0) += 1;
                    return Ok(targets);
                }
            }
        }
        
        stats.routing_failures += 1;
        Err(anyhow!("No matching route found for message"))
    }
    
    /// Check if message matches pattern
    async fn matches_pattern(&self, pattern: &RoutePattern, message: &Message) -> bool {
        match pattern {
            RoutePattern::ContentField { field, pattern } => {
                if let Some(value) = message.content.get(field) {
                    if let Some(text) = value.as_str() {
                        return pattern.is_match(text);
                    }
                }
                false
            }
            RoutePattern::Sender(sender_id) => message.from == *sender_id,
            RoutePattern::Priority(priority) => message.priority == *priority,
            RoutePattern::Custom(predicate) => predicate.matches(message),
            RoutePattern::All => true,
        }
    }
    
    /// Resolve target agents based on routing strategy
    async fn resolve_targets(&self, route: &Route, message: &Message) -> Result<Vec<Uuid>> {
        let registry = self.agent_registry.read().await;
        
        match route.strategy {
            RoutingStrategy::Direct => {
                // Filter out unavailable agents
                Ok(route.targets.iter()
                    .filter(|id| registry.is_available(**id))
                    .copied()
                    .collect())
            }
            
            RoutingStrategy::Broadcast => {
                // Send to all available targets
                Ok(route.targets.iter()
                    .filter(|id| registry.is_available(**id))
                    .copied()
                    .collect())
            }
            
            RoutingStrategy::RoundRobin => {
                let available_targets: Vec<Uuid> = route.targets.iter()
                    .filter(|id| registry.is_available(**id))
                    .copied()
                    .collect();
                
                if available_targets.is_empty() {
                    return Ok(vec![]);
                }
                
                let mut rr_state = self.round_robin_state.write().await;
                let index = rr_state.entry(route.id).or_insert(0);
                let target = available_targets[*index % available_targets.len()];
                *index += 1;
                
                Ok(vec![target])
            }
            
            RoutingStrategy::LoadBalanced => {
                // Find least loaded agent
                let mut best_agent = None;
                let mut best_load = f64::MAX;
                
                for target_id in &route.targets {
                    if let Some(info) = registry.agents.get(target_id) {
                        if info.available && info.load < best_load {
                            best_load = info.load;
                            best_agent = Some(*target_id);
                        }
                    }
                }
                
                Ok(best_agent.into_iter().collect())
            }
            
            RoutingStrategy::CapabilityBased => {
                // Extract required capability from message
                if let Some(required_cap) = self.extract_required_capability(message) {
                    if let Some(agents) = registry.capability_index.get(&required_cap) {
                        // Return first available agent with capability
                        for agent_id in agents {
                            if registry.is_available(*agent_id) {
                                return Ok(vec![*agent_id]);
                            }
                        }
                    }
                }
                Ok(vec![])
            }
            
            RoutingStrategy::ContentBased => {
                // Route based on message content
                // This is a simplified example - real implementation would be more sophisticated
                let hash = self.hash_message_content(message);
                let available_targets: Vec<Uuid> = route.targets.iter()
                    .filter(|id| registry.is_available(**id))
                    .copied()
                    .collect();
                
                if available_targets.is_empty() {
                    return Ok(vec![]);
                }
                
                let index = (hash as usize) % available_targets.len();
                Ok(vec![available_targets[index]])
            }
        }
    }
    
    fn extract_required_capability(&self, message: &Message) -> Option<AgentCapability> {
        // Extract capability requirement from message
        if let Some(cap_str) = message.content.get("required_capability") {
            if let Some(cap) = cap_str.as_str() {
                // Parse capability from string
                match cap {
                    "validation" => Some(AgentCapability::ValidationExpert),
                    "enhancement" => Some(AgentCapability::ContentEnhancement),
                    "formatting" => Some(AgentCapability::FormatConversion),
                    _ => None,
                }
            } else {
                None
            }
        } else {
            None
        }
    }
    
    fn hash_message_content(&self, message: &Message) -> u64 {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        
        let mut hasher = DefaultHasher::new();
        format!("{:?}", message.content).hash(&mut hasher);
        hasher.finish()
    }
    
    pub async fn get_stats(&self) -> RoutingStats {
        let stats = self.stats.read().await;
        RoutingStats {
            messages_routed: stats.messages_routed,
            routing_failures: stats.routing_failures,
            route_hits: stats.route_hits.clone(),
        }
    }
}

impl AgentRegistry {
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
            capability_index: HashMap::new(),
        }
    }
    
    pub fn register(&mut self, agent_info: AgentInfo) {
        // Update capability index
        for capability in &agent_info.capabilities {
            self.capability_index
                .entry(capability.clone())
                .or_insert_with(Vec::new)
                .push(agent_info.id);
        }
        
        self.agents.insert(agent_info.id, agent_info);
    }
    
    pub fn unregister(&mut self, agent_id: Uuid) {
        if let Some(agent_info) = self.agents.remove(&agent_id) {
            // Update capability index
            for capability in &agent_info.capabilities {
                if let Some(agents) = self.capability_index.get_mut(capability) {
                    agents.retain(|id| *id != agent_id);
                }
            }
        }
    }
    
    pub fn update_load(&mut self, agent_id: Uuid, load: f64) -> Result<()> {
        if let Some(agent) = self.agents.get_mut(&agent_id) {
            agent.load = load;
            Ok(())
        } else {
            Err(anyhow!("Agent not found"))
        }
    }
    
    pub fn is_available(&self, agent_id: Uuid) -> bool {
        self.agents.get(&agent_id)
            .map(|info| info.available)
            .unwrap_or(false)
    }
}

/// Predefined route builders
impl Route {
    pub fn direct(name: String, target: Uuid) -> Self {
        Route {
            id: Uuid::new_v4(),
            name,
            pattern: RoutePattern::All,
            targets: vec![target],
            strategy: RoutingStrategy::Direct,
            priority: 0,
            enabled: true,
        }
    }
    
    pub fn broadcast(name: String, targets: Vec<Uuid>) -> Self {
        Route {
            id: Uuid::new_v4(),
            name,
            pattern: RoutePattern::All,
            targets,
            strategy: RoutingStrategy::Broadcast,
            priority: 0,
            enabled: true,
        }
    }
    
    pub fn capability_based(name: String) -> Self {
        Route {
            id: Uuid::new_v4(),
            name,
            pattern: RoutePattern::All,
            targets: vec![], // Will be resolved dynamically
            strategy: RoutingStrategy::CapabilityBased,
            priority: 10,
            enabled: true,
        }
    }
}