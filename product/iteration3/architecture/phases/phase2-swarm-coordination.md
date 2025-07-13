# Phase 2: Swarm Coordination - Multi-Agent Infrastructure

## Overview

The Swarm Coordination phase introduces distributed processing capabilities through a multi-agent system. Agents work collaboratively to process documents in parallel while maintaining coordination and state consistency.

## Architecture

```rust
// Swarm module structure
pub mod neuralflow {
    pub mod swarm {
        pub mod agent;       // Agent lifecycle and behavior
        pub mod coordinator; // Swarm coordination logic
        pub mod messaging;   // Inter-agent communication
        pub mod scheduler;   // Task scheduling and distribution
        pub mod state;       // Distributed state management
    }
}
```

## Key Components

### 1. Agent System

```rust
use tokio::sync::mpsc;
use uuid::Uuid;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AgentType {
    Parser,      // PDF parsing specialist
    Extractor,   // Text extraction specialist
    Analyzer,    // Layout analysis specialist
    Aggregator,  // Result aggregation specialist
}

pub struct Agent {
    id: Uuid,
    agent_type: AgentType,
    state: AgentState,
    inbox: mpsc::Receiver<Message>,
    outbox: mpsc::Sender<Message>,
    capabilities: Vec<Capability>,
}

impl Agent {
    pub async fn run(mut self) {
        loop {
            tokio::select! {
                Some(msg) = self.inbox.recv() => {
                    self.handle_message(msg).await;
                }
                _ = tokio::time::sleep(Duration::from_millis(100)) => {
                    self.heartbeat().await;
                }
            }
        }
    }
    
    async fn handle_message(&mut self, msg: Message) {
        match msg {
            Message::Task(task) => self.process_task(task).await,
            Message::Query(query) => self.respond_to_query(query).await,
            Message::Coordination(coord) => self.handle_coordination(coord).await,
        }
    }
}
```

### 2. Swarm Coordinator

```rust
pub struct SwarmCoordinator {
    agents: HashMap<Uuid, AgentHandle>,
    topology: SwarmTopology,
    scheduler: TaskScheduler,
    state_manager: StateManager,
}

impl SwarmCoordinator {
    pub fn new(config: SwarmConfig) -> Self {
        Self {
            agents: HashMap::new(),
            topology: config.topology,
            scheduler: TaskScheduler::new(config.scheduling_strategy),
            state_manager: StateManager::new(),
        }
    }
    
    pub async fn spawn_agent(&mut self, agent_type: AgentType) -> Result<Uuid> {
        let (tx, rx) = mpsc::channel(1000);
        let agent_id = Uuid::new_v4();
        
        let agent = Agent::new(agent_id, agent_type, rx, self.message_bus.clone());
        let handle = tokio::spawn(agent.run());
        
        self.agents.insert(agent_id, AgentHandle {
            id: agent_id,
            agent_type,
            sender: tx,
            handle,
        });
        
        self.topology.register_agent(agent_id, agent_type)?;
        
        Ok(agent_id)
    }
    
    pub async fn distribute_document(&mut self, doc: DocumentTask) -> Result<ProcessingResult> {
        // Break document into subtasks
        let subtasks = self.scheduler.create_subtasks(&doc)?;
        
        // Distribute subtasks to agents
        let futures: Vec<_> = subtasks
            .into_iter()
            .map(|task| self.assign_task(task))
            .collect();
        
        // Wait for all subtasks to complete
        let results = futures::future::join_all(futures).await;
        
        // Aggregate results
        self.aggregate_results(results)
    }
}
```

### 3. Message Bus

```rust
#[derive(Clone, Debug)]
pub enum Message {
    Task(ProcessingTask),
    Query(AgentQuery),
    Coordination(CoordinationMessage),
    Result(TaskResult),
    Heartbeat(HeartbeatData),
}

pub struct MessageBus {
    topics: HashMap<String, Vec<mpsc::Sender<Message>>>,
    broadcast: broadcast::Sender<Message>,
}

impl MessageBus {
    pub async fn publish(&self, topic: &str, message: Message) -> Result<()> {
        if let Some(subscribers) = self.topics.get(topic) {
            for subscriber in subscribers {
                subscriber.send(message.clone()).await?;
            }
        }
        Ok(())
    }
    
    pub async fn broadcast(&self, message: Message) -> Result<()> {
        self.broadcast.send(message)?;
        Ok(())
    }
}
```

### 4. Task Scheduler

```rust
pub struct TaskScheduler {
    strategy: SchedulingStrategy,
    queue: Arc<Mutex<TaskQueue>>,
    agent_loads: HashMap<Uuid, AgentLoad>,
}

impl TaskScheduler {
    pub fn schedule(&mut self, task: ProcessingTask) -> Result<Uuid> {
        match self.strategy {
            SchedulingStrategy::RoundRobin => self.round_robin_schedule(task),
            SchedulingStrategy::LeastLoaded => self.least_loaded_schedule(task),
            SchedulingStrategy::Capability => self.capability_based_schedule(task),
            SchedulingStrategy::Adaptive => self.adaptive_schedule(task),
        }
    }
    
    fn adaptive_schedule(&mut self, task: ProcessingTask) -> Result<Uuid> {
        // Analyze task requirements
        let requirements = self.analyze_task_requirements(&task)?;
        
        // Find best agent based on:
        // - Current load
        // - Capabilities
        // - Past performance
        // - Network distance
        let best_agent = self.find_optimal_agent(&requirements)?;
        
        self.assign_to_agent(best_agent, task)
    }
}
```

### 5. State Management

```rust
pub struct StateManager {
    local_state: HashMap<String, StateValue>,
    distributed_state: Arc<DashMap<String, StateValue>>,
    consistency_protocol: ConsistencyProtocol,
}

impl StateManager {
    pub async fn get(&self, key: &str) -> Option<StateValue> {
        // Check local cache first
        if let Some(value) = self.local_state.get(key) {
            return Some(value.clone());
        }
        
        // Check distributed state
        self.distributed_state.get(key).map(|v| v.clone())
    }
    
    pub async fn set(&mut self, key: String, value: StateValue) -> Result<()> {
        match self.consistency_protocol {
            ConsistencyProtocol::Eventual => {
                self.distributed_state.insert(key.clone(), value.clone());
                self.local_state.insert(key, value);
            }
            ConsistencyProtocol::Strong => {
                // Use consensus protocol
                self.consensus_set(key, value).await?;
            }
        }
        Ok(())
    }
}
```

## Coordination Patterns

### 1. Task Distribution Pattern

```rust
impl SwarmCoordinator {
    async fn distribute_large_document(&mut self, doc: LargeDocument) -> Result<ProcessedDocument> {
        // Split document into pages
        let pages = doc.split_into_pages();
        
        // Create page processing tasks
        let tasks: Vec<_> = pages
            .into_iter()
            .enumerate()
            .map(|(idx, page)| ProcessingTask {
                id: Uuid::new_v4(),
                task_type: TaskType::ProcessPage,
                data: page,
                priority: Priority::Normal,
                page_number: idx,
            })
            .collect();
        
        // Distribute tasks to parser agents
        let mut handles = Vec::new();
        for task in tasks {
            let agent_id = self.scheduler.schedule(task.clone())?;
            let handle = self.send_task_to_agent(agent_id, task);
            handles.push(handle);
        }
        
        // Collect results
        let results = futures::future::join_all(handles).await;
        
        // Send to aggregator agent
        let aggregator = self.get_agent_by_type(AgentType::Aggregator)?;
        aggregator.aggregate_results(results).await
    }
}
```

### 2. Pipeline Pattern

```rust
pub struct ProcessingPipeline {
    stages: Vec<PipelineStage>,
    coordinator: Arc<SwarmCoordinator>,
}

impl ProcessingPipeline {
    pub async fn process(&self, input: DocumentInput) -> Result<DocumentOutput> {
        let mut current = PipelineData::from(input);
        
        for stage in &self.stages {
            let agent = self.coordinator.get_agent_for_stage(&stage)?;
            current = agent.process_stage(current).await?;
        }
        
        Ok(current.into())
    }
}
```

### 3. Consensus Pattern

```rust
pub struct ConsensusManager {
    agents: Vec<Uuid>,
    threshold: f32, // e.g., 0.66 for 2/3 consensus
}

impl ConsensusManager {
    pub async fn reach_consensus<T: Consensus>(&self, proposal: T) -> Result<T::Decision> {
        // Send proposal to all agents
        let votes = self.collect_votes(&proposal).await?;
        
        // Count votes
        let total_votes = votes.len() as f32;
        let positive_votes = votes.iter().filter(|v| v.is_positive()).count() as f32;
        
        if positive_votes / total_votes >= self.threshold {
            Ok(proposal.accept())
        } else {
            Ok(proposal.reject())
        }
    }
}
```

## Fault Tolerance

### 1. Agent Health Monitoring

```rust
pub struct HealthMonitor {
    agents: HashMap<Uuid, AgentHealth>,
    check_interval: Duration,
}

impl HealthMonitor {
    pub async fn monitor_health(&mut self) {
        loop {
            tokio::time::sleep(self.check_interval).await;
            
            for (agent_id, health) in &mut self.agents {
                if health.last_heartbeat.elapsed() > Duration::from_secs(30) {
                    self.handle_agent_failure(*agent_id).await;
                }
            }
        }
    }
    
    async fn handle_agent_failure(&mut self, agent_id: Uuid) {
        // Mark agent as failed
        self.agents.get_mut(&agent_id).unwrap().status = HealthStatus::Failed;
        
        // Redistribute agent's tasks
        if let Some(tasks) = self.get_agent_tasks(agent_id) {
            for task in tasks {
                self.coordinator.reschedule_task(task).await;
            }
        }
        
        // Spawn replacement agent
        let agent_type = self.get_agent_type(agent_id);
        self.coordinator.spawn_agent(agent_type).await;
    }
}
```

### 2. Task Recovery

```rust
pub struct TaskRecovery {
    checkpoints: HashMap<Uuid, TaskCheckpoint>,
    recovery_strategy: RecoveryStrategy,
}

impl TaskRecovery {
    pub async fn checkpoint(&mut self, task_id: Uuid, state: TaskState) {
        let checkpoint = TaskCheckpoint {
            task_id,
            state,
            timestamp: Instant::now(),
        };
        
        self.checkpoints.insert(task_id, checkpoint);
    }
    
    pub async fn recover_task(&mut self, task_id: Uuid) -> Result<TaskState> {
        match self.checkpoints.get(&task_id) {
            Some(checkpoint) => {
                match self.recovery_strategy {
                    RecoveryStrategy::FromCheckpoint => Ok(checkpoint.state.clone()),
                    RecoveryStrategy::Restart => Ok(TaskState::default()),
                    RecoveryStrategy::Skip => Err(anyhow!("Task skipped")),
                }
            }
            None => Err(anyhow!("No checkpoint found")),
        }
    }
}
```

## Performance Optimizations

### 1. Load Balancing

```rust
pub struct LoadBalancer {
    agents: Vec<AgentLoad>,
    strategy: LoadBalancingStrategy,
}

impl LoadBalancer {
    pub fn select_agent(&self) -> Uuid {
        match self.strategy {
            LoadBalancingStrategy::LeastConnections => {
                self.agents
                    .iter()
                    .min_by_key(|a| a.active_tasks)
                    .map(|a| a.agent_id)
                    .unwrap()
            }
            LoadBalancingStrategy::WeightedRoundRobin => {
                self.weighted_selection()
            }
            LoadBalancingStrategy::ResponseTime => {
                self.agents
                    .iter()
                    .min_by_key(|a| a.avg_response_time)
                    .map(|a| a.agent_id)
                    .unwrap()
            }
        }
    }
}
```

### 2. Caching

```rust
pub struct SwarmCache {
    local_cache: HashMap<CacheKey, CacheValue>,
    distributed_cache: Arc<DashMap<CacheKey, CacheValue>>,
    eviction_policy: EvictionPolicy,
}

impl SwarmCache {
    pub async fn get_or_compute<F, Fut>(&self, key: CacheKey, compute: F) -> Result<CacheValue>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<CacheValue>>,
    {
        // Check local cache
        if let Some(value) = self.local_cache.get(&key) {
            return Ok(value.clone());
        }
        
        // Check distributed cache
        if let Some(value) = self.distributed_cache.get(&key) {
            return Ok(value.clone());
        }
        
        // Compute and cache
        let value = compute().await?;
        self.distributed_cache.insert(key.clone(), value.clone());
        Ok(value)
    }
}
```

## Testing Strategy

### Integration Tests

```rust
#[tokio::test]
async fn test_swarm_document_processing() {
    let mut coordinator = SwarmCoordinator::new(SwarmConfig::default());
    
    // Spawn multiple agents
    for _ in 0..5 {
        coordinator.spawn_agent(AgentType::Parser).await.unwrap();
    }
    for _ in 0..3 {
        coordinator.spawn_agent(AgentType::Extractor).await.unwrap();
    }
    
    // Process document
    let doc = load_test_document("large_document.pdf");
    let result = coordinator.distribute_document(doc).await.unwrap();
    
    assert_eq!(result.pages.len(), 100);
    assert!(result.processing_time < Duration::from_secs(5));
}

#[tokio::test]
async fn test_agent_failure_recovery() {
    let mut coordinator = SwarmCoordinator::new(SwarmConfig::default());
    let agent_id = coordinator.spawn_agent(AgentType::Parser).await.unwrap();
    
    // Simulate agent failure
    coordinator.kill_agent(agent_id).await;
    
    // Verify new agent spawned
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert_eq!(coordinator.agent_count(AgentType::Parser), 1);
}
```

## Deliverables

1. **Swarm Library**: `neuralflow-swarm` crate
2. **Coordination Protocol**: Message passing and state management
3. **Agent Templates**: Reusable agent implementations
4. **Monitoring Tools**: Health checks and performance metrics
5. **Test Harness**: Distributed testing framework

## Success Criteria

- ✅ Spawn and manage 50+ agents concurrently
- ✅ Process 100 documents in parallel
- ✅ Automatic failure recovery (< 1s recovery time)
- ✅ Linear scalability up to 10 nodes
- ✅ Message latency < 1ms within swarm
- ✅ Zero message loss under normal operation