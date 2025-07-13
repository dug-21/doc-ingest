use std::{
    sync::Arc,
    time::Duration,
    collections::HashMap,
};
use tokio::sync::{RwLock, Mutex};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use anyhow::{Result, anyhow};

use crate::messaging::Message;

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,     // Normal operation
    Open,       // Failing, reject requests
    HalfOpen,   // Testing if service recovered
}

/// Circuit breaker for fault tolerance
pub struct CircuitBreaker {
    id: Uuid,
    name: String,
    state: Arc<RwLock<CircuitState>>,
    failure_count: Arc<Mutex<u32>>,
    success_count: Arc<Mutex<u32>>,
    last_failure_time: Arc<RwLock<Option<DateTime<Utc>>>>,
    config: CircuitBreakerConfig,
    state_listeners: Arc<RwLock<Vec<Box<dyn StateChangeListener>>>>,
}

#[derive(Clone)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: u32,
    pub success_threshold: u32,
    pub timeout: Duration,
    pub half_open_max_attempts: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 3,
            timeout: Duration::from_secs(60),
            half_open_max_attempts: 3,
        }
    }
}

#[async_trait::async_trait]
pub trait StateChangeListener: Send + Sync {
    async fn on_state_change(&self, breaker_id: Uuid, old_state: CircuitState, new_state: CircuitState);
}

impl CircuitBreaker {
    pub fn new(name: String, config: CircuitBreakerConfig) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            state: Arc::new(RwLock::new(CircuitState::Closed)),
            failure_count: Arc::new(Mutex::new(0)),
            success_count: Arc::new(Mutex::new(0)),
            last_failure_time: Arc::new(RwLock::new(None)),
            config,
            state_listeners: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    pub async fn add_listener(&self, listener: Box<dyn StateChangeListener>) {
        let mut listeners = self.state_listeners.write().await;
        listeners.push(listener);
    }
    
    async fn change_state(&self, new_state: CircuitState) {
        let old_state = *self.state.read().await;
        if old_state != new_state {
            *self.state.write().await = new_state;
            
            // Notify listeners
            let listeners = self.state_listeners.read().await;
            for listener in listeners.iter() {
                listener.on_state_change(self.id, old_state, new_state).await;
            }
        }
    }
    
    pub async fn call<F, T>(&self, operation: F) -> Result<T>
    where
        F: std::future::Future<Output = Result<T>> + Send,
        T: Send,
    {
        let current_state = *self.state.read().await;
        
        match current_state {
            CircuitState::Open => {
                // Check if timeout has passed
                if let Some(last_failure) = *self.last_failure_time.read().await {
                    if Utc::now() - last_failure > chrono::Duration::from_std(self.config.timeout).unwrap() {
                        self.change_state(CircuitState::HalfOpen).await;
                        *self.success_count.lock().await = 0;
                    } else {
                        return Err(anyhow!("Circuit breaker is open"));
                    }
                } else {
                    return Err(anyhow!("Circuit breaker is open"));
                }
            }
            _ => {}
        }
        
        // Try the operation
        match operation.await {
            Ok(result) => {
                self.on_success().await;
                Ok(result)
            }
            Err(e) => {
                self.on_failure().await;
                Err(e)
            }
        }
    }
    
    async fn on_success(&self) {
        let mut success_count = self.success_count.lock().await;
        *success_count += 1;
        
        let current_state = *self.state.read().await;
        
        match current_state {
            CircuitState::HalfOpen => {
                if *success_count >= self.config.success_threshold {
                    self.change_state(CircuitState::Closed).await;
                    *self.failure_count.lock().await = 0;
                }
            }
            CircuitState::Closed => {
                // Reset failure count on success in closed state
                *self.failure_count.lock().await = 0;
            }
            _ => {}
        }
    }
    
    async fn on_failure(&self) {
        let mut failure_count = self.failure_count.lock().await;
        *failure_count += 1;
        *self.last_failure_time.write().await = Some(Utc::now());
        
        let current_state = *self.state.read().await;
        
        match current_state {
            CircuitState::Closed => {
                if *failure_count >= self.config.failure_threshold {
                    self.change_state(CircuitState::Open).await;
                }
            }
            CircuitState::HalfOpen => {
                // Immediately open on failure in half-open state
                self.change_state(CircuitState::Open).await;
                *self.success_count.lock().await = 0;
            }
            _ => {}
        }
    }
    
    pub async fn get_state(&self) -> CircuitState {
        *self.state.read().await
    }
    
    pub async fn reset(&self) {
        self.change_state(CircuitState::Closed).await;
        *self.failure_count.lock().await = 0;
        *self.success_count.lock().await = 0;
        *self.last_failure_time.write().await = None;
    }
}

/// Retry mechanism with exponential backoff
pub struct RetryPolicy {
    max_attempts: u32,
    initial_delay: Duration,
    max_delay: Duration,
    exponential_base: f64,
    jitter: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            exponential_base: 2.0,
            jitter: true,
        }
    }
}

impl RetryPolicy {
    pub async fn execute<F, T>(&self, operation: F) -> Result<T>
    where
        F: Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<T>> + Send>> + Send,
        T: Send,
    {
        let mut attempt = 0;
        let mut delay = self.initial_delay;
        
        loop {
            attempt += 1;
            
            match operation().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    if attempt >= self.max_attempts {
                        return Err(anyhow!("Max retry attempts ({}) exceeded: {}", self.max_attempts, e));
                    }
                    
                    // Apply jitter if enabled
                    let actual_delay = if self.jitter {
                        let jitter_range = delay.as_millis() as f64 * 0.2;
                        let jitter = (rand::random::<f64>() - 0.5) * jitter_range;
                        Duration::from_millis((delay.as_millis() as f64 + jitter) as u64)
                    } else {
                        delay
                    };
                    
                    tokio::time::sleep(actual_delay).await;
                    
                    // Calculate next delay with exponential backoff
                    delay = Duration::from_millis(
                        (delay.as_millis() as f64 * self.exponential_base) as u64
                    ).min(self.max_delay);
                }
            }
        }
    }
}

/// Bulkhead pattern for resource isolation
pub struct Bulkhead {
    name: String,
    semaphore: Arc<tokio::sync::Semaphore>,
    max_concurrent: usize,
    active_count: Arc<Mutex<usize>>,
    queue_size: Option<usize>,
    queued_count: Arc<Mutex<usize>>,
}

impl Bulkhead {
    pub fn new(name: String, max_concurrent: usize, queue_size: Option<usize>) -> Self {
        Self {
            name,
            semaphore: Arc::new(tokio::sync::Semaphore::new(max_concurrent)),
            max_concurrent,
            active_count: Arc::new(Mutex::new(0)),
            queue_size,
            queued_count: Arc::new(Mutex::new(0)),
        }
    }
    
    pub async fn execute<F, T>(&self, operation: F) -> Result<T>
    where
        F: std::future::Future<Output = Result<T>> + Send,
        T: Send,
    {
        // Check queue size if limited
        if let Some(max_queue) = self.queue_size {
            let queued = *self.queued_count.lock().await;
            if queued >= max_queue {
                return Err(anyhow!("Bulkhead queue full"));
            }
            *self.queued_count.lock().await += 1;
        }
        
        // Acquire permit
        let permit = self.semaphore.acquire().await
            .map_err(|e| anyhow!("Failed to acquire bulkhead permit: {}", e))?;
        
        // Update counters
        if self.queue_size.is_some() {
            *self.queued_count.lock().await -= 1;
        }
        *self.active_count.lock().await += 1;
        
        // Execute operation
        let result = operation.await;
        
        // Release resources
        drop(permit);
        *self.active_count.lock().await -= 1;
        
        result
    }
    
    pub async fn get_stats(&self) -> BulkheadStats {
        BulkheadStats {
            name: self.name.clone(),
            max_concurrent: self.max_concurrent,
            active: *self.active_count.lock().await,
            queued: *self.queued_count.lock().await,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BulkheadStats {
    pub name: String,
    pub max_concurrent: usize,
    pub active: usize,
    pub queued: usize,
}

/// Fault tolerance manager combining multiple patterns
pub struct FaultToleranceManager {
    circuit_breakers: Arc<RwLock<HashMap<String, Arc<CircuitBreaker>>>>,
    retry_policies: Arc<RwLock<HashMap<String, Arc<RetryPolicy>>>>,
    bulkheads: Arc<RwLock<HashMap<String, Arc<Bulkhead>>>>,
}

impl FaultToleranceManager {
    pub fn new() -> Self {
        Self {
            circuit_breakers: Arc::new(RwLock::new(HashMap::new())),
            retry_policies: Arc::new(RwLock::new(HashMap::new())),
            bulkheads: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn register_circuit_breaker(&self, name: String, config: CircuitBreakerConfig) -> Arc<CircuitBreaker> {
        let breaker = Arc::new(CircuitBreaker::new(name.clone(), config));
        let mut breakers = self.circuit_breakers.write().await;
        breakers.insert(name, breaker.clone());
        breaker
    }
    
    pub async fn register_retry_policy(&self, name: String, policy: RetryPolicy) -> Arc<RetryPolicy> {
        let policy = Arc::new(policy);
        let mut policies = self.retry_policies.write().await;
        policies.insert(name, policy.clone());
        policy
    }
    
    pub async fn register_bulkhead(&self, name: String, max_concurrent: usize, queue_size: Option<usize>) -> Arc<Bulkhead> {
        let bulkhead = Arc::new(Bulkhead::new(name.clone(), max_concurrent, queue_size));
        let mut bulkheads = self.bulkheads.write().await;
        bulkheads.insert(name, bulkhead.clone());
        bulkhead
    }
    
    pub async fn get_circuit_breaker(&self, name: &str) -> Option<Arc<CircuitBreaker>> {
        let breakers = self.circuit_breakers.read().await;
        breakers.get(name).cloned()
    }
    
    pub async fn get_retry_policy(&self, name: &str) -> Option<Arc<RetryPolicy>> {
        let policies = self.retry_policies.read().await;
        policies.get(name).cloned()
    }
    
    pub async fn get_bulkhead(&self, name: &str) -> Option<Arc<Bulkhead>> {
        let bulkheads = self.bulkheads.read().await;
        bulkheads.get(name).cloned()
    }
    
    /// Execute with full fault tolerance (circuit breaker + retry + bulkhead)
    pub async fn execute_protected<F, T>(
        &self,
        breaker_name: &str,
        retry_name: &str,
        bulkhead_name: &str,
        operation: F,
    ) -> Result<T>
    where
        F: Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<T>> + Send>> + Send + Sync,
        T: Send + 'static,
    {
        let breaker = self.get_circuit_breaker(breaker_name).await
            .ok_or_else(|| anyhow!("Circuit breaker '{}' not found", breaker_name))?;
        
        let retry = self.get_retry_policy(retry_name).await
            .ok_or_else(|| anyhow!("Retry policy '{}' not found", retry_name))?;
        
        let bulkhead = self.get_bulkhead(bulkhead_name).await
            .ok_or_else(|| anyhow!("Bulkhead '{}' not found", bulkhead_name))?;
        
        // Execute with all protections
        breaker.call(async {
            retry.execute(|| {
                let bulkhead = bulkhead.clone();
                let op = operation();
                Box::pin(async move {
                    bulkhead.execute(op).await
                })
            }).await
        }).await
    }
}