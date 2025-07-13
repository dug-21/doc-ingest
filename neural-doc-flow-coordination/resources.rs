use std::{
    sync::Arc,
    collections::HashMap,
    time::{Duration, Instant},
};
use tokio::sync::{RwLock, Mutex, Semaphore};
use uuid::Uuid;
use anyhow::{Result, anyhow};
use serde::{Serialize, Deserialize};

/// Resource types that can be managed
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceType {
    CPU,
    Memory,
    Storage,
    Network,
    Custom(String),
}

/// Resource requirement specification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResourceRequirement {
    CPU(f64),           // CPU cores (e.g., 0.5 = 50% of one core)
    Memory(u64),        // Memory in MB
    Storage(u64),       // Storage in MB
    Network(u64),       // Network bandwidth in Mbps
    Custom(String, f64), // Custom resource with amount
}

/// Resource allocation result
#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    pub id: Uuid,
    pub owner: Uuid,
    pub resources: Vec<AllocatedResource>,
    pub allocated_at: Instant,
    pub lease_duration: Option<Duration>,
}

#[derive(Debug, Clone)]
pub struct AllocatedResource {
    pub resource_type: ResourceType,
    pub amount: f64,
}

/// Resource pool for managing available resources
pub struct ResourcePool {
    pool_id: Uuid,
    resources: Arc<RwLock<HashMap<ResourceType, ResourceInfo>>>,
    allocations: Arc<RwLock<HashMap<Uuid, ResourceAllocation>>>,
    allocation_semaphores: Arc<RwLock<HashMap<ResourceType, Arc<Semaphore>>>>,
    stats: Arc<Mutex<ResourceStats>>,
}

#[derive(Clone)]
struct ResourceInfo {
    total: f64,
    available: f64,
    reserved: f64,
}

#[derive(Default)]
pub struct ResourceStats {
    pub total_allocations: u64,
    pub failed_allocations: u64,
    pub released_allocations: u64,
    pub expired_leases: u64,
}

impl ResourcePool {
    pub fn new() -> Self {
        Self {
            pool_id: Uuid::new_v4(),
            resources: Arc::new(RwLock::new(HashMap::new())),
            allocations: Arc::new(RwLock::new(HashMap::new())),
            allocation_semaphores: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(Mutex::new(ResourceStats::default())),
        }
    }
    
    /// Initialize resource pool with available resources
    pub async fn initialize(&self, resources: Vec<(ResourceType, f64)>) -> Result<()> {
        let mut pool = self.resources.write().await;
        let mut semaphores = self.allocation_semaphores.write().await;
        
        for (resource_type, amount) in resources {
            pool.insert(resource_type.clone(), ResourceInfo {
                total: amount,
                available: amount,
                reserved: 0.0,
            });
            
            // Create semaphore for resource limiting
            let permits = (amount * 100.0) as usize; // Scale to handle fractional resources
            semaphores.insert(resource_type, Arc::new(Semaphore::new(permits)));
        }
        
        Ok(())
    }
    
    /// Request resource allocation
    pub async fn allocate(&self, owner: Uuid, requirements: Vec<ResourceRequirement>, lease_duration: Option<Duration>) -> Result<ResourceAllocation> {
        // Convert requirements to resource amounts
        let mut needed_resources = HashMap::new();
        for req in &requirements {
            let (resource_type, amount) = match req {
                ResourceRequirement::CPU(cores) => (ResourceType::CPU, *cores),
                ResourceRequirement::Memory(mb) => (ResourceType::Memory, *mb as f64),
                ResourceRequirement::Storage(mb) => (ResourceType::Storage, *mb as f64),
                ResourceRequirement::Network(mbps) => (ResourceType::Network, *mbps as f64),
                ResourceRequirement::Custom(name, amount) => (ResourceType::Custom(name.clone()), *amount),
            };
            needed_resources.insert(resource_type, amount);
        }
        
        // Check availability
        {
            let pool = self.resources.read().await;
            for (resource_type, needed) in &needed_resources {
                if let Some(info) = pool.get(resource_type) {
                    if info.available < *needed {
                        let mut stats = self.stats.lock().await;
                        stats.failed_allocations += 1;
                        return Err(anyhow!(
                            "Insufficient {} resources: needed {}, available {}",
                            format!("{:?}", resource_type),
                            needed,
                            info.available
                        ));
                    }
                } else {
                    return Err(anyhow!("Resource type {:?} not available in pool", resource_type));
                }
            }
        }
        
        // Perform allocation
        let allocation_id = Uuid::new_v4();
        let mut allocated_resources = Vec::new();
        
        // Update available resources
        {
            let mut pool = self.resources.write().await;
            for (resource_type, amount) in &needed_resources {
                if let Some(info) = pool.get_mut(resource_type) {
                    info.available -= amount;
                    allocated_resources.push(AllocatedResource {
                        resource_type: resource_type.clone(),
                        amount: *amount,
                    });
                }
            }
        }
        
        let allocation = ResourceAllocation {
            id: allocation_id,
            owner,
            resources: allocated_resources,
            allocated_at: Instant::now(),
            lease_duration,
        };
        
        // Store allocation
        {
            let mut allocations = self.allocations.write().await;
            allocations.insert(allocation_id, allocation.clone());
        }
        
        // Update stats
        {
            let mut stats = self.stats.lock().await;
            stats.total_allocations += 1;
        }
        
        // Set up lease expiration if specified
        if let Some(duration) = lease_duration {
            let pool = self.clone();
            tokio::spawn(async move {
                tokio::time::sleep(duration).await;
                if pool.release(allocation_id).await.is_ok() {
                    let mut stats = pool.stats.lock().await;
                    stats.expired_leases += 1;
                }
            });
        }
        
        Ok(allocation)
    }
    
    /// Release allocated resources
    pub async fn release(&self, allocation_id: Uuid) -> Result<()> {
        let allocation = {
            let mut allocations = self.allocations.write().await;
            allocations.remove(&allocation_id)
                .ok_or_else(|| anyhow!("Allocation not found"))?
        };
        
        // Return resources to pool
        {
            let mut pool = self.resources.write().await;
            for resource in &allocation.resources {
                if let Some(info) = pool.get_mut(&resource.resource_type) {
                    info.available += resource.amount;
                }
            }
        }
        
        // Update stats
        {
            let mut stats = self.stats.lock().await;
            stats.released_allocations += 1;
        }
        
        Ok(())
    }
    
    /// Get current resource usage
    pub async fn get_usage(&self) -> HashMap<ResourceType, ResourceUsage> {
        let pool = self.resources.read().await;
        let mut usage = HashMap::new();
        
        for (resource_type, info) in pool.iter() {
            usage.insert(resource_type.clone(), ResourceUsage {
                total: info.total,
                used: info.total - info.available,
                available: info.available,
                reserved: info.reserved,
                utilization: ((info.total - info.available) / info.total * 100.0),
            });
        }
        
        usage
    }
    
    /// Reserve resources for future allocation
    pub async fn reserve(&self, requirements: Vec<ResourceRequirement>) -> Result<Uuid> {
        let reservation_id = Uuid::new_v4();
        let mut pool = self.resources.write().await;
        
        // Check and reserve resources
        for req in &requirements {
            let (resource_type, amount) = match req {
                ResourceRequirement::CPU(cores) => (ResourceType::CPU, *cores),
                ResourceRequirement::Memory(mb) => (ResourceType::Memory, *mb as f64),
                ResourceRequirement::Storage(mb) => (ResourceType::Storage, *mb as f64),
                ResourceRequirement::Network(mbps) => (ResourceType::Network, *mbps as f64),
                ResourceRequirement::Custom(name, amount) => (ResourceType::Custom(name.clone()), *amount),
            };
            
            if let Some(info) = pool.get_mut(&resource_type) {
                if info.available >= amount {
                    info.available -= amount;
                    info.reserved += amount;
                } else {
                    // Rollback previous reservations
                    for prev_req in &requirements {
                        if prev_req == req {
                            break;
                        }
                        let (prev_type, prev_amount) = match prev_req {
                            ResourceRequirement::CPU(cores) => (ResourceType::CPU, *cores),
                            ResourceRequirement::Memory(mb) => (ResourceType::Memory, *mb as f64),
                            ResourceRequirement::Storage(mb) => (ResourceType::Storage, *mb as f64),
                            ResourceRequirement::Network(mbps) => (ResourceType::Network, *mbps as f64),
                            ResourceRequirement::Custom(name, amount) => (ResourceType::Custom(name.clone()), *amount),
                        };
                        
                        if let Some(prev_info) = pool.get_mut(&prev_type) {
                            prev_info.available += prev_amount;
                            prev_info.reserved -= prev_amount;
                        }
                    }
                    
                    return Err(anyhow!("Insufficient resources for reservation"));
                }
            }
        }
        
        Ok(reservation_id)
    }
    
    /// Get resource pool statistics
    pub async fn get_stats(&self) -> ResourceStats {
        let stats = self.stats.lock().await;
        ResourceStats {
            total_allocations: stats.total_allocations,
            failed_allocations: stats.failed_allocations,
            released_allocations: stats.released_allocations,
            expired_leases: stats.expired_leases,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ResourceUsage {
    pub total: f64,
    pub used: f64,
    pub available: f64,
    pub reserved: f64,
    pub utilization: f64,
}

/// Resource monitor for tracking usage over time
pub struct ResourceMonitor {
    pool: Arc<ResourcePool>,
    history: Arc<RwLock<Vec<ResourceSnapshot>>>,
    max_history: usize,
}

#[derive(Clone)]
pub struct ResourceSnapshot {
    pub timestamp: Instant,
    pub usage: HashMap<ResourceType, ResourceUsage>,
}

impl ResourceMonitor {
    pub fn new(pool: Arc<ResourcePool>, max_history: usize) -> Self {
        Self {
            pool,
            history: Arc::new(RwLock::new(Vec::with_capacity(max_history))),
            max_history,
        }
    }
    
    pub async fn start_monitoring(&self, interval: Duration) {
        let monitor = self.clone();
        tokio::spawn(async move {
            loop {
                monitor.take_snapshot().await;
                tokio::time::sleep(interval).await;
            }
        });
    }
    
    async fn take_snapshot(&self) {
        let usage = self.pool.get_usage().await;
        let snapshot = ResourceSnapshot {
            timestamp: Instant::now(),
            usage,
        };
        
        let mut history = self.history.write().await;
        if history.len() >= self.max_history {
            history.remove(0);
        }
        history.push(snapshot);
    }
    
    pub async fn get_history(&self) -> Vec<ResourceSnapshot> {
        let history = self.history.read().await;
        history.clone()
    }
    
    pub async fn get_average_utilization(&self, resource_type: &ResourceType, duration: Duration) -> Option<f64> {
        let history = self.history.read().await;
        let cutoff = Instant::now() - duration;
        
        let recent_snapshots: Vec<&ResourceSnapshot> = history.iter()
            .filter(|s| s.timestamp > cutoff)
            .collect();
        
        if recent_snapshots.is_empty() {
            return None;
        }
        
        let total_utilization: f64 = recent_snapshots.iter()
            .filter_map(|s| s.usage.get(resource_type))
            .map(|u| u.utilization)
            .sum();
        
        Some(total_utilization / recent_snapshots.len() as f64)
    }
}

impl Clone for ResourcePool {
    fn clone(&self) -> Self {
        Self {
            pool_id: self.pool_id,
            resources: self.resources.clone(),
            allocations: self.allocations.clone(),
            allocation_semaphores: self.allocation_semaphores.clone(),
            stats: self.stats.clone(),
        }
    }
}

impl Clone for ResourceMonitor {
    fn clone(&self) -> Self {
        Self {
            pool: self.pool.clone(),
            history: self.history.clone(),
            max_history: self.max_history,
        }
    }
}