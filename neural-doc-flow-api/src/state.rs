//! Application state management

use std::sync::Arc;
use tokio::sync::RwLock;
use sqlx::{Pool, Sqlite, SqlitePool};
use dashmap::DashMap;
use parking_lot::Mutex;
use governor::{Quota, RateLimiter};
use std::num::NonZeroU32;
use std::time::Duration;

use neural_doc_flow_core::DocumentProcessor;
use neural_doc_flow_coordination::agents::controller::CoordinationController;
use crate::config::ServerConfig;
use crate::jobs::{JobQueue, JobId, JobStatus};
use crate::auth::AuthManager;
use crate::monitoring::MetricsCollector;

/// Global application state
#[derive(Clone)]
pub struct AppState {
    /// Server configuration
    pub config: ServerConfig,
    
    /// Database connection pool
    pub db: Pool<Sqlite>,
    
    /// Document processor instance
    pub processor: Arc<DocumentProcessor>,
    
    /// DAA coordination controller
    pub coordinator: Arc<CoordinationController>,
    
    /// Job queue and tracking
    pub job_queue: Arc<JobQueue>,
    pub job_statuses: Arc<DashMap<JobId, JobStatus>>,
    
    /// Authentication manager
    pub auth: Arc<AuthManager>,
    
    /// Rate limiters by IP address
    pub rate_limiters: Arc<DashMap<String, Arc<RateLimiter<String, dashmap::DashMap<String, governor::state::InMemoryState>, governor::clock::DefaultClock>>>>,
    
    /// Metrics collector
    pub metrics: Arc<Mutex<MetricsCollector>>,
    
    /// Cache for processed documents
    pub document_cache: Arc<DashMap<String, CachedResult>>,
    
    /// Active processing jobs
    pub active_jobs: Arc<RwLock<std::collections::HashMap<JobId, tokio::task::JoinHandle<()>>>>,
    
    /// System health status
    pub health_status: Arc<RwLock<HealthStatus>>,
}

/// Cached processing result
#[derive(Debug, Clone)]
pub struct CachedResult {
    pub result: neural_doc_flow_processors::ProcessingResult,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub access_count: std::sync::atomic::AtomicU64,
}

/// System health status
#[derive(Debug, Clone)]
pub struct HealthStatus {
    pub overall_status: ServiceStatus,
    pub database_status: ServiceStatus,
    pub processor_status: ServiceStatus,
    pub coordinator_status: ServiceStatus,
    pub last_check: chrono::DateTime<chrono::Utc>,
    pub uptime: Duration,
    pub version: String,
}

/// Service status enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum ServiceStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

impl AppState {
    /// Create new application state
    pub async fn new(config: ServerConfig) -> Result<Self, Box<dyn std::error::Error>> {
        tracing::info!("Initializing application state...");

        // Initialize database
        let db = SqlitePool::connect_with(config.database_options()).await?;
        sqlx::migrate!("./migrations").run(&db).await?;
        tracing::info!("Database initialized");

        // Initialize document processor
        let processing_config = config.processing_config();
        let processor = Arc::new(DocumentProcessor::new(processing_config)?);
        tracing::info!("Document processor initialized");

        // Initialize DAA coordination
        let coordinator = Arc::new(CoordinationController::new().await?);
        tracing::info!("DAA coordination controller initialized");

        // Initialize job queue
        let job_queue = Arc::new(JobQueue::new(
            config.jobs.queue_size,
            config.jobs.workers,
        ).await?);
        tracing::info!("Job queue initialized");

        // Initialize authentication manager
        let auth = Arc::new(AuthManager::new(config.jwt_secret.clone(), db.clone()).await?);
        tracing::info!("Authentication manager initialized");

        // Initialize metrics collector
        let metrics = Arc::new(Mutex::new(MetricsCollector::new()));
        tracing::info!("Metrics collector initialized");

        // Initialize health status
        let health_status = Arc::new(RwLock::new(HealthStatus {
            overall_status: ServiceStatus::Healthy,
            database_status: ServiceStatus::Healthy,
            processor_status: ServiceStatus::Healthy,
            coordinator_status: ServiceStatus::Healthy,
            last_check: chrono::Utc::now(),
            uptime: Duration::from_secs(0),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }));

        Ok(Self {
            config,
            db,
            processor,
            coordinator,
            job_queue,
            job_statuses: Arc::new(DashMap::new()),
            auth,
            rate_limiters: Arc::new(DashMap::new()),
            metrics,
            document_cache: Arc::new(DashMap::new()),
            active_jobs: Arc::new(RwLock::new(std::collections::HashMap::new())),
            health_status,
        })
    }

    /// Get or create rate limiter for IP address
    pub fn get_rate_limiter(&self, ip: &str) -> Arc<RateLimiter<String, dashmap::DashMap<String, governor::state::InMemoryState>, governor::clock::DefaultClock>> {
        self.rate_limiters
            .entry(ip.to_string())
            .or_insert_with(|| {
                let quota = Quota::per_minute(NonZeroU32::new(self.config.rate_limit_rpm).unwrap());
                Arc::new(RateLimiter::keyed(quota))
            })
            .clone()
    }

    /// Check if a document is cached
    pub fn get_cached_result(&self, cache_key: &str) -> Option<neural_doc_flow_processors::ProcessingResult> {
        if let Some(cached) = self.document_cache.get(cache_key) {
            // Check if cache entry is still valid (e.g., within 1 hour)
            let age = chrono::Utc::now() - cached.timestamp;
            if age.num_seconds() < 3600 {
                cached.access_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Some(cached.result.clone());
            } else {
                // Remove expired entry
                self.document_cache.remove(cache_key);
            }
        }
        None
    }

    /// Cache a processing result
    pub fn cache_result(&self, cache_key: String, result: neural_doc_flow_processors::ProcessingResult) {
        let cached = CachedResult {
            result,
            timestamp: chrono::Utc::now(),
            access_count: std::sync::atomic::AtomicU64::new(0),
        };
        self.document_cache.insert(cache_key, cached);

        // Cleanup old cache entries if cache is getting too large
        if self.document_cache.len() > 1000 {
            self.cleanup_cache();
        }
    }

    /// Cleanup old cache entries
    fn cleanup_cache(&self) {
        let cutoff = chrono::Utc::now() - chrono::Duration::hours(1);
        self.document_cache.retain(|_, cached| cached.timestamp > cutoff);
    }

    /// Generate cache key for document
    pub fn generate_cache_key(&self, content_hash: &str, config: &neural_doc_flow_core::ProcessingConfig) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        content_hash.hash(&mut hasher);
        config.neural_enabled().hash(&mut hasher);
        config.security_level().hash(&mut hasher);
        // Add other relevant config parameters
        
        format!("doc_{}_{:x}", content_hash, hasher.finish())
    }

    /// Update health status
    pub async fn update_health_status(&self) {
        let mut status = self.health_status.write().await;
        
        // Check database health
        status.database_status = match sqlx::query("SELECT 1").fetch_one(&self.db).await {
            Ok(_) => ServiceStatus::Healthy,
            Err(_) => ServiceStatus::Unhealthy,
        };

        // Check processor health
        status.processor_status = match self.processor.get_statistics() {
            Ok(_) => ServiceStatus::Healthy,
            Err(_) => ServiceStatus::Degraded,
        };

        // Check coordinator health
        status.coordinator_status = if self.coordinator.is_healthy().await {
            ServiceStatus::Healthy
        } else {
            ServiceStatus::Degraded
        };

        // Determine overall status
        status.overall_status = match (
            &status.database_status,
            &status.processor_status,
            &status.coordinator_status,
        ) {
            (ServiceStatus::Healthy, ServiceStatus::Healthy, ServiceStatus::Healthy) => ServiceStatus::Healthy,
            (ServiceStatus::Unhealthy, _, _) | (_, ServiceStatus::Unhealthy, _) | (_, _, ServiceStatus::Unhealthy) => ServiceStatus::Unhealthy,
            _ => ServiceStatus::Degraded,
        };

        status.last_check = chrono::Utc::now();
    }

    /// Get system statistics
    pub async fn get_statistics(&self) -> Result<SystemStatistics, Box<dyn std::error::Error>> {
        let health = self.health_status.read().await;
        let processor_stats = self.processor.get_statistics()?;
        let job_stats = self.job_queue.get_statistics().await;
        let metrics = self.metrics.lock().collect_current_metrics();

        Ok(SystemStatistics {
            health_status: health.clone(),
            processor_stats,
            job_stats,
            cache_size: self.document_cache.len(),
            active_jobs: self.active_jobs.read().await.len(),
            rate_limiter_count: self.rate_limiters.len(),
            memory_usage: metrics.memory_usage_bytes,
            cpu_usage: metrics.cpu_usage_percent,
        })
    }

    /// Shutdown gracefully
    pub async fn shutdown(&self) -> Result<(), Box<dyn std::error::Error>> {
        tracing::info!("Starting graceful shutdown...");

        // Cancel all active jobs
        let mut active_jobs = self.active_jobs.write().await;
        for (job_id, handle) in active_jobs.drain() {
            tracing::info!("Cancelling job: {}", job_id);
            handle.abort();
        }

        // Shutdown job queue
        self.job_queue.shutdown().await?;

        // Shutdown coordinator
        self.coordinator.shutdown().await?;

        // Close database connections
        self.db.close().await;

        tracing::info!("Graceful shutdown completed");
        Ok(())
    }
}

/// System statistics aggregation
#[derive(Debug, Clone)]
pub struct SystemStatistics {
    pub health_status: HealthStatus,
    pub processor_stats: neural_doc_flow_core::ProcessorStatistics,
    pub job_stats: crate::jobs::JobStatistics,
    pub cache_size: usize,
    pub active_jobs: usize,
    pub rate_limiter_count: usize,
    pub memory_usage: u64,
    pub cpu_usage: f64,
}