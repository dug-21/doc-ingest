//! Memory profiling and monitoring tools for neural document flow
//!
//! This module provides comprehensive memory profiling capabilities
//! to ensure documents stay under 2MB memory usage.

use crate::memory::*;
use crate::optimized_types::*;
use crate::error::{Result, NeuralDocFlowError};
use std::sync::{Arc, Weak};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use chrono::{DateTime, Utc};
use tokio::sync::RwLock;
use parking_lot::Mutex;
use uuid::Uuid;
use serde::{Serialize, Deserialize};

/// Memory profiler for tracking document processing memory usage
pub struct MemoryProfiler {
    /// Active allocations
    allocations: Arc<RwLock<HashMap<Uuid, AllocationInfo>>>,
    
    /// Memory usage history
    history: Arc<Mutex<VecDeque<MemorySnapshot>>>,
    
    /// Configuration
    config: ProfilerConfig,
    
    /// Statistics
    stats: Arc<RwLock<ProfilerStats>>,
    
    /// Alert handlers
    alert_handlers: Arc<RwLock<Vec<Box<dyn AlertHandler + Send + Sync>>>>,
}

/// Allocation information
#[derive(Debug, Clone)]
struct AllocationInfo {
    id: Uuid,
    size: usize,
    allocation_type: AllocationType,
    allocated_at: Instant,
    call_stack: Vec<String>,
    document_id: Option<Uuid>,
}

/// Types of memory allocations
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AllocationType {
    Document,
    Text,
    Image,
    Table,
    ProcessingBuffer,
    Cache,
    Other(String),
}

/// Memory snapshot at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    pub timestamp: DateTime<Utc>,
    pub total_usage: usize,
    pub allocation_count: usize,
    pub largest_allocation: usize,
    pub allocations_by_type: HashMap<AllocationType, usize>,
    pub document_count: usize,
    pub cache_usage: usize,
}

/// Profiler configuration
#[derive(Debug, Clone)]
pub struct ProfilerConfig {
    /// Maximum history entries to keep
    pub max_history_entries: usize,
    
    /// Memory usage warning threshold (0.0 to 1.0)
    pub warning_threshold: f64,
    
    /// Memory usage critical threshold (0.0 to 1.0)
    pub critical_threshold: f64,
    
    /// Enable call stack tracking
    pub track_call_stacks: bool,
    
    /// Snapshot interval
    pub snapshot_interval: Duration,
    
    /// Enable memory leak detection
    pub detect_leaks: bool,
    
    /// Leak detection threshold (allocations older than this)
    pub leak_threshold: Duration,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            max_history_entries: 1000,
            warning_threshold: 0.8,
            critical_threshold: 0.95,
            track_call_stacks: false,
            snapshot_interval: Duration::from_secs(10),
            detect_leaks: true,
            leak_threshold: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// Profiler statistics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ProfilerStats {
    pub total_allocations: u64,
    pub total_deallocations: u64,
    pub peak_memory_usage: usize,
    pub average_memory_usage: f64,
    pub memory_leak_count: u64,
    pub alert_count: u64,
    pub uptime: Duration,
    pub efficiency_score: f64,
}

/// Alert handler trait
pub trait AlertHandler {
    fn handle_alert(&self, alert: MemoryAlert);
}

/// Memory alert types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryAlert {
    WarningThresholdExceeded {
        usage: usize,
        limit: usize,
        ratio: f64,
    },
    CriticalThresholdExceeded {
        usage: usize,
        limit: usize,
        ratio: f64,
    },
    MemoryLeakDetected {
        allocation_id: Uuid,
        size: usize,
        age: Duration,
    },
    LargeAllocation {
        allocation_id: Uuid,
        size: usize,
        allocation_type: AllocationType,
    },
    DocumentSizeExceeded {
        document_id: Uuid,
        size: usize,
        limit: usize,
    },
}

impl MemoryProfiler {
    /// Create new memory profiler
    pub fn new(config: ProfilerConfig) -> Self {
        Self {
            allocations: Arc::new(RwLock::new(HashMap::new())),
            history: Arc::new(Mutex::new(VecDeque::new())),
            config,
            stats: Arc::new(RwLock::new(ProfilerStats::default())),
            alert_handlers: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    /// Record memory allocation
    pub async fn record_allocation(
        &self,
        id: Uuid,
        size: usize,
        allocation_type: AllocationType,
        document_id: Option<Uuid>,
    ) -> Result<()> {
        let call_stack = if self.config.track_call_stacks {
            self.capture_call_stack()
        } else {
            Vec::new()
        };
        
        let allocation = AllocationInfo {
            id,
            size,
            allocation_type: allocation_type.clone(),
            allocated_at: Instant::now(),
            call_stack,
            document_id,
        };
        
        self.allocations.write().await.insert(id, allocation);
        
        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.total_allocations += 1;
        }
        
        // Check for large allocations
        if size > 100 * 1024 { // 100KB threshold
            self.send_alert(MemoryAlert::LargeAllocation {
                allocation_id: id,
                size,
                allocation_type,
            }).await;
        }
        
        // Take snapshot if needed
        self.maybe_take_snapshot().await;
        
        Ok(())
    }
    
    /// Record memory deallocation
    pub async fn record_deallocation(&self, id: Uuid) {
        self.allocations.write().await.remove(&id);
        
        let mut stats = self.stats.write().await;
        stats.total_deallocations += 1;
    }
    
    /// Get current memory usage
    pub async fn current_usage(&self) -> usize {
        self.allocations.read().await
            .values()
            .map(|alloc| alloc.size)
            .sum()
    }
    
    /// Get memory usage by type
    pub async fn usage_by_type(&self) -> HashMap<AllocationType, usize> {
        let mut usage = HashMap::new();
        
        for allocation in self.allocations.read().await.values() {
            *usage.entry(allocation.allocation_type.clone()).or_insert(0) += allocation.size;
        }
        
        usage
    }
    
    /// Get memory usage by document
    pub async fn usage_by_document(&self) -> HashMap<Uuid, usize> {
        let mut usage = HashMap::new();
        
        for allocation in self.allocations.read().await.values() {
            if let Some(doc_id) = allocation.document_id {
                *usage.entry(doc_id).or_insert(0) += allocation.size;
            }
        }
        
        usage
    }
    
    /// Check for memory leaks
    pub async fn detect_leaks(&self) -> Vec<Uuid> {
        if !self.config.detect_leaks {
            return Vec::new();
        }
        
        let now = Instant::now();
        let mut leaks = Vec::new();
        
        for allocation in self.allocations.read().await.values() {
            let age = now.duration_since(allocation.allocated_at);
            if age > self.config.leak_threshold {
                leaks.push(allocation.id);
                
                // Send leak alert
                self.send_alert(MemoryAlert::MemoryLeakDetected {
                    allocation_id: allocation.id,
                    size: allocation.size,
                    age,
                }).await;
            }
        }
        
        // Update leak count
        {
            let mut stats = self.stats.write().await;
            stats.memory_leak_count += leaks.len() as u64;
        }
        
        leaks
    }
    
    /// Take memory snapshot
    pub async fn take_snapshot(&self) -> MemorySnapshot {
        let allocations = self.allocations.read().await;
        
        let total_usage = allocations.values().map(|a| a.size).sum();
        let allocation_count = allocations.len();
        let largest_allocation = allocations.values().map(|a| a.size).max().unwrap_or(0);
        
        let mut allocations_by_type = HashMap::new();
        let mut document_count = 0;
        let mut cache_usage = 0;
        
        for allocation in allocations.values() {
            *allocations_by_type.entry(allocation.allocation_type.clone()).or_insert(0) += allocation.size;
            
            if allocation.document_id.is_some() {
                document_count += 1;
            }
            
            if matches!(allocation.allocation_type, AllocationType::Cache) {
                cache_usage += allocation.size;
            }
        }
        
        let snapshot = MemorySnapshot {
            timestamp: Utc::now(),
            total_usage,
            allocation_count,
            largest_allocation,
            allocations_by_type,
            document_count,
            cache_usage,
        };
        
        // Add to history
        {
            let mut history = self.history.lock();
            history.push_back(snapshot.clone());
            
            // Maintain history size limit
            while history.len() > self.config.max_history_entries {
                history.pop_front();
            }
        }
        
        // Update peak usage
        {
            let mut stats = self.stats.write().await;
            if total_usage > stats.peak_memory_usage {
                stats.peak_memory_usage = total_usage;
            }
            
            // Update average
            let count = stats.total_allocations as f64;
            if count > 0.0 {
                stats.average_memory_usage = (stats.average_memory_usage * (count - 1.0) + 
                    total_usage as f64) / count;
            }
        }
        
        snapshot
    }
    
    /// Get memory usage history
    pub fn get_history(&self) -> Vec<MemorySnapshot> {
        self.history.lock().iter().cloned().collect()
    }
    
    /// Get profiler statistics
    pub async fn get_stats(&self) -> ProfilerStats {
        self.stats.read().await.clone()
    }
    
    /// Generate memory report
    pub async fn generate_report(&self) -> MemoryReport {
        let current_usage = self.current_usage().await;
        let usage_by_type = self.usage_by_type().await;
        let usage_by_document = self.usage_by_document().await;
        let leaks = self.detect_leaks().await;
        let stats = self.get_stats().await;
        let history = self.get_history();
        
        MemoryReport {
            timestamp: Utc::now(),
            current_usage,
            usage_by_type,
            usage_by_document,
            potential_leaks: leaks,
            statistics: stats,
            recent_snapshots: history.into_iter().rev().take(10).collect(),
            recommendations: self.generate_recommendations(current_usage, &usage_by_type).await,
        }
    }
    
    /// Add alert handler
    pub async fn add_alert_handler(&self, handler: Box<dyn AlertHandler + Send + Sync>) {
        self.alert_handlers.write().await.push(handler);
    }
    
    /// Check memory thresholds and send alerts
    pub async fn check_thresholds(&self, limit: usize) {
        let current_usage = self.current_usage().await;
        let ratio = current_usage as f64 / limit as f64;
        
        if ratio >= self.config.critical_threshold {
            self.send_alert(MemoryAlert::CriticalThresholdExceeded {
                usage: current_usage,
                limit,
                ratio,
            }).await;
        } else if ratio >= self.config.warning_threshold {
            self.send_alert(MemoryAlert::WarningThresholdExceeded {
                usage: current_usage,
                limit,
                ratio,
            }).await;
        }
    }
    
    /// Monitor document memory usage
    pub async fn monitor_document(&self, doc: &OptimizedDocument, limit: usize) -> Result<()> {
        let doc_usage = doc.memory_usage().await;
        
        if doc_usage > limit {
            self.send_alert(MemoryAlert::DocumentSizeExceeded {
                document_id: doc.id,
                size: doc_usage,
                limit,
            }).await;
        }
        
        Ok(())
    }
    
    // Private helper methods
    
    async fn maybe_take_snapshot(&self) {
        // Simple interval-based snapshots
        // In a real implementation, this would use a timer
        let history = self.history.lock();
        let should_snapshot = history.is_empty() || 
            history.back().unwrap().timestamp.elapsed() > self.config.snapshot_interval;
        drop(history);
        
        if should_snapshot {
            self.take_snapshot().await;
        }
    }
    
    async fn send_alert(&self, alert: MemoryAlert) {
        let handlers = self.alert_handlers.read().await;
        for handler in handlers.iter() {
            handler.handle_alert(alert.clone());
        }
        
        // Update alert count
        {
            let mut stats = self.stats.write().await;
            stats.alert_count += 1;
        }
    }
    
    fn capture_call_stack(&self) -> Vec<String> {
        // Simplified call stack capture
        // In a real implementation, this would use backtrace crate
        vec!["capture_call_stack".to_string()]
    }
    
    async fn generate_recommendations(&self, current_usage: usize, usage_by_type: &HashMap<AllocationType, usize>) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        // Large allocation recommendations
        if current_usage > 1024 * 1024 { // 1MB
            recommendations.push("Consider using streaming processing for large documents".to_string());
        }
        
        // Type-specific recommendations
        for (alloc_type, usage) in usage_by_type {
            match alloc_type {
                AllocationType::Image if *usage > 500 * 1024 => {
                    recommendations.push("Consider compressing or lazy-loading images".to_string());
                }
                AllocationType::Table if *usage > 200 * 1024 => {
                    recommendations.push("Consider compressing large table data".to_string());
                }
                AllocationType::Cache if *usage > 100 * 1024 => {
                    recommendations.push("Consider reducing cache size or implementing LRU eviction".to_string());
                }
                _ => {}
            }
        }
        
        recommendations
    }
}

/// Memory report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryReport {
    pub timestamp: DateTime<Utc>,
    pub current_usage: usize,
    pub usage_by_type: HashMap<AllocationType, usize>,
    pub usage_by_document: HashMap<Uuid, usize>,
    pub potential_leaks: Vec<Uuid>,
    pub statistics: ProfilerStats,
    pub recent_snapshots: Vec<MemorySnapshot>,
    pub recommendations: Vec<String>,
}

/// Console alert handler
pub struct ConsoleAlertHandler;

impl AlertHandler for ConsoleAlertHandler {
    fn handle_alert(&self, alert: MemoryAlert) {
        match alert {
            MemoryAlert::WarningThresholdExceeded { usage, limit, ratio } => {
                eprintln!("⚠️  Memory warning: {}% usage ({}/{} bytes)", 
                    (ratio * 100.0) as u32, usage, limit);
            }
            MemoryAlert::CriticalThresholdExceeded { usage, limit, ratio } => {
                eprintln!("🚨 Memory critical: {}% usage ({}/{} bytes)", 
                    (ratio * 100.0) as u32, usage, limit);
            }
            MemoryAlert::MemoryLeakDetected { allocation_id, size, age } => {
                eprintln!("🔍 Potential memory leak: {} bytes, age: {:?}, ID: {}", 
                    size, age, allocation_id);
            }
            MemoryAlert::LargeAllocation { allocation_id, size, allocation_type } => {
                eprintln!("📊 Large allocation: {} bytes ({:?}), ID: {}", 
                    size, allocation_type, allocation_id);
            }
            MemoryAlert::DocumentSizeExceeded { document_id, size, limit } => {
                eprintln!("📄 Document size exceeded: {} bytes > {} limit, ID: {}", 
                    size, limit, document_id);
            }
        }
    }
}

/// File alert handler
pub struct FileAlertHandler {
    file_path: std::path::PathBuf,
}

impl FileAlertHandler {
    pub fn new(file_path: std::path::PathBuf) -> Self {
        Self { file_path }
    }
}

impl AlertHandler for FileAlertHandler {
    fn handle_alert(&self, alert: MemoryAlert) {
        let alert_json = serde_json::to_string(&alert).unwrap_or_else(|_| "Failed to serialize alert".to_string());
        
        if let Err(e) = std::fs::write(&self.file_path, format!("{}
", alert_json)) {
            eprintln!("Failed to write alert to file: {}", e);
        }
    }
}

/// Memory optimization analyzer
pub struct MemoryOptimizer {
    profiler: Arc<MemoryProfiler>,
}

impl MemoryOptimizer {
    /// Create new memory optimizer
    pub fn new(profiler: Arc<MemoryProfiler>) -> Self {
        Self { profiler }
    }
    
    /// Analyze memory usage and suggest optimizations
    pub async fn analyze_and_optimize(&self) -> OptimizationSuggestions {
        let report = self.profiler.generate_report().await;
        let mut suggestions = OptimizationSuggestions {
            immediate_actions: Vec::new(),
            long_term_improvements: Vec::new(),
            estimated_savings: 0,
        };
        
        // Analyze current usage patterns
        if report.current_usage > 1500 * 1024 { // 1.5MB
            suggestions.immediate_actions.push(
                "Memory usage is approaching 2MB limit - consider document compaction".to_string()
            );
        }
        
        // Analyze usage by type
        for (alloc_type, usage) in &report.usage_by_type {
            match alloc_type {
                AllocationType::Image if *usage > 800 * 1024 => {
                    suggestions.immediate_actions.push(
                        "Image data using significant memory - implement lazy loading".to_string()
                    );
                    suggestions.estimated_savings += usage / 2; // Assume 50% savings
                }
                AllocationType::Table if *usage > 400 * 1024 => {
                    suggestions.immediate_actions.push(
                        "Table data using significant memory - enable compression".to_string()
                    );
                    suggestions.estimated_savings += usage / 4; // Assume 25% savings
                }
                AllocationType::Cache if *usage > 200 * 1024 => {
                    suggestions.immediate_actions.push(
                        "Cache using significant memory - implement LRU eviction".to_string()
                    );
                    suggestions.estimated_savings += usage / 3; // Assume 33% savings
                }
                _ => {}
            }
        }
        
        // Check for memory leaks
        if !report.potential_leaks.is_empty() {
            suggestions.immediate_actions.push(
                format!("Found {} potential memory leaks - investigate and fix", 
                    report.potential_leaks.len())
            );
        }
        
        // Long-term improvements
        if report.statistics.efficiency_score < 0.8 {
            suggestions.long_term_improvements.push(
                "Consider implementing arena allocation for temporary objects".to_string()
            );
        }
        
        suggestions.long_term_improvements.push(
            "Implement zero-copy operations where possible".to_string()
        );
        
        suggestions.long_term_improvements.push(
            "Consider using memory-mapped files for large documents".to_string()
        );
        
        suggestions
    }
}

/// Optimization suggestions
#[derive(Debug, Clone)]
pub struct OptimizationSuggestions {
    pub immediate_actions: Vec<String>,
    pub long_term_improvements: Vec<String>,
    pub estimated_savings: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_memory_profiler() {
        let profiler = MemoryProfiler::new(ProfilerConfig::default());
        
        let id = Uuid::new_v4();
        profiler.record_allocation(
            id,
            1024,
            AllocationType::Document,
            Some(Uuid::new_v4()),
        ).await.unwrap();
        
        let usage = profiler.current_usage().await;
        assert_eq!(usage, 1024);
        
        profiler.record_deallocation(id).await;
        let usage = profiler.current_usage().await;
        assert_eq!(usage, 0);
    }
    
    #[tokio::test]
    async fn test_memory_snapshot() {
        let profiler = MemoryProfiler::new(ProfilerConfig::default());
        
        let snapshot = profiler.take_snapshot().await;
        assert_eq!(snapshot.total_usage, 0);
        assert_eq!(snapshot.allocation_count, 0);
    }
    
    #[test]
    fn test_console_alert_handler() {
        let handler = ConsoleAlertHandler;
        
        handler.handle_alert(MemoryAlert::WarningThresholdExceeded {
            usage: 1024,
            limit: 2048,
            ratio: 0.5,
        });
        
        // Test passes if no panic occurs
    }
}
