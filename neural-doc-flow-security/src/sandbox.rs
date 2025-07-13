//! Plugin sandboxing and isolation

use neural_doc_flow_core::ProcessingError;
use nix::unistd::Pid;
use nix::sys::signal::{self, Signal};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;
use tokio::time::timeout;

/// Sandbox manager for plugin isolation
pub struct SandboxManager {
    config: SandboxConfig,
    active_sandboxes: HashMap<String, SandboxInstance>,
}

/// Sandbox configuration
#[derive(Clone)]
pub struct SandboxConfig {
    pub memory_limit_mb: usize,
    pub cpu_quota_percent: f32,
    pub timeout_seconds: u64,
    pub allowed_paths: Vec<PathBuf>,
    pub blocked_syscalls: Vec<String>,
}

/// Active sandbox instance
pub struct SandboxInstance {
    plugin_id: String,
    process_id: u32,
    resource_limits: ResourceLimits,
    start_time: std::time::Instant,
}

/// Resource limits for sandboxed processes
#[derive(Clone)]
pub struct ResourceLimits {
    pub memory_bytes: usize,
    pub cpu_shares: u32,
    pub max_files: usize,
    pub max_processes: usize,
}

/// Sandbox execution result
pub struct SandboxResult<T> {
    pub output: T,
    pub resource_usage: ResourceUsage,
    pub execution_time: Duration,
}

/// Resource usage statistics
pub struct ResourceUsage {
    pub memory_peak_bytes: usize,
    pub cpu_time_seconds: f32,
    pub files_opened: usize,
}

impl SandboxManager {
    /// Create a new sandbox manager
    pub fn new() -> Result<Self, ProcessingError> {
        let config = SandboxConfig {
            memory_limit_mb: 500,
            cpu_quota_percent: 50.0,
            timeout_seconds: 30,
            allowed_paths: vec![PathBuf::from("/tmp")],
            blocked_syscalls: vec![
                "fork".to_string(),
                "execve".to_string(),
                "socket".to_string(),
            ],
        };
        
        Ok(Self {
            config,
            active_sandboxes: HashMap::new(),
        })
    }
    
    /// Execute a plugin in a sandbox
    pub async fn execute_sandboxed<F, T>(
        &mut self,
        plugin_id: &str,
        operation: F,
    ) -> Result<SandboxResult<T>, ProcessingError>
    where
        F: FnOnce() -> Result<T, ProcessingError> + Send + 'static,
        T: Send + 'static,
    {
        let start_time = std::time::Instant::now();
        
        // Create sandbox instance
        let sandbox = self.create_sandbox(plugin_id)?;
        let resource_limits = sandbox.resource_limits.clone();
        
        // Execute with timeout
        let result = timeout(
            Duration::from_secs(self.config.timeout_seconds),
            tokio::task::spawn_blocking(move || {
                // Apply resource limits
                Self::apply_resource_limits(&resource_limits)?;
                
                // Execute operation
                operation()
            }),
        )
        .await
        .map_err(|_| ProcessingError::Timeout)?
        .map_err(|e| ProcessingError::SandboxError(e.to_string()))??;
        
        // Collect resource usage
        let resource_usage = self.collect_resource_usage(&sandbox)?;
        
        // Clean up sandbox
        self.cleanup_sandbox(plugin_id)?;
        
        Ok(SandboxResult {
            output: result,
            resource_usage,
            execution_time: start_time.elapsed(),
        })
    }
    
    /// Create a new sandbox instance
    fn create_sandbox(&mut self, plugin_id: &str) -> Result<SandboxInstance, ProcessingError> {
        let resource_limits = ResourceLimits {
            memory_bytes: self.config.memory_limit_mb * 1024 * 1024,
            cpu_shares: (self.config.cpu_quota_percent * 10.24) as u32,
            max_files: 100,
            max_processes: 10,
        };
        
        let sandbox = SandboxInstance {
            plugin_id: plugin_id.to_string(),
            process_id: std::process::id(),
            resource_limits,
            start_time: std::time::Instant::now(),
        };
        
        self.active_sandboxes.insert(plugin_id.to_string(), sandbox);
        
        Ok(self.active_sandboxes[plugin_id].clone())
    }
    
    /// Apply resource limits to current process
    fn apply_resource_limits(limits: &ResourceLimits) -> Result<(), ProcessingError> {
        // Note: In production, use cgroups v2 or similar for proper resource limiting
        // This is a simplified implementation
        
        // Set memory limit using rlimit
        #[cfg(unix)]
        {
            use nix::sys::resource::{setrlimit, Resource};
            
            // Set memory limit
            setrlimit(
                Resource::RLIMIT_AS,
                limits.memory_bytes as u64,
                limits.memory_bytes as u64,
            ).map_err(|e| ProcessingError::SandboxError(format!("Failed to set memory limit: {}", e)))?;
            
            // Set file descriptor limit
            setrlimit(
                Resource::RLIMIT_NOFILE,
                limits.max_files as u64,
                limits.max_files as u64,
            ).map_err(|e| ProcessingError::SandboxError(format!("Failed to set file limit: {}", e)))?;
            
            // Set process limit
            setrlimit(
                Resource::RLIMIT_NPROC,
                limits.max_processes as u64,
                limits.max_processes as u64,
            ).map_err(|e| ProcessingError::SandboxError(format!("Failed to set process limit: {}", e)))?;
        }
        
        Ok(())
    }
    
    /// Collect resource usage statistics
    fn collect_resource_usage(&self, sandbox: &SandboxInstance) -> Result<ResourceUsage, ProcessingError> {
        // In production, read from /proc or cgroup statistics
        Ok(ResourceUsage {
            memory_peak_bytes: 100 * 1024 * 1024, // Placeholder
            cpu_time_seconds: sandbox.start_time.elapsed().as_secs_f32(),
            files_opened: 10, // Placeholder
        })
    }
    
    /// Clean up sandbox after execution
    fn cleanup_sandbox(&mut self, plugin_id: &str) -> Result<(), ProcessingError> {
        self.active_sandboxes.remove(plugin_id);
        Ok(())
    }
    
    /// Kill a misbehaving sandbox
    pub fn kill_sandbox(&mut self, plugin_id: &str) -> Result<(), ProcessingError> {
        if let Some(sandbox) = self.active_sandboxes.get(plugin_id) {
            // Send SIGKILL to the process
            signal::kill(
                Pid::from_raw(sandbox.process_id as i32),
                Signal::SIGKILL,
            ).map_err(|e| ProcessingError::SandboxError(format!("Failed to kill process: {}", e)))?;
            
            self.cleanup_sandbox(plugin_id)?;
        }
        
        Ok(())
    }
}

impl Clone for SandboxInstance {
    fn clone(&self) -> Self {
        Self {
            plugin_id: self.plugin_id.clone(),
            process_id: self.process_id,
            resource_limits: self.resource_limits.clone(),
            start_time: self.start_time,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sandbox_manager_creation() {
        let manager = SandboxManager::new();
        assert!(manager.is_ok());
    }
}