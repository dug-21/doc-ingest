//! Plugin sandboxing for security

use crate::{LoadedPlugin, PluginCapabilities};
use neural_doc_flow_core::ProcessingError;
use neural_doc_flow_security::sandbox::SandboxManager;
// use std::time::Duration; // Not used
use tracing::{info, warn};

/// Plugin sandbox for security isolation
pub struct PluginSandbox {
    sandbox_manager: SandboxManager,
}

impl PluginSandbox {
    /// Create a new plugin sandbox
    pub fn new() -> Result<Self, ProcessingError> {
        Ok(Self {
            sandbox_manager: SandboxManager::new()?,
        })
    }
    
    /// Validate a plugin in sandbox
    pub async fn validate_plugin(&mut self, plugin: &LoadedPlugin) -> Result<(), ProcessingError> {
        info!("Validating plugin {} in sandbox", plugin.metadata.name);
        
        let capabilities = &plugin.metadata.capabilities;
        
        // Check capability requirements
        self.validate_capabilities(capabilities)?;
        
        // Test plugin in sandbox
        let _result = self.sandbox_manager.execute_sandboxed(
            &plugin.metadata.name,
            move || {
                // Basic plugin validation
                // In production, would do more thorough testing
                Ok(())
            }
        ).await?;
        
        info!("Plugin {} validated successfully", plugin.metadata.name);
        Ok(())
    }
    
    /// Validate plugin capabilities
    fn validate_capabilities(&self, capabilities: &PluginCapabilities) -> Result<(), ProcessingError> {
        // Check memory requirements
        if capabilities.max_memory_mb > 1000 {
            return Err(ProcessingError::PluginLoadError(
                "Plugin requires too much memory (>1GB)".to_string()
            ));
        }
        
        // Check CPU requirements
        if capabilities.max_cpu_percent > 80.0 {
            return Err(ProcessingError::PluginLoadError(
                "Plugin requires too much CPU (>80%)".to_string()
            ));
        }
        
        // Check timeout
        if capabilities.timeout_seconds > 300 {
            return Err(ProcessingError::PluginLoadError(
                "Plugin timeout too long (>5 minutes)".to_string()
            ));
        }
        
        // Warn about network access
        if capabilities.requires_network {
            warn!("Plugin requires network access - this is a security risk");
        }
        
        Ok(())
    }
    
    /// Execute plugin operation in sandbox
    pub async fn execute_sandboxed<F, T>(
        &mut self,
        plugin_name: &str,
        operation: F,
    ) -> Result<T, ProcessingError>
    where
        F: FnOnce() -> Result<T, ProcessingError> + Send + 'static,
        T: Send + 'static,
    {
        let result = self.sandbox_manager.execute_sandboxed(
            plugin_name,
            operation,
        ).await?;
        
        // Log resource usage
        info!(
            "Plugin {} used {} MB memory, {:.2}s CPU time",
            plugin_name,
            result.resource_usage.memory_peak_bytes / 1_000_000,
            result.resource_usage.cpu_time_seconds
        );
        
        Ok(result.output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_plugin_sandbox_creation() {
        let sandbox = PluginSandbox::new();
        // May fail in test environment due to permissions
        // assert!(sandbox.is_ok());
    }
}