//! Plugin system with hot-reload support for neural document flow
//!
//! This module provides:
//! - Dynamic plugin loading at runtime
//! - Hot-reload capability for plugin updates
//! - Security sandboxing for plugin execution
//! - Plugin lifecycle management

pub mod loader;
pub mod manager;
pub mod discovery;
pub mod sandbox;
pub mod registry;
pub mod signature;
pub mod sdk;
pub mod builtin;

use neural_doc_flow_core::{DocumentSource, ProcessingError};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Plugin metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginMetadata {
    pub name: String,
    pub version: String,
    pub author: String,
    pub description: String,
    pub supported_formats: Vec<String>,
    pub capabilities: PluginCapabilities,
}

/// Plugin capabilities and requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginCapabilities {
    pub requires_network: bool,
    pub requires_filesystem: bool,
    pub max_memory_mb: usize,
    pub max_cpu_percent: f32,
    pub timeout_seconds: u64,
}

/// Plugin interface that all plugins must implement
pub trait Plugin: Send + Sync {
    /// Get plugin metadata
    fn metadata(&self) -> &PluginMetadata;
    
    /// Initialize the plugin
    fn initialize(&mut self) -> Result<(), ProcessingError>;
    
    /// Shutdown the plugin gracefully
    fn shutdown(&mut self) -> Result<(), ProcessingError>;
    
    /// Get the document source implementation
    fn document_source(&self) -> Box<dyn DocumentSource>;
}

/// Plugin loading result
pub struct LoadedPlugin {
    pub metadata: PluginMetadata,
    pub plugin: Box<dyn Plugin>,
    pub library: libloading::Library,
    pub path: std::path::PathBuf,
}

/// Plugin manager for the system
pub use manager::PluginManager;

/// Create plugin manager with default configuration
pub fn create_plugin_manager() -> Result<PluginManager, ProcessingError> {
    PluginManager::new(PluginConfig::default())
}

/// Create plugin manager with built-in plugins registered
pub async fn create_plugin_manager_with_builtins() -> Result<PluginManager, ProcessingError> {
    let mut manager = PluginManager::new(PluginConfig::default())?;
    builtin::register_builtin_plugins(&mut manager).await?;
    Ok(manager)
}

/// Plugin system configuration
#[derive(Debug, Clone)]
pub struct PluginConfig {
    pub plugin_dir: PathBuf,
    pub enable_hot_reload: bool,
    pub enable_sandboxing: bool,
    pub max_plugins: usize,
}

impl Default for PluginConfig {
    fn default() -> Self {
        Self {
            plugin_dir: PathBuf::from("./plugins"),
            enable_hot_reload: true,
            enable_sandboxing: true,
            max_plugins: 50,
        }
    }
}

/// Plugin API version for compatibility checking
pub const PLUGIN_API_VERSION: &str = "1.0.0";

/// Plugin entry point function name
pub const PLUGIN_ENTRY_POINT: &[u8] = b"create_plugin\0";

/// Type alias for plugin constructor function
pub type PluginConstructor = unsafe extern "C" fn() -> *mut dyn Plugin;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_plugin_config_default() {
        let config = PluginConfig::default();
        assert_eq!(config.enable_hot_reload, true);
        assert_eq!(config.enable_sandboxing, true);
        assert_eq!(config.max_plugins, 50);
    }
}