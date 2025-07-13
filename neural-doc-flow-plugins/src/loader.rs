//! Dynamic plugin loading

use crate::{Plugin, LoadedPlugin, PluginConstructor, PLUGIN_ENTRY_POINT};
use neural_doc_flow_core::ProcessingError;
use libloading::{Library, Symbol};
use std::path::Path;
use tracing::{info, error};

/// Plugin loader for dynamic library loading
pub struct PluginLoader;

impl PluginLoader {
    /// Create a new plugin loader
    pub fn new() -> Self {
        Self
    }
    
    /// Load a plugin from a dynamic library
    pub fn load_plugin(&self, path: &Path) -> Result<LoadedPlugin, ProcessingError> {
        info!("Loading plugin from: {:?}", path);
        
        // Load the dynamic library
        let library = unsafe {
            Library::new(path).map_err(|e| {
                error!("Failed to load library: {}", e);
                ProcessingError::PluginLoadError(format!("Failed to load library: {}", e))
            })?
        };
        
        // Get the plugin constructor function
        let constructor: Symbol<PluginConstructor> = unsafe {
            library.get(PLUGIN_ENTRY_POINT).map_err(|e| {
                error!("Failed to find plugin entry point: {}", e);
                ProcessingError::PluginLoadError(format!("Plugin entry point not found: {}", e))
            })?
        };
        
        // Create the plugin instance
        let plugin_ptr = unsafe { constructor() };
        if plugin_ptr.is_null() {
            return Err(ProcessingError::PluginLoadError(
                "Plugin constructor returned null".to_string()
            ));
        }
        
        let mut plugin: Box<dyn Plugin> = unsafe { Box::from_raw(plugin_ptr) };
        
        // Initialize the plugin
        plugin.initialize().map_err(|e| {
            error!("Plugin initialization failed: {}", e);
            e
        })?;
        
        // Get metadata
        let metadata = plugin.metadata().clone();
        
        info!("Successfully loaded plugin: {} v{}", metadata.name, metadata.version);
        
        Ok(LoadedPlugin {
            metadata,
            plugin,
            library,
            path: path.to_path_buf(),
        })
    }
    
    /// Validate plugin compatibility
    pub fn validate_plugin(&self, plugin: &dyn Plugin) -> Result<(), ProcessingError> {
        let metadata = plugin.metadata();
        
        // Check plugin API version compatibility
        // TODO: Implement version checking
        
        // Validate metadata
        if metadata.name.is_empty() {
            return Err(ProcessingError::PluginLoadError(
                "Plugin name cannot be empty".to_string()
            ));
        }
        
        if metadata.supported_formats.is_empty() {
            return Err(ProcessingError::PluginLoadError(
                "Plugin must support at least one format".to_string()
            ));
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_plugin_loader_creation() {
        let loader = PluginLoader::new();
        // Basic creation test
    }
}