//! Plugin registry for managing loaded plugins

use crate::{PluginMetadata, LoadedPlugin};
use neural_doc_flow_core::ProcessingError;
use dashmap::DashMap;
// use std::collections::HashMap; // Not used
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{info, warn};

/// Plugin registry for managing loaded plugins
pub struct PluginRegistry {
    plugins: DashMap<String, Arc<LoadedPlugin>>,
    paths: DashMap<PathBuf, String>,
}

/// Plugin information
#[derive(Clone)]
pub struct PluginInfo {
    pub name: String,
    pub path: PathBuf,
    pub metadata: PluginMetadata,
}

impl PluginRegistry {
    /// Create a new plugin registry
    pub fn new() -> Self {
        Self {
            plugins: DashMap::new(),
            paths: DashMap::new(),
        }
    }
    
    /// Register a new plugin
    pub fn register_plugin(&mut self, plugin: LoadedPlugin) -> Result<(), ProcessingError> {
        let name = plugin.metadata.name.clone();
        let path = plugin.path.clone();
        
        // Check if plugin already exists
        if self.plugins.contains_key(&name) {
            return Err(ProcessingError::PluginLoadError(
                format!("Plugin '{}' is already registered", name)
            ));
        }
        
        info!("Registering plugin: {}", name);
        self.paths.insert(path, name.clone());
        self.plugins.insert(name.clone(), Arc::new(plugin));
        
        Ok(())
    }
    
    /// Update an existing plugin (for hot-reload)
    pub fn update_plugin(&mut self, plugin: LoadedPlugin) -> Result<(), ProcessingError> {
        let name = plugin.metadata.name.clone();
        
        // Remove old version if exists
        if let Some((_, old_plugin)) = self.plugins.remove(&name) {
            // Shutdown old plugin
            if let Ok(mut plugin) = Arc::try_unwrap(old_plugin) {
                let _ = plugin.plugin.shutdown();
            }
        }
        
        // Register new version
        self.register_plugin(plugin)
    }
    
    /// Unregister a plugin
    pub fn unregister_plugin(&mut self, name: &str) -> Result<(), ProcessingError> {
        info!("Unregistering plugin: {}", name);
        
        if let Some((_, plugin)) = self.plugins.remove(name) {
            // Shutdown plugin
            if let Ok(mut loaded) = Arc::try_unwrap(plugin) {
                loaded.plugin.shutdown()?;
            }
            Ok(())
        } else {
            Err(ProcessingError::PluginNotFound(name.to_string()))
        }
    }
    
    /// Get a plugin by name
    pub fn get_plugin(&self, name: &str) -> Option<Arc<LoadedPlugin>> {
        self.plugins.get(name)
            .map(|entry| Arc::clone(&entry))
    }
    
    /// Get plugin information
    pub fn get_plugin_info(&self, name: &str) -> Option<PluginInfo> {
        self.plugins.get(name).map(|entry| {
            PluginInfo {
                name: name.to_string(),
                path: entry.path.clone(),
                metadata: entry.metadata.clone(),
            }
        })
    }
    
    /// List all registered plugins
    pub fn list_plugins(&self) -> Vec<PluginMetadata> {
        self.plugins.iter()
            .map(|entry| entry.value().metadata.clone())
            .collect()
    }
    
    /// Remove plugin by path
    pub fn remove_by_path(&mut self, path: &PathBuf) {
        if let Some((_, name)) = self.paths.remove(path) {
            let _ = self.unregister_plugin(&name);
        }
    }
    
    /// Get plugin by supported format
    pub fn get_plugins_for_format(&self, format: &str) -> Vec<Arc<LoadedPlugin>> {
        self.plugins.iter()
            .filter(|entry| {
                entry.value().metadata.supported_formats.contains(&format.to_string())
            })
            .map(|entry| Arc::clone(entry.value()))
            .collect()
    }
    
    /// Shutdown all plugins
    pub async fn shutdown_all(&mut self) -> Result<(), ProcessingError> {
        info!("Shutting down all plugins");
        
        let plugin_names: Vec<String> = self.plugins.iter()
            .map(|entry| entry.key().clone())
            .collect();
        
        for name in plugin_names {
            if let Err(e) = self.unregister_plugin(&name) {
                warn!("Failed to shutdown plugin {}: {}", name, e);
            }
        }
        
        Ok(())
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_plugin_registry_creation() {
        let registry = PluginRegistry::new();
        assert_eq!(registry.list_plugins().len(), 0);
    }
}