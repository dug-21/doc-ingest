//! Plugin manager with hot-reload capability

use crate::{
    PluginConfig, PluginMetadata, LoadedPlugin,
    loader::PluginLoader,
    discovery::PluginDiscovery,
    registry::PluginRegistry,
    sandbox::PluginSandbox,
};
use neural_doc_flow_core::ProcessingError;
// use neural_doc_flow_security::sandbox::SandboxManager; // Not used directly
use notify::{Watcher, RecommendedWatcher, RecursiveMode, Event};
// use std::collections::HashMap; // Not used
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use tracing::{info, warn, error};

/// Plugin manager with hot-reload support
pub struct PluginManager {
    config: PluginConfig,
    registry: Arc<RwLock<PluginRegistry>>,
    loader: Arc<PluginLoader>,
    discovery: Arc<PluginDiscovery>,
    sandbox: Arc<tokio::sync::Mutex<PluginSandbox>>,
    
    // Hot-reload components
    watcher: Option<RecommendedWatcher>,
    reload_tx: mpsc::Sender<PluginReload>,
    reload_handle: Option<tokio::task::JoinHandle<()>>,
}

/// Plugin reload event
#[derive(Debug)]
struct PluginReload {
    plugin_path: PathBuf,
    action: ReloadAction,
}

#[derive(Debug)]
enum ReloadAction {
    Load,
    Update,
    Remove,
}

impl PluginManager {
    /// Create a new plugin manager
    pub fn new(config: PluginConfig) -> Result<Self, ProcessingError> {
        let registry = Arc::new(RwLock::new(PluginRegistry::new()));
        let loader = Arc::new(PluginLoader::new());
        let discovery = Arc::new(PluginDiscovery::new(config.plugin_dir.clone()));
        let sandbox = Arc::new(tokio::sync::Mutex::new(PluginSandbox::new()?));
        
        let (reload_tx, reload_rx) = mpsc::channel(100);
        
        let mut manager = Self {
            config,
            registry,
            loader,
            discovery,
            sandbox,
            watcher: None,
            reload_tx,
            reload_handle: None,
        };
        
        // Start hot-reload monitor if enabled
        if manager.config.enable_hot_reload {
            manager.start_hot_reload_monitor(reload_rx)?;
        }
        
        Ok(manager)
    }
    
    /// Initialize and discover plugins
    pub async fn initialize(&mut self) -> Result<(), ProcessingError> {
        info!("Initializing plugin manager");
        
        // Discover plugins in plugin directory
        let discovered = self.discovery.discover_plugins().await?;
        info!("Discovered {} plugins", discovered.len());
        
        // Load each discovered plugin
        for plugin_info in discovered {
            match self.load_plugin(&plugin_info.path).await {
                Ok(()) => info!("Loaded plugin: {}", plugin_info.name),
                Err(e) => warn!("Failed to load plugin {}: {}", plugin_info.name, e),
            }
        }
        
        Ok(())
    }
    
    /// Load a plugin from path
    pub async fn load_plugin(&self, path: &Path) -> Result<(), ProcessingError> {
        info!("Loading plugin from: {:?}", path);
        
        // Load the plugin library
        let loaded = self.loader.load_plugin(path)?;
        
        // Validate plugin in sandbox if enabled
        if self.config.enable_sandboxing {
            self.sandbox.lock().await.validate_plugin(&loaded).await?;
        }
        
        // Register the plugin
        let mut registry = self.registry.write().await;
        registry.register_plugin(loaded)?;
        
        Ok(())
    }
    
    /// Hot-reload a plugin
    pub async fn hot_reload(&self, plugin_name: &str) -> Result<(), ProcessingError> {
        info!("Hot-reloading plugin: {}", plugin_name);
        
        let mut registry = self.registry.write().await;
        
        // Get current plugin info
        let plugin_info = registry.get_plugin_info(plugin_name)
            .ok_or_else(|| ProcessingError::PluginNotFound(plugin_name.to_string()))?;
        
        let plugin_path = plugin_info.path.clone();
        
        // Unload current version
        registry.unregister_plugin(plugin_name)?;
        
        // Load new version
        drop(registry); // Release lock before loading
        self.load_plugin(&plugin_path).await?;
        
        info!("Successfully hot-reloaded plugin: {}", plugin_name);
        Ok(())
    }
    
    /// Start file watcher for hot-reload
    fn start_hot_reload_monitor(
        &mut self,
        mut reload_rx: mpsc::Receiver<PluginReload>,
    ) -> Result<(), ProcessingError> {
        info!("Starting hot-reload monitor");
        
        // Create file watcher
        let reload_tx = self.reload_tx.clone();
        let mut watcher = notify::recommended_watcher(move |res: Result<Event, notify::Error>| {
            match res {
                Ok(event) => {
                    if let Some(path) = event.paths.first() {
                        if path.extension().and_then(|s| s.to_str()) == Some("so") ||
                           path.extension().and_then(|s| s.to_str()) == Some("dll") ||
                           path.extension().and_then(|s| s.to_str()) == Some("dylib") {
                            let action = match event.kind {
                                notify::EventKind::Create(_) => ReloadAction::Load,
                                notify::EventKind::Modify(_) => ReloadAction::Update,
                                notify::EventKind::Remove(_) => ReloadAction::Remove,
                                _ => return,
                            };
                            
                            let _ = reload_tx.blocking_send(PluginReload {
                                plugin_path: path.to_path_buf(),
                                action,
                            });
                        }
                    }
                }
                Err(e) => error!("Watch error: {:?}", e),
            }
        }).map_err(|e| ProcessingError::ProcessorFailed {
            processor_name: "file_watcher".to_string(),
            reason: e.to_string(),
        })?;
        
        // Watch plugin directory
        watcher.watch(&self.config.plugin_dir, RecursiveMode::Recursive)
            .map_err(|e| ProcessingError::ProcessorFailed {
                processor_name: "file_watcher".to_string(),
                reason: e.to_string(),
            })?;
        self.watcher = Some(watcher);
        
        // Spawn reload handler task
        let registry = self.registry.clone();
        let loader = self.loader.clone();
        let sandbox = self.sandbox.clone();
        let enable_sandboxing = self.config.enable_sandboxing;
        
        let handle = tokio::spawn(async move {
            while let Some(reload) = reload_rx.recv().await {
                match reload.action {
                    ReloadAction::Load | ReloadAction::Update => {
                        info!("Plugin file changed: {:?}", reload.plugin_path);
                        
                        // Wait a bit for file write to complete
                        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                        
                        // Load the plugin
                        match loader.load_plugin(&reload.plugin_path) {
                            Ok(loaded) => {
                                // Validate in sandbox if enabled
                                if enable_sandboxing {
                                    if let Err(e) = sandbox.lock().await.validate_plugin(&loaded).await {
                                        error!("Plugin validation failed: {}", e);
                                        continue;
                                    }
                                }
                                
                                // Register or update plugin
                                let mut reg = registry.write().await;
                                if let Err(e) = reg.update_plugin(loaded) {
                                    error!("Failed to update plugin: {}", e);
                                }
                            }
                            Err(e) => error!("Failed to load plugin: {}", e),
                        }
                    }
                    ReloadAction::Remove => {
                        info!("Plugin removed: {:?}", reload.plugin_path);
                        
                        let mut reg = registry.write().await;
                        reg.remove_by_path(&reload.plugin_path);
                    }
                }
            }
        });
        
        self.reload_handle = Some(handle);
        
        Ok(())
    }
    
    /// Get a plugin by name
    pub async fn get_plugin(&self, name: &str) -> Option<Arc<LoadedPlugin>> {
        let registry = self.registry.read().await;
        registry.get_plugin(name)
    }
    
    /// List all loaded plugins
    pub async fn list_plugins(&self) -> Vec<PluginMetadata> {
        let registry = self.registry.read().await;
        registry.list_plugins()
    }
    
    /// Shutdown the plugin manager
    pub async fn shutdown(&mut self) -> Result<(), ProcessingError> {
        info!("Shutting down plugin manager");
        
        // Stop file watcher
        self.watcher.take();
        
        // Stop reload handler
        if let Some(handle) = self.reload_handle.take() {
            handle.abort();
        }
        
        // Shutdown all plugins
        let mut registry = self.registry.write().await;
        registry.shutdown_all().await?;
        
        Ok(())
    }
}

impl Drop for PluginManager {
    fn drop(&mut self) {
        // Ensure cleanup
        let _ = tokio::runtime::Handle::current().block_on(self.shutdown());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_plugin_manager_creation() {
        let config = PluginConfig {
            enable_hot_reload: false, // Disable for test
            ..Default::default()
        };
        
        let manager = PluginManager::new(config);
        assert!(manager.is_ok());
    }
}