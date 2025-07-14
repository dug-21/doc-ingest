//! Plugin manager with hot-reload capability

use crate::{
    PluginConfig, PluginMetadata, LoadedPlugin,
    loader::PluginLoader,
    discovery::PluginDiscovery,
    registry::PluginRegistry,
    sandbox::PluginSandbox,
    signature::{PluginSignatureVerifier, VerificationResult},
};
use neural_doc_flow_core::ProcessingError;
// use neural_doc_flow_security::sandbox::SandboxManager; // Temporarily disabled
use notify::{Watcher, RecommendedWatcher, RecursiveMode, Event};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::collections::{HashMap, HashSet, VecDeque};
use tokio::sync::{RwLock, mpsc};
use tracing::{info, warn, error};

/// Plugin manager with hot-reload support
pub struct PluginManager {
    config: PluginConfig,
    registry: Arc<RwLock<PluginRegistry>>,
    loader: Arc<PluginLoader>,
    discovery: Arc<PluginDiscovery>,
    sandbox: Arc<tokio::sync::Mutex<PluginSandbox>>,
    signature_verifier: Arc<tokio::sync::Mutex<PluginSignatureVerifier>>,
    
    // Hot-reload components
    watcher: Option<RecommendedWatcher>,
    reload_tx: mpsc::Sender<PluginReload>,
    reload_handle: Option<tokio::task::JoinHandle<()>>,
    
    // Performance monitoring
    reload_metrics: Arc<RwLock<ReloadMetrics>>,
}

/// Plugin reload event
#[derive(Debug)]
struct PluginReload {
    plugin_path: PathBuf,
    action: ReloadAction,
    timestamp: std::time::Instant,
    retry_count: u32,
}

#[derive(Debug)]
enum ReloadAction {
    Load,
    Update,
    Remove,
    Verify,
}

/// Hot-reload performance metrics
#[derive(Debug, Default, Clone)]
struct ReloadMetrics {
    total_reloads: u64,
    successful_reloads: u64,
    failed_reloads: u64,
    average_reload_time_ms: f64,
    last_reload_time: Option<std::time::Instant>,
    reload_history: std::collections::VecDeque<ReloadEvent>,
}

#[derive(Debug, Clone)]
struct ReloadEvent {
    plugin_name: String,
    action: String,
    duration_ms: u64,
    success: bool,
    timestamp: std::time::Instant,
    error: Option<String>,
}

impl PluginManager {
    /// Create a new plugin manager
    pub fn new(config: PluginConfig) -> Result<Self, ProcessingError> {
        let registry = Arc::new(RwLock::new(PluginRegistry::new()));
        let loader = Arc::new(PluginLoader::new());
        let discovery = Arc::new(PluginDiscovery::new(config.plugin_dir.clone()));
        let sandbox = Arc::new(tokio::sync::Mutex::new(PluginSandbox::new()?));
        
        // Initialize signature verifier
        let mut signature_verifier = PluginSignatureVerifier::new(config.enable_sandboxing);
        
        // Load trusted keys if they exist
        let keys_file = config.plugin_dir.join("trusted_keys.txt");
        if let Err(e) = signature_verifier.load_trusted_keys(&keys_file) {
            warn!("Failed to load trusted keys: {}", e);
        }
        
        let signature_verifier = Arc::new(tokio::sync::Mutex::new(signature_verifier));
        
        let (reload_tx, reload_rx) = mpsc::channel(1000); // Increased buffer
        
        let mut manager = Self {
            config,
            registry,
            loader,
            discovery,
            sandbox,
            signature_verifier,
            watcher: None,
            reload_tx,
            reload_handle: None,
            reload_metrics: Arc::new(RwLock::new(ReloadMetrics::default())),
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
    
    /// Load a plugin from path with full security validation
    pub async fn load_plugin(&self, path: &Path) -> Result<(), ProcessingError> {
        let start_time = std::time::Instant::now();
        info!("Loading plugin from: {:?}", path);
        
        // Verify plugin signature first
        let verification_result = {
            let mut verifier = self.signature_verifier.lock().await;
            verifier.verify_plugin(path)?
        };
        
        if !verification_result.is_valid {
            let error_msg = format!(
                "Plugin signature verification failed: {}", 
                verification_result.errors.join(", ")
            );
            error!("{}", error_msg);
            self.record_reload_event(
                path.file_stem().unwrap_or_default().to_string_lossy().to_string(),
                "load".to_string(),
                start_time.elapsed().as_millis() as u64,
                false,
                Some(error_msg.clone()),
            ).await;
            return Err(ProcessingError::PluginLoadError(error_msg));
        }
        
        // Log verification warnings
        for warning in &verification_result.warnings {
            warn!("Plugin verification warning: {}", warning);
        }
        
        // Load the plugin library
        let loaded = self.loader.load_plugin(path)?;
        
        // Validate plugin capabilities match signature
        if !verification_result.capabilities.is_empty() {
            let plugin_capabilities = &loaded.metadata.supported_formats;
            for declared_cap in &verification_result.capabilities {
                if !plugin_capabilities.contains(declared_cap) {
                    warn!(
                        "Plugin {} declares capability '{}' in signature but not in metadata",
                        loaded.metadata.name, declared_cap
                    );
                }
            }
        }
        
        // Validate plugin in sandbox if enabled
        if self.config.enable_sandboxing {
            self.sandbox.lock().await.validate_plugin(&loaded).await?;
        }
        
        // Register the plugin
        let mut registry = self.registry.write().await;
        registry.register_plugin(loaded)?;
        
        // Record successful load
        self.record_reload_event(
            path.file_stem().unwrap_or_default().to_string_lossy().to_string(),
            "load".to_string(),
            start_time.elapsed().as_millis() as u64,
            true,
            None,
        ).await;
        
        info!("Successfully loaded plugin from: {:?}", path);
        Ok(())
    }
    
    /// Hot-reload a plugin with zero-downtime strategy
    pub async fn hot_reload(&self, plugin_name: &str) -> Result<(), ProcessingError> {
        let start_time = std::time::Instant::now();
        info!("Hot-reloading plugin: {}", plugin_name);
        
        let mut registry = self.registry.write().await;
        
        // Get current plugin info
        let plugin_info = registry.get_plugin_info(plugin_name)
            .ok_or_else(|| ProcessingError::PluginNotFound(plugin_name.to_string()))?;
        
        let plugin_path = plugin_info.path.clone();
        let old_metadata = plugin_info.metadata.clone();
        
        drop(registry); // Release lock early
        
        // Pre-validate new version before unloading current
        let temp_loaded = self.loader.load_plugin(&plugin_path);
        if let Err(e) = temp_loaded {
            self.record_reload_event(
                plugin_name.to_string(),
                "hot_reload".to_string(),
                start_time.elapsed().as_millis() as u64,
                false,
                Some(format!("Pre-validation failed: {}", e)),
            ).await;
            return Err(e);
        }
        
        let new_loaded = temp_loaded.unwrap();
        
        // Check version compatibility
        if self.is_incompatible_version(&old_metadata, &new_loaded.metadata) {
            let error_msg = "Incompatible plugin version for hot-reload".to_string();
            self.record_reload_event(
                plugin_name.to_string(),
                "hot_reload".to_string(),
                start_time.elapsed().as_millis() as u64,
                false,
                Some(error_msg.clone()),
            ).await;
            return Err(ProcessingError::PluginLoadError(error_msg));
        }
        
        // Verify new version signature
        let verification_result = {
            let mut verifier = self.signature_verifier.lock().await;
            verifier.verify_plugin(&plugin_path)?
        };
        
        if !verification_result.is_valid {
            let error_msg = format!(
                "New plugin version signature verification failed: {}", 
                verification_result.errors.join(", ")
            );
            self.record_reload_event(
                plugin_name.to_string(),
                "hot_reload".to_string(),
                start_time.elapsed().as_millis() as u64,
                false,
                Some(error_msg.clone()),
            ).await;
            return Err(ProcessingError::PluginLoadError(error_msg));
        }
        
        // Now safely unload and reload
        let mut registry = self.registry.write().await;
        
        // Store old version for potential rollback
        let old_plugin = registry.get_plugin(plugin_name).cloned();
        
        // Unload current version
        if let Err(e) = registry.unregister_plugin(plugin_name) {
            warn!("Failed to unregister old plugin version: {}", e);
        }
        
        // Register new version (already loaded and validated)
        match registry.register_plugin(new_loaded) {
            Ok(()) => {
                drop(registry);
                self.record_reload_event(
                    plugin_name.to_string(),
                    "hot_reload".to_string(),
                    start_time.elapsed().as_millis() as u64,
                    true,
                    None,
                ).await;
                info!("Successfully hot-reloaded plugin: {} in {}ms", 
                     plugin_name, start_time.elapsed().as_millis());
                Ok(())
            }
            Err(e) => {
                // Rollback to old version if possible
                if let Some(old) = old_plugin {
                    warn!("Rolling back to previous plugin version due to error: {}", e);
                    if let Ok(old_loaded) = Arc::try_unwrap(old) {
                        let _ = registry.register_plugin(old_loaded);
                    }
                }
                
                self.record_reload_event(
                    plugin_name.to_string(),
                    "hot_reload".to_string(),
                    start_time.elapsed().as_millis() as u64,
                    false,
                    Some(format!("Registration failed: {}", e)),
                ).await;
                
                Err(e)
            }
        }
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
                                timestamp: std::time::Instant::now(),
                                retry_count: 0,
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
                        info!("Plugin file changed: {:?}, age: {:?}", 
                             reload.plugin_path, reload.timestamp.elapsed());
                        
                        // Debounce rapid file changes (wait for write to complete)
                        let debounce_delay = if reload.retry_count > 0 {
                            tokio::time::Duration::from_millis(500 + reload.retry_count as u64 * 200)
                        } else {
                            tokio::time::Duration::from_millis(200)
                        };
                        tokio::time::sleep(debounce_delay).await;
                        
                        // Check if file still exists and is readable
                        if !reload.plugin_path.exists() {
                            warn!("Plugin file no longer exists: {:?}", reload.plugin_path);
                            continue;
                        }
                        
                        // Load the plugin with full validation
                        match loader.load_plugin(&reload.plugin_path) {
                            Ok(loaded) => {
                                // Validate in sandbox if enabled
                                if enable_sandboxing {
                                    if let Err(e) = sandbox.lock().await.validate_plugin(&loaded).await {
                                        error!("Plugin validation failed: {}", e);
                                        
                                        // Retry with backoff if not too many attempts
                                        if reload.retry_count < 3 {
                                            let retry_reload = PluginReload {
                                                plugin_path: reload.plugin_path.clone(),
                                                action: reload.action,
                                                timestamp: reload.timestamp,
                                                retry_count: reload.retry_count + 1,
                                            };
                                            
                                            // Schedule retry
                                            tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                                            if reload_tx.send(retry_reload).await.is_err() {
                                                error!("Failed to schedule plugin reload retry");
                                            }
                                        }
                                        continue;
                                    }
                                }
                                
                                // Register or update plugin
                                let mut reg = registry.write().await;
                                if let Err(e) = reg.update_plugin(loaded) {
                                    error!("Failed to update plugin: {}", e);
                                } else {
                                    info!("Successfully hot-reloaded plugin: {:?}", reload.plugin_path);
                                }
                            }
                            Err(e) => {
                                error!("Failed to load plugin: {}", e);
                                
                                // Retry with backoff for transient errors
                                if reload.retry_count < 2 && e.to_string().contains("busy") {
                                    let retry_reload = PluginReload {
                                        plugin_path: reload.plugin_path.clone(),
                                        action: reload.action,
                                        timestamp: reload.timestamp,
                                        retry_count: reload.retry_count + 1,
                                    };
                                    
                                    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                                    if reload_tx.send(retry_reload).await.is_err() {
                                        error!("Failed to schedule plugin reload retry");
                                    }
                                }
                            },
                        }
                    }
                    ReloadAction::Remove => {
                        info!("Plugin removed: {:?}", reload.plugin_path);
                        
                        let mut reg = registry.write().await;
                        reg.remove_by_path(&reload.plugin_path);
                    }
                    ReloadAction::Verify => {
                        info!("Verifying plugin: {:?}", reload.plugin_path);
                        
                        // Re-verify plugin signature (useful after key updates)
                        // This doesn't reload the plugin, just checks if it's still valid
                        // Implementation would check signature and log results
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
    
    /// Check if plugin version is incompatible for hot-reload
    fn is_incompatible_version(&self, old: &PluginMetadata, new: &PluginMetadata) -> bool {
        // Check major version changes
        let old_version = semver::Version::parse(&old.version).unwrap_or(semver::Version::new(0, 0, 0));
        let new_version = semver::Version::parse(&new.version).unwrap_or(semver::Version::new(0, 0, 0));
        
        // Major version changes are incompatible
        if old_version.major != new_version.major {
            return true;
        }
        
        // Check if capabilities changed significantly
        let old_caps: std::collections::HashSet<_> = old.supported_formats.iter().collect();
        let new_caps: std::collections::HashSet<_> = new.supported_formats.iter().collect();
        
        // If new version removes capabilities, it might be incompatible
        if !old_caps.is_subset(&new_caps) {
            warn!("Plugin {} removes capabilities in new version", new.name);
            return true;
        }
        
        false
    }
    
    /// Record reload event for metrics
    async fn record_reload_event(
        &self,
        plugin_name: String,
        action: String,
        duration_ms: u64,
        success: bool,
        error: Option<String>,
    ) {
        let mut metrics = self.reload_metrics.write().await;
        
        let event = ReloadEvent {
            plugin_name,
            action,
            duration_ms,
            success,
            timestamp: std::time::Instant::now(),
            error,
        };
        
        // Update counters
        metrics.total_reloads += 1;
        if success {
            metrics.successful_reloads += 1;
        } else {
            metrics.failed_reloads += 1;
        }
        
        // Update average duration
        metrics.average_reload_time_ms = 
            (metrics.average_reload_time_ms * (metrics.total_reloads - 1) as f64 + duration_ms as f64) 
            / metrics.total_reloads as f64;
        
        metrics.last_reload_time = Some(event.timestamp);
        
        // Keep history (last 100 events)
        metrics.reload_history.push_back(event);
        if metrics.reload_history.len() > 100 {
            metrics.reload_history.pop_front();
        }
    }
    
    /// Get hot-reload metrics
    pub async fn get_reload_metrics(&self) -> ReloadMetrics {
        self.reload_metrics.read().await.clone()
    }
    
    /// Clear signature cache (useful after updating trusted keys)
    pub async fn clear_signature_cache(&self) {
        let mut verifier = self.signature_verifier.lock().await;
        verifier.clear_cache();
        info!("Plugin signature cache cleared");
    }
    
    /// Refresh all plugin signatures
    pub async fn refresh_all_signatures(&self) -> Result<(), ProcessingError> {
        info!("Refreshing all plugin signatures");
        
        let plugins: Vec<_> = {
            let registry = self.registry.read().await;
            registry.list_plugins().into_iter()
                .map(|meta| (meta.name.clone(), registry.get_plugin_info(&meta.name)))
                .filter_map(|(name, info)| info.map(|i| (name, i.path)))
                .collect()
        };
        
        let mut total_verified = 0;
        let mut failed_verifications = Vec::new();
        
        for (plugin_name, plugin_path) in plugins {
            let verification_result = {
                let mut verifier = self.signature_verifier.lock().await;
                verifier.verify_plugin(&plugin_path)
            };
            
            match verification_result {
                Ok(result) if result.is_valid => {
                    total_verified += 1;
                    info!("Plugin {} signature verified", plugin_name);
                }
                Ok(result) => {
                    failed_verifications.push(format!(
                        "{}: {}", 
                        plugin_name, 
                        result.errors.join(", ")
                    ));
                }
                Err(e) => {
                    failed_verifications.push(format!("{}: {}", plugin_name, e));
                }
            }
        }
        
        info!("Signature refresh complete: {} verified, {} failed", 
             total_verified, failed_verifications.len());
        
        if !failed_verifications.is_empty() {
            warn!("Failed signature verifications: {:?}", failed_verifications);
        }
        
        Ok(())
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
        
        // Log final metrics
        let metrics = self.reload_metrics.read().await;
        info!(
            "Plugin manager shutdown - Total reloads: {}, Success rate: {:.1}%",
            metrics.total_reloads,
            if metrics.total_reloads > 0 {
                (metrics.successful_reloads as f64 / metrics.total_reloads as f64) * 100.0
            } else {
                100.0
            }
        );
        
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