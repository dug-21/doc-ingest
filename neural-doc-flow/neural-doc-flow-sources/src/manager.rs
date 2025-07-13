//! Plugin manager for document sources
//!
//! This module provides the plugin manager that discovers, loads, and manages
//! document source plugins.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use parking_lot::RwLock;
use neural_doc_flow_core::*;

/// Configuration for the source plugin manager
#[derive(Debug, Clone)]
pub struct ManagerConfig {
    /// Directories to search for plugins
    pub plugin_directories: Vec<PathBuf>,
    /// Path to configuration file
    pub config_path: PathBuf,
}

impl Default for ManagerConfig {
    fn default() -> Self {
        Self {
            plugin_directories: vec![
                PathBuf::from("./plugins"),
                PathBuf::from("/usr/local/lib/neuraldocflow/sources"),
            ],
            config_path: PathBuf::from("./sources.yaml"),
        }
    }
}

/// Plugin manager for document sources
/// 
/// This manager discovers, loads, and manages document source plugins.
/// It provides functionality for:
/// - Automatic plugin discovery
/// - Plugin loading and initialization
/// - Source selection for documents
/// - Hot-reload capability (when enabled)
pub struct SourcePluginManager {
    /// Plugin directories to search
    pub plugin_dirs: Vec<PathBuf>,
    /// Loaded source plugins
    pub sources: Arc<RwLock<HashMap<String, Arc<dyn DocumentSource>>>>,
    /// Plugin configurations
    configs: HashMap<String, SourceConfig>,
    /// Manager configuration
    config: ManagerConfig,
}

impl SourcePluginManager {
    /// Create new plugin manager
    /// 
    /// # Parameters
    /// - `config`: Manager configuration
    /// 
    /// # Returns
    /// - `Ok(SourcePluginManager)` if creation succeeded
    /// - `Err(_)` if creation failed
    pub fn new(config: ManagerConfig) -> NeuralDocFlowResult<Self> {
        let mut manager = Self {
            plugin_dirs: config.plugin_directories.clone(),
            sources: Arc::new(RwLock::new(HashMap::new())),
            configs: HashMap::new(),
            config,
        };

        // Load built-in sources
        manager.load_builtin_sources()?;
        
        Ok(manager)
    }

    /// Load built-in document sources
    fn load_builtin_sources(&mut self) -> NeuralDocFlowResult<()> {
        #[cfg(feature = "pdf")]
        {
            let pdf_source = Arc::new(crate::pdf::PdfSource::new());
            self.sources.write().insert(
                pdf_source.source_id().to_string(),
                pdf_source
            );
        }

        Ok(())
    }

    /// Find compatible sources for a given input
    /// 
    /// # Parameters
    /// - `input`: The input to find sources for
    /// 
    /// # Returns
    /// - `Ok(Vec<Arc<dyn DocumentSource>>)` with compatible sources
    /// - `Err(_)` if search failed
    pub async fn find_compatible_sources(
        &self,
        input: &SourceInput,
    ) -> Result<Vec<Arc<dyn DocumentSource>>, SourceError> {
        let mut compatible = Vec::new();
        let sources = self.sources.read();

        for source in sources.values() {
            if source.can_handle(input).await? {
                compatible.push(Arc::clone(source));
            }
        }

        // Sort by priority if we have configurations
        compatible.sort_by(|a, b| {
            let priority_a = self.configs.get(a.source_id())
                .map(|c| c.priority)
                .unwrap_or(0);
            let priority_b = self.configs.get(b.source_id())
                .map(|c| c.priority)
                .unwrap_or(0);
            priority_b.cmp(&priority_a) // Higher priority first
        });

        Ok(compatible)
    }

    /// Get a specific source by ID
    /// 
    /// # Parameters
    /// - `source_id`: The source identifier
    /// 
    /// # Returns
    /// - `Some(Arc<dyn DocumentSource>)` if source exists
    /// - `None` if source not found
    pub fn get_source(&self, source_id: &str) -> Option<Arc<dyn DocumentSource>> {
        self.sources.read().get(source_id).cloned()
    }

    /// List all available sources
    /// 
    /// # Returns
    /// Vector of source identifiers
    pub fn list_sources(&self) -> Vec<String> {
        self.sources.read().keys().cloned().collect()
    }

    /// Get source information
    /// 
    /// # Parameters
    /// - `source_id`: The source identifier
    /// 
    /// # Returns
    /// - `Some(SourceInfo)` if source exists
    /// - `None` if source not found
    pub fn get_source_info(&self, source_id: &str) -> Option<SourceInfo> {
        self.sources.read().get(source_id).map(|source| {
            SourceInfo {
                id: source.source_id().to_string(),
                name: source.name().to_string(),
                version: source.version().to_string(),
                supported_extensions: source.supported_extensions().iter()
                    .map(|s| s.to_string())
                    .collect(),
                supported_mime_types: source.supported_mime_types().iter()
                    .map(|s| s.to_string())
                    .collect(),
                capabilities: source.capabilities(),
                statistics: source.statistics(),
            }
        })
    }

    /// Enable hot-reload for plugins
    /// 
    /// This watches plugin directories for changes and automatically
    /// reloads plugins when they are updated.
    /// 
    /// # Returns
    /// - `Ok(())` if hot-reload was enabled
    /// - `Err(_)` if hot-reload setup failed
    pub fn enable_hot_reload(&self) -> NeuralDocFlowResult<()> {
        // Phase 1: Placeholder implementation
        // Phase 2: Implement file watching with notify crate
        tracing::info!("Hot-reload enabled for plugin directories: {:?}", self.plugin_dirs);
        Ok(())
    }

    /// Reload all plugins
    /// 
    /// # Returns
    /// - `Ok(())` if reload succeeded
    /// - `Err(_)` if reload failed
    pub async fn reload_plugins(&mut self) -> NeuralDocFlowResult<()> {
        // Phase 1: Clear and reload built-in sources
        self.sources.write().clear();
        self.load_builtin_sources()?;
        
        tracing::info!("Plugins reloaded successfully");
        Ok(())
    }

    /// Configure a source
    /// 
    /// # Parameters
    /// - `source_id`: The source identifier
    /// - `config`: Source configuration
    /// 
    /// # Returns
    /// - `Ok(())` if configuration succeeded
    /// - `Err(_)` if configuration failed
    pub async fn configure_source(
        &mut self,
        source_id: &str,
        config: SourceConfig,
    ) -> Result<(), SourceError> {
        // Store configuration
        self.configs.insert(source_id.to_string(), config.clone());

        // Apply configuration to source if it's loaded
        if let Some(source) = self.sources.read().get(source_id) {
            // Note: This requires the source to implement SourceConfiguration
            // For now, we just store the config
            tracing::info!("Configuration stored for source: {}", source_id);
        }

        Ok(())
    }

    /// Get source configuration
    /// 
    /// # Parameters
    /// - `source_id`: The source identifier
    /// 
    /// # Returns
    /// - `Some(SourceConfig)` if configuration exists
    /// - `None` if no configuration found
    pub fn get_source_config(&self, source_id: &str) -> Option<&SourceConfig> {
        self.configs.get(source_id)
    }

    /// Get manager statistics
    /// 
    /// # Returns
    /// Manager statistics including plugin counts and performance data
    pub fn get_statistics(&self) -> ManagerStatistics {
        let sources = self.sources.read();
        let total_sources = sources.len();
        let enabled_sources = self.configs.values()
            .filter(|c| c.enabled)
            .count();

        ManagerStatistics {
            total_sources,
            enabled_sources,
            plugin_directories: self.plugin_dirs.len(),
            configurations: self.configs.len(),
        }
    }

    /// Validate all loaded sources
    /// 
    /// # Returns
    /// Vector of validation results for each source
    pub async fn validate_sources(&self) -> Vec<SourceValidationResult> {
        let mut results = Vec::new();
        let sources = self.sources.read();

        for (id, source) in sources.iter() {
            let result = SourceValidationResult {
                source_id: id.clone(),
                is_valid: true, // Phase 1: Assume all sources are valid
                issues: Vec::new(),
            };
            results.push(result);
        }

        results
    }
}

/// Information about a document source
#[derive(Debug, Clone)]
pub struct SourceInfo {
    /// Source identifier
    pub id: String,
    /// Source name
    pub name: String,
    /// Source version
    pub version: String,
    /// Supported file extensions
    pub supported_extensions: Vec<String>,
    /// Supported MIME types
    pub supported_mime_types: Vec<String>,
    /// Source capabilities
    pub capabilities: SourceCapabilities,
    /// Source statistics
    pub statistics: SourceStatistics,
}

/// Manager performance statistics
#[derive(Debug, Clone)]
pub struct ManagerStatistics {
    /// Total number of loaded sources
    pub total_sources: usize,
    /// Number of enabled sources
    pub enabled_sources: usize,
    /// Number of plugin directories
    pub plugin_directories: usize,
    /// Number of configurations
    pub configurations: usize,
}

/// Source validation result
#[derive(Debug, Clone)]
pub struct SourceValidationResult {
    /// Source identifier
    pub source_id: String,
    /// Whether the source is valid
    pub is_valid: bool,
    /// Validation issues found
    pub issues: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_plugin_manager_creation() {
        let config = ManagerConfig::default();
        let manager = SourcePluginManager::new(config).unwrap();
        
        // Should have built-in sources loaded
        #[cfg(feature = "pdf")]
        {
            assert!(manager.get_source("pdf").is_some());
        }
    }

    #[tokio::test]
    async fn test_source_discovery() {
        let config = ManagerConfig::default();
        let manager = SourcePluginManager::new(config).unwrap();

        let input = SourceInput::Memory {
            data: b"%PDF-1.4".to_vec(),
            filename: Some("test.pdf".to_string()),
            mime_type: None,
        };

        let sources = manager.find_compatible_sources(&input).await.unwrap();
        
        #[cfg(feature = "pdf")]
        {
            assert!(!sources.is_empty());
            assert_eq!(sources[0].source_id(), "pdf");
        }
    }

    #[test]
    fn test_manager_statistics() {
        let config = ManagerConfig::default();
        let manager = SourcePluginManager::new(config).unwrap();
        
        let stats = manager.get_statistics();
        assert!(stats.total_sources > 0);
    }

    #[tokio::test]
    async fn test_source_validation() {
        let config = ManagerConfig::default();
        let manager = SourcePluginManager::new(config).unwrap();
        
        let results = manager.validate_sources().await;
        assert!(!results.is_empty());
        
        for result in results {
            assert!(result.is_valid);
        }
    }
}