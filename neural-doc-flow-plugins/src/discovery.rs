//! Plugin discovery mechanism

use neural_doc_flow_core::ProcessingError;
use std::path::{Path, PathBuf};
use tokio::fs;
use tracing::info;

/// Plugin discovery service
pub struct PluginDiscovery {
    plugin_dir: PathBuf,
}

/// Discovered plugin information
#[derive(Debug, Clone)]
pub struct DiscoveredPlugin {
    pub name: String,
    pub path: PathBuf,
    pub metadata_path: Option<PathBuf>,
}

impl PluginDiscovery {
    /// Create a new plugin discovery service
    pub fn new(plugin_dir: PathBuf) -> Self {
        Self { plugin_dir }
    }
    
    /// Discover all plugins in the plugin directory
    pub async fn discover_plugins(&self) -> Result<Vec<DiscoveredPlugin>, ProcessingError> {
        info!("Discovering plugins in: {:?}", self.plugin_dir);
        
        // Ensure plugin directory exists
        if !self.plugin_dir.exists() {
            fs::create_dir_all(&self.plugin_dir).await?;
            info!("Created plugin directory: {:?}", self.plugin_dir);
            return Ok(Vec::new());
        }
        
        let mut plugins = Vec::new();
        let mut entries = fs::read_dir(&self.plugin_dir).await?;
        
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            
            // Check if it's a plugin file
            if self.is_plugin_file(&path) {
                let name = path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string();
                
                // Look for associated metadata file
                let metadata_path = self.find_metadata_file(&path).await;
                
                plugins.push(DiscoveredPlugin {
                    name,
                    path,
                    metadata_path,
                });
            }
        }
        
        info!("Discovered {} plugins", plugins.len());
        Ok(plugins)
    }
    
    /// Check if a file is a plugin library
    fn is_plugin_file(&self, path: &Path) -> bool {
        if !path.is_file() {
            return false;
        }
        
        match path.extension().and_then(|s| s.to_str()) {
            Some("so") => true,   // Linux
            Some("dll") => true,  // Windows
            Some("dylib") => true, // macOS
            _ => false,
        }
    }
    
    /// Find metadata file for a plugin
    async fn find_metadata_file(&self, plugin_path: &Path) -> Option<PathBuf> {
        let stem = plugin_path.file_stem()?;
        let metadata_path = plugin_path.with_file_name(format!("{}.toml", stem.to_str()?));
        
        if metadata_path.exists() {
            Some(metadata_path)
        } else {
            None
        }
    }
    
    /// Watch for new plugins
    pub async fn watch_for_plugins(&self) -> Result<(), ProcessingError> {
        // This is handled by the PluginManager's hot-reload monitor
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[tokio::test]
    async fn test_plugin_discovery() {
        let temp_dir = TempDir::new().unwrap();
        let discovery = PluginDiscovery::new(temp_dir.path().to_path_buf());
        
        let plugins = discovery.discover_plugins().await.unwrap();
        assert_eq!(plugins.len(), 0);
    }
}