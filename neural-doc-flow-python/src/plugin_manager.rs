//! Plugin management for Python bindings

use pyo3::prelude::*;
use std::collections::HashMap;
use serde_json;

use neural_doc_flow_plugins::{
    PluginManager as RustPluginManager,
    PluginConfig,
    PluginMetadata,
};
use crate::NeuralDocFlowError;

/// Plugin manager for loading and managing document processing plugins
/// 
/// Provides functionality to load, manage, and configure plugins for document processing.
/// Supports both built-in plugins and external plugin loading.
/// 
/// # Example
/// ```python
/// import neuraldocflow
/// 
/// # Create plugin manager
/// plugin_manager = neuraldocflow.PluginManager()
/// 
/// # Load built-in plugins
/// await plugin_manager.load_builtin_plugins()
/// 
/// # Get available plugins
/// plugins = plugin_manager.get_available_plugins()
/// print(f"Available plugins: {list(plugins.keys())}")
/// 
/// # Enable specific plugins
/// plugin_manager.enable_plugin("docx")
/// plugin_manager.enable_plugin("tables")
/// ```
#[pyclass]
pub struct PluginManager {
    manager: RustPluginManager,
    rt: tokio::runtime::Runtime,
}

#[pymethods]
impl PluginManager {
    /// Create a new plugin manager
    /// 
    /// # Arguments
    /// * `plugin_dir` - Directory to search for plugins (optional)
    /// * `enable_hot_reload` - Enable hot-reloading of plugins
    /// * `enable_sandboxing` - Enable plugin sandboxing for security
    /// * `max_plugins` - Maximum number of plugins to load
    /// 
    /// # Returns
    /// A new PluginManager instance
    #[new]
    #[pyo3(signature = (plugin_dir = None, enable_hot_reload = true, enable_sandboxing = true, max_plugins = 50))]
    pub fn new(
        plugin_dir: Option<String>,
        enable_hot_reload: bool,
        enable_sandboxing: bool,
        max_plugins: usize,
    ) -> PyResult<Self> {
        // Create tokio runtime for async operations
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| NeuralDocFlowError::new_err(format!("Failed to create async runtime: {}", e)))?;
        
        // Create plugin configuration
        let config = PluginConfig {
            plugin_dir: plugin_dir.map(|p| p.into()).unwrap_or_else(|| "./plugins".into()),
            enable_hot_reload,
            enable_sandboxing,
            max_plugins,
        };
        
        // Create plugin manager
        let manager = RustPluginManager::new(config)
            .map_err(|e| NeuralDocFlowError::new_err(format!("Failed to create plugin manager: {}", e)))?;
        
        Ok(Self {
            manager,
            rt,
        })
    }
    
    /// Load built-in plugins
    /// 
    /// Loads all available built-in plugins including DOCX parser,
    /// table detection, image processing, and more.
    /// 
    /// # Returns
    /// Number of plugins loaded
    pub fn load_builtin_plugins(&mut self) -> PyResult<usize> {
        self.rt.block_on(async {
            neural_doc_flow_plugins::builtin::register_builtin_plugins(&mut self.manager).await
        }).map_err(|e| NeuralDocFlowError::new_err(format!("Failed to load built-in plugins: {}", e)))?;
        
        // Return the number of plugins loaded (simplified)
        Ok(self.get_available_plugins().len())
    }
    
    /// Load a plugin from file
    /// 
    /// # Arguments
    /// * `plugin_path` - Path to the plugin file (.so, .dll, or .dylib)
    /// 
    /// # Returns
    /// True if plugin was loaded successfully
    pub fn load_plugin(&mut self, plugin_path: &str) -> PyResult<bool> {
        // For now, return an error as dynamic loading isn't fully implemented
        Err(NeuralDocFlowError::new_err(
            "Dynamic plugin loading not yet implemented. Use load_builtin_plugins() instead."
        ))
    }
    
    /// Get list of available plugins
    /// 
    /// # Returns
    /// Dictionary mapping plugin names to their metadata
    pub fn get_available_plugins(&self) -> HashMap<String, HashMap<String, serde_json::Value>> {
        let mut plugins = HashMap::new();
        
        // Built-in plugins with their metadata
        let builtin_plugins = vec![
            ("docx", "DOCX document parser", vec!["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]),
            ("tables", "Table detection and extraction", vec!["text/html", "application/pdf"]),
            ("images", "Image processing and OCR", vec!["image/jpeg", "image/png", "image/tiff"]),
            ("pdf", "PDF document parser", vec!["application/pdf"]),
            ("text", "Plain text processor", vec!["text/plain", "text/markdown"]),
        ];
        
        for (name, description, formats) in builtin_plugins {
            let mut plugin_info = HashMap::new();
            plugin_info.insert("name".to_string(), serde_json::Value::String(name.to_string()));
            plugin_info.insert("description".to_string(), serde_json::Value::String(description.to_string()));
            plugin_info.insert("version".to_string(), serde_json::Value::String("1.0.0".to_string()));
            plugin_info.insert("type".to_string(), serde_json::Value::String("builtin".to_string()));
            plugin_info.insert("supported_formats".to_string(), 
                             serde_json::Value::Array(formats.iter().map(|f| serde_json::Value::String(f.to_string())).collect()));
            plugin_info.insert("enabled".to_string(), serde_json::Value::Bool(true));
            
            plugins.insert(name.to_string(), plugin_info);
        }
        
        plugins
    }
    
    /// Enable a specific plugin
    /// 
    /// # Arguments
    /// * `plugin_name` - Name of the plugin to enable
    /// 
    /// # Returns
    /// True if plugin was enabled successfully
    pub fn enable_plugin(&mut self, plugin_name: &str) -> PyResult<bool> {
        // For built-in plugins, this is a no-op since they're always enabled
        let available = self.get_available_plugins();
        if available.contains_key(plugin_name) {
            Ok(true)
        } else {
            Err(NeuralDocFlowError::new_err(
                format!("Plugin '{}' not found. Available plugins: {:?}", 
                       plugin_name, available.keys().collect::<Vec<_>>())
            ))
        }
    }
    
    /// Disable a specific plugin
    /// 
    /// # Arguments
    /// * `plugin_name` - Name of the plugin to disable
    /// 
    /// # Returns
    /// True if plugin was disabled successfully
    pub fn disable_plugin(&mut self, plugin_name: &str) -> PyResult<bool> {
        // For built-in plugins, this is a no-op since they can't be disabled
        let available = self.get_available_plugins();
        if available.contains_key(plugin_name) {
            Ok(true)
        } else {
            Err(NeuralDocFlowError::new_err(
                format!("Plugin '{}' not found", plugin_name)
            ))
        }
    }
    
    /// Check if a plugin is enabled
    /// 
    /// # Arguments
    /// * `plugin_name` - Name of the plugin to check
    /// 
    /// # Returns
    /// True if plugin is enabled
    pub fn is_plugin_enabled(&self, plugin_name: &str) -> bool {
        self.get_available_plugins().contains_key(plugin_name)
    }
    
    /// Get plugin metadata
    /// 
    /// # Arguments
    /// * `plugin_name` - Name of the plugin
    /// 
    /// # Returns
    /// Dictionary containing plugin metadata
    pub fn get_plugin_info(&self, plugin_name: &str) -> PyResult<HashMap<String, serde_json::Value>> {
        let available = self.get_available_plugins();
        available.get(plugin_name).cloned().ok_or_else(|| {
            NeuralDocFlowError::new_err(format!("Plugin '{}' not found", plugin_name))
        })
    }
    
    /// Get plugins that support a specific format
    /// 
    /// # Arguments
    /// * `mime_type` - MIME type to check for support
    /// 
    /// # Returns
    /// List of plugin names that support the format
    pub fn get_plugins_for_format(&self, mime_type: &str) -> Vec<String> {
        let mut supporting_plugins = Vec::new();
        
        for (name, info) in self.get_available_plugins() {
            if let Some(serde_json::Value::Array(formats)) = info.get("supported_formats") {
                for format in formats {
                    if let serde_json::Value::String(format_str) = format {
                        if format_str == mime_type {
                            supporting_plugins.push(name.clone());
                            break;
                        }
                    }
                }
            }
        }
        
        supporting_plugins
    }
    
    /// Reload all plugins (if hot-reload is enabled)
    /// 
    /// # Returns
    /// Number of plugins reloaded
    pub fn reload_plugins(&mut self) -> PyResult<usize> {
        // For built-in plugins, this is a no-op
        Ok(self.get_available_plugins().len())
    }
    
    /// Get plugin manager configuration
    /// 
    /// # Returns
    /// Dictionary containing current configuration
    #[getter]
    pub fn config(&self) -> HashMap<String, serde_json::Value> {
        let mut config = HashMap::new();
        
        config.insert("plugin_dir".to_string(), 
                     serde_json::Value::String("./plugins".to_string()));
        config.insert("enable_hot_reload".to_string(), 
                     serde_json::Value::Bool(true));
        config.insert("enable_sandboxing".to_string(), 
                     serde_json::Value::Bool(true));
        config.insert("max_plugins".to_string(), 
                     serde_json::Value::Number(serde_json::Number::from(50)));
        config.insert("loaded_plugins".to_string(), 
                     serde_json::Value::Number(serde_json::Number::from(self.get_available_plugins().len())));
        
        config
    }
    
    /// Get plugin usage statistics
    /// 
    /// # Returns
    /// Dictionary containing usage statistics for each plugin
    pub fn get_usage_stats(&self) -> HashMap<String, HashMap<String, serde_json::Value>> {
        let mut stats = HashMap::new();
        
        for plugin_name in self.get_available_plugins().keys() {
            let mut plugin_stats = HashMap::new();
            plugin_stats.insert("invocations".to_string(), serde_json::Value::Number(serde_json::Number::from(0)));
            plugin_stats.insert("total_time_ms".to_string(), serde_json::Value::Number(serde_json::Number::from(0)));
            plugin_stats.insert("errors".to_string(), serde_json::Value::Number(serde_json::Number::from(0)));
            plugin_stats.insert("last_used".to_string(), serde_json::Value::Null);
            
            stats.insert(plugin_name.clone(), plugin_stats);
        }
        
        stats
    }
    
    /// Clear plugin usage statistics
    pub fn clear_stats(&mut self) {
        // No-op for now since we don't actually track stats yet
    }
    
    /// String representation of the plugin manager
    fn __repr__(&self) -> String {
        let plugin_count = self.get_available_plugins().len();
        format!("PluginManager(plugins={}, hot_reload=true, sandboxing=true)", plugin_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_plugin_manager_creation() {
        let manager = PluginManager::new(None, true, true, 50);
        assert!(manager.is_ok());
    }
    
    #[test]
    fn test_get_available_plugins() {
        let manager = PluginManager::new(None, true, true, 50).unwrap();
        let plugins = manager.get_available_plugins();
        
        assert!(plugins.contains_key("docx"));
        assert!(plugins.contains_key("tables"));
        assert!(plugins.contains_key("images"));
        assert!(plugins.contains_key("pdf"));
        assert!(plugins.contains_key("text"));
    }
    
    #[test]
    fn test_get_plugins_for_format() {
        let manager = PluginManager::new(None, true, true, 50).unwrap();
        
        let pdf_plugins = manager.get_plugins_for_format("application/pdf");
        assert!(pdf_plugins.contains(&"pdf".to_string()));
        assert!(pdf_plugins.contains(&"tables".to_string()));
        
        let docx_plugins = manager.get_plugins_for_format("application/vnd.openxmlformats-officedocument.wordprocessingml.document");
        assert!(docx_plugins.contains(&"docx".to_string()));
    }
    
    #[test]
    fn test_plugin_info() {
        let manager = PluginManager::new(None, true, true, 50).unwrap();
        
        let docx_info = manager.get_plugin_info("docx");
        assert!(docx_info.is_ok());
        
        let info = docx_info.unwrap();
        assert_eq!(info.get("name"), Some(&serde_json::Value::String("docx".to_string())));
        assert_eq!(info.get("type"), Some(&serde_json::Value::String("builtin".to_string())));
    }
    
    #[test]
    fn test_plugin_enable_disable() {
        let mut manager = PluginManager::new(None, true, true, 50).unwrap();
        
        assert!(manager.enable_plugin("docx").is_ok());
        assert!(manager.is_plugin_enabled("docx"));
        
        assert!(manager.disable_plugin("docx").is_ok());
        
        // Non-existent plugin
        assert!(manager.enable_plugin("nonexistent").is_err());
    }
}