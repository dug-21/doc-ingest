//! Main document processor for Python bindings

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use neural_doc_flow_core::{
    engine::DocumentEngine,
    config::{NeuralDocFlowConfig, SecurityConfig, SecurityScanMode},
    error::ProcessingError,
};
use neural_doc_flow_security::SecurityProcessor as RustSecurityProcessor;
use neural_doc_flow_plugins::{PluginManager, PluginConfig};

use crate::{ProcessingResult, NeuralDocFlowError, SecurityAnalysis};

/// Main document processor class
/// 
/// This class provides the primary interface for processing documents
/// with neural enhancement, security analysis, and plugin support.
/// 
/// # Example
/// ```python
/// import neuraldocflow
/// 
/// # Create processor with high security
/// processor = neuraldocflow.Processor(
///     security_level="high",
///     enable_neural=True,
///     plugins=["docx", "tables", "images"]
/// )
/// 
/// # Process a document
/// result = processor.process_document("document.pdf", output_format="json")
/// 
/// # Access results
/// print(result.text)
/// print(result.tables)
/// print(result.security_analysis)
/// ```
#[pyclass]
pub struct Processor {
    engine: DocumentEngine,
    plugin_manager: Option<PluginManager>,
    config: NeuralDocFlowConfig,
    rt: tokio::runtime::Runtime,
}

#[pymethods]
impl Processor {
    /// Create a new document processor
    /// 
    /// # Arguments
    /// * `security_level` - Security level: "disabled", "basic", "standard", "high"
    /// * `enable_neural` - Enable neural processing features
    /// * `plugins` - List of plugin names to enable
    /// 
    /// # Returns
    /// A new Processor instance
    #[new]
    #[pyo3(signature = (security_level = "standard", enable_neural = true, plugins = None))]
    pub fn new(
        security_level: &str,
        enable_neural: bool,
        plugins: Option<Vec<String>>,
    ) -> PyResult<Self> {
        // Create tokio runtime for async operations
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| NeuralDocFlowError::new_err(format!("Failed to create async runtime: {}", e)))?;
        
        // Create configuration
        let mut config = NeuralDocFlowConfig::default();
        
        // Configure security
        config.security = SecurityConfig {
            enabled: security_level != "disabled",
            scan_mode: match security_level {
                "disabled" => SecurityScanMode::Disabled,
                "basic" => SecurityScanMode::Basic,
                "standard" => SecurityScanMode::Standard,
                "high" | "comprehensive" => SecurityScanMode::Comprehensive,
                _ => return Err(NeuralDocFlowError::new_err(
                    format!("Invalid security level: {}. Use 'disabled', 'basic', 'standard', or 'high'", security_level)
                )),
            },
            policies: Default::default(),
        };
        
        // Configure neural features
        config.neural.enabled = enable_neural;
        
        // Create document engine
        let mut engine = DocumentEngine::new(config.clone())
            .map_err(|e| NeuralDocFlowError::new_err(format!("Failed to create document engine: {}", e)))?;
        
        // Set up security processor if enabled
        if config.security.enabled {
            let security_processor = RustSecurityProcessor::new()
                .map_err(|e| NeuralDocFlowError::new_err(format!("Failed to create security processor: {}", e)))?;
            engine.set_security_processor(Arc::new(RwLock::new(security_processor)));
        }
        
        // Set up plugin manager if plugins are specified
        let plugin_manager = if let Some(plugin_names) = plugins {
            let mut manager = PluginManager::new(PluginConfig::default())
                .map_err(|e| NeuralDocFlowError::new_err(format!("Failed to create plugin manager: {}", e)))?;
            
            // Load built-in plugins
            rt.block_on(async {
                neural_doc_flow_plugins::builtin::register_builtin_plugins(&mut manager).await
            }).map_err(|e| NeuralDocFlowError::new_err(format!("Failed to register built-in plugins: {}", e)))?;
            
            Some(manager)
        } else {
            None
        };
        
        Ok(Self {
            engine,
            plugin_manager,
            config,
            rt,
        })
    }
    
    /// Process a document from file path
    /// 
    /// # Arguments
    /// * `file_path` - Path to the document file
    /// * `output_format` - Output format: "json", "markdown", "html", "xml"
    /// 
    /// # Returns
    /// ProcessingResult containing extracted content and metadata
    /// 
    /// # Example
    /// ```python
    /// result = processor.process_document("document.pdf", output_format="json")
    /// print(result.text)
    /// ```
    #[pyo3(signature = (file_path, output_format = "json"))]
    pub fn process_document(&self, file_path: &str, output_format: &str) -> PyResult<ProcessingResult> {
        // Read file
        let content = std::fs::read(file_path)
            .map_err(|e| NeuralDocFlowError::new_err(format!("Failed to read file {}: {}", file_path, e)))?;
        
        // Detect MIME type from file extension
        let mime_type = self.detect_mime_type(file_path);
        
        // Process the document
        self.process_bytes(&content, &mime_type, output_format)
    }
    
    /// Process a document from byte data
    /// 
    /// # Arguments
    /// * `data` - Document data as bytes
    /// * `mime_type` - MIME type of the document
    /// * `output_format` - Output format: "json", "markdown", "html", "xml"
    /// 
    /// # Returns
    /// ProcessingResult containing extracted content and metadata
    /// 
    /// # Example
    /// ```python
    /// with open("document.pdf", "rb") as f:
    ///     data = f.read()
    /// result = processor.process_bytes(data, "application/pdf", "json")
    /// ```
    #[pyo3(signature = (data, mime_type, output_format = "json"))]
    pub fn process_bytes(&self, data: &PyBytes, mime_type: &str, output_format: &str) -> PyResult<ProcessingResult> {
        let bytes = data.as_bytes().to_vec();
        
        // Process the document asynchronously
        let document = self.rt.block_on(async {
            self.engine.process(bytes, mime_type).await
        }).map_err(|e| NeuralDocFlowError::new_err(format!("Document processing failed: {}", e)))?;
        
        // Convert to ProcessingResult
        ProcessingResult::from_document(document, output_format)
    }
    
    /// Process multiple documents in batch
    /// 
    /// # Arguments
    /// * `file_paths` - List of file paths to process
    /// * `output_format` - Output format for all documents
    /// 
    /// # Returns
    /// List of ProcessingResult objects
    /// 
    /// # Example
    /// ```python
    /// results = processor.process_batch(
    ///     ["doc1.pdf", "doc2.docx", "doc3.txt"],
    ///     output_format="json"
    /// )
    /// for result in results:
    ///     print(result.text)
    /// ```
    #[pyo3(signature = (file_paths, output_format = "json"))]
    pub fn process_batch(&self, file_paths: Vec<&str>, output_format: &str) -> PyResult<Vec<ProcessingResult>> {
        let mut results = Vec::new();
        
        for file_path in file_paths {
            match self.process_document(file_path, output_format) {
                Ok(result) => results.push(result),
                Err(e) => {
                    // For batch processing, we collect errors in the result rather than failing
                    let error_result = ProcessingResult::new_error(
                        file_path.to_string(),
                        e.to_string()
                    );
                    results.push(error_result);
                }
            }
        }
        
        Ok(results)
    }
    
    /// Get processor configuration
    /// 
    /// # Returns
    /// Dictionary containing current processor configuration
    #[getter]
    pub fn config(&self) -> HashMap<String, serde_json::Value> {
        let mut config_map = HashMap::new();
        
        config_map.insert("security_enabled".to_string(), 
                         serde_json::Value::Bool(self.config.security.enabled));
        config_map.insert("security_scan_mode".to_string(),
                         serde_json::Value::String(format!("{:?}", self.config.security.scan_mode)));
        config_map.insert("neural_enabled".to_string(),
                         serde_json::Value::Bool(self.config.neural.enabled));
        config_map.insert("plugins_enabled".to_string(),
                         serde_json::Value::Bool(self.plugin_manager.is_some()));
        
        config_map
    }
    
    /// Get list of available plugins
    /// 
    /// # Returns
    /// List of plugin names and their status
    pub fn get_available_plugins(&self) -> HashMap<String, HashMap<String, serde_json::Value>> {
        let mut plugins = HashMap::new();
        
        // Built-in plugins
        let builtin_plugins = vec![
            ("docx", "DOCX document parser"),
            ("tables", "Table detection and extraction"),
            ("images", "Image processing and OCR"),
            ("pdf", "PDF document parser"),
            ("text", "Plain text processor"),
        ];
        
        for (name, description) in builtin_plugins {
            let mut plugin_info = HashMap::new();
            plugin_info.insert("description".to_string(), serde_json::Value::String(description.to_string()));
            plugin_info.insert("type".to_string(), serde_json::Value::String("builtin".to_string()));
            plugin_info.insert("enabled".to_string(), serde_json::Value::Bool(self.plugin_manager.is_some()));
            plugins.insert(name.to_string(), plugin_info);
        }
        
        plugins
    }
    
    /// Enable or disable neural processing
    /// 
    /// # Arguments
    /// * `enabled` - Whether to enable neural processing
    pub fn set_neural_enabled(&mut self, enabled: bool) {
        self.config.neural.enabled = enabled;
    }
    
    /// Get current neural processing status
    /// 
    /// # Returns
    /// True if neural processing is enabled
    #[getter]
    pub fn neural_enabled(&self) -> bool {
        self.config.neural.enabled
    }
    
    /// Get current security status
    /// 
    /// # Returns
    /// Dictionary with security configuration
    #[getter]
    pub fn security_status(&self) -> HashMap<String, serde_json::Value> {
        let mut status = HashMap::new();
        
        status.insert("enabled".to_string(), 
                     serde_json::Value::Bool(self.config.security.enabled));
        status.insert("scan_mode".to_string(),
                     serde_json::Value::String(format!("{:?}", self.config.security.scan_mode)));
        status.insert("max_file_size_mb".to_string(),
                     serde_json::Value::Number(serde_json::Number::from(self.config.security.policies.max_file_size_mb)));
        
        status
    }
    
    /// String representation of the processor
    fn __repr__(&self) -> String {
        format!(
            "Processor(security={}, neural={}, plugins={})",
            self.config.security.enabled,
            self.config.neural.enabled,
            self.plugin_manager.is_some()
        )
    }
}

impl Processor {
    /// Detect MIME type from file extension
    fn detect_mime_type(&self, file_path: &str) -> String {
        let extension = std::path::Path::new(file_path)
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_lowercase();
            
        match extension.as_str() {
            "pdf" => "application/pdf",
            "docx" => "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "doc" => "application/msword",
            "xlsx" => "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "xls" => "application/vnd.ms-excel",
            "pptx" => "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "ppt" => "application/vnd.ms-powerpoint",
            "txt" => "text/plain",
            "md" => "text/markdown",
            "html" | "htm" => "text/html",
            "json" => "application/json",
            "xml" => "application/xml",
            "jpg" | "jpeg" => "image/jpeg",
            "png" => "image/png",
            "tiff" | "tif" => "image/tiff",
            "bmp" => "image/bmp",
            _ => "application/octet-stream",
        }.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_processor_creation() {
        let processor = Processor::new("standard", true, None);
        assert!(processor.is_ok());
        
        let processor = processor.unwrap();
        assert!(processor.config.security.enabled);
        assert!(processor.config.neural.enabled);
    }
    
    #[test]
    fn test_mime_type_detection() {
        let processor = Processor::new("disabled", false, None).unwrap();
        
        assert_eq!(processor.detect_mime_type("test.pdf"), "application/pdf");
        assert_eq!(processor.detect_mime_type("test.docx"), "application/vnd.openxmlformats-officedocument.wordprocessingml.document");
        assert_eq!(processor.detect_mime_type("test.txt"), "text/plain");
        assert_eq!(processor.detect_mime_type("test.unknown"), "application/octet-stream");
    }
}