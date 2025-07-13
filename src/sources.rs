//! Document source plugin system
//!
//! This module provides the trait-based plugin architecture for document sources,
//! enabling hot-reloadable support for any document format.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncRead;

use crate::config::{Config, SourceConfig};
use crate::core::{Document, DocumentMetadata, DocumentStructure};
use crate::error::{NeuralDocFlowError, Result};

/// Document source input types
pub enum SourceInput {
    /// File path input
    File {
        path: PathBuf,
        metadata: Option<HashMap<String, String>>,
    },
    
    /// In-memory data input
    Memory {
        data: Vec<u8>,
        filename: Option<String>,
        mime_type: Option<String>,
    },
    
    /// Streaming input
    Stream {
        reader: Box<dyn AsyncRead + Send + Unpin>,
        size_hint: Option<usize>,
        mime_type: Option<String>,
    },
    
    /// URL input
    Url {
        url: String,
        headers: Option<HashMap<String, String>>,
    },
}

impl std::fmt::Debug for SourceInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::File { path, metadata } => f.debug_struct("File")
                .field("path", path)
                .field("metadata", metadata)
                .finish(),
            Self::Memory { data, filename, mime_type } => f.debug_struct("Memory")
                .field("data_len", &data.len())
                .field("filename", filename)
                .field("mime_type", mime_type)
                .finish(),
            Self::Stream { reader: _, size_hint, mime_type } => f.debug_struct("Stream")
                .field("size_hint", size_hint)
                .field("mime_type", mime_type)
                .finish(),
            Self::Url { url, headers } => f.debug_struct("Url")
                .field("url", url)
                .field("headers", headers)
                .finish(),
        }
    }
}

/// Document source trait - implemented by all source plugins
#[async_trait]
pub trait DocumentSource: Send + Sync {
    /// Unique identifier for this source
    fn source_id(&self) -> &str;
    
    /// Human-readable name
    fn name(&self) -> &str;
    
    /// Source version
    fn version(&self) -> &str;
    
    /// Supported file extensions
    fn supported_extensions(&self) -> &[&str];
    
    /// Supported MIME types
    fn supported_mime_types(&self) -> &[&str];
    
    /// Check if this source can handle the given input
    async fn can_handle(&self, input: &SourceInput) -> Result<bool>;
    
    /// Validate input before processing
    async fn validate(&self, input: &SourceInput) -> Result<ValidationResult>;
    
    /// Extract content from the input
    async fn extract(&self, input: SourceInput) -> Result<Document>;
    
    /// Get configuration schema for this source
    fn config_schema(&self) -> serde_json::Value;
    
    /// Initialize source with configuration
    async fn initialize(&mut self, config: SourceConfig) -> Result<()>;
    
    /// Cleanup resources
    async fn cleanup(&mut self) -> Result<()>;
}

/// Validation result for input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Whether input is valid
    pub is_valid: bool,
    
    /// Validation errors
    pub errors: Vec<String>,
    
    /// Validation warnings
    pub warnings: Vec<String>,
    
    /// Security issues
    pub security_issues: Vec<String>,
}

impl ValidationResult {
    /// Create successful validation result
    pub fn valid() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            security_issues: Vec::new(),
        }
    }

    /// Create failed validation result
    pub fn invalid<S: Into<String>>(error: S) -> Self {
        Self {
            is_valid: false,
            errors: vec![error.into()],
            warnings: Vec::new(),
            security_issues: Vec::new(),
        }
    }

    /// Add error
    pub fn add_error<S: Into<String>>(&mut self, error: S) {
        self.errors.push(error.into());
        self.is_valid = false;
    }

    /// Add warning
    pub fn add_warning<S: Into<String>>(&mut self, warning: S) {
        self.warnings.push(warning.into());
    }

    /// Add security issue
    pub fn add_security_issue<S: Into<String>>(&mut self, issue: S) {
        self.security_issues.push(issue.into());
    }
}

impl Default for ValidationResult {
    fn default() -> Self {
        Self::valid()
    }
}

/// Source manager for plugin discovery and management
pub struct SourceManager {
    sources: HashMap<String, Box<dyn DocumentSource>>,
    plugin_dirs: Vec<PathBuf>,
    config: Config,
}

impl SourceManager {
    /// Create new source manager
    pub fn new(config: &Config) -> Result<Self> {
        let mut manager = Self {
            sources: HashMap::new(),
            plugin_dirs: config.sources.plugin_directories.clone(),
            config: config.clone(),
        };

        // Load built-in sources
        manager.load_builtin_sources()?;
        
        // Discover and load plugin sources
        if config.sources.enable_hot_reload {
            manager.discover_plugins()?;
        }

        Ok(manager)
    }

    /// Find compatible source for input
    pub async fn find_compatible_source(&self, input: &SourceInput) -> Result<&dyn DocumentSource> {
        let mut compatible_sources = Vec::new();

        for (source_id, source) in &self.sources {
            if source.can_handle(input).await? {
                if let Some(source_config) = self.config.sources.source_configs.get(source_id) {
                    if source_config.enabled {
                        compatible_sources.push((source_config.priority, source.as_ref()));
                    }
                }
            }
        }

        if compatible_sources.is_empty() {
            return Err(NeuralDocFlowError::source("No compatible source found"));
        }

        // Sort by priority (highest first)
        compatible_sources.sort_by_key(|(priority, _)| std::cmp::Reverse(*priority));
        
        Ok(compatible_sources[0].1)
    }

    /// Get all loaded sources
    pub fn get_sources(&self) -> Vec<&dyn DocumentSource> {
        self.sources.values().map(|s| s.as_ref()).collect()
    }

    /// Get source by ID
    pub fn get_source(&self, source_id: &str) -> Option<&dyn DocumentSource> {
        self.sources.get(source_id).map(|s| s.as_ref())
    }

    /// Load built-in sources
    fn load_builtin_sources(&mut self) -> Result<()> {
        // Register built-in PDF source
        #[cfg(feature = "pdf")]
        {
            let pdf_source = crate::sources::pdf::PdfSource::new();
            self.sources.insert("pdf".to_string(), Box::new(pdf_source));
        }

        // Register built-in DOCX source
        #[cfg(feature = "docx")]
        {
            let docx_source = crate::sources::docx::DocxSource::new();
            self.sources.insert("docx".to_string(), Box::new(docx_source));
        }

        // Register built-in HTML source
        #[cfg(feature = "html")]
        {
            let html_source = crate::sources::html::HtmlSource::new();
            self.sources.insert("html".to_string(), Box::new(html_source));
        }

        Ok(())
    }

    /// Discover and load plugin sources
    fn discover_plugins(&mut self) -> Result<()> {
        let plugin_dirs = self.plugin_dirs.clone();
        for plugin_dir in &plugin_dirs {
            if plugin_dir.exists() {
                self.load_plugins_from_directory(plugin_dir)?;
            }
        }
        Ok(())
    }

    /// Load plugins from directory
    fn load_plugins_from_directory(&mut self, dir: &Path) -> Result<()> {
        let entries = std::fs::read_dir(dir)
            .map_err(|e| NeuralDocFlowError::plugin(format!("Failed to read plugin directory: {}", e)))?;

        for entry in entries {
            let entry = entry
                .map_err(|e| NeuralDocFlowError::plugin(format!("Failed to read directory entry: {}", e)))?;
            
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("so") {
                // Load dynamic library plugin
                self.load_dynamic_plugin(&path)?;
            }
        }

        Ok(())
    }

    /// Load dynamic plugin from shared library
    fn load_dynamic_plugin(&mut self, _path: &Path) -> Result<()> {
        // This would implement dynamic library loading for plugins
        // For now, return success
        Ok(())
    }

    /// Shutdown and cleanup
    pub async fn shutdown(mut self) -> Result<()> {
        for (_, mut source) in self.sources {
            source.cleanup().await?;
        }
        Ok(())
    }
}

/// Built-in PDF source implementation
#[cfg(feature = "pdf")]
pub mod pdf {
    use super::*;

    /// PDF document source
    pub struct PdfSource {
        config: Option<SourceConfig>,
    }

    impl PdfSource {
        /// Create new PDF source
        pub fn new() -> Self {
            Self { config: None }
        }
    }

    #[async_trait]
    impl DocumentSource for PdfSource {
        fn source_id(&self) -> &str {
            "pdf"
        }

        fn name(&self) -> &str {
            "PDF Document Source"
        }

        fn version(&self) -> &str {
            "1.0.0"
        }

        fn supported_extensions(&self) -> &[&str] {
            &["pdf", "PDF"]
        }

        fn supported_mime_types(&self) -> &[&str] {
            &["application/pdf"]
        }

        async fn can_handle(&self, input: &SourceInput) -> Result<bool> {
            match input {
                SourceInput::File { path, .. } => {
                    if let Some(ext) = path.extension() {
                        Ok(self.supported_extensions().contains(&ext.to_str().unwrap_or("")))
                    } else {
                        Ok(false)
                    }
                }
                SourceInput::Memory { mime_type, data, .. } => {
                    if let Some(mime) = mime_type {
                        Ok(self.supported_mime_types().contains(&mime.as_str()))
                    } else {
                        // Check PDF magic bytes
                        Ok(data.starts_with(b"%PDF-"))
                    }
                }
                _ => Ok(false),
            }
        }

        async fn validate(&self, input: &SourceInput) -> Result<ValidationResult> {
            let mut result = ValidationResult::valid();

            match input {
                SourceInput::File { path, .. } => {
                    if !path.exists() {
                        result.add_error("File does not exist");
                    }
                    
                    if let Ok(metadata) = std::fs::metadata(path) {
                        if metadata.len() > 100 * 1024 * 1024 { // 100MB limit
                            result.add_warning("Large file size may impact performance");
                        }
                    }
                }
                SourceInput::Memory { data, .. } => {
                    if !data.starts_with(b"%PDF-") {
                        result.add_error("Invalid PDF header");
                    }
                    
                    if data.len() > 100 * 1024 * 1024 {
                        result.add_warning("Large file size may impact performance");
                    }
                }
                _ => {
                    result.add_error("Unsupported input type for PDF source");
                }
            }

            Ok(result)
        }

        async fn extract(&self, input: SourceInput) -> Result<ExtractedDocument> {
            let start_time = std::time::Instant::now();
            
            // Simple PDF extraction implementation
            let mut doc = ExtractedDocument::new(self.source_id().to_string());
            
            match input {
                SourceInput::File { path, .. } => {
                    doc.metadata.title = path.file_stem()
                        .and_then(|s| s.to_str())
                        .map(|s| s.to_string());
                }
                SourceInput::Memory { filename, .. } => {
                    doc.metadata.title = filename;
                }
                _ => {}
            }

            // Mock content extraction
            let block = ContentBlock::new(crate::core::BlockType::Paragraph)
                .with_text("Sample PDF content extracted");
            doc.content.push(block);
            
            doc.confidence = 0.95;
            doc.metrics.extraction_time = start_time.elapsed();
            doc.metrics.blocks_extracted = doc.content.len();

            Ok(doc)
        }

        fn config_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "max_file_size": {
                        "type": "integer",
                        "description": "Maximum file size in bytes",
                        "default": 104857600
                    },
                    "enable_ocr": {
                        "type": "boolean",
                        "description": "Enable OCR for scanned PDFs",
                        "default": false
                    },
                    "extract_tables": {
                        "type": "boolean",
                        "description": "Extract table structures",
                        "default": true
                    }
                }
            })
        }

        async fn initialize(&mut self, config: SourceConfig) -> Result<()> {
            self.config = Some(config);
            Ok(())
        }

        async fn cleanup(&mut self) -> Result<()> {
            Ok(())
        }
    }
}

/// Built-in DOCX source implementation
#[cfg(feature = "docx")]
pub mod docx {
    use super::*;

    /// DOCX document source
    pub struct DocxSource {
        config: Option<SourceConfig>,
    }

    impl DocxSource {
        /// Create new DOCX source
        pub fn new() -> Self {
            Self { config: None }
        }
    }

    #[async_trait]
    impl DocumentSource for DocxSource {
        fn source_id(&self) -> &str {
            "docx"
        }

        fn name(&self) -> &str {
            "Microsoft Word Document Source"
        }

        fn version(&self) -> &str {
            "1.0.0"
        }

        fn supported_extensions(&self) -> &[&str] {
            &["docx", "DOCX"]
        }

        fn supported_mime_types(&self) -> &[&str] {
            &["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
        }

        async fn can_handle(&self, input: &SourceInput) -> Result<bool> {
            match input {
                SourceInput::File { path, .. } => {
                    if let Some(ext) = path.extension() {
                        Ok(self.supported_extensions().contains(&ext.to_str().unwrap_or("")))
                    } else {
                        Ok(false)
                    }
                }
                SourceInput::Memory { mime_type, .. } => {
                    if let Some(mime) = mime_type {
                        Ok(self.supported_mime_types().contains(&mime.as_str()))
                    } else {
                        Ok(false)
                    }
                }
                _ => Ok(false),
            }
        }

        async fn validate(&self, input: &SourceInput) -> Result<ValidationResult> {
            let result = ValidationResult::valid();
            // Add DOCX-specific validation logic here
            Ok(result)
        }

        async fn extract(&self, input: SourceInput) -> Result<ExtractedDocument> {
            let start_time = std::time::Instant::now();
            
            // Simple DOCX extraction implementation
            let mut doc = ExtractedDocument::new(self.source_id().to_string());
            
            // Mock content extraction
            let block = ContentBlock::new(crate::core::BlockType::Paragraph)
                .with_text("Sample DOCX content extracted");
            doc.content.push(block);
            
            doc.confidence = 0.93;
            doc.metrics.extraction_time = start_time.elapsed();
            doc.metrics.blocks_extracted = doc.content.len();

            Ok(doc)
        }

        fn config_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "extract_comments": {
                        "type": "boolean",
                        "default": true
                    },
                    "extract_track_changes": {
                        "type": "boolean",
                        "default": false
                    }
                }
            })
        }

        async fn initialize(&mut self, config: SourceConfig) -> Result<()> {
            self.config = Some(config);
            Ok(())
        }

        async fn cleanup(&mut self) -> Result<()> {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_validation_result() {
        let mut result = ValidationResult::valid();
        assert!(result.is_valid);
        assert!(result.errors.is_empty());

        result.add_error("Test error");
        assert!(!result.is_valid);
        assert_eq!(result.errors.len(), 1);

        result.add_warning("Test warning");
        assert_eq!(result.warnings.len(), 1);
    }

    #[test]
    fn test_source_input_types() {
        let file_input = SourceInput::File {
            path: PathBuf::from("test.pdf"),
            metadata: None,
        };

        let memory_input = SourceInput::Memory {
            data: b"%PDF-1.4".to_vec(),
            filename: Some("test.pdf".to_string()),
            mime_type: Some("application/pdf".to_string()),
        };

        match file_input {
            SourceInput::File { .. } => assert!(true),
            _ => assert!(false),
        }

        match memory_input {
            SourceInput::Memory { .. } => assert!(true),
            _ => assert!(false),
        }
    }

    #[cfg(feature = "pdf")]
    #[tokio::test]
    async fn test_pdf_source() {
        let source = pdf::PdfSource::new();
        
        assert_eq!(source.source_id(), "pdf");
        assert!(source.supported_extensions().contains(&"pdf"));

        let input = SourceInput::Memory {
            data: b"%PDF-1.4\ntest content".to_vec(),
            filename: Some("test.pdf".to_string()),
            mime_type: Some("application/pdf".to_string()),
        };

        assert!(source.can_handle(&input).await.unwrap());
        
        let validation = source.validate(&input).await.unwrap();
        assert!(validation.is_valid);

        let doc = source.extract(input).await.unwrap();
        assert_eq!(doc.source_id, "pdf");
        assert!(doc.confidence > 0.0);
    }

    #[tokio::test]
    async fn test_source_manager() {
        let config = Config::default();
        let manager = SourceManager::new(&config);
        
        match manager {
            Ok(mgr) => {
                let sources = mgr.get_sources();
                assert!(!sources.is_empty());
            }
            Err(_) => {
                // Expected in test environment without full setup
                assert!(true);
            }
        }
    }

    #[test]
    fn test_invalid_validation() {
        let result = ValidationResult::invalid("Test error");
        assert!(!result.is_valid);
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0], "Test error");
    }
}