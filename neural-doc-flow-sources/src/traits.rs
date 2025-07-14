//! Source traits and common types
//!
//! Re-exports the DocumentSource trait from core and provides common
//! implementations and utilities for source plugins.

use neural_doc_flow_core::prelude::*;
use async_trait::async_trait;
use std::collections::HashMap;

// Re-export the DocumentSource trait from core
pub use neural_doc_flow_core::traits::DocumentSource;

/// Source capabilities that can be reported by plugins
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum SourceCapability {
    /// Can parse PDF documents
    Pdf,
    /// Can parse plain text documents
    PlainText,
    /// Can parse DOCX documents
    Docx,
    /// Can parse Markdown documents
    Markdown,
    /// Can parse HTML documents
    Html,
    /// Can parse XML documents
    Xml,
    /// Can fetch remote documents
    Remote,
    /// Can monitor file system changes
    FileSystemMonitoring,
    /// Can extract metadata
    MetadataExtraction,
    /// Can extract embedded resources
    ResourceExtraction,
    /// Can perform security validation on sources
    SecurityValidation,
    /// Can scan for malware before processing
    MalwareScanning,
}

/// Source metadata for plugin registration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SourceMetadata {
    /// Unique identifier for the source
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Version of the source implementation
    pub version: String,
    /// List of capabilities this source provides
    pub capabilities: Vec<SourceCapability>,
    /// Supported file extensions (if applicable)
    pub file_extensions: Vec<String>,
    /// Supported MIME types
    pub mime_types: Vec<String>,
}

/// Common source configuration options
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SourceConfig {
    /// Maximum file size to process (in bytes)
    pub max_file_size: Option<usize>,
    /// Encoding to use for text extraction
    pub encoding: Option<String>,
    /// Whether to extract metadata
    pub extract_metadata: bool,
    /// Whether to extract embedded resources
    pub extract_resources: bool,
    /// Enable security validation before processing
    pub enable_security_validation: bool,
    /// Security scan timeout in seconds
    pub security_scan_timeout: Option<u64>,
    /// Custom configuration options
    pub custom: HashMap<String, serde_json::Value>,
}

impl Default for SourceConfig {
    fn default() -> Self {
        Self {
            max_file_size: Some(100 * 1024 * 1024), // 100MB default
            encoding: None,
            extract_metadata: true,
            extract_resources: false,
            enable_security_validation: true,
            security_scan_timeout: Some(30), // 30 seconds default
            custom: HashMap::new(),
        }
    }
}

/// Base implementation helper for document sources
#[async_trait]
pub trait BaseDocumentSource: DocumentSource {
    /// Get metadata about this source
    fn metadata(&self) -> SourceMetadata;
    
    /// Validate that this source can handle the given input
    async fn can_handle(&self, input: &str) -> bool;
    
    /// Get the current configuration
    fn config(&self) -> &SourceConfig;
    
    /// Update the configuration
    fn set_config(&mut self, config: SourceConfig);
    
    /// Perform security validation on the source before processing
    async fn validate_security(&self, input: &str) -> SourceResult<bool> {
        // Default implementation - can be overridden by sources
        if self.config().enable_security_validation {
            // Basic security checks
            self.basic_security_validation(input).await
        } else {
            Ok(true)
        }
    }
    
    /// Basic security validation (can be overridden)
    async fn basic_security_validation(&self, input: &str) -> SourceResult<bool> {
        // Check for obvious security issues
        if input.contains("javascript:") 
           || input.contains("vbscript:") 
           || input.contains("data:text/html") {
            return Ok(false);
        }
        
        // Check file size if it's a file path
        if let Ok(metadata) = tokio::fs::metadata(input).await {
            if let Some(max_size) = self.config().max_file_size {
                if metadata.len() as usize > max_size {
                    return Err(SourceError::FileTooLarge {
                        size: metadata.len() as usize,
                        max: max_size,
                    });
                }
            }
        }
        
        Ok(true)
    }
}

/// Error types specific to source operations
#[derive(Debug, thiserror::Error)]
pub enum SourceError {
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),
    
    #[error("File too large: {size} bytes (max: {max} bytes)")]
    FileTooLarge { size: usize, max: usize },
    
    #[error("Encoding error: {0}")]
    EncodingError(String),
    
    #[error("Parse error: {0}")]
    ParseError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Network error: {0}")]
    NetworkError(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("Security validation failed: {0}")]
    SecurityValidationFailed(String),
}

/// Result type for source operations
pub type SourceResult<T> = Result<T, SourceError>;