//! Document source trait definitions following iteration5 architecture

use crate::{Document, SourceResult};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Core trait for document sources - exactly as specified in iteration5
#[async_trait]
pub trait DocumentSource: Send + Sync {
    /// Get the source type identifier
    fn source_type(&self) -> &'static str;
    
    /// Check if this source can handle the given input
    async fn can_handle(&self, input: &str) -> bool;
    
    /// Load a document from the given input path/URL
    async fn load_document(&self, input: &str) -> SourceResult<Document>;
    
    /// Load multiple documents in batch
    async fn load_documents(&self, inputs: &[String]) -> Vec<SourceResult<Document>> {
        let mut results = Vec::new();
        for input in inputs {
            results.push(self.load_document(input).await);
        }
        results
    }
    
    /// Get metadata without loading the full document
    async fn get_metadata(&self, input: &str) -> SourceResult<DocumentMetadata>;
    
    /// Validate the input source
    async fn validate(&self, input: &str) -> SourceResult<ValidationResult>;
    
    /// Get supported file extensions
    fn supported_extensions(&self) -> Vec<&'static str>;
    
    /// Get supported MIME types
    fn supported_mime_types(&self) -> Vec<&'static str>;
}

/// Document metadata for preview/validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    /// File name or identifier
    pub name: String,
    
    /// File size in bytes
    pub size: Option<u64>,
    
    /// MIME type
    pub mime_type: String,
    
    /// Last modified timestamp
    pub modified: Option<chrono::DateTime<chrono::Utc>>,
    
    /// Additional metadata
    pub attributes: HashMap<String, serde_json::Value>,
}

/// Validation result for input sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Whether the source is valid
    pub is_valid: bool,
    
    /// Validation issues found
    pub issues: Vec<ValidationIssue>,
    
    /// Estimated processing time in seconds
    pub estimated_processing_time: Option<f64>,
    
    /// Confidence in validation (0.0 to 1.0)
    pub confidence: f64,
}

/// Individual validation issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationIssue {
    /// Issue severity
    pub severity: ValidationSeverity,
    
    /// Issue description
    pub message: String,
    
    /// Suggested fix
    pub suggestion: Option<String>,
}

/// Validation issue severity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationSeverity {
    /// Informational
    Info,
    
    /// Warning - processing may have issues
    Warning,
    
    /// Error - processing will likely fail
    Error,
    
    /// Critical - processing will definitely fail
    Critical,
}

/// Factory trait for creating document sources
pub trait DocumentSourceFactory: Send + Sync {
    /// Create a new document source instance
    fn create_source(&self, config: &SourceConfig) -> SourceResult<Box<dyn DocumentSource>>;
    
    /// Get the source type this factory creates
    fn source_type(&self) -> &'static str;
    
    /// Get configuration schema for this source type
    fn config_schema(&self) -> serde_json::Value;
}

/// Configuration for document sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceConfig {
    /// Source type
    pub source_type: String,
    
    /// Source-specific parameters
    pub parameters: HashMap<String, serde_json::Value>,
    
    /// Processing options
    pub options: SourceOptions,
}

/// Options for source processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceOptions {
    /// Enable parallel processing
    pub parallel: bool,
    
    /// Maximum file size to process (bytes)
    pub max_file_size: Option<u64>,
    
    /// Timeout for processing (seconds)
    pub timeout: Option<u64>,
    
    /// Enable caching
    pub cache: bool,
    
    /// Custom processing parameters
    pub custom: HashMap<String, serde_json::Value>,
}

impl Default for SourceOptions {
    fn default() -> Self {
        Self {
            parallel: true,
            max_file_size: Some(100 * 1024 * 1024), // 100MB default
            timeout: Some(300), // 5 minutes default
            cache: true,
            custom: HashMap::new(),
        }
    }
}

impl ValidationResult {
    /// Create a successful validation result
    pub fn success() -> Self {
        Self {
            is_valid: true,
            issues: Vec::new(),
            estimated_processing_time: None,
            confidence: 1.0,
        }
    }
    
    /// Create a failed validation result
    pub fn failure(issues: Vec<ValidationIssue>) -> Self {
        Self {
            is_valid: false,
            issues,
            estimated_processing_time: None,
            confidence: 0.0,
        }
    }
    
    /// Add an issue to the validation result
    pub fn add_issue(&mut self, issue: ValidationIssue) {
        if matches!(issue.severity, ValidationSeverity::Error | ValidationSeverity::Critical) {
            self.is_valid = false;
        }
        self.issues.push(issue);
    }
    
    /// Check if there are any critical issues
    pub fn has_critical_issues(&self) -> bool {
        self.issues.iter().any(|i| matches!(i.severity, ValidationSeverity::Critical))
    }
}