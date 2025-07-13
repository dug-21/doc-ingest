//! Error types for Neural Document Flow
//!
//! This module provides comprehensive error handling for all components
//! of the document processing system.

use std::fmt;
use thiserror::Error;

/// Main error type for document source operations
#[derive(Error, Debug)]
pub enum SourceError {
    /// Invalid document format detected
    #[error("Invalid document format: {0}")]
    InvalidFormat(String),

    /// Unsupported input type
    #[error("Unsupported input type")]
    UnsupportedInput,

    /// Unsupported document version
    #[error("Unsupported document version: {0}")]
    UnsupportedVersion(u32),

    /// Document parsing error
    #[error("Parse error: {0}")]
    ParseError(String),

    /// I/O operation failed
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Security validation failed
    #[error("Security error: {0}")]
    SecurityError(String),

    /// External API error
    #[error("API error: {0}")]
    ApiError(String),

    /// JSON serialization/deserialization error
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// HTTP request error
    #[error("HTTP error: {0}")]
    HttpError(String),

    /// Timeout error
    #[error("Operation timed out")]
    Timeout,

    /// Memory limit exceeded
    #[error("Memory limit exceeded")]
    MemoryLimitExceeded,

    /// Plugin not found
    #[error("Plugin not found: {0}")]
    PluginNotFound(String),

    /// Plugin initialization failed
    #[error("Plugin initialization failed: {0}")]
    PluginInitError(String),

    /// Custom error for domain-specific issues
    #[error("Custom error: {0}")]
    Custom(String),
}

/// Error type for neural processing operations
#[derive(Error, Debug)]
pub enum NeuralError {
    /// Model loading failed
    #[error("Model loading error: {0}")]
    ModelLoadError(String),

    /// Model inference failed
    #[error("Inference error: {0}")]
    InferenceError(String),

    /// Training data error
    #[error("Training data error: {0}")]
    TrainingDataError(String),

    /// Model configuration error
    #[error("Model config error: {0}")]
    ConfigError(String),

    /// GPU/Hardware acceleration error
    #[error("Hardware acceleration error: {0}")]
    HardwareError(String),

    /// Memory allocation error
    #[error("Memory allocation error: {0}")]
    MemoryError(String),

    /// Model architecture mismatch
    #[error("Architecture mismatch: expected {expected}, got {actual}")]
    ArchitectureMismatch {
        expected: String,
        actual: String,
    },

    /// Custom neural processing error
    #[error("Neural processing error: {0}")]
    Custom(String),
}

/// Error type for pipeline processing operations
#[derive(Error, Debug)]
pub enum PipelineError {
    /// Stage execution failed
    #[error("Stage '{stage}' failed: {error}")]
    StageError {
        stage: String,
        error: String,
    },

    /// Pipeline configuration error
    #[error("Pipeline configuration error: {0}")]
    ConfigError(String),

    /// Resource allocation failed
    #[error("Resource allocation failed: {0}")]
    ResourceError(String),

    /// Dependency resolution failed
    #[error("Dependency error: {0}")]
    DependencyError(String),

    /// Pipeline timeout
    #[error("Pipeline timeout after {seconds} seconds")]
    Timeout { seconds: u64 },

    /// Data validation failed between stages
    #[error("Data validation failed at stage '{stage}': {error}")]
    ValidationError {
        stage: String,
        error: String,
    },

    /// Source error propagated from document sources
    #[error("Source error: {0}")]
    SourceError(#[from] SourceError),

    /// Neural error propagated from neural processors
    #[error("Neural error: {0}")]
    NeuralError(#[from] NeuralError),

    /// Custom pipeline error
    #[error("Pipeline error: {0}")]
    Custom(String),
}

/// Error type for output formatting operations
#[derive(Error, Debug)]
pub enum FormatError {
    /// Template not found
    #[error("Template not found: {0}")]
    TemplateNotFound(String),

    /// Formatter not found
    #[error("Formatter not found: {0}")]
    FormatterNotFound(String),

    /// Template compilation error
    #[error("Template compilation error: {0}")]
    TemplateError(String),

    /// Data serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Invalid output format specification
    #[error("Invalid format specification: {0}")]
    InvalidFormat(String),

    /// Schema validation failed
    #[error("Schema validation failed: {0}")]
    SchemaError(String),

    /// JSON formatting error
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Custom formatting error
    #[error("Format error: {0}")]
    Custom(String),
}

/// Unified result type for all neural document flow operations
pub type NeuralDocFlowResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Extension trait for converting errors to NeuralDocFlowResult
pub trait IntoNeuralDocFlowResult<T> {
    /// Convert this result into a NeuralDocFlowResult
    fn into_neural_result(self) -> NeuralDocFlowResult<T>;
}

impl<T, E> IntoNeuralDocFlowResult<T> for Result<T, E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn into_neural_result(self) -> NeuralDocFlowResult<T> {
        self.map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
    }
}

/// Error context for better debugging
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Operation that failed
    pub operation: String,
    /// File or resource being processed
    pub resource: Option<String>,
    /// Additional context information
    pub context: Vec<(String, String)>,
}

impl ErrorContext {
    /// Create new error context
    pub fn new(operation: impl Into<String>) -> Self {
        Self {
            operation: operation.into(),
            resource: None,
            context: Vec::new(),
        }
    }

    /// Add resource information
    pub fn with_resource(mut self, resource: impl Into<String>) -> Self {
        self.resource = Some(resource.into());
        self
    }

    /// Add context key-value pair
    pub fn with_context(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.context.push((key.into(), value.into()));
        self
    }
}

impl fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Operation: {}", self.operation)?;
        
        if let Some(resource) = &self.resource {
            write!(f, ", Resource: {}", resource)?;
        }
        
        if !self.context.is_empty() {
            write!(f, ", Context: ")?;
            for (i, (key, value)) in self.context.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}={}", key, value)?;
            }
        }
        
        Ok(())
    }
}

/// Trait for adding context to errors
pub trait WithContext<T> {
    /// Add context to this result
    fn with_context(self, context: ErrorContext) -> Result<T, String>;
    
    /// Add operation context
    fn with_operation(self, operation: impl Into<String>) -> Result<T, String>;
}

impl<T, E: fmt::Display> WithContext<T> for Result<T, E> {
    fn with_context(self, context: ErrorContext) -> Result<T, String> {
        self.map_err(|e| format!("{}: {}", context, e))
    }
    
    fn with_operation(self, operation: impl Into<String>) -> Result<T, String> {
        self.with_context(ErrorContext::new(operation))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_context() {
        let context = ErrorContext::new("parsing PDF")
            .with_resource("document.pdf")
            .with_context("page", "5")
            .with_context("size", "1.2MB");
        
        let context_str = context.to_string();
        assert!(context_str.contains("parsing PDF"));
        assert!(context_str.contains("document.pdf"));
        assert!(context_str.contains("page=5"));
        assert!(context_str.contains("size=1.2MB"));
    }

    #[test]
    fn test_with_context() {
        let result: Result<(), std::io::Error> = Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "file not found"
        ));
        
        let with_context = result.with_operation("loading config");
        assert!(with_context.is_err());
        assert!(with_context.unwrap_err().contains("loading config"));
    }

    #[test]
    fn test_error_conversion() {
        let source_error = SourceError::InvalidFormat("bad magic bytes".to_string());
        let pipeline_error = PipelineError::SourceError(source_error);
        
        assert!(pipeline_error.to_string().contains("bad magic bytes"));
    }
}