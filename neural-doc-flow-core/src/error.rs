//! Error types for the neural document flow framework

use thiserror::Error;

/// Main error type for the neural document flow framework
#[derive(Error, Debug)]
pub enum NeuralDocFlowError {
    /// Document source errors
    #[error("Document source error: {0}")]
    SourceError(#[from] SourceError),

    /// Processing pipeline errors
    #[error("Processing error: {0}")]
    ProcessingError(#[from] ProcessingError),

    /// Neural processing errors
    #[error("Neural processing error: {0}")]
    NeuralError(#[from] NeuralError),

    /// Output formatting errors
    #[error("Output formatting error: {0}")]
    OutputError(#[from] OutputError),

    /// Configuration errors
    #[error("Configuration error: {message}")]
    ConfigError { message: String },

    /// IO errors
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Serialization errors
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// Memory management errors
    #[error("Memory error: {message}")]
    MemoryError { message: String },

    /// Generic errors
    #[error("Internal error: {0}")]
    Internal(#[from] anyhow::Error),
}

/// Document source specific errors
#[derive(Error, Debug)]
pub enum SourceError {
    #[error("Document not found: {path}")]
    DocumentNotFound { path: String },

    #[error("Unsupported document format: {format}")]
    UnsupportedFormat { format: String },

    #[error("Document parsing failed: {reason}")]
    ParseError { reason: String },

    #[error("Access denied to document: {path}")]
    AccessDenied { path: String },

    #[error("Document corrupted: {path}")]
    Corrupted { path: String },
}

/// Processing pipeline specific errors
#[derive(Error, Debug)]
pub enum ProcessingError {
    #[error("Pipeline configuration invalid: {reason}")]
    InvalidConfiguration { reason: String },

    #[error("Processor failed: {processor_name} - {reason}")]
    ProcessorFailed {
        processor_name: String,
        reason: String,
    },

    #[error("Pipeline execution timeout")]
    Timeout,

    #[error("Insufficient resources for processing")]
    InsufficientResources,

    #[error("Processing interrupted: {reason}")]
    Interrupted { reason: String },

    #[error("Plugin not found: {0}")]
    PluginNotFound(String),

    #[error("Plugin load error: {0}")]
    PluginLoadError(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Regex error: {0}")]
    Regex(String),

    #[error("Aho-Corasick build error: {0}")]
    AhoCorasick(String),

    #[error("Sandbox error: {0}")]
    SandboxError(String),

    #[error("Model creation error: {0}")]
    ModelCreation(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Prediction error: {0}")]
    Prediction(String),

    #[error("Training error: {0}")]
    Training(String),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Deserialization error: {0}")]
    Deserialization(String),

    #[error("Security violation: {0}")]
    SecurityViolation(String),
}

/// Neural processing specific errors
#[derive(Error, Debug)]
pub enum NeuralError {
    #[error("Neural model not found: {model_name}")]
    ModelNotFound { model_name: String },

    #[error("Neural model loading failed: {reason}")]
    ModelLoadingFailed { reason: String },

    #[error("Neural inference failed: {reason}")]
    InferenceFailed { reason: String },

    #[error("Invalid neural input dimensions: expected {expected}, got {actual}")]
    InvalidInputDimensions { expected: usize, actual: usize },

    #[error("Neural model training failed: {reason}")]
    TrainingFailed { reason: String },

    #[error("Neural initialization error: {0}")]
    InitializationError(String),

    #[error("Tensor operation failed: operation={operation}, details={details}")]
    TensorOperationFailed { operation: String, details: String },

    #[error("CUDA/GPU error: {reason}")]
    GpuError { reason: String },
}

/// Output formatting specific errors
#[derive(Error, Debug)]
pub enum OutputError {
    #[error("Unsupported output format: {format}")]
    UnsupportedFormat { format: String },

    #[error("Output generation failed: {reason}")]
    GenerationFailed { reason: String },

    #[error("Template not found: {template_name}")]
    TemplateNotFound { template_name: String },

    #[error("Output validation failed: {reason}")]
    ValidationFailed { reason: String },
}

/// Result type alias for convenience
pub type Result<T> = std::result::Result<T, NeuralDocFlowError>;
pub type SourceResult<T> = std::result::Result<T, SourceError>;
pub type ProcessingResult<T> = std::result::Result<T, ProcessingError>;
pub type NeuralResult<T> = std::result::Result<T, NeuralError>;
pub type OutputResult<T> = std::result::Result<T, OutputError>;

// Error conversion implementations for security module compatibility
impl From<regex::Error> for ProcessingError {
    fn from(err: regex::Error) -> Self {
        ProcessingError::Regex(err.to_string())
    }
}

impl From<aho_corasick::BuildError> for ProcessingError {
    fn from(err: aho_corasick::BuildError) -> Self {
        ProcessingError::AhoCorasick(err.to_string())
    }
}

impl From<serde_json::Error> for ProcessingError {
    fn from(err: serde_json::Error) -> Self {
        ProcessingError::ProcessorFailed {
            processor_name: "serialization".to_string(),
            reason: err.to_string(),
        }
    }
}

#[cfg(feature = "sandbox")]
impl From<nix::errno::Errno> for ProcessingError {
    fn from(err: nix::errno::Errno) -> Self {
        ProcessingError::ProcessorFailed {
            processor_name: "sandbox".to_string(),
            reason: err.to_string(),
        }
    }
}

// Note: notify::Error conversion is implemented in the plugins module

