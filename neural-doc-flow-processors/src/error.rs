//! Error types for neural document flow processors

use thiserror::Error;
use std::fmt;

/// Result type alias for neural processing operations
pub type Result<T> = std::result::Result<T, NeuralError>;

/// Neural processing error types
#[derive(Error, Debug)]
pub enum NeuralError {
    /// Neural engine not initialized
    #[error("Neural engine not initialized")]
    NotInitialized,

    /// Model file not found
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    /// Model loading error
    #[error("Failed to load model: {0}")]
    ModelLoad(String),

    /// Model saving error
    #[error("Failed to save model: {0}")]
    ModelSave(String),

    /// Network creation error
    #[error("Failed to create neural network: {0}")]
    NetworkCreation(String),

    /// Training error
    #[error("Training failed: {0}")]
    Training(String),

    /// Inference error
    #[error("Inference failed: {0}")]
    Inference(String),

    /// Invalid input data
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// ruv-FANN specific errors
    #[error("ruv-FANN error: {0}")]
    RuvFann(String),

    /// ONNX Runtime error (when enabled)
    #[cfg(feature = "onnx")]
    #[error("ONNX Runtime error: {0}")]
    OnnxRuntime(String),

    /// OpenCV error (when enabled)
    #[cfg(feature = "opencv-support")]
    #[error("OpenCV error: {0}")]
    OpenCV(String),

    /// Memory allocation error
    #[error("Memory allocation failed: {0}")]
    Memory(String),

    /// Feature extraction error
    #[error("Feature extraction failed: {0}")]
    FeatureExtraction(String),

    /// Model compatibility error
    #[error("Model incompatibility: {0}")]
    ModelIncompatibility(String),

    /// Unsupported operation
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    /// Timeout error
    #[error("Operation timed out: {0}")]
    Timeout(String),

    /// DAA coordination error (when enabled)
    #[cfg(feature = "daa-coordination")]
    #[error("DAA coordination error: {0}")]
    DaaCoordination(String),
}

/// Convert from ruv-FANN error types
impl From<ruv_fann::Error> for NeuralError {
    fn from(err: ruv_fann::Error) -> Self {
        NeuralError::RuvFann(err.to_string())
    }
}

/// Convert from bincode error
impl From<bincode::Error> for NeuralError {
    fn from(err: bincode::Error) -> Self {
        NeuralError::Serialization(serde_json::Error::custom(err.to_string()))
    }
}

/// Convert from reqwest error (for API calls)
impl From<reqwest::Error> for NeuralError {
    fn from(err: reqwest::Error) -> Self {
        NeuralError::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            err.to_string(),
        ))
    }
}

/// Convert from join error (for async tasks)
impl From<tokio::task::JoinError> for NeuralError {
    fn from(err: tokio::task::JoinError) -> Self {
        NeuralError::Configuration(format!("Task join error: {}", err))
    }
}

#[cfg(feature = "onnx")]
impl From<ort::Error> for NeuralError {
    fn from(err: ort::Error) -> Self {
        NeuralError::OnnxRuntime(err.to_string())
    }
}

#[cfg(feature = "opencv-support")]
impl From<opencv::Error> for NeuralError {
    fn from(err: opencv::Error) -> Self {
        NeuralError::OpenCV(err.to_string())
    }
}

/// Neural processing warning types (non-fatal)
#[derive(Debug, Clone)]
pub enum NeuralWarning {
    /// Low confidence in processing result
    LowConfidence {
        component: String,
        confidence: f32,
        threshold: f32,
    },

    /// Model version mismatch
    ModelVersionMismatch {
        expected: String,
        found: String,
    },

    /// Performance degradation detected
    PerformanceDegradation {
        component: String,
        expected_ms: u64,
        actual_ms: u64,
    },

    /// Memory usage high
    HighMemoryUsage {
        component: String,
        usage_mb: usize,
        limit_mb: usize,
    },

    /// Training data quality issues
    TrainingDataQuality {
        issue: String,
        samples_affected: usize,
    },
}

impl fmt::Display for NeuralWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NeuralWarning::LowConfidence { component, confidence, threshold } => {
                write!(f, "Low confidence in {}: {:.2} < {:.2}", component, confidence, threshold)
            }
            NeuralWarning::ModelVersionMismatch { expected, found } => {
                write!(f, "Model version mismatch: expected {}, found {}", expected, found)
            }
            NeuralWarning::PerformanceDegradation { component, expected_ms, actual_ms } => {
                write!(f, "Performance degradation in {}: {}ms > {}ms", component, actual_ms, expected_ms)
            }
            NeuralWarning::HighMemoryUsage { component, usage_mb, limit_mb } => {
                write!(f, "High memory usage in {}: {}MB > {}MB", component, usage_mb, limit_mb)
            }
            NeuralWarning::TrainingDataQuality { issue, samples_affected } => {
                write!(f, "Training data quality issue: {} ({} samples affected)", issue, samples_affected)
            }
        }
    }
}

/// Error context for better debugging
#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub operation: String,
    pub component: String,
    pub input_size: Option<usize>,
    pub model_type: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl ErrorContext {
    pub fn new(operation: &str, component: &str) -> Self {
        Self {
            operation: operation.to_string(),
            component: component.to_string(),
            input_size: None,
            model_type: None,
            timestamp: chrono::Utc::now(),
        }
    }

    pub fn with_input_size(mut self, size: usize) -> Self {
        self.input_size = Some(size);
        self
    }

    pub fn with_model_type(mut self, model_type: &str) -> Self {
        self.model_type = Some(model_type.to_string());
        self
    }
}

/// Enhanced error with context
#[derive(Debug)]
pub struct ContextualError {
    pub error: NeuralError,
    pub context: ErrorContext,
}

impl fmt::Display for ContextualError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} [{}::{} at {}]",
            self.error,
            self.context.component,
            self.context.operation,
            self.context.timestamp.format("%Y-%m-%d %H:%M:%S UTC")
        )
    }
}

impl std::error::Error for ContextualError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// Helper macro for creating contextual errors
#[macro_export]
macro_rules! neural_error {
    ($err:expr, $op:expr, $comp:expr) => {
        ContextualError {
            error: $err,
            context: ErrorContext::new($op, $comp),
        }
    };
    ($err:expr, $op:expr, $comp:expr, size: $size:expr) => {
        ContextualError {
            error: $err,
            context: ErrorContext::new($op, $comp).with_input_size($size),
        }
    };
    ($err:expr, $op:expr, $comp:expr, model: $model:expr) => {
        ContextualError {
            error: $err,
            context: ErrorContext::new($op, $comp).with_model_type($model),
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let error = NeuralError::ModelNotFound("test_model".to_string());
        assert_eq!(error.to_string(), "Model not found: test_model");
    }

    #[test]
    fn test_warning_display() {
        let warning = NeuralWarning::LowConfidence {
            component: "text_extractor".to_string(),
            confidence: 0.65,
            threshold: 0.8,
        };
        assert_eq!(
            warning.to_string(),
            "Low confidence in text_extractor: 0.65 < 0.80"
        );
    }

    #[test]
    fn test_contextual_error() {
        let error = NeuralError::Training("Test training error".to_string());
        let context = ErrorContext::new("train_network", "text_model");
        let contextual = ContextualError { error, context };
        
        let display = contextual.to_string();
        assert!(display.contains("Training failed: Test training error"));
        assert!(display.contains("text_model::train_network"));
    }

    #[test]
    fn test_error_macro() {
        let error = neural_error!(
            NeuralError::InvalidInput("bad data".to_string()),
            "process",
            "neural_engine",
            size: 1024
        );
        
        assert_eq!(error.context.operation, "process");
        assert_eq!(error.context.component, "neural_engine");
        assert_eq!(error.context.input_size, Some(1024));
    }
}