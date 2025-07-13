//! Error types and handling for NeuralDocFlow
//!
//! This module provides comprehensive error handling for all components of the
//! document extraction platform, with structured error types that enable
//! proper error recovery and user feedback.

use std::fmt;
use thiserror::Error;

/// Main result type used throughout NeuralDocFlow
pub type Result<T> = std::result::Result<T, NeuralDocFlowError>;

/// Comprehensive error type for all NeuralDocFlow operations
#[derive(Error, Debug)]
pub enum NeuralDocFlowError {
    /// Configuration-related errors
    #[error("Configuration error: {message}")]
    Config { message: String },

    /// Source plugin errors
    #[error("Source error: {message}")]
    Source { message: String },

    /// DAA coordination errors
    #[error("DAA coordination error: {message}")]
    Coordination { message: String },

    /// Neural processing errors
    #[error("Neural processing error: {message}")]
    Neural { message: String },

    /// Document parsing errors
    #[error("Parse error: {message}")]
    Parse { message: String },

    /// Validation errors
    #[error("Validation error: {message}")]
    Validation { message: String },

    /// I/O related errors
    #[error("I/O error: {source}")]
    Io {
        #[from]
        source: std::io::Error,
    },

    /// Serialization/deserialization errors
    #[error("Serialization error: {source}")]
    Serialization {
        #[from]
        source: serde_json::Error,
    },

    /// Network/HTTP errors
    #[error("Network error: {source}")]
    Network {
        #[from]
        source: reqwest::Error,
    },

    /// Memory allocation errors
    #[error("Memory error: {message}")]
    Memory { message: String },

    /// Security violations
    #[error("Security error: {message}")]
    Security { message: String },

    /// Plugin loading errors
    #[error("Plugin error: {message}")]
    Plugin { message: String },

    /// Timeout errors
    #[error("Timeout error: operation took longer than {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },

    /// Capacity exceeded errors
    #[error("Capacity exceeded: {message}")]
    Capacity { message: String },

    /// Unsupported operation errors
    #[error("Unsupported operation: {message}")]
    Unsupported { message: String },

    /// Internal errors (should not occur in normal operation)
    #[error("Internal error: {message}")]
    Internal { message: String },
}

impl NeuralDocFlowError {
    /// Create a configuration error
    pub fn config<S: Into<String>>(message: S) -> Self {
        Self::Config {
            message: message.into(),
        }
    }

    /// Create a source error
    pub fn source<S: Into<String>>(message: S) -> Self {
        Self::Source {
            message: message.into(),
        }
    }

    /// Create a coordination error
    pub fn coordination<S: Into<String>>(message: S) -> Self {
        Self::Coordination {
            message: message.into(),
        }
    }

    /// Create a neural processing error
    pub fn neural<S: Into<String>>(message: S) -> Self {
        Self::Neural {
            message: message.into(),
        }
    }

    /// Create a parse error
    pub fn parse<S: Into<String>>(message: S) -> Self {
        Self::Parse {
            message: message.into(),
        }
    }

    /// Create a validation error
    pub fn validation<S: Into<String>>(message: S) -> Self {
        Self::Validation {
            message: message.into(),
        }
    }

    /// Create a memory error
    pub fn memory<S: Into<String>>(message: S) -> Self {
        Self::Memory {
            message: message.into(),
        }
    }

    /// Create a security error
    pub fn security<S: Into<String>>(message: S) -> Self {
        Self::Security {
            message: message.into(),
        }
    }

    /// Create a plugin error
    pub fn plugin<S: Into<String>>(message: S) -> Self {
        Self::Plugin {
            message: message.into(),
        }
    }

    /// Create a timeout error
    pub fn timeout(timeout_ms: u64) -> Self {
        Self::Timeout { timeout_ms }
    }

    /// Create a capacity exceeded error
    pub fn capacity<S: Into<String>>(message: S) -> Self {
        Self::Capacity {
            message: message.into(),
        }
    }

    /// Create an unsupported operation error
    pub fn unsupported<S: Into<String>>(message: S) -> Self {
        Self::Unsupported {
            message: message.into(),
        }
    }

    /// Create an internal error
    pub fn internal<S: Into<String>>(message: S) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }

    /// Check if this error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::Config { .. } 
            | Self::Validation { .. } 
            | Self::Unsupported { .. }
            | Self::Security { .. } => false,
            
            Self::Source { .. } 
            | Self::Coordination { .. }
            | Self::Neural { .. }
            | Self::Parse { .. }
            | Self::Io { .. }
            | Self::Network { .. }
            | Self::Memory { .. }
            | Self::Plugin { .. }
            | Self::Timeout { .. }
            | Self::Capacity { .. } => true,
            
            Self::Serialization { .. } 
            | Self::Internal { .. } => false,
        }
    }

    /// Get error category for metrics and monitoring
    pub fn category(&self) -> &'static str {
        match self {
            Self::Config { .. } => "config",
            Self::Source { .. } => "source",
            Self::Coordination { .. } => "coordination",
            Self::Neural { .. } => "neural",
            Self::Parse { .. } => "parse",
            Self::Validation { .. } => "validation",
            Self::Io { .. } => "io",
            Self::Serialization { .. } => "serialization",
            Self::Network { .. } => "network",
            Self::Memory { .. } => "memory",
            Self::Security { .. } => "security",
            Self::Plugin { .. } => "plugin",
            Self::Timeout { .. } => "timeout",
            Self::Capacity { .. } => "capacity",
            Self::Unsupported { .. } => "unsupported",
            Self::Internal { .. } => "internal",
        }
    }

    /// Get severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            Self::Internal { .. } | Self::Security { .. } => ErrorSeverity::Critical,
            Self::Memory { .. } | Self::Capacity { .. } => ErrorSeverity::High,
            Self::Coordination { .. } | Self::Neural { .. } | Self::Parse { .. } => ErrorSeverity::Medium,
            Self::Source { .. } | Self::Validation { .. } | Self::Timeout { .. } => ErrorSeverity::Low,
            _ => ErrorSeverity::Low,
        }
    }
}

/// Error severity levels for monitoring and alerting
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    /// Low severity - expected operational issues
    Low,
    /// Medium severity - degraded performance
    Medium,
    /// High severity - significant issues
    High,
    /// Critical severity - system failure
    Critical,
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => write!(f, "LOW"),
            Self::Medium => write!(f, "MEDIUM"),
            Self::High => write!(f, "HIGH"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Error context for enhanced debugging
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Component where error occurred
    pub component: String,
    /// Operation being performed
    pub operation: String,
    /// Document ID if applicable
    pub document_id: Option<String>,
    /// Additional context data
    pub context: std::collections::HashMap<String, String>,
}

impl ErrorContext {
    /// Create new error context
    pub fn new<S: Into<String>>(component: S, operation: S) -> Self {
        Self {
            component: component.into(),
            operation: operation.into(),
            document_id: None,
            context: std::collections::HashMap::new(),
        }
    }

    /// Set document ID
    pub fn with_document_id<S: Into<String>>(mut self, doc_id: S) -> Self {
        self.document_id = Some(doc_id.into());
        self
    }

    /// Add context data
    pub fn with_context<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.context.insert(key.into(), value.into());
        self
    }
}

/// Result with enhanced error context
pub type ContextResult<T> = std::result::Result<T, (NeuralDocFlowError, ErrorContext)>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = NeuralDocFlowError::config("test message");
        assert!(matches!(err, NeuralDocFlowError::Config { .. }));
        assert_eq!(err.category(), "config");
    }

    #[test]
    fn test_error_severity() {
        let critical = NeuralDocFlowError::security("security breach");
        assert_eq!(critical.severity(), ErrorSeverity::Critical);

        let low = NeuralDocFlowError::validation("validation failed");
        assert_eq!(low.severity(), ErrorSeverity::Low);
    }

    #[test]
    fn test_error_recoverability() {
        let recoverable = NeuralDocFlowError::timeout(1000);
        assert!(recoverable.is_recoverable());

        let not_recoverable = NeuralDocFlowError::config("bad config");
        assert!(!not_recoverable.is_recoverable());
    }

    #[test]
    fn test_error_context() {
        let ctx = ErrorContext::new("test_component", "test_operation")
            .with_document_id("doc123")
            .with_context("key", "value");
        
        assert_eq!(ctx.component, "test_component");
        assert_eq!(ctx.document_id, Some("doc123".to_string()));
        assert_eq!(ctx.context.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_error_display() {
        let err = NeuralDocFlowError::parse("test parse error");
        let display = format!("{}", err);
        assert!(display.contains("Parse error"));
        assert!(display.contains("test parse error"));
    }

    #[test]
    fn test_error_categories() {
        let errors = vec![
            (NeuralDocFlowError::config("test"), "config"),
            (NeuralDocFlowError::source("test"), "source"),
            (NeuralDocFlowError::coordination("test"), "coordination"),
            (NeuralDocFlowError::neural("test"), "neural"),
            (NeuralDocFlowError::parse("test"), "parse"),
            (NeuralDocFlowError::validation("test"), "validation"),
            (NeuralDocFlowError::memory("test"), "memory"),
            (NeuralDocFlowError::security("test"), "security"),
            (NeuralDocFlowError::plugin("test"), "plugin"),
            (NeuralDocFlowError::timeout(1000), "timeout"),
            (NeuralDocFlowError::capacity("test"), "capacity"),
            (NeuralDocFlowError::unsupported("test"), "unsupported"),
            (NeuralDocFlowError::internal("test"), "internal"),
        ];

        for (error, expected_category) in errors {
            assert_eq!(error.category(), expected_category);
        }
    }
}