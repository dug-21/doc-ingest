//! Error handling for neural document processing

use thiserror::Error;
use uuid::Uuid;

/// Result type alias for neural doc flow operations
pub type Result<T> = std::result::Result<T, CoreError>;

/// Core error types for neural document processing
#[derive(Error, Debug)]
pub enum CoreError {
    /// Coordination-related errors (DAA)
    #[error("Coordination error: {0}")]
    CoordinationError(String),
    
    /// Agent-related errors
    #[error("Agent error: {0}")]
    AgentError(String),
    
    /// Neural processing errors
    #[error("Neural processing error: {0}")]
    NeuralError(String),
    
    /// Document parsing errors
    #[error("Parse error: {0}")]
    ParseError(String),
    
    /// Content extraction errors
    #[error("Extraction error: {0}")]
    ExtractionError(String),
    
    /// Validation errors
    #[error("Validation error: {0}")]
    ValidationError(String),
    
    /// IO errors
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    /// Serialization errors
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    /// Timeout errors
    #[error("Operation timed out after {timeout_ms}ms")]
    TimeoutError { timeout_ms: u64 },
    
    /// Configuration errors
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    /// Plugin errors
    #[error("Plugin error in {plugin}: {message}")]
    PluginError { plugin: String, message: String },
    
    /// Memory/storage errors
    #[error("Memory error: {0}")]
    MemoryError(String),
    
    /// Network/communication errors
    #[error("Network error: {0}")]
    NetworkError(String),
    
    /// Resource exhaustion errors
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),
    
    /// Invalid operation errors
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
    
    /// Not found errors
    #[error("Not found: {0}")]
    NotFound(String),
    
    /// Already exists errors
    #[error("Already exists: {0}")]
    AlreadyExists(String),
    
    /// Permission denied errors
    #[error("Permission denied: {0}")]
    PermissionDenied(String),
    
    /// Unsupported operation errors
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),
    
    /// Internal errors (should not occur in normal operation)
    #[error("Internal error: {0}")]
    InternalError(String),
}

impl CoreError {
    /// Check if the error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            CoreError::TimeoutError { .. } => true,
            CoreError::NetworkError(_) => true,
            CoreError::ResourceExhausted(_) => true,
            CoreError::IoError(_) => true,
            CoreError::MemoryError(_) => false,
            CoreError::InternalError(_) => false,
            CoreError::CoordinationError(_) => true,
            CoreError::AgentError(_) => true,
            CoreError::NeuralError(_) => false,
            CoreError::ParseError(_) => false,
            CoreError::ExtractionError(_) => false,
            CoreError::ValidationError(_) => false,
            CoreError::SerializationError(_) => false,
            CoreError::ConfigError(_) => false,
            CoreError::PluginError { .. } => true,
            CoreError::InvalidOperation(_) => false,
            CoreError::NotFound(_) => false,
            CoreError::AlreadyExists(_) => false,
            CoreError::PermissionDenied(_) => false,
            CoreError::UnsupportedOperation(_) => false,
        }
    }
    
    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            CoreError::InternalError(_) => ErrorSeverity::Critical,
            CoreError::MemoryError(_) => ErrorSeverity::Critical,
            CoreError::ConfigError(_) => ErrorSeverity::High,
            CoreError::PermissionDenied(_) => ErrorSeverity::High,
            CoreError::CoordinationError(_) => ErrorSeverity::Medium,
            CoreError::AgentError(_) => ErrorSeverity::Medium,
            CoreError::NeuralError(_) => ErrorSeverity::Medium,
            CoreError::ParseError(_) => ErrorSeverity::Medium,
            CoreError::ExtractionError(_) => ErrorSeverity::Medium,
            CoreError::ValidationError(_) => ErrorSeverity::Low,
            CoreError::TimeoutError { .. } => ErrorSeverity::Low,
            CoreError::NetworkError(_) => ErrorSeverity::Low,
            CoreError::ResourceExhausted(_) => ErrorSeverity::Medium,
            CoreError::IoError(_) => ErrorSeverity::Low,
            CoreError::SerializationError(_) => ErrorSeverity::Low,
            CoreError::PluginError { .. } => ErrorSeverity::Medium,
            CoreError::InvalidOperation(_) => ErrorSeverity::Medium,
            CoreError::NotFound(_) => ErrorSeverity::Low,
            CoreError::AlreadyExists(_) => ErrorSeverity::Low,
            CoreError::UnsupportedOperation(_) => ErrorSeverity::Medium,
        }
    }
    
    /// Get error category
    pub fn category(&self) -> ErrorCategory {
        match self {
            CoreError::CoordinationError(_) => ErrorCategory::Coordination,
            CoreError::AgentError(_) => ErrorCategory::Agent,
            CoreError::NeuralError(_) => ErrorCategory::Neural,
            CoreError::ParseError(_) => ErrorCategory::Processing,
            CoreError::ExtractionError(_) => ErrorCategory::Processing,
            CoreError::ValidationError(_) => ErrorCategory::Validation,
            CoreError::IoError(_) => ErrorCategory::System,
            CoreError::SerializationError(_) => ErrorCategory::System,
            CoreError::TimeoutError { .. } => ErrorCategory::System,
            CoreError::ConfigError(_) => ErrorCategory::Configuration,
            CoreError::PluginError { .. } => ErrorCategory::Plugin,
            CoreError::MemoryError(_) => ErrorCategory::System,
            CoreError::NetworkError(_) => ErrorCategory::Network,
            CoreError::ResourceExhausted(_) => ErrorCategory::System,
            CoreError::InvalidOperation(_) => ErrorCategory::Usage,
            CoreError::NotFound(_) => ErrorCategory::Usage,
            CoreError::AlreadyExists(_) => ErrorCategory::Usage,
            CoreError::PermissionDenied(_) => ErrorCategory::Security,
            CoreError::UnsupportedOperation(_) => ErrorCategory::Usage,
            CoreError::InternalError(_) => ErrorCategory::Internal,
        }
    }
    
    /// Create a coordination error
    pub fn coordination_error(message: impl Into<String>) -> Self {
        CoreError::CoordinationError(message.into())
    }
    
    /// Create an agent error
    pub fn agent_error(message: impl Into<String>) -> Self {
        CoreError::AgentError(message.into())
    }
    
    /// Create a neural error
    pub fn neural_error(message: impl Into<String>) -> Self {
        CoreError::NeuralError(message.into())
    }
    
    /// Create a plugin error
    pub fn plugin_error(plugin: impl Into<String>, message: impl Into<String>) -> Self {
        CoreError::PluginError {
            plugin: plugin.into(),
            message: message.into(),
        }
    }
    
    /// Create a timeout error
    pub fn timeout_error(timeout_ms: u64) -> Self {
        CoreError::TimeoutError { timeout_ms }
    }
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Error categories for classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    Coordination,
    Agent,
    Neural,
    Processing,
    Validation,
    System,
    Configuration,
    Plugin,
    Network,
    Security,
    Usage,
    Internal,
}

/// Error context for debugging and telemetry
#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub error_id: Uuid,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub agent_id: Option<Uuid>,
    pub document_id: Option<Uuid>,
    pub task_id: Option<Uuid>,
    pub additional_info: std::collections::HashMap<String, String>,
}

impl ErrorContext {
    pub fn new() -> Self {
        Self {
            error_id: Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            agent_id: None,
            document_id: None,
            task_id: None,
            additional_info: std::collections::HashMap::new(),
        }
    }
    
    pub fn with_agent_id(mut self, agent_id: Uuid) -> Self {
        self.agent_id = Some(agent_id);
        self
    }
    
    pub fn with_document_id(mut self, document_id: Uuid) -> Self {
        self.document_id = Some(document_id);
        self
    }
    
    pub fn with_task_id(mut self, task_id: Uuid) -> Self {
        self.task_id = Some(task_id);
        self
    }
    
    pub fn with_info(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.additional_info.insert(key.into(), value.into());
        self
    }
}

impl Default for ErrorContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Enhanced error with context
#[derive(Debug)]
pub struct ContextualError {
    pub error: CoreError,
    pub context: ErrorContext,
}

impl ContextualError {
    pub fn new(error: CoreError, context: ErrorContext) -> Self {
        Self { error, context }
    }
    
    pub fn with_context(error: CoreError) -> Self {
        Self {
            error,
            context: ErrorContext::new(),
        }
    }
}

impl std::fmt::Display for ContextualError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Error {} at {}: {}",
            self.context.error_id,
            self.context.timestamp.format("%Y-%m-%d %H:%M:%S UTC"),
            self.error
        )
    }
}

impl std::error::Error for ContextualError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// Error reporter for telemetry and monitoring
pub trait ErrorReporter: Send + Sync {
    fn report(&self, error: &ContextualError);
}

/// Default error reporter that logs to tracing
pub struct TracingErrorReporter;

impl ErrorReporter for TracingErrorReporter {
    fn report(&self, error: &ContextualError) {
        match error.error.severity() {
            ErrorSeverity::Critical => tracing::error!("{}", error),
            ErrorSeverity::High => tracing::error!("{}", error),
            ErrorSeverity::Medium => tracing::warn!("{}", error),
            ErrorSeverity::Low => tracing::info!("{}", error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_properties() {
        let error = CoreError::coordination_error("Test coordination error");
        assert!(error.is_recoverable());
        assert_eq!(error.severity(), ErrorSeverity::Medium);
        assert_eq!(error.category(), ErrorCategory::Coordination);
    }
    
    #[test]
    fn test_contextual_error() {
        let error = CoreError::agent_error("Test agent error");
        let context = ErrorContext::new()
            .with_agent_id(Uuid::new_v4())
            .with_info("test_key", "test_value");
        
        let contextual = ContextualError::new(error, context);
        assert!(!contextual.to_string().is_empty());
    }
}