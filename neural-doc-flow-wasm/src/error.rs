//! Error handling for WASM bindings

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// WASM-specific error types
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum WasmError {
    #[error("Processing error: {message}")]
    ProcessingError { message: String },

    #[error("Invalid input: {message}")]
    InvalidInput { message: String },

    #[error("Configuration error: {message}")]
    ConfigError { message: String },

    #[error("Security error: {message}")]
    SecurityError { message: String },

    #[error("Memory error: {message}")]
    MemoryError { message: String },

    #[error("Timeout error: operation took too long")]
    TimeoutError,

    #[error("Network error: {message}")]
    NetworkError { message: String },

    #[error("Serialization error: {message}")]
    SerializationError { message: String },

    #[error("JavaScript error: {message}")]
    JavaScriptError { message: String },

    #[error("Unknown error: {message}")]
    Unknown { message: String },
}

impl From<WasmError> for JsValue {
    fn from(error: WasmError) -> Self {
        let error_obj = js_sys::Error::new(&error.to_string());
        
        // Add error type as a property
        let type_name = match error {
            WasmError::ProcessingError { .. } => "ProcessingError",
            WasmError::InvalidInput { .. } => "InvalidInput",
            WasmError::ConfigError { .. } => "ConfigError",
            WasmError::SecurityError { .. } => "SecurityError",
            WasmError::MemoryError { .. } => "MemoryError",
            WasmError::TimeoutError => "TimeoutError",
            WasmError::NetworkError { .. } => "NetworkError",
            WasmError::SerializationError { .. } => "SerializationError",
            WasmError::JavaScriptError { .. } => "JavaScriptError",
            WasmError::Unknown { .. } => "Unknown",
        };
        
        js_sys::Reflect::set(
            &error_obj,
            &JsValue::from_str("type"),
            &JsValue::from_str(type_name),
        ).unwrap_or_default();

        error_obj.into()
    }
}

impl From<JsValue> for WasmError {
    fn from(js_value: JsValue) -> Self {
        if let Some(error_string) = js_value.as_string() {
            WasmError::JavaScriptError {
                message: error_string,
            }
        } else if let Ok(error) = js_value.dyn_into::<js_sys::Error>() {
            WasmError::JavaScriptError {
                message: error.message().into(),
            }
        } else {
            WasmError::JavaScriptError {
                message: format!("Unknown JavaScript error: {:?}", js_value),
            }
        }
    }
}

/// Convert various error types to WasmError
impl From<neural_doc_flow_core::error::ProcessingError> for WasmError {
    fn from(error: neural_doc_flow_core::error::ProcessingError) -> Self {
        WasmError::ProcessingError {
            message: error.to_string(),
        }
    }
}

impl From<serde_wasm_bindgen::Error> for WasmError {
    fn from(error: serde_wasm_bindgen::Error) -> Self {
        WasmError::SerializationError {
            message: error.to_string(),
        }
    }
}

impl From<std::io::Error> for WasmError {
    fn from(error: std::io::Error) -> Self {
        WasmError::ProcessingError {
            message: format!("IO error: {}", error),
        }
    }
}

/// Result type alias for WASM operations
pub type WasmResult<T> = Result<T, WasmError>;

/// Error context for better debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContext {
    pub operation: String,
    pub file_name: Option<String>,
    pub file_size: Option<u64>,
    pub timestamp: String,
    pub stack_trace: Option<String>,
}

impl ErrorContext {
    pub fn new(operation: &str) -> Self {
        Self {
            operation: operation.to_string(),
            file_name: None,
            file_size: None,
            timestamp: chrono::Utc::now().to_rfc3339(),
            stack_trace: None,
        }
    }

    pub fn with_file_info(mut self, name: Option<String>, size: Option<u64>) -> Self {
        self.file_name = name;
        self.file_size = size;
        self
    }

    pub fn with_stack_trace(mut self, trace: Option<String>) -> Self {
        self.stack_trace = trace;
        self
    }
}

/// Enhanced error with context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextualError {
    pub error: WasmError,
    pub context: ErrorContext,
}

impl ContextualError {
    pub fn new(error: WasmError, context: ErrorContext) -> Self {
        Self { error, context }
    }
}

impl From<ContextualError> for JsValue {
    fn from(error: ContextualError) -> Self {
        let js_error = js_sys::Error::new(&error.error.to_string());
        
        // Add context information
        if let Ok(context_value) = serde_wasm_bindgen::to_value(&error.context) {
            js_sys::Reflect::set(
                &js_error,
                &JsValue::from_str("context"),
                &context_value,
            ).unwrap_or_default();
        }

        js_error.into()
    }
}

/// Macro for creating contextual errors
#[macro_export]
macro_rules! wasm_error {
    ($error_type:ident, $message:expr, $operation:expr) => {
        ContextualError::new(
            WasmError::$error_type {
                message: $message.to_string(),
            },
            ErrorContext::new($operation),
        )
    };
    
    ($error_type:ident, $message:expr, $operation:expr, $file_name:expr, $file_size:expr) => {
        ContextualError::new(
            WasmError::$error_type {
                message: $message.to_string(),
            },
            ErrorContext::new($operation)
                .with_file_info(Some($file_name.to_string()), Some($file_size)),
        )
    };
}

/// Macro for handling results with context
#[macro_export]
macro_rules! wasm_try {
    ($expr:expr, $operation:expr) => {
        match $expr {
            Ok(value) => value,
            Err(error) => {
                let wasm_error = WasmError::from(error);
                let context = ErrorContext::new($operation);
                return Err(ContextualError::new(wasm_error, context).into());
            }
        }
    };
}

/// Error recovery strategies
pub enum RecoveryStrategy {
    Retry { max_attempts: u32 },
    Fallback { alternative_method: String },
    Abort,
    Continue { skip_invalid: bool },
}

/// Error handler with recovery capabilities
pub struct ErrorHandler {
    strategy: RecoveryStrategy,
    attempt_count: u32,
}

impl ErrorHandler {
    pub fn new(strategy: RecoveryStrategy) -> Self {
        Self {
            strategy,
            attempt_count: 0,
        }
    }

    pub fn handle_error(&mut self, error: WasmError) -> WasmResult<Option<String>> {
        match &self.strategy {
            RecoveryStrategy::Retry { max_attempts } => {
                self.attempt_count += 1;
                if self.attempt_count < *max_attempts {
                    crate::console_warn!("Retrying operation (attempt {})", self.attempt_count);
                    Ok(Some("retry".to_string()))
                } else {
                    Err(error)
                }
            }
            RecoveryStrategy::Fallback { alternative_method } => {
                crate::console_warn!("Falling back to alternative method: {}", alternative_method);
                Ok(Some(alternative_method.clone()))
            }
            RecoveryStrategy::Abort => {
                crate::console_error!("Aborting operation due to error");
                Err(error)
            }
            RecoveryStrategy::Continue { skip_invalid } => {
                if *skip_invalid {
                    crate::console_warn!("Skipping invalid input and continuing");
                    Ok(None)
                } else {
                    Err(error)
                }
            }
        }
    }
}

/// Async error handling utilities
pub struct AsyncErrorHandler;

impl AsyncErrorHandler {
    pub async fn with_timeout<F, T>(
        future: F,
        timeout_ms: u32,
    ) -> WasmResult<T>
    where
        F: std::future::Future<Output = WasmResult<T>>,
    {
        // Note: In a real implementation, we'd use something like tokio::timeout
        // For WASM, we might need to use JavaScript's setTimeout
        future.await
    }

    pub async fn with_retry<F, T, Fut>(
        mut operation: F,
        max_attempts: u32,
        delay_ms: u32,
    ) -> WasmResult<T>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = WasmResult<T>>,
    {
        let mut last_error = None;
        
        for attempt in 1..=max_attempts {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(error) => {
                    last_error = Some(error);
                    if attempt < max_attempts {
                        crate::console_warn!("Attempt {} failed, retrying in {}ms", attempt, delay_ms);
                        // In a real implementation, we'd await a delay here
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| WasmError::Unknown {
            message: "All retry attempts failed".to_string(),
        }))
    }
}