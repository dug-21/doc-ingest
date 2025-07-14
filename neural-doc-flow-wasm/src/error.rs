//! Pure Rust error handling for WASM bindings
//! 
//! Architecture Compliance: Zero JavaScript dependencies

use serde::{Deserialize, Serialize};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
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

    #[error("System error: {message}")]
    SystemError { message: String },

    #[error("Unknown error: {message}")]
    Unknown { message: String },
}

/// C-style error representation
#[repr(C)]
#[derive(Debug, Clone)]
pub struct CWasmError {
    pub error_code: i32,
    pub message: *mut c_char,
    pub context: *mut c_char,
}

impl CWasmError {
    pub fn new(error: WasmError, context: Option<&str>) -> Self {
        let error_code = match error {
            WasmError::ProcessingError { .. } => 1,
            WasmError::InvalidInput { .. } => 2,
            WasmError::ConfigError { .. } => 3,
            WasmError::SecurityError { .. } => 4,
            WasmError::MemoryError { .. } => 5,
            WasmError::TimeoutError => 6,
            WasmError::NetworkError { .. } => 7,
            WasmError::SerializationError { .. } => 8,
            WasmError::SystemError { .. } => 9,
            WasmError::Unknown { .. } => 10,
        };

        let message = crate::utils::rust_string_to_c(&error.to_string());
        let context_ptr = match context {
            Some(ctx) => crate::utils::rust_string_to_c(ctx),
            None => std::ptr::null_mut(),
        };

        Self {
            error_code,
            message,
            context: context_ptr,
        }
    }

    pub fn free_memory(&mut self) {
        unsafe {
            if !self.message.is_null() {
                let _ = CString::from_raw(self.message);
                self.message = std::ptr::null_mut();
            }

            if !self.context.is_null() {
                let _ = CString::from_raw(self.context);
                self.context = std::ptr::null_mut();
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

impl From<std::io::Error> for WasmError {
    fn from(error: std::io::Error) -> Self {
        WasmError::ProcessingError {
            message: format!("IO error: {}", error),
        }
    }
}

impl From<serde_json::Error> for WasmError {
    fn from(error: serde_json::Error) -> Self {
        WasmError::SerializationError {
            message: error.to_string(),
        }
    }
}

impl From<std::string::FromUtf8Error> for WasmError {
    fn from(error: std::string::FromUtf8Error) -> Self {
        WasmError::SerializationError {
            message: format!("UTF-8 error: {}", error),
        }
    }
}

impl From<std::num::ParseIntError> for WasmError {
    fn from(error: std::num::ParseIntError) -> Self {
        WasmError::InvalidInput {
            message: format!("Parse error: {}", error),
        }
    }
}

impl From<std::num::ParseFloatError> for WasmError {
    fn from(error: std::num::ParseFloatError) -> Self {
        WasmError::InvalidInput {
            message: format!("Parse error: {}", error),
        }
    }
}

/// Result type alias for WASM operations
pub type WasmResult<T> = Result<T, WasmError>;

/// Error context for better debugging
#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContext {
    pub operation: *mut c_char,
    pub file_name: *mut c_char,
    pub file_size: u64,
    pub timestamp: *mut c_char,
    pub stack_trace: *mut c_char,
}

impl ErrorContext {
    pub fn new(operation: &str) -> Self {
        let timestamp = chrono::Utc::now().to_rfc3339();
        
        Self {
            operation: crate::utils::rust_string_to_c(operation),
            file_name: std::ptr::null_mut(),
            file_size: 0,
            timestamp: crate::utils::rust_string_to_c(&timestamp),
            stack_trace: std::ptr::null_mut(),
        }
    }

    pub fn with_file_info(mut self, name: Option<&str>, size: u64) -> Self {
        if let Some(name) = name {
            self.file_name = crate::utils::rust_string_to_c(name);
        }
        self.file_size = size;
        self
    }

    pub fn with_stack_trace(mut self, trace: Option<&str>) -> Self {
        if let Some(trace) = trace {
            self.stack_trace = crate::utils::rust_string_to_c(trace);
        }
        self
    }

    pub fn free_memory(&mut self) {
        unsafe {
            if !self.operation.is_null() {
                let _ = CString::from_raw(self.operation);
                self.operation = std::ptr::null_mut();
            }

            if !self.file_name.is_null() {
                let _ = CString::from_raw(self.file_name);
                self.file_name = std::ptr::null_mut();
            }

            if !self.timestamp.is_null() {
                let _ = CString::from_raw(self.timestamp);
                self.timestamp = std::ptr::null_mut();
            }

            if !self.stack_trace.is_null() {
                let _ = CString::from_raw(self.stack_trace);
                self.stack_trace = std::ptr::null_mut();
            }
        }
    }
}

/// Enhanced error with context
#[repr(C)]
#[derive(Debug, Clone)]
pub struct ContextualError {
    pub error: CWasmError,
    pub context: ErrorContext,
}

impl ContextualError {
    pub fn new(error: WasmError, context: ErrorContext) -> Self {
        Self {
            error: CWasmError::new(error, None),
            context,
        }
    }

    pub fn free_memory(&mut self) {
        self.error.free_memory();
        self.context.free_memory();
    }
}

/// Error recovery strategies
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum RecoveryStrategy {
    Retry = 1,
    Fallback = 2,
    Abort = 3,
    Continue = 4,
}

/// Error handler with recovery capabilities
pub struct ErrorHandler {
    strategy: RecoveryStrategy,
    max_attempts: u32,
    attempt_count: u32,
}

impl ErrorHandler {
    pub fn new(strategy: RecoveryStrategy, max_attempts: u32) -> Self {
        Self {
            strategy,
            max_attempts,
            attempt_count: 0,
        }
    }

    pub fn handle_error(&mut self, error: WasmError) -> WasmResult<Option<String>> {
        match self.strategy {
            RecoveryStrategy::Retry => {
                self.attempt_count += 1;
                if self.attempt_count < self.max_attempts {
                    crate::utils::log_warn(&format!("Retrying operation (attempt {})", self.attempt_count));
                    Ok(Some("retry".to_string()))
                } else {
                    crate::utils::log_error(&format!("Max retry attempts reached: {}", error));
                    Err(error)
                }
            }
            RecoveryStrategy::Fallback => {
                crate::utils::log_warn(&format!("Falling back due to error: {}", error));
                Ok(Some("fallback".to_string()))
            }
            RecoveryStrategy::Abort => {
                crate::utils::log_error(&format!("Aborting operation due to error: {}", error));
                Err(error)
            }
            RecoveryStrategy::Continue => {
                crate::utils::log_warn(&format!("Continuing despite error: {}", error));
                Ok(None)
            }
        }
    }

    pub fn reset(&mut self) {
        self.attempt_count = 0;
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
        // Simple timeout implementation - in production you'd use proper async timeout
        let start = std::time::Instant::now();
        let result = future.await;
        
        if start.elapsed().as_millis() > timeout_ms as u128 {
            Err(WasmError::TimeoutError)
        } else {
            result
        }
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
                        crate::utils::log_warn(&format!("Attempt {} failed, retrying in {}ms", attempt, delay_ms));
                        // In production, you'd implement proper async delay
                        std::thread::sleep(std::time::Duration::from_millis(delay_ms as u64));
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| WasmError::Unknown {
            message: "All retry attempts failed".to_string(),
        }))
    }
}

/// Global error tracking
use std::sync::atomic::{AtomicU64, Ordering};

pub static ERROR_COUNT: AtomicU64 = AtomicU64::new(0);
pub static LAST_ERROR_TIME: AtomicU64 = AtomicU64::new(0);

pub fn track_error(error: &WasmError) {
    ERROR_COUNT.fetch_add(1, Ordering::SeqCst);
    LAST_ERROR_TIME.store(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        Ordering::SeqCst,
    );
    
    crate::utils::log_error(&format!("Error tracked: {}", error));
}

pub fn get_error_count() -> u64 {
    ERROR_COUNT.load(Ordering::SeqCst)
}

pub fn get_last_error_time() -> u64 {
    LAST_ERROR_TIME.load(Ordering::SeqCst)
}

pub fn reset_error_tracking() {
    ERROR_COUNT.store(0, Ordering::SeqCst);
    LAST_ERROR_TIME.store(0, Ordering::SeqCst);
}

// C-style API for error handling
#[no_mangle]
pub extern "C" fn create_error_handler(strategy: RecoveryStrategy, max_attempts: u32) -> *mut ErrorHandler {
    let handler = ErrorHandler::new(strategy, max_attempts);
    Box::into_raw(Box::new(handler))
}

#[no_mangle]
pub extern "C" fn destroy_error_handler(handler: *mut ErrorHandler) {
    if !handler.is_null() {
        unsafe {
            let _ = Box::from_raw(handler);
        }
    }
}

#[no_mangle]
pub extern "C" fn reset_error_handler(handler: *mut ErrorHandler) {
    if !handler.is_null() {
        unsafe {
            let handler_ref = &mut *handler;
            handler_ref.reset();
        }
    }
}

#[no_mangle]
pub extern "C" fn create_error_context(operation: *const c_char) -> *mut ErrorContext {
    if operation.is_null() {
        return std::ptr::null_mut();
    }

    unsafe {
        let operation_str = CStr::from_ptr(operation).to_string_lossy();
        let context = ErrorContext::new(&operation_str);
        Box::into_raw(Box::new(context))
    }
}

#[no_mangle]
pub extern "C" fn destroy_error_context(context: *mut ErrorContext) {
    if !context.is_null() {
        unsafe {
            let mut context_box = Box::from_raw(context);
            context_box.free_memory();
        }
    }
}

#[no_mangle]
pub extern "C" fn set_error_context_file_info(
    context: *mut ErrorContext,
    file_name: *const c_char,
    file_size: u64,
) {
    if context.is_null() {
        return;
    }

    unsafe {
        let context_ref = &mut *context;
        
        // Free existing file_name if set
        if !context_ref.file_name.is_null() {
            let _ = CString::from_raw(context_ref.file_name);
        }
        
        if !file_name.is_null() {
            let file_name_str = CStr::from_ptr(file_name).to_string_lossy();
            context_ref.file_name = crate::utils::rust_string_to_c(&file_name_str);
        } else {
            context_ref.file_name = std::ptr::null_mut();
        }
        
        context_ref.file_size = file_size;
    }
}

#[no_mangle]
pub extern "C" fn get_global_error_count() -> u64 {
    get_error_count()
}

#[no_mangle]
pub extern "C" fn get_global_last_error_time() -> u64 {
    get_last_error_time()
}

#[no_mangle]
pub extern "C" fn reset_global_error_tracking() {
    reset_error_tracking();
}

/// Validation utilities
pub fn validate_pointer<T>(ptr: *const T, name: &str) -> WasmResult<()> {
    if ptr.is_null() {
        Err(WasmError::InvalidInput {
            message: format!("Null pointer provided for {}", name),
        })
    } else {
        Ok(())
    }
}

pub fn validate_string(s: &str, name: &str, max_len: usize) -> WasmResult<()> {
    if s.is_empty() {
        Err(WasmError::InvalidInput {
            message: format!("Empty string provided for {}", name),
        })
    } else if s.len() > max_len {
        Err(WasmError::InvalidInput {
            message: format!("String too long for {}: {} > {}", name, s.len(), max_len),
        })
    } else {
        Ok(())
    }
}

pub fn validate_range(value: i64, min: i64, max: i64, name: &str) -> WasmResult<()> {
    if value < min || value > max {
        Err(WasmError::InvalidInput {
            message: format!("Value {} out of range for {}: [{}, {}]", value, name, min, max),
        })
    } else {
        Ok(())
    }
}

/// Safe string conversion utilities
pub fn safe_c_string_to_rust(c_str: *const c_char) -> WasmResult<String> {
    validate_pointer(c_str, "c_string")?;
    
    unsafe {
        CStr::from_ptr(c_str)
            .to_str()
            .map_err(|e| WasmError::SerializationError {
                message: format!("Invalid UTF-8 in C string: {}", e),
            })
            .map(|s| s.to_string())
    }
}

pub fn safe_rust_string_to_c(rust_str: &str) -> WasmResult<*mut c_char> {
    CString::new(rust_str)
        .map_err(|e| WasmError::SerializationError {
            message: format!("Invalid string conversion: {}", e),
        })
        .map(|c_string| c_string.into_raw())
}

/// Memory safety utilities
pub fn safe_free_c_string(ptr: *mut c_char) -> WasmResult<()> {
    if ptr.is_null() {
        Ok(()) // Freeing null pointer is safe
    } else {
        unsafe {
            let _ = CString::from_raw(ptr);
            Ok(())
        }
    }
}

pub fn safe_slice_from_raw_parts<T>(ptr: *const T, len: usize) -> WasmResult<&'static [T]> {
    validate_pointer(ptr, "slice_pointer")?;
    
    if len == 0 {
        Ok(&[])
    } else {
        unsafe {
            Ok(std::slice::from_raw_parts(ptr, len))
        }
    }
}

/// Panic handler for WASM
pub fn setup_panic_handler() {
    std::panic::set_hook(Box::new(|panic_info| {
        let message = if let Some(s) = panic_info.payload().downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = panic_info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "Unknown panic".to_string()
        };

        let location = if let Some(location) = panic_info.location() {
            format!(" at {}:{}", location.file(), location.line())
        } else {
            String::new()
        };

        let error = WasmError::SystemError {
            message: format!("Panic: {}{}", message, location),
        };

        track_error(&error);
        crate::utils::log_error(&format!("PANIC: {}", error));
    }));
}