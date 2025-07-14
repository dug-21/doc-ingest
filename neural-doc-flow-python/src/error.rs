//! Error types for Python bindings

use pyo3::prelude::*;
use pyo3::exceptions::PyException;

/// Custom exception type for neural document flow errors
/// 
/// This exception is raised when errors occur during document processing,
/// plugin operations, or security analysis.
/// 
/// # Example
/// ```python
/// import neuraldocflow
/// 
/// try:
///     processor = neuraldocflow.Processor(security_level="invalid")
/// except neuraldocflow.NeuralDocFlowError as e:
///     print(f"Error: {e}")
/// ```
#[pyclass(extends=PyException)]
#[derive(Debug, Clone)]
pub struct NeuralDocFlowError {
    #[pyo3(get)]
    pub message: String,
    
    #[pyo3(get)]
    pub error_type: String,
    
    #[pyo3(get)]
    pub error_code: Option<i32>,
}

#[pymethods]
impl NeuralDocFlowError {
    /// Create a new NeuralDocFlowError
    /// 
    /// # Arguments
    /// * `message` - Error message
    /// * `error_type` - Type of error (optional)
    /// * `error_code` - Error code (optional)
    #[new]
    #[pyo3(signature = (message, error_type = None, error_code = None))]
    pub fn new(
        message: String,
        error_type: Option<String>,
        error_code: Option<i32>,
    ) -> Self {
        Self {
            message,
            error_type: error_type.unwrap_or_else(|| "NeuralDocFlowError".to_string()),
            error_code,
        }
    }
    
    /// String representation of the error
    fn __str__(&self) -> String {
        if let Some(code) = self.error_code {
            format!("[{}:{}] {}", self.error_type, code, self.message)
        } else {
            format!("[{}] {}", self.error_type, self.message)
        }
    }
    
    /// Representation of the error
    fn __repr__(&self) -> String {
        format!(
            "NeuralDocFlowError(message='{}', error_type='{}', error_code={:?})",
            self.message, self.error_type, self.error_code
        )
    }
}

impl NeuralDocFlowError {
    /// Create a new error with just a message
    pub fn new_err(message: String) -> PyErr {
        PyErr::new::<NeuralDocFlowError, _>((message.clone(), None::<String>, None::<i32>))
    }
    
    /// Create a processing error
    pub fn processing_error(message: String) -> PyErr {
        PyErr::new::<NeuralDocFlowError, _>((
            message.clone(),
            Some("ProcessingError".to_string()),
            Some(1001)
        ))
    }
    
    /// Create a security error
    pub fn security_error(message: String) -> PyErr {
        PyErr::new::<NeuralDocFlowError, _>((
            message.clone(),
            Some("SecurityError".to_string()),
            Some(2001)
        ))
    }
    
    /// Create a plugin error
    pub fn plugin_error(message: String) -> PyErr {
        PyErr::new::<NeuralDocFlowError, _>((
            message.clone(),
            Some("PluginError".to_string()),
            Some(3001)
        ))
    }
    
    /// Create a configuration error
    pub fn config_error(message: String) -> PyErr {
        PyErr::new::<NeuralDocFlowError, _>((
            message.clone(),
            Some("ConfigurationError".to_string()),
            Some(4001)
        ))
    }
    
    /// Create an I/O error
    pub fn io_error(message: String) -> PyErr {
        PyErr::new::<NeuralDocFlowError, _>((
            message.clone(),
            Some("IOError".to_string()),
            Some(5001)
        ))
    }
}

/// Specific error types for different categories

/// Processing-related errors
#[pyclass(extends=NeuralDocFlowError)]
pub struct ProcessingError;

/// Security-related errors
#[pyclass(extends=NeuralDocFlowError)]
pub struct SecurityError;

/// Plugin-related errors
#[pyclass(extends=NeuralDocFlowError)]
pub struct PluginError;

/// Configuration-related errors
#[pyclass(extends=NeuralDocFlowError)]
pub struct ConfigurationError;

/// I/O related errors
#[pyclass(extends=NeuralDocFlowError)]
pub struct IOError;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_creation() {
        let error = NeuralDocFlowError::new(
            "Test error".to_string(),
            Some("TestError".to_string()),
            Some(1234)
        );
        
        assert_eq!(error.message, "Test error");
        assert_eq!(error.error_type, "TestError");
        assert_eq!(error.error_code, Some(1234));
    }
    
    #[test]
    fn test_error_display() {
        let error = NeuralDocFlowError::new(
            "Test message".to_string(),
            Some("TestType".to_string()),
            Some(123)
        );
        
        let display = error.__str__();
        assert!(display.contains("Test message"));
        assert!(display.contains("TestType"));
        assert!(display.contains("123"));
    }
    
    #[test]
    fn test_error_without_code() {
        let error = NeuralDocFlowError::new(
            "Test message".to_string(),
            Some("TestType".to_string()),
            None
        );
        
        let display = error.__str__();
        assert!(display.contains("Test message"));
        assert!(display.contains("TestType"));
        assert!(!display.contains(":"));
    }
}