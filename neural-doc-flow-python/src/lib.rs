//! # Neural Document Flow Python Bindings
//!
//! Python bindings for the neural document processing framework using PyO3.
//! Provides a Pythonic API for document processing, security analysis, and plugin management.

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::{exceptions::PyException, wrap_pyfunction};

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

// Import the core Rust modules
use neural_doc_flow_core::{
    engine::DocumentEngine,
    config::{NeuralDocFlowConfig, SecurityConfig, SecurityScanMode, SecurityAction},
    document::Document,
    error::ProcessingError,
};
use neural_doc_flow_security::SecurityProcessor as RustSecurityProcessor;
use neural_doc_flow_plugins::{PluginManager, PluginConfig};

mod processor;
mod security;
mod plugin_manager;
mod result;
mod error;
mod types;

pub use processor::Processor;
pub use security::{SecurityAnalysis, ThreatLevel};
pub use plugin_manager::PluginManager as PyPluginManager;
pub use result::ProcessingResult;
pub use error::NeuralDocFlowError;
pub use types::*;

/// Initialize the neural document flow Python module
#[pymodule]
fn neuraldocflow(_py: Python, m: &PyModule) -> PyResult<()> {
    // Add the main processor class
    m.add_class::<Processor>()?;
    
    // Add security analysis classes
    m.add_class::<SecurityAnalysis>()?;
    m.add_class::<ThreatLevel>()?;
    
    // Add plugin management
    m.add_class::<PyPluginManager>()?;
    
    // Add result classes
    m.add_class::<ProcessingResult>()?;
    
    // Add error types
    m.add("NeuralDocFlowError", _py.get_type::<NeuralDocFlowError>())?;
    
    // Add utility functions
    m.add_function(wrap_pyfunction!(create_processor, m)?)?;
    m.add_function(wrap_pyfunction!(get_supported_formats, m)?)?;
    m.add_function(wrap_pyfunction!(get_version, m)?)?;
    
    // Add constants
    m.add("__version__", "0.1.0")?;
    m.add("API_VERSION", "1.0.0")?;
    
    Ok(())
}

/// Create a new document processor with default settings
/// 
/// # Arguments
/// * `security_level` - Security level: "disabled", "basic", "standard", "high"
/// * `enable_neural` - Enable neural processing features
/// * `plugins` - List of plugin names to enable
/// 
/// # Returns
/// A new Processor instance
/// 
/// # Example
/// ```python
/// import neuraldocflow
/// 
/// processor = neuraldocflow.create_processor(
///     security_level="high",
///     enable_neural=True,
///     plugins=["docx", "tables", "images"]
/// )
/// ```
#[pyfunction]
#[pyo3(signature = (security_level = "standard", enable_neural = true, plugins = None))]
fn create_processor(
    security_level: &str,
    enable_neural: bool,
    plugins: Option<Vec<String>>,
) -> PyResult<Processor> {
    Processor::new(security_level, enable_neural, plugins)
}

/// Get list of supported document formats
/// 
/// # Returns
/// List of supported MIME types and file extensions
/// 
/// # Example
/// ```python
/// import neuraldocflow
/// 
/// formats = neuraldocflow.get_supported_formats()
/// print(f"Supported formats: {formats}")
/// ```
#[pyfunction]
fn get_supported_formats() -> HashMap<String, Vec<String>> {
    let mut formats = HashMap::new();
    
    // Document formats
    formats.insert("text".to_string(), vec![
        "text/plain".to_string(),
        "text/markdown".to_string(),
        "text/html".to_string(),
    ]);
    
    formats.insert("pdf".to_string(), vec![
        "application/pdf".to_string(),
    ]);
    
    formats.insert("office".to_string(), vec![
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document".to_string(),
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet".to_string(),
        "application/vnd.openxmlformats-officedocument.presentationml.presentation".to_string(),
        "application/msword".to_string(),
        "application/vnd.ms-excel".to_string(),
        "application/vnd.ms-powerpoint".to_string(),
    ]);
    
    formats.insert("images".to_string(), vec![
        "image/jpeg".to_string(),
        "image/png".to_string(),
        "image/tiff".to_string(),
        "image/bmp".to_string(),
    ]);
    
    formats
}

/// Get the version of the neural document flow library
/// 
/// # Returns
/// Version information including core library, Python bindings, and API versions
/// 
/// # Example
/// ```python
/// import neuraldocflow
/// 
/// version_info = neuraldocflow.get_version()
/// print(f"Library version: {version_info}")
/// ```
#[pyfunction]
fn get_version() -> HashMap<String, String> {
    let mut version_info = HashMap::new();
    
    version_info.insert("neuraldocflow".to_string(), "0.1.0".to_string());
    version_info.insert("python_bindings".to_string(), "0.1.0".to_string());
    version_info.insert("api_version".to_string(), "1.0.0".to_string());
    version_info.insert("rust_core".to_string(), "0.1.0".to_string());
    
    version_info
}

/// Convert Rust ProcessingError to Python exception
impl From<ProcessingError> for PyErr {
    fn from(err: ProcessingError) -> PyErr {
        NeuralDocFlowError::new_err(err.to_string())
    }
}

/// Convert anyhow::Error to Python exception
impl From<anyhow::Error> for PyErr {
    fn from(err: anyhow::Error) -> PyErr {
        NeuralDocFlowError::new_err(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_supported_formats() {
        let formats = get_supported_formats();
        assert!(formats.contains_key("text"));
        assert!(formats.contains_key("pdf"));
        assert!(formats.contains_key("office"));
        assert!(formats.contains_key("images"));
    }
    
    #[test]
    fn test_version_info() {
        let version = get_version();
        assert!(version.contains_key("neuraldocflow"));
        assert!(version.contains_key("python_bindings"));
        assert!(version.contains_key("api_version"));
        assert!(version.contains_key("rust_core"));
    }
}