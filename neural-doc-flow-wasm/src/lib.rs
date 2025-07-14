//! Pure Rust WebAssembly bindings for neural document processing
//! 
//! This crate provides a pure Rust WASM implementation for the neural document
//! processing system, enabling high-performance document processing without
//! JavaScript runtime dependencies.
//! 
//! Architecture Compliance: Zero JavaScript dependencies - Pure Rust WASM

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

// Import our core processing modules
use neural_doc_flow_core::{DocumentProcessor, ProcessingConfig, Document};
use neural_doc_flow_processors::neural_engine::NeuralEngine;
use neural_doc_flow_sources::manager::SourceManager;
use neural_doc_flow_outputs::OutputFormat;
use neural_doc_flow_security::SecurityEngine;

mod utils;
mod streaming;
mod error;
mod types;

pub use utils::*;
pub use streaming::*;
pub use error::*;
pub use types::*;

// Global allocator for smaller WASM binary size
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

/// Initialize the WASM module with pure Rust implementation
#[no_mangle]
pub extern "C" fn init_neural_doc_flow() -> i32 {
    // Set up panic hook for better error reporting
    utils::set_panic_hook();
    
    // Log initialization (pure Rust logging)
    println!("Neural Document Flow WASM initialized (Pure Rust)");
    
    0 // Success
}

/// Main processor interface for pure Rust WASM
#[repr(C)]
pub struct WasmDocumentProcessor {
    inner: Arc<DocumentProcessor>,
    config: ProcessingConfig,
}

/// Configuration options for document processing
#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmProcessingConfig {
    /// Enable neural enhancement
    pub neural_enhancement: bool,
    /// Maximum file size in bytes
    pub max_file_size: u32,
    /// Processing timeout in milliseconds
    pub timeout_ms: u32,
    /// Security scanning level (0-3)
    pub security_level: u8,
    /// Output formats count
    pub output_formats_count: usize,
    /// Output formats as C-style strings
    pub output_formats: [*const c_char; 16], // Max 16 formats
}

/// Document processing result
#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmProcessingResult {
    /// Processing success status
    pub success: bool,
    /// Processing time in milliseconds
    pub processing_time_ms: u32,
    /// Content length
    pub content_length: usize,
    /// Content pointer (caller must free)
    pub content_ptr: *mut c_char,
    /// Metadata count
    pub metadata_count: usize,
    /// Metadata keys pointer
    pub metadata_keys: *mut *mut c_char,
    /// Metadata values pointer
    pub metadata_values: *mut *mut c_char,
    /// Warnings count
    pub warnings_count: usize,
    /// Warnings pointer
    pub warnings: *mut *mut c_char,
}

/// Security scan result structure
#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityScanResult {
    pub is_safe: bool,
    pub threat_level: u8,
    pub scan_time_ms: u32,
    pub detected_threats_count: usize,
    pub detected_threats: *mut *mut c_char,
}

// C-style API for pure Rust WASM
#[no_mangle]
pub extern "C" fn create_processor() -> *mut WasmDocumentProcessor {
    let config = ProcessingConfig::default();
    
    match DocumentProcessor::new(config.clone()) {
        Ok(processor) => {
            let wasm_processor = Box::new(WasmDocumentProcessor {
                inner: Arc::new(processor),
                config,
            });
            Box::into_raw(wasm_processor)
        }
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn create_processor_with_config(config: *const WasmProcessingConfig) -> *mut WasmDocumentProcessor {
    if config.is_null() {
        return std::ptr::null_mut();
    }
    
    unsafe {
        let config_ref = &*config;
        
        match config_ref.to_core_config() {
            Ok(processing_config) => {
                match DocumentProcessor::new(processing_config.clone()) {
                    Ok(processor) => {
                        let wasm_processor = Box::new(WasmDocumentProcessor {
                            inner: Arc::new(processor),
                            config: processing_config,
                        });
                        Box::into_raw(wasm_processor)
                    }
                    Err(_) => std::ptr::null_mut(),
                }
            }
            Err(_) => std::ptr::null_mut(),
        }
    }
}

#[no_mangle]
pub extern "C" fn destroy_processor(processor: *mut WasmDocumentProcessor) {
    if !processor.is_null() {
        unsafe {
            let _ = Box::from_raw(processor);
        }
    }
}

#[no_mangle]
pub extern "C" fn process_bytes(
    processor: *mut WasmDocumentProcessor,
    data: *const u8,
    data_len: usize,
    filename: *const c_char,
    result: *mut WasmProcessingResult,
) -> i32 {
    if processor.is_null() || data.is_null() || result.is_null() {
        return -1;
    }
    
    unsafe {
        let processor_ref = &*processor;
        let data_slice = std::slice::from_raw_parts(data, data_len);
        
        let filename_str = if filename.is_null() {
            "unknown".to_string()
        } else {
            CStr::from_ptr(filename).to_string_lossy().to_string()
        };
        
        // Process the document
        let rt = tokio::runtime::Runtime::new().unwrap();
        let document = match Document::from_bytes(data_slice.to_vec(), filename_str) {
            Ok(doc) => doc,
            Err(_) => return -2,
        };
        
        let processing_result = match rt.block_on(processor_ref.inner.process(document)) {
            Ok(res) => res,
            Err(_) => return -3,
        };
        
        // Convert to C-style result
        let wasm_result = WasmProcessingResult::from_core_result(processing_result);
        *result = wasm_result;
        
        0 // Success
    }
}

#[no_mangle]
pub extern "C" fn process_file_data(
    processor: *mut WasmDocumentProcessor,
    data: *const u8,
    data_len: usize,
    filename: *const c_char,
    result: *mut WasmProcessingResult,
) -> i32 {
    process_bytes(processor, data, data_len, filename, result)
}

#[no_mangle]
pub extern "C" fn get_version() -> *const c_char {
    static VERSION: &str = env!("CARGO_PKG_VERSION");
    VERSION.as_ptr() as *const c_char
}

#[no_mangle]
pub extern "C" fn neural_available() -> bool {
    neural_doc_flow_processors::neural_engine::is_available()
}

#[no_mangle]
pub extern "C" fn validate_file_format(
    data: *const u8,
    data_len: usize,
    filename: *const c_char,
    max_size: u32,
) -> bool {
    if data.is_null() || filename.is_null() {
        return false;
    }
    
    if data_len > max_size as usize {
        return false;
    }
    
    unsafe {
        let data_slice = std::slice::from_raw_parts(data, data_len);
        let filename_str = CStr::from_ptr(filename).to_string_lossy();
        
        match neural_doc_flow_sources::validate_file_format(data_slice, &filename_str) {
            Ok(is_valid) => is_valid,
            Err(_) => false,
        }
    }
}

#[no_mangle]
pub extern "C" fn estimate_processing_time(file_size: u32, enable_neural: bool) -> u32 {
    let base_time = (file_size / 1024) * 10; // 10ms per KB base
    let neural_multiplier = if enable_neural { 3 } else { 1 };
    
    base_time * neural_multiplier
}

#[no_mangle]
pub extern "C" fn free_cstring(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe {
            let _ = CString::from_raw(ptr);
        }
    }
}

#[no_mangle]
pub extern "C" fn free_processing_result(result: *mut WasmProcessingResult) {
    if !result.is_null() {
        unsafe {
            let result_ref = &mut *result;
            
            // Free content
            if !result_ref.content_ptr.is_null() {
                let _ = CString::from_raw(result_ref.content_ptr);
            }
            
            // Free metadata
            if !result_ref.metadata_keys.is_null() {
                for i in 0..result_ref.metadata_count {
                    let key_ptr = *result_ref.metadata_keys.add(i);
                    let value_ptr = *result_ref.metadata_values.add(i);
                    if !key_ptr.is_null() {
                        let _ = CString::from_raw(key_ptr);
                    }
                    if !value_ptr.is_null() {
                        let _ = CString::from_raw(value_ptr);
                    }
                }
                let _ = Vec::from_raw_parts(
                    result_ref.metadata_keys,
                    result_ref.metadata_count,
                    result_ref.metadata_count,
                );
                let _ = Vec::from_raw_parts(
                    result_ref.metadata_values,
                    result_ref.metadata_count,
                    result_ref.metadata_count,
                );
            }
            
            // Free warnings
            if !result_ref.warnings.is_null() {
                for i in 0..result_ref.warnings_count {
                    let warning_ptr = *result_ref.warnings.add(i);
                    if !warning_ptr.is_null() {
                        let _ = CString::from_raw(warning_ptr);
                    }
                }
                let _ = Vec::from_raw_parts(
                    result_ref.warnings,
                    result_ref.warnings_count,
                    result_ref.warnings_count,
                );
            }
        }
    }
}

/// Pure Rust WASM utilities
pub struct WasmUtils;

impl WasmUtils {
    /// Get version information
    pub fn version() -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }

    /// Check if neural processing is available
    pub fn neural_available() -> bool {
        neural_doc_flow_processors::neural_engine::is_available()
    }

    /// Get supported input formats
    pub fn supported_formats() -> Vec<String> {
        neural_doc_flow_sources::get_supported_formats()
    }

    /// Get supported output formats
    pub fn output_formats() -> Vec<String> {
        neural_doc_flow_outputs::get_supported_formats()
    }

    /// Validate a file before processing
    pub fn validate_file(data: &[u8], filename: String, max_size: u32) -> Result<bool, Box<dyn std::error::Error>> {
        if data.len() > max_size as usize {
            return Ok(false);
        }

        let is_valid = neural_doc_flow_sources::validate_file_format(data, &filename)?;
        Ok(is_valid)
    }

    /// Estimate processing time for a document
    pub fn estimate_processing_time(file_size: u32, enable_neural: bool) -> u32 {
        let base_time = (file_size / 1024) * 10; // 10ms per KB base
        let neural_multiplier = if enable_neural { 3 } else { 1 };
        
        base_time * neural_multiplier
    }
}

/// Export the default processing configuration
pub fn default_config() -> WasmProcessingConfig {
    WasmProcessingConfig {
        neural_enhancement: true,
        max_file_size: 50 * 1024 * 1024, // 50MB
        timeout_ms: 30000,               // 30 seconds
        security_level: 2,               // Moderate security
        output_formats_count: 2,
        output_formats: [
            "text\0".as_ptr() as *const c_char,
            "json\0".as_ptr() as *const c_char,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
        ],
    }
}

// Export C-style header information
#[no_mangle]
pub extern "C" fn get_api_version() -> u32 {
    1 // API version 1
}

#[no_mangle]
pub extern "C" fn get_feature_flags() -> u32 {
    let mut flags = 0u32;
    
    if neural_doc_flow_processors::neural_engine::is_available() {
        flags |= 1; // Neural processing available
    }
    
    flags |= 2; // Security scanning always available
    flags |= 4; // Streaming processing available
    
    flags
}