//! Pure Rust type definitions and conversions for WASM bindings
//! 
//! Architecture Compliance: Zero JavaScript dependencies

use serde::{Deserialize, Serialize};

#[cfg(feature = "streaming")]
use std::collections::HashMap;

#[cfg(feature = "basic")]
use std::ffi::{CStr, CString};

#[cfg(feature = "basic")]
use std::os::raw::c_char;

#[cfg(feature = "processing")]
use neural_doc_flow_core::{ProcessingConfig, Document};

#[cfg(feature = "neural")]
use neural_doc_flow_processors::ProcessingResult;

use crate::WasmProcessingConfig;
use crate::WasmProcessingResult;

#[cfg(feature = "security")]
use crate::SecurityScanResult;

/// Convert WASM config to core config
impl WasmProcessingConfig {
    pub fn to_core_config(&self) -> Result<ProcessingConfig, Box<dyn std::error::Error>> {
        let mut config = ProcessingConfig::default();
        
        // Set neural enhancement
        config.set_neural_enabled(self.neural_enhancement);
        
        // Set size limits
        config.set_max_file_size(self.max_file_size as u64);
        
        // Set timeout
        config.set_timeout_ms(self.timeout_ms);
        
        // Set security level
        config.set_security_level(self.security_level);
        
        // Set output formats from C-style array
        for i in 0..self.output_formats_count {
            if i < self.output_formats.len() {
                let format_ptr = self.output_formats[i];
                if !format_ptr.is_null() {
                    unsafe {
                        let format_str = CStr::from_ptr(format_ptr).to_string_lossy();
                        config.add_output_format(format_str.to_string())?;
                    }
                }
            }
        }
        
        Ok(config)
    }

    pub fn from_core_config(config: &ProcessingConfig) -> Self {
        let formats = config.output_formats();
        let mut output_formats = [std::ptr::null(); 16];
        
        for (i, format) in formats.iter().enumerate() {
            if i < 16 {
                output_formats[i] = format.as_ptr() as *const c_char;
            }
        }
        
        Self {
            neural_enhancement: config.neural_enabled(),
            max_file_size: config.max_file_size() as u32,
            timeout_ms: config.timeout_ms(),
            security_level: config.security_level(),
            output_formats_count: formats.len().min(16),
            output_formats,
        }
    }
}

/// Convert core processing result to WASM result
impl WasmProcessingResult {
    pub fn from_core_result(result: ProcessingResult) -> Self {
        let content_ptr = crate::utils::rust_string_to_c(&result.content);
        let content_length = result.content.len();
        
        // Convert metadata to C-style arrays
        let metadata_count = result.metadata.len();
        let (metadata_keys, metadata_values) = if metadata_count > 0 {
            let mut keys = Vec::new();
            let mut values = Vec::new();
            
            for (key, value) in result.metadata {
                keys.push(crate::utils::rust_string_to_c(&key));
                values.push(crate::utils::rust_string_to_c(&value));
            }
            
            let keys_ptr = keys.as_mut_ptr();
            let values_ptr = values.as_mut_ptr();
            std::mem::forget(keys);
            std::mem::forget(values);
            
            (keys_ptr, values_ptr)
        } else {
            (std::ptr::null_mut(), std::ptr::null_mut())
        };
        
        // Convert warnings to C-style array
        let warnings_count = result.warnings.len();
        let warnings_ptr = if warnings_count > 0 {
            let (warnings_ptr, _) = crate::utils::rust_vec_to_c_array(result.warnings);
            warnings_ptr
        } else {
            std::ptr::null_mut()
        };

        Self {
            success: result.success,
            processing_time_ms: result.processing_time_ms,
            content_length,
            content_ptr,
            metadata_count,
            metadata_keys,
            metadata_values,
            warnings_count,
            warnings: warnings_ptr,
        }
    }

    pub fn to_core_result(&self) -> ProcessingResult {
        // Convert content back
        let content = if self.content_ptr.is_null() {
            String::new()
        } else {
            unsafe {
                CStr::from_ptr(self.content_ptr).to_string_lossy().to_string()
            }
        };
        
        // Convert metadata back
        let mut metadata = HashMap::new();
        if !self.metadata_keys.is_null() && !self.metadata_values.is_null() {
            unsafe {
                for i in 0..self.metadata_count {
                    let key_ptr = *self.metadata_keys.add(i);
                    let value_ptr = *self.metadata_values.add(i);
                    
                    if !key_ptr.is_null() && !value_ptr.is_null() {
                        let key = CStr::from_ptr(key_ptr).to_string_lossy().to_string();
                        let value = CStr::from_ptr(value_ptr).to_string_lossy().to_string();
                        metadata.insert(key, value);
                    }
                }
            }
        }
        
        // Convert warnings back
        let mut warnings = Vec::new();
        if !self.warnings.is_null() {
            unsafe {
                for i in 0..self.warnings_count {
                    let warning_ptr = *self.warnings.add(i);
                    if !warning_ptr.is_null() {
                        let warning = CStr::from_ptr(warning_ptr).to_string_lossy().to_string();
                        warnings.push(warning);
                    }
                }
            }
        }

        ProcessingResult {
            success: self.success,
            processing_time_ms: self.processing_time_ms,
            content,
            metadata,
            outputs: HashMap::new(), // Not converted for simplicity
            security_results: None, // Not converted for simplicity
            warnings,
        }
    }
}

/// Pure Rust document metadata
#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmDocumentMetadata {
    pub file_name: *mut c_char,
    pub file_size: u64,
    pub mime_type: *mut c_char,
    pub pages: u32,
    pub word_count: u32,
    pub character_count: u32,
    pub language: *mut c_char,
    pub created_at: *mut c_char,
    pub modified_at: *mut c_char,
    pub author: *mut c_char,
    pub title: *mut c_char,
    pub custom_fields_count: usize,
    pub custom_field_keys: *mut *mut c_char,
    pub custom_field_values: *mut *mut c_char,
}

impl WasmDocumentMetadata {
    pub fn new(file_name: &str, file_size: u64) -> Self {
        Self {
            file_name: crate::utils::rust_string_to_c(file_name),
            file_size,
            mime_type: std::ptr::null_mut(),
            pages: 0,
            word_count: 0,
            character_count: 0,
            language: std::ptr::null_mut(),
            created_at: std::ptr::null_mut(),
            modified_at: std::ptr::null_mut(),
            author: std::ptr::null_mut(),
            title: std::ptr::null_mut(),
            custom_fields_count: 0,
            custom_field_keys: std::ptr::null_mut(),
            custom_field_values: std::ptr::null_mut(),
        }
    }

    pub fn set_mime_type(&mut self, mime_type: Option<&str>) {
        if !self.mime_type.is_null() {
            unsafe {
                let _ = CString::from_raw(self.mime_type);
            }
        }
        
        self.mime_type = match mime_type {
            Some(mt) => crate::utils::rust_string_to_c(mt),
            None => std::ptr::null_mut(),
        };
    }

    pub fn set_custom_field(&mut self, key: &str, value: &str) {
        // For simplicity, this implementation doesn't handle dynamic arrays
        // In a real implementation, you'd manage a growable array of custom fields
        crate::utils::log_info(&format!("Custom field set: {} = {}", key, value));
    }

    pub fn free_memory(&mut self) {
        unsafe {
            if !self.file_name.is_null() {
                let _ = CString::from_raw(self.file_name);
                self.file_name = std::ptr::null_mut();
            }
            
            if !self.mime_type.is_null() {
                let _ = CString::from_raw(self.mime_type);
                self.mime_type = std::ptr::null_mut();
            }
            
            if !self.language.is_null() {
                let _ = CString::from_raw(self.language);
                self.language = std::ptr::null_mut();
            }
            
            if !self.created_at.is_null() {
                let _ = CString::from_raw(self.created_at);
                self.created_at = std::ptr::null_mut();
            }
            
            if !self.modified_at.is_null() {
                let _ = CString::from_raw(self.modified_at);
                self.modified_at = std::ptr::null_mut();
            }
            
            if !self.author.is_null() {
                let _ = CString::from_raw(self.author);
                self.author = std::ptr::null_mut();
            }
            
            if !self.title.is_null() {
                let _ = CString::from_raw(self.title);
                self.title = std::ptr::null_mut();
            }
            
            if !self.custom_field_keys.is_null() {
                crate::utils::free_c_array(self.custom_field_keys, self.custom_fields_count);
                self.custom_field_keys = std::ptr::null_mut();
            }
            
            if !self.custom_field_values.is_null() {
                crate::utils::free_c_array(self.custom_field_values, self.custom_fields_count);
                self.custom_field_values = std::ptr::null_mut();
            }
        }
    }
}

/// Processing statistics for monitoring
#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmProcessingStats {
    pub documents_processed: u64,
    pub total_processing_time_ms: u64,
    pub average_processing_time_ms: f64,
    pub memory_usage_bytes: u64,
    pub neural_operations: u64,
    pub security_scans: u64,
    pub errors_encountered: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

impl WasmProcessingStats {
    pub fn new() -> Self {
        Self {
            documents_processed: 0,
            total_processing_time_ms: 0,
            average_processing_time_ms: 0.0,
            memory_usage_bytes: 0,
            neural_operations: 0,
            security_scans: 0,
            errors_encountered: 0,
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    pub fn update_processing_time(&mut self, time_ms: u64) {
        self.documents_processed += 1;
        self.total_processing_time_ms += time_ms;
        self.average_processing_time_ms = self.total_processing_time_ms as f64 / self.documents_processed as f64;
    }

    pub fn cache_hit_ratio(&self) -> f64 {
        let total_cache_operations = self.cache_hits + self.cache_misses;
        if total_cache_operations == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total_cache_operations as f64
        }
    }

    pub fn error_rate(&self) -> f64 {
        if self.documents_processed == 0 {
            0.0
        } else {
            self.errors_encountered as f64 / self.documents_processed as f64
        }
    }
}

/// Batch processing configuration
#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmBatchConfig {
    pub max_concurrent: u32,
    pub chunk_size: u32,
    pub progress_reporting: bool,
    pub fail_fast: bool,
    pub preserve_order: bool,
}

impl WasmBatchConfig {
    pub fn new() -> Self {
        Self {
            max_concurrent: 4,
            chunk_size: 10,
            progress_reporting: true,
            fail_fast: false,
            preserve_order: true,
        }
    }

    pub fn set_max_concurrent(&mut self, value: u32) {
        self.max_concurrent = value.max(1).min(16); // Limit to reasonable range
    }
}

/// Batch processing result
#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmBatchResult {
    pub total_documents: u32,
    pub successful_documents: u32,
    pub failed_documents: u32,
    pub total_processing_time_ms: u64,
    pub average_processing_time_ms: f64,
    pub results_count: usize,
    pub results: *mut WasmProcessingResult,
    pub errors_count: usize,
    pub errors: *mut *mut c_char,
}

impl WasmBatchResult {
    pub fn new(results: Vec<Result<WasmProcessingResult, String>>) -> Self {
        let total_documents = results.len() as u32;
        let mut successful_documents = 0;
        let mut failed_documents = 0;
        let mut total_time = 0u64;
        let mut wasm_results = Vec::new();
        let mut error_messages = Vec::new();

        for result in results {
            match result {
                Ok(wasm_result) => {
                    successful_documents += 1;
                    total_time += wasm_result.processing_time_ms as u64;
                    wasm_results.push(wasm_result);
                }
                Err(error) => {
                    failed_documents += 1;
                    error_messages.push(crate::utils::rust_string_to_c(&error));
                }
            }
        }

        let average_time = if successful_documents > 0 {
            total_time as f64 / successful_documents as f64
        } else {
            0.0
        };

        let results_ptr = if !wasm_results.is_empty() {
            let ptr = wasm_results.as_mut_ptr();
            std::mem::forget(wasm_results);
            ptr
        } else {
            std::ptr::null_mut()
        };

        let errors_ptr = if !error_messages.is_empty() {
            let ptr = error_messages.as_mut_ptr();
            std::mem::forget(error_messages);
            ptr
        } else {
            std::ptr::null_mut()
        };

        Self {
            total_documents,
            successful_documents,
            failed_documents,
            total_processing_time_ms: total_time,
            average_processing_time_ms: average_time,
            results_count: wasm_results.len(),
            results: results_ptr,
            errors_count: error_messages.len(),
            errors: errors_ptr,
        }
    }

    pub fn success_rate(&self) -> f64 {
        if self.total_documents == 0 {
            0.0
        } else {
            self.successful_documents as f64 / self.total_documents as f64
        }
    }

    pub fn free_memory(&mut self) {
        unsafe {
            if !self.results.is_null() {
                for i in 0..self.results_count {
                    let result_ptr = self.results.add(i);
                    // Free individual WasmProcessingResult
                    crate::free_processing_result(result_ptr);
                }
                let _ = Vec::from_raw_parts(self.results, self.results_count, self.results_count);
                self.results = std::ptr::null_mut();
            }

            if !self.errors.is_null() {
                crate::utils::free_c_array(self.errors, self.errors_count);
                self.errors = std::ptr::null_mut();
            }
        }
    }
}

/// File validation result
#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmValidationResult {
    pub is_valid: bool,
    pub mime_type: *mut c_char,
    pub estimated_size: u64,
    pub estimated_processing_time_ms: u32,
    pub security_risk_level: u8,
    pub warnings_count: usize,
    pub warnings: *mut *mut c_char,
    pub supported_features_count: usize,
    pub supported_features: *mut *mut c_char,
}

impl WasmValidationResult {
    pub fn new(is_valid: bool) -> Self {
        Self {
            is_valid,
            mime_type: std::ptr::null_mut(),
            estimated_size: 0,
            estimated_processing_time_ms: 0,
            security_risk_level: 0,
            warnings_count: 0,
            warnings: std::ptr::null_mut(),
            supported_features_count: 0,
            supported_features: std::ptr::null_mut(),
        }
    }

    pub fn set_mime_type(&mut self, mime_type: Option<&str>) {
        if !self.mime_type.is_null() {
            unsafe {
                let _ = CString::from_raw(self.mime_type);
            }
        }
        
        self.mime_type = match mime_type {
            Some(mt) => crate::utils::rust_string_to_c(mt),
            None => std::ptr::null_mut(),
        };
    }

    pub fn set_warnings(&mut self, warnings: Vec<String>) {
        if !self.warnings.is_null() {
            unsafe {
                crate::utils::free_c_array(self.warnings, self.warnings_count);
            }
        }

        let (warnings_ptr, warnings_count) = crate::utils::rust_vec_to_c_array(warnings);
        self.warnings = warnings_ptr;
        self.warnings_count = warnings_count;
    }

    pub fn set_supported_features(&mut self, features: Vec<String>) {
        if !self.supported_features.is_null() {
            unsafe {
                crate::utils::free_c_array(self.supported_features, self.supported_features_count);
            }
        }

        let (features_ptr, features_count) = crate::utils::rust_vec_to_c_array(features);
        self.supported_features = features_ptr;
        self.supported_features_count = features_count;
    }

    pub fn free_memory(&mut self) {
        unsafe {
            if !self.mime_type.is_null() {
                let _ = CString::from_raw(self.mime_type);
                self.mime_type = std::ptr::null_mut();
            }

            if !self.warnings.is_null() {
                crate::utils::free_c_array(self.warnings, self.warnings_count);
                self.warnings = std::ptr::null_mut();
                self.warnings_count = 0;
            }

            if !self.supported_features.is_null() {
                crate::utils::free_c_array(self.supported_features, self.supported_features_count);
                self.supported_features = std::ptr::null_mut();
                self.supported_features_count = 0;
            }
        }
    }
}

// C-style API functions for type management
#[no_mangle]
pub extern "C" fn create_metadata(file_name: *const c_char, file_size: u64) -> *mut WasmDocumentMetadata {
    if file_name.is_null() {
        return std::ptr::null_mut();
    }

    unsafe {
        let file_name_str = CStr::from_ptr(file_name).to_string_lossy();
        let metadata = WasmDocumentMetadata::new(&file_name_str, file_size);
        Box::into_raw(Box::new(metadata))
    }
}

#[no_mangle]
pub extern "C" fn destroy_metadata(metadata: *mut WasmDocumentMetadata) {
    if !metadata.is_null() {
        unsafe {
            let mut metadata_box = Box::from_raw(metadata);
            metadata_box.free_memory();
        }
    }
}

#[no_mangle]
pub extern "C" fn create_stats() -> *mut WasmProcessingStats {
    let stats = WasmProcessingStats::new();
    Box::into_raw(Box::new(stats))
}

#[no_mangle]
pub extern "C" fn destroy_stats(stats: *mut WasmProcessingStats) {
    if !stats.is_null() {
        unsafe {
            let _ = Box::from_raw(stats);
        }
    }
}

#[no_mangle]
pub extern "C" fn create_batch_config() -> *mut WasmBatchConfig {
    let config = WasmBatchConfig::new();
    Box::into_raw(Box::new(config))
}

#[no_mangle]
pub extern "C" fn destroy_batch_config(config: *mut WasmBatchConfig) {
    if !config.is_null() {
        unsafe {
            let _ = Box::from_raw(config);
        }
    }
}

#[no_mangle]
pub extern "C" fn destroy_batch_result(result: *mut WasmBatchResult) {
    if !result.is_null() {
        unsafe {
            let mut result_box = Box::from_raw(result);
            result_box.free_memory();
        }
    }
}

#[no_mangle]
pub extern "C" fn create_validation_result(is_valid: bool) -> *mut WasmValidationResult {
    let result = WasmValidationResult::new(is_valid);
    Box::into_raw(Box::new(result))
}

#[no_mangle]
pub extern "C" fn destroy_validation_result(result: *mut WasmValidationResult) {
    if !result.is_null() {
        unsafe {
            let mut result_box = Box::from_raw(result);
            result_box.free_memory();
        }
    }
}