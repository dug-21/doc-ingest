//! Type definitions and conversions for WASM bindings

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use neural_doc_flow_core::{ProcessingConfig, Document};
use neural_doc_flow_processors::ProcessingResult;
use crate::{WasmProcessingConfig, WasmProcessingResult, SecurityScanResult};

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
        
        // Set output formats
        for format in &self.output_formats {
            config.add_output_format(format.clone())?;
        }
        
        // Set custom options
        for (key, value) in &self.custom_options {
            config.set_custom_option(key.clone(), value.clone());
        }
        
        Ok(config)
    }

    pub fn from_core_config(config: &ProcessingConfig) -> Self {
        Self {
            neural_enhancement: config.neural_enabled(),
            max_file_size: config.max_file_size() as u32,
            timeout_ms: config.timeout_ms(),
            security_level: config.security_level(),
            output_formats: config.output_formats().clone(),
            custom_options: config.custom_options().clone(),
        }
    }
}

/// Convert core processing result to WASM result
impl WasmProcessingResult {
    pub fn from_core_result(result: ProcessingResult) -> Self {
        let security_results = result.security_results.map(|sr| SecurityScanResult {
            is_safe: sr.is_safe,
            threat_level: sr.threat_level,
            detected_threats: sr.detected_threats,
            scan_time_ms: sr.scan_time_ms,
        });

        Self {
            success: result.success,
            processing_time_ms: result.processing_time_ms,
            content: result.content,
            metadata: result.metadata,
            outputs: result.outputs,
            security_results,
            warnings: result.warnings,
        }
    }

    pub fn to_core_result(&self) -> ProcessingResult {
        let security_results = self.security_results.as_ref().map(|sr| {
            neural_doc_flow_processors::SecurityScanResult {
                is_safe: sr.is_safe,
                threat_level: sr.threat_level,
                detected_threats: sr.detected_threats.clone(),
                scan_time_ms: sr.scan_time_ms,
            }
        });

        ProcessingResult {
            success: self.success,
            processing_time_ms: self.processing_time_ms,
            content: self.content.clone(),
            metadata: self.metadata.clone(),
            outputs: self.outputs.clone(),
            security_results,
            warnings: self.warnings.clone(),
        }
    }
}

/// JavaScript-compatible document metadata
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmDocumentMetadata {
    pub file_name: String,
    pub file_size: u64,
    pub mime_type: Option<String>,
    pub pages: Option<u32>,
    pub word_count: Option<u32>,
    pub character_count: Option<u32>,
    pub language: Option<String>,
    pub created_at: Option<String>,
    pub modified_at: Option<String>,
    pub author: Option<String>,
    pub title: Option<String>,
    #[wasm_bindgen(skip)]
    pub custom_fields: HashMap<String, String>,
}

#[wasm_bindgen]
impl WasmDocumentMetadata {
    #[wasm_bindgen(constructor)]
    pub fn new(file_name: String, file_size: u64) -> Self {
        Self {
            file_name,
            file_size,
            mime_type: None,
            pages: None,
            word_count: None,
            character_count: None,
            language: None,
            created_at: None,
            modified_at: None,
            author: None,
            title: None,
            custom_fields: HashMap::new(),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn file_name(&self) -> String {
        self.file_name.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn file_size(&self) -> u64 {
        self.file_size
    }

    #[wasm_bindgen(getter)]
    pub fn mime_type(&self) -> Option<String> {
        self.mime_type.clone()
    }

    #[wasm_bindgen(setter)]
    pub fn set_mime_type(&mut self, mime_type: Option<String>) {
        self.mime_type = mime_type;
    }

    #[wasm_bindgen(getter)]
    pub fn pages(&self) -> Option<u32> {
        self.pages
    }

    #[wasm_bindgen(setter)]
    pub fn set_pages(&mut self, pages: Option<u32>) {
        self.pages = pages;
    }

    /// Get custom field value
    #[wasm_bindgen]
    pub fn get_custom_field(&self, key: &str) -> Option<String> {
        self.custom_fields.get(key).cloned()
    }

    /// Set custom field value
    #[wasm_bindgen]
    pub fn set_custom_field(&mut self, key: String, value: String) {
        self.custom_fields.insert(key, value);
    }

    /// Get all custom field keys
    #[wasm_bindgen]
    pub fn custom_field_keys(&self) -> Vec<String> {
        self.custom_fields.keys().cloned().collect()
    }
}

/// Processing statistics for monitoring
#[wasm_bindgen]
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

#[wasm_bindgen]
impl WasmProcessingStats {
    #[wasm_bindgen(constructor)]
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

    #[wasm_bindgen(getter)]
    pub fn documents_processed(&self) -> u64 {
        self.documents_processed
    }

    #[wasm_bindgen(getter)]
    pub fn total_processing_time_ms(&self) -> u64 {
        self.total_processing_time_ms
    }

    #[wasm_bindgen(getter)]
    pub fn average_processing_time_ms(&self) -> f64 {
        self.average_processing_time_ms
    }

    #[wasm_bindgen(getter)]
    pub fn memory_usage_bytes(&self) -> u64 {
        self.memory_usage_bytes
    }

    #[wasm_bindgen(getter)]
    pub fn cache_hit_ratio(&self) -> f64 {
        let total_cache_operations = self.cache_hits + self.cache_misses;
        if total_cache_operations == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total_cache_operations as f64
        }
    }

    #[wasm_bindgen(getter)]
    pub fn error_rate(&self) -> f64 {
        if self.documents_processed == 0 {
            0.0
        } else {
            self.errors_encountered as f64 / self.documents_processed as f64
        }
    }
}

/// Batch processing configuration
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmBatchConfig {
    pub max_concurrent: u32,
    pub chunk_size: u32,
    pub progress_reporting: bool,
    pub fail_fast: bool,
    pub preserve_order: bool,
}

#[wasm_bindgen]
impl WasmBatchConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            max_concurrent: 4,
            chunk_size: 10,
            progress_reporting: true,
            fail_fast: false,
            preserve_order: true,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn max_concurrent(&self) -> u32 {
        self.max_concurrent
    }

    #[wasm_bindgen(setter)]
    pub fn set_max_concurrent(&mut self, value: u32) {
        self.max_concurrent = value.max(1).min(16); // Limit to reasonable range
    }
}

/// Batch processing result
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmBatchResult {
    pub total_documents: u32,
    pub successful_documents: u32,
    pub failed_documents: u32,
    pub total_processing_time_ms: u64,
    pub average_processing_time_ms: f64,
    #[wasm_bindgen(skip)]
    pub results: Vec<WasmProcessingResult>,
    #[wasm_bindgen(skip)]
    pub errors: Vec<String>,
}

#[wasm_bindgen]
impl WasmBatchResult {
    #[wasm_bindgen(getter)]
    pub fn total_documents(&self) -> u32 {
        self.total_documents
    }

    #[wasm_bindgen(getter)]
    pub fn successful_documents(&self) -> u32 {
        self.successful_documents
    }

    #[wasm_bindgen(getter)]
    pub fn failed_documents(&self) -> u32 {
        self.failed_documents
    }

    #[wasm_bindgen(getter)]
    pub fn success_rate(&self) -> f64 {
        if self.total_documents == 0 {
            0.0
        } else {
            self.successful_documents as f64 / self.total_documents as f64
        }
    }

    /// Get result for a specific document index
    #[wasm_bindgen]
    pub fn get_result(&self, index: u32) -> Result<JsValue, JsValue> {
        if let Some(result) = self.results.get(index as usize) {
            serde_wasm_bindgen::to_value(result)
                .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
        } else {
            Err(JsValue::from_str("Index out of bounds"))
        }
    }

    /// Get error for a specific document index
    #[wasm_bindgen]
    pub fn get_error(&self, index: u32) -> Option<String> {
        self.errors.get(index as usize).cloned()
    }

    /// Get all successful results
    #[wasm_bindgen]
    pub fn get_successful_results(&self) -> Result<JsValue, JsValue> {
        let successful: Vec<_> = self.results.iter()
            .filter(|r| r.success)
            .collect();
            
        serde_wasm_bindgen::to_value(&successful)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }
}

/// File validation result
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmValidationResult {
    pub is_valid: bool,
    pub mime_type: Option<String>,
    pub estimated_size: u64,
    pub estimated_processing_time_ms: u32,
    pub security_risk_level: u8,
    pub warnings: Vec<String>,
    pub supported_features: Vec<String>,
}

#[wasm_bindgen]
impl WasmValidationResult {
    #[wasm_bindgen(getter)]
    pub fn is_valid(&self) -> bool {
        self.is_valid
    }

    #[wasm_bindgen(getter)]
    pub fn mime_type(&self) -> Option<String> {
        self.mime_type.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn estimated_size(&self) -> u64 {
        self.estimated_size
    }

    #[wasm_bindgen(getter)]
    pub fn security_risk_level(&self) -> u8 {
        self.security_risk_level
    }

    #[wasm_bindgen]
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }

    #[wasm_bindgen]
    pub fn warning_count(&self) -> u32 {
        self.warnings.len() as u32
    }

    #[wasm_bindgen]
    pub fn get_warning(&self, index: u32) -> Option<String> {
        self.warnings.get(index as usize).cloned()
    }

    #[wasm_bindgen]
    pub fn feature_count(&self) -> u32 {
        self.supported_features.len() as u32
    }

    #[wasm_bindgen]
    pub fn get_feature(&self, index: u32) -> Option<String> {
        self.supported_features.get(index as usize).cloned()
    }
}

/// Utility functions for type conversion
pub fn js_array_to_vec<T>(array: &js_sys::Array) -> Result<Vec<T>, JsValue>
where
    T: wasm_bindgen::JsCast + Clone,
{
    let mut result = Vec::new();
    for i in 0..array.length() {
        let item = array.get(i);
        if let Ok(typed_item) = item.dyn_into::<T>() {
            result.push(typed_item);
        } else {
            return Err(JsValue::from_str(&format!("Invalid item type at index {}", i)));
        }
    }
    Ok(result)
}

pub fn vec_to_js_array<T>(vec: &[T]) -> js_sys::Array
where
    T: wasm_bindgen::JsCast + Clone,
{
    let array = js_sys::Array::new();
    for item in vec {
        array.push(&JsValue::from(item.clone()));
    }
    array
}