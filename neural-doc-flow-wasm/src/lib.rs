//! WebAssembly bindings for neural document processing
//! 
//! This crate provides JavaScript-compatible WASM bindings for the neural document
//! processing system, enabling high-performance document processing in web browsers
//! and Node.js environments.

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use js_sys::{Array, Object, Promise, Uint8Array};
use web_sys::{console, Blob, File, ReadableStream};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use futures::StreamExt;

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

/// Initialize the WASM module
#[wasm_bindgen(start)]
pub fn init() {
    utils::set_panic_hook();
    console::log_1(&"Neural Document Flow WASM initialized".into());
}

/// Main processor interface for WASM
#[wasm_bindgen]
pub struct WasmDocumentProcessor {
    inner: Arc<DocumentProcessor>,
    config: ProcessingConfig,
}

/// Configuration options for document processing
#[wasm_bindgen]
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
    /// Output formats to generate
    pub output_formats: Vec<String>,
    /// Custom processing options
    #[wasm_bindgen(skip)]
    pub custom_options: HashMap<String, String>,
}

/// Document processing result
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmProcessingResult {
    /// Processing success status
    pub success: bool,
    /// Processing time in milliseconds
    pub processing_time_ms: u32,
    /// Extracted text content
    pub content: String,
    /// Document metadata
    #[wasm_bindgen(skip)]
    pub metadata: HashMap<String, String>,
    /// Generated outputs
    #[wasm_bindgen(skip)]
    pub outputs: HashMap<String, Vec<u8>>,
    /// Security scan results
    #[wasm_bindgen(skip)]
    pub security_results: Option<SecurityScanResult>,
    /// Processing warnings
    pub warnings: Vec<String>,
}

/// Security scan result structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityScanResult {
    pub is_safe: bool,
    pub threat_level: u8,
    pub detected_threats: Vec<String>,
    pub scan_time_ms: u32,
}

#[wasm_bindgen]
impl WasmDocumentProcessor {
    /// Create a new document processor with default configuration
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<WasmDocumentProcessor, JsValue> {
        let config = ProcessingConfig::default();
        let processor = DocumentProcessor::new(config.clone())
            .map_err(|e| JsValue::from_str(&format!("Failed to create processor: {}", e)))?;

        Ok(WasmDocumentProcessor {
            inner: Arc::new(processor),
            config,
        })
    }

    /// Create a processor with custom configuration
    #[wasm_bindgen]
    pub fn with_config(config: &WasmProcessingConfig) -> Result<WasmDocumentProcessor, JsValue> {
        let processing_config = config.to_core_config()
            .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
        
        let processor = DocumentProcessor::new(processing_config.clone())
            .map_err(|e| JsValue::from_str(&format!("Failed to create processor: {}", e)))?;

        Ok(WasmDocumentProcessor {
            inner: Arc::new(processor),
            config: processing_config,
        })
    }

    /// Process a single document from bytes
    #[wasm_bindgen]
    pub async fn process_bytes(&self, data: &[u8], filename: Option<String>) -> Result<JsValue, JsValue> {
        let document = Document::from_bytes(data.to_vec(), filename.unwrap_or_default())
            .map_err(|e| JsValue::from_str(&format!("Failed to create document: {}", e)))?;

        let result = self.inner.process(document).await
            .map_err(|e| JsValue::from_str(&format!("Processing failed: {}", e)))?;

        let wasm_result = WasmProcessingResult::from_core_result(result);
        
        // Convert to JsValue for return
        serde_wasm_bindgen::to_value(&wasm_result)
            .map_err(|e| JsValue::from_str(&format!("Serialization failed: {}", e)))
    }

    /// Process a File object from the browser
    #[wasm_bindgen]
    pub async fn process_file(&self, file: &File) -> Result<JsValue, JsValue> {
        let filename = file.name();
        let array_buffer = JsFuture::from(file.array_buffer()).await?;
        let uint8_array = Uint8Array::new(&array_buffer);
        let bytes = uint8_array.to_vec();

        self.process_bytes(&bytes, Some(filename)).await
    }

    /// Process a Blob object
    #[wasm_bindgen]
    pub async fn process_blob(&self, blob: &Blob) -> Result<JsValue, JsValue> {
        let array_buffer = JsFuture::from(blob.array_buffer()).await?;
        let uint8_array = Uint8Array::new(&array_buffer);
        let bytes = uint8_array.to_vec();

        self.process_bytes(&bytes, None).await
    }

    /// Process multiple documents in batch
    #[wasm_bindgen]
    pub async fn process_batch(&self, files: &Array) -> Result<JsValue, JsValue> {
        let mut results = Vec::new();
        
        for i in 0..files.length() {
            let file_val = files.get(i);
            
            // Try to convert to File first, then fall back to other types
            if let Ok(file) = file_val.dyn_into::<File>() {
                match self.process_file(&file).await {
                    Ok(result) => results.push(result),
                    Err(e) => {
                        console::error_1(&format!("Failed to process file {}: {:?}", i, e).into());
                        results.push(JsValue::NULL);
                    }
                }
            } else if let Ok(blob) = file_val.dyn_into::<Blob>() {
                match self.process_blob(&blob).await {
                    Ok(result) => results.push(result),
                    Err(e) => {
                        console::error_1(&format!("Failed to process blob {}: {:?}", i, e).into());
                        results.push(JsValue::NULL);
                    }
                }
            } else {
                console::error_1(&format!("Unsupported file type at index {}", i).into());
                results.push(JsValue::NULL);
            }
        }

        serde_wasm_bindgen::to_value(&results)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize results: {}", e)))
    }

    /// Get processor statistics
    #[wasm_bindgen]
    pub fn get_stats(&self) -> Result<JsValue, JsValue> {
        let stats = self.inner.get_statistics()
            .map_err(|e| JsValue::from_str(&format!("Failed to get stats: {}", e)))?;

        serde_wasm_bindgen::to_value(&stats)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize stats: {}", e)))
    }

    /// Reset processor state
    #[wasm_bindgen]
    pub fn reset(&mut self) -> Result<(), JsValue> {
        self.inner.reset()
            .map_err(|e| JsValue::from_str(&format!("Failed to reset processor: {}", e)))
    }
}

/// Streaming processor for large documents
#[wasm_bindgen]
pub struct WasmStreamingProcessor {
    inner: StreamingDocumentProcessor,
}

#[wasm_bindgen]
impl WasmStreamingProcessor {
    /// Create a new streaming processor
    #[wasm_bindgen(constructor)]
    pub fn new(config: &WasmProcessingConfig) -> Result<WasmStreamingProcessor, JsValue> {
        let core_config = config.to_core_config()
            .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
            
        let processor = StreamingDocumentProcessor::new(core_config)
            .map_err(|e| JsValue::from_str(&format!("Failed to create streaming processor: {}", e)))?;

        Ok(WasmStreamingProcessor { inner: processor })
    }

    /// Process a ReadableStream
    #[wasm_bindgen]
    pub async fn process_stream(&mut self, stream: &ReadableStream) -> Result<JsValue, JsValue> {
        // Convert ReadableStream to Rust stream
        let js_stream = wasm_streams::ReadableStream::from_raw(stream.clone());
        let rust_stream = js_stream.into_stream();

        let mut chunks = Vec::new();
        let mut stream = rust_stream.map(|chunk| {
            chunk.map(|js_val| {
                let uint8_array = js_val.dyn_into::<Uint8Array>().unwrap();
                uint8_array.to_vec()
            }).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{:?}", e)))
        });

        // Collect all chunks
        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(chunk) => chunks.extend_from_slice(&chunk),
                Err(e) => return Err(JsValue::from_str(&format!("Stream error: {}", e))),
            }
        }

        // Process the accumulated data
        let result = self.inner.process_bytes(&chunks).await
            .map_err(|e| JsValue::from_str(&format!("Processing failed: {}", e)))?;

        serde_wasm_bindgen::to_value(&result)
            .map_err(|e| JsValue::from_str(&format!("Serialization failed: {}", e)))
    }
}

/// Utility functions for WASM integration
#[wasm_bindgen]
pub struct WasmUtils;

#[wasm_bindgen]
impl WasmUtils {
    /// Get version information
    #[wasm_bindgen]
    pub fn version() -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }

    /// Check if neural processing is available
    #[wasm_bindgen]
    pub fn neural_available() -> bool {
        neural_doc_flow_processors::neural_engine::is_available()
    }

    /// Get supported input formats
    #[wasm_bindgen]
    pub fn supported_formats() -> Vec<String> {
        neural_doc_flow_sources::get_supported_formats()
    }

    /// Get supported output formats
    #[wasm_bindgen]
    pub fn output_formats() -> Vec<String> {
        neural_doc_flow_outputs::get_supported_formats()
    }

    /// Validate a file before processing
    #[wasm_bindgen]
    pub fn validate_file(data: &[u8], filename: String, max_size: u32) -> Result<bool, JsValue> {
        if data.len() > max_size as usize {
            return Ok(false);
        }

        // Check file extension and magic bytes
        let is_valid = neural_doc_flow_sources::validate_file_format(&data, &filename)
            .map_err(|e| JsValue::from_str(&format!("Validation error: {}", e)))?;

        Ok(is_valid)
    }

    /// Estimate processing time for a document
    #[wasm_bindgen]
    pub fn estimate_processing_time(file_size: u32, enable_neural: bool) -> u32 {
        // Base processing time estimation
        let base_time = (file_size / 1024) * 10; // 10ms per KB base
        let neural_multiplier = if enable_neural { 3 } else { 1 };
        
        base_time * neural_multiplier
    }
}

/// Export the default processing configuration
#[wasm_bindgen]
pub fn default_config() -> WasmProcessingConfig {
    WasmProcessingConfig {
        neural_enhancement: true,
        max_file_size: 50 * 1024 * 1024, // 50MB
        timeout_ms: 30000,               // 30 seconds
        security_level: 2,               // Moderate security
        output_formats: vec!["text".to_string(), "json".to_string()],
        custom_options: HashMap::new(),
    }
}

/// Export type definitions for TypeScript
#[wasm_bindgen(typescript_custom_section)]
const TS_APPEND_CONTENT: &'static str = r#"
export interface ProcessingResult {
    success: boolean;
    processing_time_ms: number;
    content: string;
    metadata: Record<string, string>;
    outputs: Record<string, Uint8Array>;
    security_results?: SecurityScanResult;
    warnings: string[];
}

export interface ProcessingConfig {
    neural_enhancement: boolean;
    max_file_size: number;
    timeout_ms: number;
    security_level: number;
    output_formats: string[];
    custom_options: Record<string, string>;
}

export interface SecurityScanResult {
    is_safe: boolean;
    threat_level: number;
    detected_threats: string[];
    scan_time_ms: number;
}

export interface ProcessorStats {
    documents_processed: number;
    total_processing_time_ms: number;
    average_processing_time_ms: number;
    memory_usage_bytes: number;
    neural_operations: number;
}
"#;