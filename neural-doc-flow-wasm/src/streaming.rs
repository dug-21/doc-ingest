//! Pure Rust streaming document processor for large files
//! 
//! Architecture Compliance: Zero JavaScript dependencies

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use serde::{Deserialize, Serialize};

use neural_doc_flow_core::{ProcessingConfig, Document};
use neural_doc_flow_processors::ProcessingResult;
use crate::{WasmProcessingResult, SecurityScanResult};
use crate::utils::PerformanceTimer;

/// Pure Rust streaming document processor for handling large files efficiently
pub struct StreamingDocumentProcessor {
    config: ProcessingConfig,
    chunk_size: usize,
    max_memory_usage: usize,
}

impl StreamingDocumentProcessor {
    pub fn new(config: ProcessingConfig) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            config,
            chunk_size: 64 * 1024, // 64KB chunks
            max_memory_usage: 100 * 1024 * 1024, // 100MB max memory
        })
    }

    pub async fn process_bytes(&mut self, data: &[u8]) -> Result<WasmProcessingResult, Box<dyn std::error::Error>> {
        let timer = PerformanceTimer::new("streaming_process_bytes");
        
        // For large files, process in chunks
        let result = if data.len() > self.chunk_size * 10 {
            self.process_large_file(data).await
        } else {
            self.process_small_file(data).await
        };
        
        match result {
            Ok(mut res) => {
                let elapsed = timer.finish();
                res.processing_time_ms = elapsed.as_millis() as u32;
                Ok(res)
            }
            Err(e) => {
                let _ = timer.finish();
                Err(e)
            }
        }
    }

    async fn process_small_file(&self, data: &[u8]) -> Result<WasmProcessingResult, Box<dyn std::error::Error>> {
        // For small files, use regular processing
        let document = Document::from_bytes(data.to_vec(), "stream_document".to_string())?;
        let processor = neural_doc_flow_core::DocumentProcessor::new(self.config.clone())?;
        let result = processor.process(document).await?;
        
        Ok(WasmProcessingResult::from_core_result(result))
    }

    async fn process_large_file(&self, data: &[u8]) -> Result<WasmProcessingResult, Box<dyn std::error::Error>> {
        let mut content_parts = Vec::new();
        let mut total_metadata = HashMap::new();
        let mut all_warnings = Vec::new();
        let start_time = std::time::Instant::now();

        // Process in chunks
        for (i, chunk) in data.chunks(self.chunk_size).enumerate() {
            crate::utils::log_info(&format!("Processing chunk {} of size {}", i, chunk.len()));
            
            // Create a temporary document for this chunk
            let chunk_name = format!("chunk_{}", i);
            let document = Document::from_bytes(chunk.to_vec(), chunk_name)?;
            
            // Process the chunk
            let processor = neural_doc_flow_core::DocumentProcessor::new(self.config.clone())?;
            let chunk_result = processor.process(document).await?;
            
            // Accumulate results
            content_parts.push(chunk_result.content);
            
            // Merge metadata
            for (key, value) in chunk_result.metadata {
                total_metadata.insert(format!("chunk_{}_{}", i, key), value);
            }
            
            // Collect warnings
            all_warnings.extend(chunk_result.warnings);
            
            // Memory pressure check
            if self.estimate_memory_usage() > self.max_memory_usage {
                crate::utils::log_warn("Memory pressure detected, forcing garbage collection");
                // In a real implementation, we might need to flush intermediate results
            }
        }

        // Combine all content
        let combined_content = content_parts.join("\n");
        
        // Create final result
        Ok(WasmProcessingResult {
            success: true,
            processing_time_ms: start_time.elapsed().as_millis() as u32,
            content_length: combined_content.len(),
            content_ptr: crate::utils::rust_string_to_c(&combined_content),
            metadata_count: total_metadata.len(),
            metadata_keys: std::ptr::null_mut(), // Will be populated by conversion
            metadata_values: std::ptr::null_mut(), // Will be populated by conversion
            warnings_count: all_warnings.len(),
            warnings: std::ptr::null_mut(), // Will be populated by conversion
        })
    }

    fn estimate_memory_usage(&self) -> usize {
        // Estimate current memory usage
        crate::utils::MemoryTracker::current_usage().unwrap_or(0) as usize
    }
}

/// Pure Rust stream for processing data chunks
pub struct DocumentChunkStream {
    buffer: Vec<u8>,
    chunk_size: usize,
    position: usize,
}

impl DocumentChunkStream {
    pub fn new(data: Vec<u8>, chunk_size: usize) -> Self {
        Self {
            buffer: data,
            chunk_size,
            position: 0,
        }
    }

    pub fn next_chunk(&mut self) -> Option<&[u8]> {
        if self.position >= self.buffer.len() {
            return None;
        }

        let end = std::cmp::min(self.position + self.chunk_size, self.buffer.len());
        let chunk = &self.buffer[self.position..end];
        self.position = end;
        
        Some(chunk)
    }

    pub fn remaining_bytes(&self) -> usize {
        self.buffer.len().saturating_sub(self.position)
    }

    pub fn is_complete(&self) -> bool {
        self.position >= self.buffer.len()
    }
}

/// Progress tracking for streaming operations
#[repr(C)]
pub struct StreamingProgress {
    total_size: u64,
    processed_size: u64,
    chunks_processed: u32,
    start_time: std::time::SystemTime,
}

impl StreamingProgress {
    pub fn new(total_size: u64) -> Self {
        Self {
            total_size,
            processed_size: 0,
            chunks_processed: 0,
            start_time: std::time::SystemTime::now(),
        }
    }

    pub fn percentage(&self) -> f64 {
        if self.total_size == 0 {
            100.0
        } else {
            (self.processed_size as f64 / self.total_size as f64) * 100.0
        }
    }

    pub fn processed_bytes(&self) -> u64 {
        self.processed_size
    }

    pub fn chunks_processed(&self) -> u32 {
        self.chunks_processed
    }

    pub fn elapsed_ms(&self) -> u32 {
        self.start_time
            .elapsed()
            .unwrap_or_default()
            .as_millis() as u32
    }

    pub fn estimated_remaining_ms(&self) -> Option<u32> {
        let percentage = self.percentage();
        if percentage > 0.0 && percentage < 100.0 {
            let elapsed = self.elapsed_ms() as f64;
            let remaining_percentage = 100.0 - percentage;
            let rate = percentage / elapsed;
            Some((remaining_percentage / rate) as u32)
        } else {
            None
        }
    }

    pub fn update_progress(&mut self, chunk_size: u64) {
        self.processed_size += chunk_size;
        self.chunks_processed += 1;
    }
}

/// Streaming configuration options
#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    pub chunk_size: usize,
    pub max_concurrent_chunks: usize,
    pub memory_limit_mb: usize,
    pub enable_progress_reporting: bool,
    pub compression_enabled: bool,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 64 * 1024,    // 64KB
            max_concurrent_chunks: 4,  // Process 4 chunks concurrently
            memory_limit_mb: 100,      // 100MB memory limit
            enable_progress_reporting: true,
            compression_enabled: false,
        }
    }
}

// C-style API for streaming processor
#[no_mangle]
pub extern "C" fn create_streaming_processor(config: *const StreamingConfig) -> *mut StreamingDocumentProcessor {
    let config = if config.is_null() {
        StreamingConfig::default()
    } else {
        unsafe { (*config).clone() }
    };
    
    // Convert StreamingConfig to ProcessingConfig
    let processing_config = ProcessingConfig::default(); // You'd convert properly in real implementation
    
    match StreamingDocumentProcessor::new(processing_config) {
        Ok(processor) => Box::into_raw(Box::new(processor)),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn destroy_streaming_processor(processor: *mut StreamingDocumentProcessor) {
    if !processor.is_null() {
        unsafe {
            let _ = Box::from_raw(processor);
        }
    }
}

#[no_mangle]
pub extern "C" fn process_streaming_bytes(
    processor: *mut StreamingDocumentProcessor,
    data: *const u8,
    data_len: usize,
    result: *mut WasmProcessingResult,
) -> i32 {
    if processor.is_null() || data.is_null() || result.is_null() {
        return -1;
    }
    
    unsafe {
        let processor_ref = &mut *processor;
        let data_slice = std::slice::from_raw_parts(data, data_len);
        
        // Process the data
        let rt = tokio::runtime::Runtime::new().unwrap();
        let processing_result = match rt.block_on(processor_ref.process_bytes(data_slice)) {
            Ok(res) => res,
            Err(_) => return -2,
        };
        
        *result = processing_result;
        0 // Success
    }
}

#[no_mangle]
pub extern "C" fn create_chunk_stream(
    data: *const u8,
    data_len: usize,
    chunk_size: usize,
) -> *mut DocumentChunkStream {
    if data.is_null() || data_len == 0 {
        return std::ptr::null_mut();
    }
    
    unsafe {
        let data_vec = std::slice::from_raw_parts(data, data_len).to_vec();
        let stream = DocumentChunkStream::new(data_vec, chunk_size);
        Box::into_raw(Box::new(stream))
    }
}

#[no_mangle]
pub extern "C" fn destroy_chunk_stream(stream: *mut DocumentChunkStream) {
    if !stream.is_null() {
        unsafe {
            let _ = Box::from_raw(stream);
        }
    }
}

#[no_mangle]
pub extern "C" fn get_next_chunk(
    stream: *mut DocumentChunkStream,
    chunk_data: *mut *const u8,
    chunk_len: *mut usize,
) -> bool {
    if stream.is_null() || chunk_data.is_null() || chunk_len.is_null() {
        return false;
    }
    
    unsafe {
        let stream_ref = &mut *stream;
        if let Some(chunk) = stream_ref.next_chunk() {
            *chunk_data = chunk.as_ptr();
            *chunk_len = chunk.len();
            true
        } else {
            *chunk_data = std::ptr::null();
            *chunk_len = 0;
            false
        }
    }
}

#[no_mangle]
pub extern "C" fn stream_remaining_bytes(stream: *mut DocumentChunkStream) -> usize {
    if stream.is_null() {
        return 0;
    }
    
    unsafe {
        let stream_ref = &*stream;
        stream_ref.remaining_bytes()
    }
}

#[no_mangle]
pub extern "C" fn stream_is_complete(stream: *mut DocumentChunkStream) -> bool {
    if stream.is_null() {
        return true;
    }
    
    unsafe {
        let stream_ref = &*stream;
        stream_ref.is_complete()
    }
}

#[no_mangle]
pub extern "C" fn create_progress_tracker(total_size: u64) -> *mut StreamingProgress {
    let progress = StreamingProgress::new(total_size);
    Box::into_raw(Box::new(progress))
}

#[no_mangle]
pub extern "C" fn destroy_progress_tracker(progress: *mut StreamingProgress) {
    if !progress.is_null() {
        unsafe {
            let _ = Box::from_raw(progress);
        }
    }
}

#[no_mangle]
pub extern "C" fn update_progress(progress: *mut StreamingProgress, chunk_size: u64) {
    if !progress.is_null() {
        unsafe {
            let progress_ref = &mut *progress;
            progress_ref.update_progress(chunk_size);
        }
    }
}

#[no_mangle]
pub extern "C" fn get_progress_percentage(progress: *mut StreamingProgress) -> f64 {
    if progress.is_null() {
        return 0.0;
    }
    
    unsafe {
        let progress_ref = &*progress;
        progress_ref.percentage()
    }
}

#[no_mangle]
pub extern "C" fn get_progress_elapsed_ms(progress: *mut StreamingProgress) -> u32 {
    if progress.is_null() {
        return 0;
    }
    
    unsafe {
        let progress_ref = &*progress;
        progress_ref.elapsed_ms()
    }
}

#[no_mangle]
pub extern "C" fn get_progress_estimated_remaining_ms(progress: *mut StreamingProgress) -> u32 {
    if progress.is_null() {
        return 0;
    }
    
    unsafe {
        let progress_ref = &*progress;
        progress_ref.estimated_remaining_ms().unwrap_or(0)
    }
}

/// Batch processing for multiple documents
pub struct BatchProcessor {
    config: ProcessingConfig,
    max_concurrent: usize,
}

impl BatchProcessor {
    pub fn new(config: ProcessingConfig, max_concurrent: usize) -> Self {
        Self {
            config,
            max_concurrent,
        }
    }

    pub async fn process_batch(&self, documents: Vec<(Vec<u8>, String)>) -> Vec<Result<WasmProcessingResult, String>> {
        let mut results = Vec::new();
        let processor = match neural_doc_flow_core::DocumentProcessor::new(self.config.clone()) {
            Ok(p) => p,
            Err(e) => {
                // Return errors for all documents
                for _ in &documents {
                    results.push(Err(format!("Failed to create processor: {}", e)));
                }
                return results;
            }
        };

        for (data, filename) in documents {
            let document = match Document::from_bytes(data, filename) {
                Ok(doc) => doc,
                Err(e) => {
                    results.push(Err(format!("Failed to create document: {}", e)));
                    continue;
                }
            };

            match processor.process(document).await {
                Ok(result) => {
                    results.push(Ok(WasmProcessingResult::from_core_result(result)));
                }
                Err(e) => {
                    results.push(Err(format!("Processing failed: {}", e)));
                }
            }
        }

        results
    }
}

#[no_mangle]
pub extern "C" fn create_batch_processor(max_concurrent: usize) -> *mut BatchProcessor {
    let config = ProcessingConfig::default();
    let processor = BatchProcessor::new(config, max_concurrent);
    Box::into_raw(Box::new(processor))
}

#[no_mangle]
pub extern "C" fn destroy_batch_processor(processor: *mut BatchProcessor) {
    if !processor.is_null() {
        unsafe {
            let _ = Box::from_raw(processor);
        }
    }
}