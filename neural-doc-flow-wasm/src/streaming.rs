//! Streaming document processor for large files

use wasm_bindgen::prelude::*;
use futures::{Stream, StreamExt};
use std::pin::Pin;
use std::task::{Context, Poll};
use serde::{Deserialize, Serialize};

use neural_doc_flow_core::{ProcessingConfig, Document};
use neural_doc_flow_processors::ProcessingResult;
use crate::{WasmProcessingResult, SecurityScanResult};
use crate::utils::PerformanceTimer;

/// Streaming document processor for handling large files efficiently
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
        if data.len() > self.chunk_size * 10 {
            self.process_large_file(data).await
        } else {
            self.process_small_file(data).await
        }
        .map(|mut result| {
            result.processing_time_ms = timer.finish() as u32;
            result
        })
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
        let mut total_metadata = std::collections::HashMap::new();
        let mut all_warnings = Vec::new();
        let start_time = std::time::Instant::now();

        // Process in chunks
        for (i, chunk) in data.chunks(self.chunk_size).enumerate() {
            crate::console_info!("Processing chunk {} of size {}", i, chunk.len());
            
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
                crate::console_warn!("Memory pressure detected, forcing garbage collection");
                // In a real implementation, we might need to flush intermediate results
            }
        }

        // Combine all content
        let combined_content = content_parts.join("\n");
        
        // Create final result
        Ok(WasmProcessingResult {
            success: true,
            processing_time_ms: start_time.elapsed().as_millis() as u32,
            content: combined_content,
            metadata: total_metadata,
            outputs: std::collections::HashMap::new(), // Would be populated in real implementation
            security_results: None, // Would run security scan on combined result
            warnings: all_warnings,
        })
    }

    fn estimate_memory_usage(&self) -> usize {
        // Estimate current memory usage
        // In WASM, we can check memory.size() but this is a simplified estimation
        crate::utils::MemoryTracker::current_usage().unwrap_or(0) as usize
    }
}

/// Stream adapter for processing readable streams
pub struct DocumentStream {
    stream: Pin<Box<dyn Stream<Item = Result<Vec<u8>, JsValue>> + Send>>,
    buffer: Vec<u8>,
    chunk_size: usize,
}

impl DocumentStream {
    pub fn new<S>(stream: S, chunk_size: usize) -> Self 
    where 
        S: Stream<Item = Result<Vec<u8>, JsValue>> + Send + 'static
    {
        Self {
            stream: Box::pin(stream),
            buffer: Vec::new(),
            chunk_size,
        }
    }

    pub async fn collect_all(&mut self) -> Result<Vec<u8>, JsValue> {
        let mut all_data = Vec::new();
        
        while let Some(chunk_result) = self.stream.next().await {
            match chunk_result {
                Ok(chunk) => all_data.extend_from_slice(&chunk),
                Err(e) => return Err(e),
            }
        }
        
        Ok(all_data)
    }

    pub async fn process_streaming<F, Fut>(&mut self, mut processor: F) -> Result<Vec<WasmProcessingResult>, JsValue>
    where
        F: FnMut(Vec<u8>) -> Fut + Send,
        Fut: std::future::Future<Output = Result<WasmProcessingResult, JsValue>> + Send,
    {
        let mut results = Vec::new();
        
        while let Some(chunk_result) = self.stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    self.buffer.extend_from_slice(&chunk);
                    
                    // Process when buffer reaches chunk size
                    if self.buffer.len() >= self.chunk_size {
                        let data_to_process = self.buffer.drain(..self.chunk_size).collect();
                        let result = processor(data_to_process).await?;
                        results.push(result);
                    }
                }
                Err(e) => return Err(e),
            }
        }
        
        // Process remaining data in buffer
        if !self.buffer.is_empty() {
            let remaining_data = std::mem::take(&mut self.buffer);
            let result = processor(remaining_data).await?;
            results.push(result);
        }
        
        Ok(results)
    }
}

/// Progress callback for streaming operations
#[wasm_bindgen]
pub struct StreamingProgress {
    total_size: Option<u64>,
    processed_size: u64,
    chunks_processed: u32,
    start_time: std::time::Instant,
}

#[wasm_bindgen]
impl StreamingProgress {
    #[wasm_bindgen(constructor)]
    pub fn new(total_size: Option<u64>) -> Self {
        Self {
            total_size,
            processed_size: 0,
            chunks_processed: 0,
            start_time: std::time::Instant::now(),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn percentage(&self) -> Option<f64> {
        self.total_size.map(|total| {
            if total == 0 {
                100.0
            } else {
                (self.processed_size as f64 / total as f64) * 100.0
            }
        })
    }

    #[wasm_bindgen(getter)]
    pub fn processed_bytes(&self) -> u64 {
        self.processed_size
    }

    #[wasm_bindgen(getter)]
    pub fn chunks_processed(&self) -> u32 {
        self.chunks_processed
    }

    #[wasm_bindgen(getter)]
    pub fn elapsed_ms(&self) -> u32 {
        self.start_time.elapsed().as_millis() as u32
    }

    #[wasm_bindgen(getter)]
    pub fn estimated_remaining_ms(&self) -> Option<u32> {
        if let Some(percentage) = self.percentage() {
            if percentage > 0.0 && percentage < 100.0 {
                let elapsed = self.elapsed_ms() as f64;
                let remaining_percentage = 100.0 - percentage;
                let rate = percentage / elapsed;
                Some((remaining_percentage / rate) as u32)
            } else {
                None
            }
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