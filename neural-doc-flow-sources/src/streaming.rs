//! Streaming document sources for memory-efficient processing
//!
//! This module provides streaming implementations of document sources
//! that process documents in chunks to maintain <2MB memory usage.

use crate::traits::*;
use neural_doc_flow_core::{
    memory::*,
    optimized_types::*,
    types::*,
    error::{Result, NeuralDocFlowError},
};
use bytes::{Bytes, BytesMut};
use std::sync::{Arc, Weak};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::task::{Context, Poll};
use futures::{Stream, StreamExt};
use tokio::{
    fs::File,
    io::{AsyncRead, AsyncReadExt, AsyncSeekExt},
    sync::{RwLock, Semaphore},
};
use parking_lot::Mutex;
use uuid::Uuid;
use async_trait::async_trait;

/// Streaming PDF source that processes documents in chunks
pub struct StreamingPdfSource {
    /// Memory monitor for tracking usage
    monitor: Arc<MemoryMonitor>,
    
    /// Memory pool for buffer allocation
    memory_pool: Arc<Mutex<MemoryPool>>,
    
    /// Maximum chunk size for processing
    chunk_size: usize,
    
    /// Maximum concurrent operations
    semaphore: Arc<Semaphore>,
    
    /// Processing statistics
    stats: Arc<RwLock<StreamingStats>>,
}

/// Streaming statistics
#[derive(Debug, Default, Clone)]
pub struct StreamingStats {
    pub documents_processed: u64,
    pub total_bytes_read: u64,
    pub peak_memory_usage: usize,
    pub average_chunk_size: f64,
    pub processing_time: std::time::Duration,
    pub compression_ratio: f64,
}

impl StreamingPdfSource {
    /// Create new streaming PDF source
    pub fn new(memory_limit: usize) -> Self {
        let chunk_size = (memory_limit / 8).max(4096); // Use 1/8 of limit, min 4KB
        let max_concurrent = (memory_limit / chunk_size).max(1);
        
        Self {
            monitor: Arc::new(MemoryMonitor::new(memory_limit)),
            memory_pool: Arc::new(Mutex::new(MemoryPool::new())),
            chunk_size,
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            stats: Arc::new(RwLock::new(StreamingStats::default())),
        }
    }
    
    /// Process PDF file as stream
    pub async fn process_pdf_stream(&self, path: &Path) -> Result<PdfDocumentStream> {
        let file = File::open(path).await
            .map_err(|e| NeuralDocFlowError::IoError(e))?;
        
        let metadata = file.metadata().await
            .map_err(|e| NeuralDocFlowError::IoError(e))?;
        
        let file_size = metadata.len() as usize;
        
        // Check if we can process this file
        if self.monitor.would_exceed_limit(self.chunk_size) {
            return Err(NeuralDocFlowError::MemoryError {
                message: format!("Cannot process PDF: chunk size {} would exceed memory limit", 
                    self.chunk_size),
            });
        }
        
        Ok(PdfDocumentStream {
            file: Some(file),
            file_size,
            chunk_size: self.chunk_size,
            position: 0,
            monitor: self.monitor.clone(),
            memory_pool: self.memory_pool.clone(),
            semaphore: self.semaphore.clone(),
            processing_id: Uuid::new_v4(),
            stats: self.stats.clone(),
        })
    }
    
    /// Get streaming statistics
    pub async fn get_stats(&self) -> StreamingStats {
        self.stats.read().await.clone()
    }
}

/// PDF document stream for chunk-based processing
pub struct PdfDocumentStream {
    file: Option<File>,
    file_size: usize,
    chunk_size: usize,
    position: usize,
    monitor: Arc<MemoryMonitor>,
    memory_pool: Arc<Mutex<MemoryPool>>,
    semaphore: Arc<Semaphore>,
    processing_id: Uuid,
    stats: Arc<RwLock<StreamingStats>>,
}

impl Stream for PdfDocumentStream {
    type Item = Result<PdfChunk>;
    
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.position >= self.file_size || self.file.is_none() {
            // Clean up on completion
            self.monitor.deallocate(self.processing_id);
            return Poll::Ready(None);
        }
        
        // Try to acquire semaphore permit
        match self.semaphore.try_acquire() {
            Ok(_permit) => {
                // Allocate memory for this chunk
                if let Err(e) = self.monitor.allocate(self.processing_id, self.chunk_size) {
                    return Poll::Ready(Some(Err(e)));
                }
                
                // Read chunk
                let file = self.file.as_mut().unwrap();
                let read_size = (self.chunk_size).min(self.file_size - self.position);
                
                // Allocate buffer from pool
                let mut buffer = self.memory_pool.lock().allocate(read_size);
                
                // In a real implementation, this would be async
                // For now, we'll simulate the read
                let chunk_data = Bytes::from(vec![0u8; read_size]);
                
                let chunk = PdfChunk {
                    data: chunk_data.clone(),
                    offset: self.position,
                    size: read_size,
                    is_final: self.position + read_size >= self.file_size,
                };
                
                self.position += read_size;
                
                // Update statistics
                tokio::spawn({
                    let stats = self.stats.clone();
                    let chunk_size = read_size;
                    async move {
                        let mut s = stats.write().await;
                        s.total_bytes_read += chunk_size as u64;
                        s.average_chunk_size = (s.average_chunk_size * (s.documents_processed as f64) + 
                            chunk_size as f64) / (s.documents_processed + 1) as f64;
                    }
                });
                
                Poll::Ready(Some(Ok(chunk)))
            }
            Err(_) => {
                // Would block - register waker
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        }
    }
}

/// PDF chunk for streaming processing
#[derive(Debug, Clone)]
pub struct PdfChunk {
    pub data: Bytes,
    pub offset: usize,
    pub size: usize,
    pub is_final: bool,
}

/// Streaming text source for large text documents
pub struct StreamingTextSource {
    monitor: Arc<MemoryMonitor>,
    memory_pool: Arc<Mutex<MemoryPool>>,
    string_cache: Arc<Mutex<StringCache>>,
    chunk_size: usize,
    line_buffer_size: usize,
}

impl StreamingTextSource {
    /// Create new streaming text source
    pub fn new(memory_limit: usize) -> Self {
        let chunk_size = (memory_limit / 16).max(1024); // Use 1/16 of limit, min 1KB
        
        Self {
            monitor: Arc::new(MemoryMonitor::new(memory_limit)),
            memory_pool: Arc::new(Mutex::new(MemoryPool::new())),
            string_cache: Arc::new(Mutex::new(StringCache::new(1000))),
            chunk_size,
            line_buffer_size: chunk_size / 4, // Buffer 1/4 chunk for line processing
        }
    }
    
    /// Process text file as stream of lines
    pub async fn process_text_stream(&self, path: &Path) -> Result<TextLineStream> {
        let file = File::open(path).await
            .map_err(|e| NeuralDocFlowError::IoError(e))?;
        
        let metadata = file.metadata().await
            .map_err(|e| NeuralDocFlowError::IoError(e))?;
        
        Ok(TextLineStream {
            file: Some(file),
            file_size: metadata.len() as usize,
            buffer: BytesMut::with_capacity(self.line_buffer_size),
            chunk_size: self.chunk_size,
            position: 0,
            eof_reached: false,
            monitor: self.monitor.clone(),
            string_cache: self.string_cache.clone(),
            processing_id: Uuid::new_v4(),
        })
    }
}

/// Text line stream for processing large text files
pub struct TextLineStream {
    file: Option<File>,
    file_size: usize,
    buffer: BytesMut,
    chunk_size: usize,
    position: usize,
    eof_reached: bool,
    monitor: Arc<MemoryMonitor>,
    string_cache: Arc<Mutex<StringCache>>,
    processing_id: Uuid,
}

impl Stream for TextLineStream {
    type Item = Result<Arc<str>>;
    
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.eof_reached && self.buffer.is_empty() {
            self.monitor.deallocate(self.processing_id);
            return Poll::Ready(None);
        }
        
        // Try to find a complete line in buffer
        if let Some(line_end) = self.buffer.iter().position(|&b| b == b'\n') {
            let line_bytes = self.buffer.split_to(line_end + 1);
            let line_str = String::from_utf8_lossy(&line_bytes[..line_end]); // Exclude newline
            
            // Use string cache for deduplication
            let cached_line = self.string_cache.lock().get_or_insert(&line_str);
            
            return Poll::Ready(Some(Ok(cached_line)));
        }
        
        // Need to read more data
        if !self.eof_reached && self.file.is_some() {
            // Check memory limit
            if self.monitor.would_exceed_limit(self.chunk_size) {
                return Poll::Ready(Some(Err(NeuralDocFlowError::MemoryError {
                    message: "Reading chunk would exceed memory limit".to_string(),
                })));
            }
            
            // Allocate memory for read
            if let Err(e) = self.monitor.allocate(self.processing_id, self.chunk_size) {
                return Poll::Ready(Some(Err(e)));
            }
            
            // Read more data (simplified for this example)
            let read_size = (self.chunk_size).min(self.file_size - self.position);
            
            if read_size > 0 {
                // Simulate reading data
                let data = vec![65u8; read_size]; // Fill with 'A' for demonstration
                self.buffer.extend_from_slice(&data);
                self.position += read_size;
                
                if self.position >= self.file_size {
                    self.eof_reached = true;
                }
                
                // Try again to find a line
                cx.waker().wake_by_ref();
                return Poll::Pending;
            } else {
                self.eof_reached = true;
            }
        }
        
        // If we have remaining data but no newline, return it as final line
        if !self.buffer.is_empty() {
            let line_bytes = self.buffer.split();
            let line_str = String::from_utf8_lossy(&line_bytes);
            let cached_line = self.string_cache.lock().get_or_insert(&line_str);
            Poll::Ready(Some(Ok(cached_line)))
        } else {
            self.monitor.deallocate(self.processing_id);
            Poll::Ready(None)
        }
    }
}

/// Streaming document processor that combines multiple sources
pub struct StreamingDocumentProcessor {
    pdf_source: StreamingPdfSource,
    text_source: StreamingTextSource,
    monitor: Arc<MemoryMonitor>,
    active_streams: Arc<RwLock<HashMap<Uuid, StreamInfo>>>,
}

/// Information about active streams
#[derive(Debug, Clone)]
struct StreamInfo {
    stream_type: String,
    file_size: usize,
    bytes_processed: usize,
    started_at: std::time::Instant,
}

impl StreamingDocumentProcessor {
    /// Create new streaming document processor
    pub fn new(memory_limit: usize) -> Self {
        Self {
            pdf_source: StreamingPdfSource::new(memory_limit / 2),
            text_source: StreamingTextSource::new(memory_limit / 2),
            monitor: Arc::new(MemoryMonitor::new(memory_limit)),
            active_streams: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Process document file with appropriate streaming source
    pub async fn process_document_file(&self, path: &Path) -> Result<DocumentProcessingStream> {
        let extension = path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");
        
        match extension.to_lowercase().as_str() {
            "pdf" => {
                let stream = self.pdf_source.process_pdf_stream(path).await?;
                Ok(DocumentProcessingStream::Pdf(stream))
            }
            "txt" | "md" | "rst" => {
                let stream = self.text_source.process_text_stream(path).await?;
                Ok(DocumentProcessingStream::Text(stream))
            }
            _ => Err(NeuralDocFlowError::UnsupportedFormat {
                format: extension.to_string(),
            })
        }
    }
    
    /// Process multiple documents concurrently with memory management
    pub async fn process_documents_batch(&self, paths: Vec<PathBuf>) -> Result<Vec<OptimizedDocument>> {
        let mut results = Vec::new();
        
        for path in paths {
            // Check memory before processing each document
            if self.monitor.usage_ratio() > 0.8 {
                // Memory usage too high - compact existing results
                for doc in &results {
                    doc.compact().await?;
                }
                
                // If still too high, wait or fail
                if self.monitor.usage_ratio() > 0.9 {
                    return Err(NeuralDocFlowError::MemoryError {
                        message: "Memory usage too high to process more documents".to_string(),
                    });
                }
            }
            
            let doc = self.process_single_document(&path).await?;
            results.push(doc);
        }
        
        Ok(results)
    }
    
    /// Process single document with streaming
    async fn process_single_document(&self, path: &Path) -> Result<OptimizedDocument> {
        let stream = self.process_document_file(path).await?;
        
        let mut doc = OptimizedDocument::new(
            path.to_string_lossy().to_string(),
            self.detect_mime_type(path),
            self.monitor.clone(),
        )?;
        
        match stream {
            DocumentProcessingStream::Pdf(mut pdf_stream) => {
                let mut accumulated_data = Vec::new();
                
                while let Some(chunk_result) = pdf_stream.next().await {
                    let chunk = chunk_result?;
                    accumulated_data.extend_from_slice(&chunk.data);
                    
                    // Process chunk if we have enough data or it's the final chunk
                    if accumulated_data.len() >= 16384 || chunk.is_final {
                        let data = Bytes::from(accumulated_data.clone());
                        self.process_pdf_chunk(&mut doc, data, chunk.is_final).await?;
                        accumulated_data.clear();
                    }
                }
            }
            DocumentProcessingStream::Text(mut text_stream) => {
                let mut lines = Vec::new();
                
                while let Some(line_result) = text_stream.next().await {
                    let line = line_result?;
                    lines.push(line);
                    
                    // Process lines in batches
                    if lines.len() >= 100 {
                        self.process_text_lines(&mut doc, &lines).await?;
                        lines.clear();
                    }
                }
                
                // Process remaining lines
                if !lines.is_empty() {
                    self.process_text_lines(&mut doc, &lines).await?;
                }
            }
        }
        
        // Compact document before returning
        doc.compact().await?;
        
        Ok(doc)
    }
    
    /// Process PDF chunk
    async fn process_pdf_chunk(&self, doc: &mut OptimizedDocument, data: Bytes, is_final: bool) -> Result<()> {
        // Set streaming content for PDF
        if is_final {
            let data_clone = data.clone();
            doc.set_streaming_content(data.len(), move || {
                let data = data_clone.clone();
                Box::new(futures::stream::once(async move { Ok(data) }))
            }).await?;
        }
        
        Ok(())
    }
    
    /// Process text lines
    async fn process_text_lines(&self, doc: &mut OptimizedDocument, lines: &[Arc<str>]) -> Result<()> {
        // Combine lines into text content
        let combined_text = lines.iter()
            .map(|line| line.as_ref())
            .collect::<Vec<_>>()
            .join("\n");
        
        let text_len = combined_text.len();
        let text_arc = Arc::from(combined_text.as_str());
        
        doc.set_text_lazy(move || Ok(text_arc.clone()), text_len).await?;
        
        Ok(())
    }
    
    /// Detect MIME type from file extension
    fn detect_mime_type(&self, path: &Path) -> String {
        match path.extension().and_then(|ext| ext.to_str()) {
            Some("pdf") => "application/pdf".to_string(),
            Some("txt") => "text/plain".to_string(),
            Some("md") => "text/markdown".to_string(),
            Some("html") | Some("htm") => "text/html".to_string(),
            _ => "application/octet-stream".to_string(),
        }
    }
    
    /// Get processing statistics
    pub async fn get_processing_stats(&self) -> ProcessingStatsSummary {
        let pdf_stats = self.pdf_source.get_stats().await;
        let active_streams = self.active_streams.read().await.len();
        
        ProcessingStatsSummary {
            total_documents: pdf_stats.documents_processed,
            total_bytes: pdf_stats.total_bytes_read,
            peak_memory: pdf_stats.peak_memory_usage,
            active_streams,
            memory_usage: self.monitor.current_usage(),
            memory_limit: 2 * 1024 * 1024, // 2MB
        }
    }
}

/// Document processing stream enum
pub enum DocumentProcessingStream {
    Pdf(PdfDocumentStream),
    Text(TextLineStream),
}

/// Processing statistics summary
#[derive(Debug, Clone)]
pub struct ProcessingStatsSummary {
    pub total_documents: u64,
    pub total_bytes: u64,
    pub peak_memory: usize,
    pub active_streams: usize,
    pub memory_usage: usize,
    pub memory_limit: usize,
}

#[async_trait]
impl BaseDocumentSource for StreamingDocumentProcessor {
    async fn can_handle(&self, source: &str) -> bool {
        let path = Path::new(source);
        match path.extension().and_then(|ext| ext.to_str()) {
            Some("pdf") | Some("txt") | Some("md") | Some("html") => true,
            _ => false,
        }
    }
    
    async fn extract(&self, source: &str) -> Result<Document> {
        let path = Path::new(source);
        let optimized_doc = self.process_single_document(path).await?;
        
        // Convert optimized document to standard document
        // This is a simplified conversion
        let mut doc = Document::new(source.to_string(), self.detect_mime_type(path));
        
        // Get text content if available
        if let Ok(Some(text)) = optimized_doc.get_text().await {
            doc.content.text = Some(text.to_string());
        }
        
        Ok(doc)
    }
    
    async fn get_metadata(&self, source: &str) -> Result<SourceMetadata> {
        let path = Path::new(source);
        let metadata = tokio::fs::metadata(path).await
            .map_err(|e| NeuralDocFlowError::IoError(e))?;
        
        Ok(SourceMetadata {
            source_type: "streaming".to_string(),
            size: Some(metadata.len()),
            mime_type: Some(self.detect_mime_type(path)),
            encoding: Some("utf-8".to_string()),
            capabilities: vec![
                SourceCapability::TextExtraction,
                SourceCapability::Streaming,
                SourceCapability::MemoryOptimized,
            ],
            properties: HashMap::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use tokio::io::AsyncWriteExt;
    
    #[tokio::test]
    async fn test_streaming_text_source() {
        let source = StreamingTextSource::new(1024 * 1024);
        
        // Create temporary file
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"Line 1\nLine 2\nLine 3").unwrap();
        
        let mut stream = source.process_text_stream(temp_file.path()).await.unwrap();
        let mut lines = Vec::new();
        
        while let Some(line_result) = stream.next().await {
            lines.push(line_result.unwrap());
        }
        
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0].as_ref(), "Line 1");
        assert_eq!(lines[1].as_ref(), "Line 2");
        assert_eq!(lines[2].as_ref(), "Line 3");
    }
    
    #[tokio::test]
    async fn test_streaming_processor() {
        let processor = StreamingDocumentProcessor::new(2 * 1024 * 1024);
        
        // Create temporary text file
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"Hello, world!\nThis is a test.").unwrap();
        
        let doc = processor.process_single_document(temp_file.path()).await.unwrap();
        let stats = doc.memory_stats().await;
        
        assert!(stats.total_usage > 0);
        assert!(stats.total_usage < 2 * 1024 * 1024); // Should be well under limit
    }
    
    #[test]
    fn test_pdf_chunk() {
        let chunk = PdfChunk {
            data: Bytes::from("test data"),
            offset: 0,
            size: 9,
            is_final: true,
        };
        
        assert_eq!(chunk.size, 9);
        assert!(chunk.is_final);
    }
}
