//! Memory-optimized document types for <2MB per document processing
//!
//! This module provides memory-efficient alternatives to the standard document types,
//! using techniques like string interning, lazy loading, and streaming processing.

use crate::memory::*;
use crate::types::*;
use crate::error::{Result, NeuralDocFlowError, ProcessingError};
use bytes::Bytes;
use std::sync::{Arc, Weak};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use parking_lot::Mutex;

/// Serde wrapper for Arc<str> serialization
mod arc_str_serde {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::sync::Arc;
    
    pub fn serialize<S>(arc: &Arc<str>, s: S) -> std::result::Result<S::Ok, S::Error>
    where S: Serializer {
        s.serialize_str(arc)
    }
    
    pub fn deserialize<'de, D>(d: D) -> std::result::Result<Arc<str>, D::Error>
    where D: Deserializer<'de> {
        String::deserialize(d).map(Arc::from)
    }
}

/// Serde wrapper for Option<Arc<str>> serialization
mod option_arc_str_serde {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::sync::Arc;
    
    pub fn serialize<S>(opt: &Option<Arc<str>>, s: S) -> std::result::Result<S::Ok, S::Error>
    where S: Serializer {
        match opt {
            Some(arc) => s.serialize_some(&**arc),
            None => s.serialize_none(),
        }
    }
    
    pub fn deserialize<'de, D>(d: D) -> std::result::Result<Option<Arc<str>>, D::Error>
    where D: Deserializer<'de> {
        let opt_string: Option<String> = Option::deserialize(d)?;
        Ok(opt_string.map(|s| Arc::from(s.as_str())))
    }
}

/// Helper function to serialize Vec<Arc<str>>
fn serialize_arc_str_vec<S>(vec: &Vec<Arc<str>>, s: S) -> std::result::Result<S::Ok, S::Error>
where S: serde::Serializer {
    use serde::ser::SerializeSeq;
    let mut seq = s.serialize_seq(Some(vec.len()))?;
    for arc in vec {
        seq.serialize_element(&**arc)?;
    }
    seq.end()
}

/// Helper function to deserialize Vec<Arc<str>>
fn deserialize_arc_str_vec<'de, D>(d: D) -> std::result::Result<Vec<Arc<str>>, D::Error>
where D: serde::Deserializer<'de> {
    use serde::Deserialize;
    let strings: Vec<String> = Vec::deserialize(d)?;
    Ok(strings.into_iter().map(|s| Arc::from(s.as_str())).collect())
}

/// Helper function to serialize HashMap<Arc<str>, CompactValue>
fn serialize_arc_str_hashmap<S>(
    map: &HashMap<Arc<str>, CompactValue>,
    s: S,
) -> std::result::Result<S::Ok, S::Error>
where S: serde::Serializer {
    use serde::ser::SerializeMap;
    let mut map_ser = s.serialize_map(Some(map.len()))?;
    for (k, v) in map {
        map_ser.serialize_entry(&**k, v)?;
    }
    map_ser.end()
}

/// Helper function to deserialize HashMap<Arc<str>, CompactValue>
fn deserialize_arc_str_hashmap<'de, D>(
    d: D,
) -> std::result::Result<HashMap<Arc<str>, CompactValue>, D::Error>
where D: serde::Deserializer<'de> {
    use serde::Deserialize;
    let map: HashMap<String, CompactValue> = HashMap::deserialize(d)?;
    Ok(map.into_iter().map(|(k, v)| (Arc::from(k.as_str()), v)).collect())
}

/// Memory-optimized document with lazy loading and streaming support
#[derive(Debug, Clone)]
pub struct OptimizedDocument {
    /// Document metadata (always in memory)
    pub id: Uuid,
    pub doc_type: DocumentType,
    pub source_type: DocumentSourceType,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    
    /// Lazy-loaded content
    content: Arc<RwLock<OptimizedDocumentContent>>,
    
    /// Memory monitor for tracking usage
    monitor: Arc<MemoryMonitor>,
}

/// Optimized document content with lazy loading
#[derive(Debug)]
struct OptimizedDocumentContent {
    /// Metadata (small, always loaded)
    metadata: CompactMetadata,
    
    /// Lazy-loaded text content
    text: Option<LazyText>,
    
    /// Streaming binary content
    raw_content: Option<StreamingContent>,
    
    /// Optimized image storage
    images: Vec<OptimizedImage>,
    
    /// Compact table storage
    tables: Vec<CompactTable>,
    
    /// Processing history (limited)
    processing_history: CircularBuffer<ProcessingEvent>,
    
    /// Current memory usage
    memory_usage: usize,
}

/// Compact metadata with string interning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactMetadata {
    /// Interned strings for common values
    #[serde(with = "option_arc_str_serde", skip_serializing_if = "Option::is_none")]
    pub title: Option<Arc<str>>,
    #[serde(with = "arc_str_serde")]
    pub source: Arc<str>,
    #[serde(with = "arc_str_serde")]
    pub mime_type: Arc<str>,
    #[serde(with = "option_arc_str_serde", skip_serializing_if = "Option::is_none")]
    pub language: Option<Arc<str>>,
    
    /// Authors as interned strings
    #[serde(
        serialize_with = "serialize_arc_str_vec",
        deserialize_with = "deserialize_arc_str_vec"
    )]
    pub authors: Vec<Arc<str>>,
    
    /// Only essential custom metadata
    #[serde(
        serialize_with = "serialize_arc_str_hashmap",
        deserialize_with = "deserialize_arc_str_hashmap"
    )]
    pub custom: HashMap<Arc<str>, CompactValue>,
    
    /// File size (no overhead)
    pub size: Option<u64>,
}

/// Compact value type for metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompactValue {
    String(#[serde(with = "arc_str_serde")] Arc<str>),
    Number(f64),
    Boolean(bool),
    Array(Vec<CompactValue>),
}

/// Lazy-loaded text content
struct LazyText {
    /// Text data (loaded on demand)
    data: Option<Arc<str>>,
    
    /// Text loader function
    loader: Option<Box<dyn Fn() -> Result<Arc<str>> + Send + Sync>>,
    
    /// Text length (cached)
    length: usize,
    
    /// Whether text is currently loaded
    loaded: bool,
}

/// Streaming content for large binary data
struct StreamingContent {
    /// Content size
    size: usize,
    
    /// Stream factory
    stream_factory: Box<dyn Fn() -> Box<dyn futures::Stream<Item = Result<Bytes>> + Unpin + Send> + Send + Sync>,
    
    /// Current chunk cache (limited size)
    chunk_cache: HashMap<usize, Bytes>,
    
    /// Max cached chunks
    max_cached_chunks: usize,
}

/// Optimized image storage
#[derive(Debug, Clone)]
pub struct OptimizedImage {
    /// Image metadata
    pub id: Arc<str>,
    pub format: Arc<str>,
    pub width: u32,
    pub height: u32,
    pub caption: Option<Arc<str>>,
    
    /// Image content (lazy-loaded)
    content: ImageContentRef,
}

/// Reference to image content (lazy-loaded)
#[derive(Clone)]
enum ImageContentRef {
    /// Inline small images (< 1KB)
    Inline(Bytes),
    
    /// Reference to external file
    FileRef(Arc<str>),
    
    /// Base64 reference (loaded on demand)
    Base64Ref {
        data: Option<Bytes>,
        loader: Arc<dyn Fn() -> Result<Bytes> + Send + Sync>,
    },
    
    /// Weak reference to shared data
    SharedRef(Weak<Bytes>),
}

// Manual Debug implementations for types with function pointers
impl std::fmt::Debug for LazyText {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazyText")
            .field("data", &self.data)
            .field("loader", &self.loader.as_ref().map(|_| "<function>"))
            .field("length", &self.length)
            .field("loaded", &self.loaded)
            .finish()
    }
}

impl std::fmt::Debug for StreamingContent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingContent")
            .field("size", &self.size)
            .field("stream_factory", &"<function>")
            .field("chunk_cache", &self.chunk_cache)
            .field("max_cached_chunks", &self.max_cached_chunks)
            .finish()
    }
}

impl std::fmt::Debug for ImageContentRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Inline(bytes) => f.debug_tuple("Inline").field(bytes).finish(),
            Self::FileRef(path) => f.debug_tuple("FileRef").field(path).finish(),
            Self::Base64Ref { data, .. } => f.debug_struct("Base64Ref")
                .field("data", data)
                .field("loader", &"<function>")
                .finish(),
            Self::SharedRef(weak) => f.debug_tuple("SharedRef").field(weak).finish(),
        }
    }
}

/// Compact table storage
#[derive(Debug, Clone)]
pub struct CompactTable {
    /// Table metadata
    pub id: Arc<str>,
    pub caption: Option<Arc<str>>,
    
    /// Headers as interned strings
    pub headers: Vec<Arc<str>>,
    
    /// Row data (compressed if large)
    pub rows: CompactTableData,
    
    /// Table dimensions
    pub row_count: usize,
    pub col_count: usize,
}

/// Compact table data storage
#[derive(Debug, Clone)]
pub enum CompactTableData {
    /// Small tables stored directly
    Direct(Vec<Vec<Arc<str>>>),
    
    /// Large tables stored compressed
    Compressed {
        data: Bytes,
        original_size: usize,
    },
    
    /// Reference to external storage
    External(Arc<str>),
}

/// Circular buffer for processing history
#[derive(Debug)]
struct CircularBuffer<T> {
    buffer: Vec<Option<T>>,
    capacity: usize,
    start: usize,
    len: usize,
}

impl<T> CircularBuffer<T> {
    fn new(capacity: usize) -> Self {
        Self {
            buffer: (0..capacity).map(|_| None).collect(),
            capacity,
            start: 0,
            len: 0,
        }
    }
    
    fn push(&mut self, item: T) {
        let index = (self.start + self.len) % self.capacity;
        self.buffer[index] = Some(item);
        
        if self.len < self.capacity {
            self.len += 1;
        } else {
            self.start = (self.start + 1) % self.capacity;
        }
    }
    
    fn iter(&self) -> impl Iterator<Item = &T> {
        (0..self.len)
            .map(move |i| {
                let index = (self.start + i) % self.capacity;
                self.buffer[index].as_ref().unwrap()
            })
    }
    
    fn len(&self) -> usize {
        self.len
    }
    
    fn clear(&mut self) {
        for item in &mut self.buffer {
            *item = None;
        }
        self.start = 0;
        self.len = 0;
    }
}

impl OptimizedDocument {
    /// Create new optimized document
    pub fn new(source: String, mime_type: String, monitor: Arc<MemoryMonitor>) -> Result<Self> {
        let id = Uuid::new_v4();
        let now = Utc::now();
        
        // Register initial allocation
        monitor.allocate(id, std::mem::size_of::<Self>())?;
        
        let doc_type = match mime_type.as_str() {
            "application/pdf" => DocumentType::Pdf,
            "text/plain" => DocumentType::Text,
            "text/html" => DocumentType::Html,
            "text/markdown" => DocumentType::Markdown,
            "application/json" => DocumentType::Json,
            "application/xml" | "text/xml" => DocumentType::Xml,
            t if t.starts_with("image/") => DocumentType::Image,
            _ => DocumentType::Unknown,
        };
        
        // Create string cache for this document
        let string_cache = Arc::new(Mutex::new(StringCache::new(100)));
        
        let metadata = CompactMetadata {
            title: None,
            source: string_cache.lock().get_or_insert(&source),
            mime_type: string_cache.lock().get_or_insert(&mime_type),
            language: None,
            authors: Vec::new(),
            custom: HashMap::new(),
            size: None,
        };
        
        let content = OptimizedDocumentContent {
            metadata,
            text: None,
            raw_content: None,
            images: Vec::new(),
            tables: Vec::new(),
            processing_history: CircularBuffer::new(10), // Limit history to 10 events
            memory_usage: std::mem::size_of::<OptimizedDocumentContent>(),
        };
        
        Ok(Self {
            id,
            doc_type,
            source_type: DocumentSourceType::File,
            created_at: now,
            updated_at: now,
            content: Arc::new(RwLock::new(content)),
            monitor,
        })
    }
    
    /// Set text content with lazy loading
    pub async fn set_text_lazy<F>(&self, loader: F, length: usize) -> Result<()>
    where
        F: Fn() -> Result<Arc<str>> + Send + Sync + 'static,
    {
        let estimated_size = length;
        
        if self.monitor.would_exceed_limit(estimated_size) {
            return Err(NeuralDocFlowError::MemoryError {
                message: "Text content would exceed memory limit".to_string(),
            });
        }
        
        let mut content = self.content.write().await;
        content.text = Some(LazyText {
            data: None,
            loader: Some(Box::new(loader)),
            length,
            loaded: false,
        });
        
        content.memory_usage += estimated_size;
        Ok(())
    }
    
    /// Get text content (loads if needed)
    pub async fn get_text(&self) -> Result<Option<Arc<str>>> {
        let mut content = self.content.write().await;
        
        if let Some(ref mut lazy_text) = content.text {
            if !lazy_text.loaded {
                if let Some(ref loader) = lazy_text.loader {
                    lazy_text.data = Some(loader()?);
                    lazy_text.loaded = true;
                    // Remove loader to free memory
                    lazy_text.loader = None;
                }
            }
            
            Ok(lazy_text.data.clone())
        } else {
            Ok(None)
        }
    }
    
    /// Set streaming content
    pub async fn set_streaming_content<F>(&self, size: usize, stream_factory: F) -> Result<()>
    where
        F: Fn() -> Box<dyn futures::Stream<Item = Result<Bytes>> + Unpin + Send> + Send + Sync + 'static,
    {
        // Only allocate memory for chunk cache, not full content
        let cache_size = (size / 1024).min(64 * 1024); // Max 64KB cache
        
        if self.monitor.would_exceed_limit(cache_size) {
            return Err(NeuralDocFlowError::MemoryError {
                message: "Streaming content cache would exceed memory limit".to_string(),
            });
        }
        
        let mut content = self.content.write().await;
        content.raw_content = Some(StreamingContent {
            size,
            stream_factory: Box::new(stream_factory),
            chunk_cache: HashMap::new(),
            max_cached_chunks: cache_size / 1024, // 1KB chunks
        });
        
        content.memory_usage += cache_size;
        Ok(())
    }
    
    /// Add optimized image
    pub async fn add_image(&self, image: OptimizedImage) -> Result<()> {
        let image_size = image.estimated_size();
        
        if self.monitor.would_exceed_limit(image_size) {
            return Err(NeuralDocFlowError::MemoryError {
                message: "Image would exceed memory limit".to_string(),
            });
        }
        
        let mut content = self.content.write().await;
        content.images.push(image);
        content.memory_usage += image_size;
        
        Ok(())
    }
    
    /// Add compact table
    pub async fn add_table(&self, table: CompactTable) -> Result<()> {
        let table_size = table.estimated_size();
        
        if self.monitor.would_exceed_limit(table_size) {
            return Err(NeuralDocFlowError::MemoryError {
                message: "Table would exceed memory limit".to_string(),
            });
        }
        
        let mut content = self.content.write().await;
        content.tables.push(table);
        content.memory_usage += table_size;
        
        Ok(())
    }
    
    /// Add processing event (with circular buffer)
    pub async fn add_processing_event(&self, event: ProcessingEvent) {
        let mut content = self.content.write().await;
        content.processing_history.push(event);
    }
    
    /// Get current memory usage
    pub async fn memory_usage(&self) -> usize {
        let content = self.content.read().await;
        content.memory_usage
    }
    
    /// Get memory usage statistics
    pub async fn memory_stats(&self) -> MemoryStats {
        let content = self.content.read().await;
        
        MemoryStats {
            total_usage: content.memory_usage,
            text_size: content.text.as_ref().map(|t| t.length).unwrap_or(0),
            image_count: content.images.len(),
            table_count: content.tables.len(),
            history_count: content.processing_history.len(),
            raw_content_size: content.raw_content.as_ref().map(|c| c.size).unwrap_or(0),
        }
    }
    
    /// Compact document by removing unused data
    pub async fn compact(&self) -> Result<usize> {
        let mut content = self.content.write().await;
        let initial_usage = content.memory_usage;
        
        // Unload text if it's large and not recently accessed
        if let Some(ref mut lazy_text) = content.text {
            if lazy_text.loaded && lazy_text.length > 10000 {
                lazy_text.data = None;
                lazy_text.loaded = false;
                content.memory_usage -= lazy_text.length;
            }
        }
        
        // Clear chunk cache for streaming content
        if let Some(ref mut streaming) = content.raw_content {
            let cache_size: usize = streaming.chunk_cache.values().map(|b| b.len()).sum();
            streaming.chunk_cache.clear();
            content.memory_usage -= cache_size;
        }
        
        // Compress large tables
        let mut memory_delta: isize = 0;
        for table in &mut content.tables {
            if let CompactTableData::Direct(ref rows) = table.rows {
                if rows.len() > 100 { // Large table
                    // Convert Arc<str> to String for serialization
                    let serializable_rows: Vec<Vec<String>> = rows.iter()
                        .map(|row| row.iter().map(|cell| cell.to_string()).collect())
                        .collect();
                    let serialized = serde_json::to_vec(&serializable_rows)
                        .map_err(|e| NeuralDocFlowError::MemoryError {
                            message: e.to_string(),
                        })?;
                    
                    // Simple compression (in real implementation, use proper compression)
                    let compressed = compress_data(&serialized);
                    
                    if compressed.len() < serialized.len() {
                        let original_size = estimate_table_size(rows);
                        table.rows = CompactTableData::Compressed {
                            data: Bytes::from(compressed),
                            original_size,
                        };
                        
                        memory_delta -= original_size as isize;
                        memory_delta += serialized.len() as isize;
                    }
                }
            }
        }
        
        // Apply memory delta after the loop
        if memory_delta < 0 {
            content.memory_usage = content.memory_usage.saturating_sub((-memory_delta) as usize);
        } else {
            content.memory_usage = content.memory_usage.saturating_add(memory_delta as usize);
        }
        
        let saved = initial_usage.saturating_sub(content.memory_usage);
        Ok(saved)
    }
}

impl OptimizedImage {
    /// Create new optimized image
    pub fn new(
        id: String,
        format: String,
        width: u32,
        height: u32,
        content: ImageContentRef,
    ) -> Self {
        let string_cache = StringCache::new(10);
        let mut cache = string_cache;
        
        Self {
            id: cache.get_or_insert(&id),
            format: cache.get_or_insert(&format),
            width,
            height,
            caption: None,
            content,
        }
    }
    
    /// Estimate memory size
    fn estimated_size(&self) -> usize {
        let base_size = std::mem::size_of::<Self>();
        let content_size = match &self.content {
            ImageContentRef::Inline(bytes) => bytes.len(),
            ImageContentRef::FileRef(_) => 0, // No memory cost for file reference
            ImageContentRef::Base64Ref { data: Some(bytes), .. } => bytes.len(),
            ImageContentRef::Base64Ref { data: None, .. } => 0,
            ImageContentRef::SharedRef(_) => 0, // Shared reference has no cost
        };
        
        base_size + content_size
    }
    
    /// Get image data (loads if needed)
    pub async fn get_data(&self) -> Result<Option<Bytes>> {
        match &self.content {
            ImageContentRef::Inline(bytes) => Ok(Some(bytes.clone())),
            ImageContentRef::FileRef(_path) => {
                // Load from file (implementation would read file)
                Ok(None) // Placeholder
            }
            ImageContentRef::Base64Ref { data, loader } => {
                if let Some(bytes) = data {
                    Ok(Some(bytes.clone()))
                } else {
                    // Load using loader
                    Ok(Some(loader()?))
                }
            }
            ImageContentRef::SharedRef(weak) => {
                if let Some(bytes) = weak.upgrade() {
                    Ok(Some((*bytes).clone()))
                } else {
                    Ok(None) // Reference expired
                }
            }
        }
    }
}

impl CompactTable {
    /// Create new compact table
    pub fn new(id: String, headers: Vec<String>, rows: Vec<Vec<String>>) -> Self {
        let string_cache = StringCache::new(headers.len() * 2);
        let mut cache = string_cache;
        
        let interned_headers: Vec<Arc<str>> = headers
            .iter()
            .map(|h| cache.get_or_insert(h))
            .collect();
        
        let interned_rows: Vec<Vec<Arc<str>>> = rows
            .iter()
            .map(|row| {
                row.iter()
                    .map(|cell| cache.get_or_insert(cell))
                    .collect()
            })
            .collect();
        
        let row_count = interned_rows.len();
        let col_count = interned_headers.len();
        
        Self {
            id: cache.get_or_insert(&id),
            caption: None,
            headers: interned_headers,
            rows: CompactTableData::Direct(interned_rows),
            row_count,
            col_count,
        }
    }
    
    /// Estimate memory size
    fn estimated_size(&self) -> usize {
        let base_size = std::mem::size_of::<Self>();
        let headers_size = self.headers.len() * std::mem::size_of::<Arc<str>>();
        
        let rows_size = match &self.rows {
            CompactTableData::Direct(rows) => estimate_table_size(rows),
            CompactTableData::Compressed { data, .. } => data.len(),
            CompactTableData::External(_) => 0,
        };
        
        base_size + headers_size + rows_size
    }
    
    /// Get table data (decompresses if needed)
    pub async fn get_rows(&self) -> Result<Vec<Vec<Arc<str>>>> {
        match &self.rows {
            CompactTableData::Direct(rows) => Ok(rows.clone()),
            CompactTableData::Compressed { data, .. } => {
                // Decompress data
                let decompressed = decompress_data(data)?;
                let string_rows: Vec<Vec<String>> = serde_json::from_slice(&decompressed)
                    .map_err(|e| NeuralDocFlowError::SerializationError(e))?;
                // Convert String to Arc<str>
                let rows = string_rows.into_iter()
                    .map(|row| row.into_iter().map(|s| Arc::from(s.as_str())).collect())
                    .collect();
                Ok(rows)
            }
            CompactTableData::External(_path) => {
                // Load from external file (placeholder)
                Ok(Vec::new())
            }
        }
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_usage: usize,
    pub text_size: usize,
    pub image_count: usize,
    pub table_count: usize,
    pub history_count: usize,
    pub raw_content_size: usize,
}

/// Optimized document processor that maintains <2MB memory usage
pub struct OptimizedProcessor {
    monitor: Arc<MemoryMonitor>,
    builder: OptimizedDocumentBuilder,
    streaming_processor: StreamingProcessor,
}

impl OptimizedProcessor {
    /// Create new optimized processor
    pub fn new() -> Self {
        let memory_limit = 2 * 1024 * 1024; // 2MB limit
        let monitor = Arc::new(MemoryMonitor::new(memory_limit));
        
        Self {
            monitor: monitor.clone(),
            builder: OptimizedDocumentBuilder::new(memory_limit),
            streaming_processor: StreamingProcessor::new(4096, memory_limit), // 4KB chunks
        }
    }
    
    /// Process document with memory optimization
    pub async fn process_document(&mut self, raw_data: Bytes, mime_type: String) -> Result<OptimizedDocument> {
        // Check if we can process this document
        if self.monitor.would_exceed_limit(raw_data.len()) {
            return Err(NeuralDocFlowError::MemoryError {
                message: format!("Document size {} exceeds memory limit", raw_data.len()),
            });
        }
        
        // Create optimized document
        let mut doc = OptimizedDocument::new(
            "memory://processed".to_string(),
            mime_type.clone(),
            self.monitor.clone(),
        )?;
        
        // Process based on document type
        match mime_type.as_str() {
            "text/plain" => {
                self.process_text_document(&mut doc, raw_data).await?
            }
            "application/pdf" => {
                self.process_pdf_document(&mut doc, raw_data).await?
            }
            _ => {
                // Generic processing with streaming
                self.process_generic_document(&mut doc, raw_data).await?
            }
        }
        
        // Compact document before returning
        doc.compact().await?;
        
        Ok(doc)
    }
    
    async fn process_text_document(&mut self, doc: &mut OptimizedDocument, data: Bytes) -> Result<()> {
        // Convert to string and set as lazy text
        let text = String::from_utf8(data.to_vec())
            .map_err(|e| NeuralDocFlowError::ProcessingError(
                ProcessingError::ProcessorFailed {
                    processor_name: "text_decoder".to_string(),
                    reason: format!("Invalid UTF-8: {}", e),
                }
            ))?;
        
        let text_len = text.len();
        let text_arc: Arc<str> = Arc::from(text.as_str());
        
        doc.set_text_lazy(move || Ok(text_arc.clone()), text_len).await?;
        
        Ok(())
    }
    
    async fn process_pdf_document(&mut self, doc: &mut OptimizedDocument, data: Bytes) -> Result<()> {
        // Set as streaming content for large PDFs
        let data_size = data.len();
        let data_clone = data.clone();
        
        doc.set_streaming_content(data_size, move || {
            let data = data_clone.clone();
            Box::new(futures::stream::once(futures::future::ready(Ok(data))))
        }).await?;
        
        Ok(())
    }
    
    async fn process_generic_document(&mut self, doc: &mut OptimizedDocument, data: Bytes) -> Result<()> {
        // Process as streaming content
        let data_size = data.len();
        let _chunks = self.streaming_processor.process_document_stream(data);
        
        // Store reference to stream factory
        doc.set_streaming_content(data_size, move || {
            Box::new(futures::stream::empty())
        }).await?;
        
        Ok(())
    }
    
    /// Get current memory usage
    pub fn memory_usage(&self) -> usize {
        self.monitor.current_usage()
    }
    
    /// Get memory usage ratio
    pub fn memory_ratio(&self) -> f64 {
        self.monitor.usage_ratio()
    }
}

// Helper functions

fn estimate_table_size(rows: &[Vec<Arc<str>>]) -> usize {
    let mut size = rows.len() * std::mem::size_of::<Vec<Arc<str>>>();
    for row in rows {
        size += row.len() * std::mem::size_of::<Arc<str>>();
        for cell in row {
            size += cell.len();
        }
    }
    size
}

fn compress_data(data: &[u8]) -> Vec<u8> {
    // Placeholder compression - in real implementation use proper compression library
    data.to_vec()
}

fn decompress_data(data: &Bytes) -> Result<Vec<u8>> {
    // Placeholder decompression
    Ok(data.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::runtime::Runtime;
    
    #[test]
    fn test_optimized_document_creation() {
        let rt = Runtime::new().unwrap();
        rt.block_on(async {
            let monitor = Arc::new(MemoryMonitor::new(1024 * 1024));
            let doc = OptimizedDocument::new(
                "test.txt".to_string(),
                "text/plain".to_string(),
                monitor,
            ).unwrap();
            
            assert_eq!(doc.doc_type, DocumentType::Text);
            assert!(doc.memory_usage().await > 0);
        });
    }
    
    #[test]
    fn test_compact_table() {
        let headers = vec!["Name".to_string(), "Age".to_string()];
        let rows = vec![
            vec!["John".to_string(), "30".to_string()],
            vec!["Jane".to_string(), "25".to_string()],
        ];
        
        let table = CompactTable::new("test_table".to_string(), headers, rows);
        assert_eq!(table.row_count, 2);
        assert_eq!(table.col_count, 2);
    }
    
    #[test]
    fn test_circular_buffer() {
        let mut buffer = CircularBuffer::new(3);
        
        buffer.push(1);
        buffer.push(2);
        buffer.push(3);
        assert_eq!(buffer.len(), 3);
        
        buffer.push(4); // Should overwrite first element
        assert_eq!(buffer.len(), 3);
        
        let items: Vec<_> = buffer.iter().cloned().collect();
        assert_eq!(items, vec![2, 3, 4]);
    }
    
    #[test]
    fn test_memory_monitor_limit() {
        let monitor = MemoryMonitor::new(100);
        let id = Uuid::new_v4();
        
        assert!(monitor.allocate(id, 50).is_ok());
        assert!(monitor.allocate(Uuid::new_v4(), 60).is_err()); // Would exceed limit
    }
}
