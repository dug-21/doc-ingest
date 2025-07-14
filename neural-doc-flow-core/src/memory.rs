//! Memory optimization utilities for neural document flow
//!
//! This module provides memory-efficient data structures and operations
//! for processing documents with minimal memory footprint (<2MB per document).

use std::sync::{Arc, Weak};
use std::collections::HashMap;
use parking_lot::{RwLock, Mutex};
use bytes::{Bytes, BytesMut};
use tokio::sync::{Semaphore, OwnedSemaphorePermit};
use std::pin::Pin;
use std::task::{Context, Poll};
use futures::Stream;
use uuid::Uuid;

/// Memory pool for efficient buffer allocation
pub struct MemoryPool {
    /// Pre-allocated buffers of different sizes
    pools: HashMap<usize, Vec<BytesMut>>,
    /// Maximum pool size for each buffer size
    max_pool_size: usize,
    /// Statistics tracking
    stats: Arc<RwLock<PoolStats>>,
}

/// Memory pool statistics
#[derive(Debug, Default)]
pub struct PoolStats {
    pub allocations: u64,
    pub deallocations: u64,
    pub pool_hits: u64,
    pub pool_misses: u64,
    pub current_usage: usize,
    pub peak_usage: usize,
}

impl MemoryPool {
    /// Create a new memory pool
    pub fn new() -> Self {
        Self::with_max_size(100) // 100 buffers per size
    }
    
    /// Create memory pool with specified max size
    pub fn with_max_size(max_pool_size: usize) -> Self {
        let mut pools = HashMap::new();
        
        // Pre-allocate common buffer sizes for document processing
        let common_sizes = vec![1024, 4096, 16384, 65536, 262144, 1048576]; // 1KB to 1MB
        
        for size in common_sizes {
            pools.insert(size, Vec::with_capacity(max_pool_size));
        }
        
        Self {
            pools,
            max_pool_size,
            stats: Arc::new(RwLock::new(PoolStats::default())),
        }
    }
    
    /// Allocate a buffer of specified size
    pub fn allocate(&mut self, size: usize) -> BytesMut {
        let buffer_size = self.round_up_to_next_power_of_2(size.max(1024));
        
        {
            let mut stats = self.stats.write();
            stats.allocations += 1;
        }
        
        if let Some(pool) = self.pools.get_mut(&buffer_size) {
            if let Some(mut buffer) = pool.pop() {
                buffer.clear();
                buffer.reserve(size);
                
                let mut stats = self.stats.write();
                stats.pool_hits += 1;
                stats.current_usage += buffer_size;
                stats.peak_usage = stats.peak_usage.max(stats.current_usage);
                
                return buffer;
            }
        }
        
        // Pool miss - allocate new buffer
        let mut stats = self.stats.write();
        stats.pool_misses += 1;
        stats.current_usage += buffer_size;
        stats.peak_usage = stats.peak_usage.max(stats.current_usage);
        
        BytesMut::with_capacity(buffer_size)
    }
    
    /// Return a buffer to the pool
    pub fn deallocate(&mut self, mut buffer: BytesMut) {
        let buffer_size = buffer.capacity();
        
        {
            let mut stats = self.stats.write();
            stats.deallocations += 1;
            stats.current_usage = stats.current_usage.saturating_sub(buffer_size);
        }
        
        if let Some(pool) = self.pools.get_mut(&buffer_size) {
            if pool.len() < self.max_pool_size {
                buffer.clear();
                pool.push(buffer);
                return;
            }
        }
        
        // Pool is full or size not in pool - just drop the buffer
        drop(buffer);
    }
    
    /// Get memory pool statistics
    pub fn stats(&self) -> PoolStats {
        self.stats.read().clone()
    }
    
    /// Clear all pools and reset statistics
    pub fn clear(&mut self) {
        for pool in self.pools.values_mut() {
            pool.clear();
        }
        *self.stats.write() = PoolStats::default();
    }
    
    fn round_up_to_next_power_of_2(&self, size: usize) -> usize {
        let mut power = 1024; // Start at 1KB minimum
        while power < size {
            power *= 2;
        }
        power.min(1048576) // Cap at 1MB
    }
}

/// Shared reference-counted document data with copy-on-write semantics
#[derive(Debug, Clone)]
pub struct SharedDocumentData {
    inner: Arc<DocumentDataInner>,
}

#[derive(Debug)]
struct DocumentDataInner {
    id: Uuid,
    data: RwLock<Option<Bytes>>,
    metadata: RwLock<HashMap<String, String>>,
    ref_count: parking_lot::Mutex<usize>,
}

impl SharedDocumentData {
    /// Create new shared document data
    pub fn new(id: Uuid) -> Self {
        Self {
            inner: Arc::new(DocumentDataInner {
                id,
                data: RwLock::new(None),
                metadata: RwLock::new(HashMap::new()),
                ref_count: parking_lot::Mutex::new(1),
            }),
        }
    }
    
    /// Get document ID
    pub fn id(&self) -> Uuid {
        self.inner.id
    }
    
    /// Set document data
    pub fn set_data(&self, data: Bytes) {
        *self.inner.data.write() = Some(data);
    }
    
    /// Get document data (zero-copy read)
    pub fn get_data(&self) -> Option<Bytes> {
        self.inner.data.read().clone()
    }
    
    /// Get metadata value
    pub fn get_metadata(&self, key: &str) -> Option<String> {
        self.inner.metadata.read().get(key).cloned()
    }
    
    /// Set metadata value
    pub fn set_metadata(&self, key: String, value: String) {
        self.inner.metadata.write().insert(key, value);
    }
    
    /// Get current reference count
    pub fn ref_count(&self) -> usize {
        *self.inner.ref_count.lock()
    }
    
    /// Create a weak reference to avoid circular dependencies
    pub fn downgrade(&self) -> WeakDocumentData {
        WeakDocumentData {
            inner: Arc::downgrade(&self.inner),
        }
    }
}

/// Weak reference to shared document data
#[derive(Debug, Clone)]
pub struct WeakDocumentData {
    inner: Weak<DocumentDataInner>,
}

impl WeakDocumentData {
    /// Upgrade to strong reference if still valid
    pub fn upgrade(&self) -> Option<SharedDocumentData> {
        self.inner.upgrade().map(|inner| SharedDocumentData { inner })
    }
}

/// Streaming document processor that processes documents in chunks
pub struct StreamingProcessor {
    chunk_size: usize,
    memory_limit: usize,
    current_memory: Arc<Mutex<usize>>,
    semaphore: Arc<Semaphore>,
}

impl StreamingProcessor {
    /// Create new streaming processor
    pub fn new(chunk_size: usize, memory_limit: usize) -> Self {
        let max_concurrent = memory_limit / chunk_size;
        
        Self {
            chunk_size,
            memory_limit,
            current_memory: Arc::new(Mutex::new(0)),
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
        }
    }
    
    /// Process document as stream of chunks
    pub fn process_document_stream(&self, data: Bytes) -> DocumentChunkStream {
        DocumentChunkStream::new(data, self.chunk_size, self.semaphore.clone())
    }
    
    /// Get current memory usage
    pub fn current_memory_usage(&self) -> usize {
        *self.current_memory.lock()
    }
    
    /// Check if memory limit would be exceeded
    pub fn can_allocate(&self, size: usize) -> bool {
        let current = *self.current_memory.lock();
        current + size <= self.memory_limit
    }
}

/// Stream of document chunks for memory-efficient processing
pub struct DocumentChunkStream {
    data: Bytes,
    chunk_size: usize,
    position: usize,
    semaphore: Arc<Semaphore>,
    permit: Option<OwnedSemaphorePermit>,
}

impl DocumentChunkStream {
    fn new(data: Bytes, chunk_size: usize, semaphore: Arc<Semaphore>) -> Self {
        Self {
            data,
            chunk_size,
            position: 0,
            semaphore,
            permit: None,
        }
    }
}

impl Stream for DocumentChunkStream {
    type Item = Result<Bytes, crate::error::NeuralDocFlowError>;
    
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Acquire semaphore permit if we don't have one
        if self.permit.is_none() {
            match self.semaphore.clone().try_acquire_owned() {
                Ok(permit) => self.permit = Some(permit),
                Err(_) => {
                    // Would block - wake up when permits available
                    cx.waker().wake_by_ref();
                    return Poll::Pending;
                }
            }
        }
        
        if self.position >= self.data.len() {
            return Poll::Ready(None);
        }
        
        let end = (self.position + self.chunk_size).min(self.data.len());
        let chunk = self.data.slice(self.position..end);
        self.position = end;
        
        Poll::Ready(Some(Ok(chunk)))
    }
}

/// Arena allocator for temporary document processing
pub struct ProcessingArena {
    buffers: Vec<BytesMut>,
    current_buffer: usize,
    current_offset: usize,
    buffer_size: usize,
    total_allocated: usize,
}

impl ProcessingArena {
    /// Create new arena with specified buffer size
    pub fn new(buffer_size: usize) -> Self {
        let mut arena = Self {
            buffers: Vec::new(),
            current_buffer: 0,
            current_offset: 0,
            buffer_size,
            total_allocated: 0,
        };
        
        // Allocate first buffer
        arena.buffers.push(BytesMut::with_capacity(buffer_size));
        arena
    }
    
    /// Allocate memory from arena
    pub fn allocate(&mut self, size: usize) -> Option<*mut u8> {
        if size > self.buffer_size {
            return None; // Request too large
        }
        
        // Check if current buffer has enough space
        if self.current_offset + size > self.buffer_size {
            // Need new buffer
            self.current_buffer += 1;
            self.current_offset = 0;
            
            if self.current_buffer >= self.buffers.len() {
                self.buffers.push(BytesMut::with_capacity(self.buffer_size));
            }
        }
        
        let buffer = &mut self.buffers[self.current_buffer];
        if buffer.len() < self.current_offset + size {
            buffer.resize(self.current_offset + size, 0);
        }
        
        let ptr = unsafe { buffer.as_mut_ptr().add(self.current_offset) };
        self.current_offset += size;
        self.total_allocated += size;
        
        Some(ptr)
    }
    
    /// Reset arena (keeps buffers but resets offsets)
    pub fn reset(&mut self) {
        self.current_buffer = 0;
        self.current_offset = 0;
        self.total_allocated = 0;
        
        for buffer in &mut self.buffers {
            buffer.clear();
        }
    }
    
    /// Get total allocated memory
    pub fn total_allocated(&self) -> usize {
        self.total_allocated
    }
    
    /// Get number of buffers
    pub fn buffer_count(&self) -> usize {
        self.buffers.len()
    }
}

/// Smart string cache for deduplicating repeated text content
pub struct StringCache {
    cache: HashMap<u64, Arc<str>>,
    max_entries: usize,
    hit_count: u64,
    miss_count: u64,
}

impl StringCache {
    /// Create new string cache
    pub fn new(max_entries: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(max_entries),
            max_entries,
            hit_count: 0,
            miss_count: 0,
        }
    }
    
    /// Get or insert string into cache
    pub fn get_or_insert(&mut self, s: &str) -> Arc<str> {
        let hash = self.calculate_hash(s);
        
        if let Some(cached) = self.cache.get(&hash) {
            self.hit_count += 1;
            return cached.clone();
        }
        
        self.miss_count += 1;
        
        // Evict old entries if cache is full
        if self.cache.len() >= self.max_entries {
            // Simple random eviction - remove first entry
            if let Some(key) = self.cache.keys().next().copied() {
                self.cache.remove(&key);
            }
        }
        
        let arc_str: Arc<str> = Arc::from(s);
        self.cache.insert(hash, arc_str.clone());
        arc_str
    }
    
    /// Get cache hit ratio
    pub fn hit_ratio(&self) -> f64 {
        let total = self.hit_count + self.miss_count;
        if total == 0 {
            0.0
        } else {
            self.hit_count as f64 / total as f64
        }
    }
    
    /// Clear cache
    pub fn clear(&mut self) {
        self.cache.clear();
        self.hit_count = 0;
        self.miss_count = 0;
    }
    
    fn calculate_hash(&self, s: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        hasher.finish()
    }
}

/// Memory-optimized document builder
pub struct OptimizedDocumentBuilder {
    pool: Arc<Mutex<MemoryPool>>,
    string_cache: Arc<Mutex<StringCache>>,
    arena: ProcessingArena,
    current_size: usize,
    size_limit: usize,
}

impl OptimizedDocumentBuilder {
    /// Create new optimized document builder
    pub fn new(size_limit: usize) -> Self {
        Self {
            pool: Arc::new(Mutex::new(MemoryPool::new())),
            string_cache: Arc::new(Mutex::new(StringCache::new(1000))),
            arena: ProcessingArena::new(64 * 1024), // 64KB buffers
            current_size: 0,
            size_limit,
        }
    }
    
    /// Check if we can add more data
    pub fn can_add(&self, size: usize) -> bool {
        self.current_size + size <= self.size_limit
    }
    
    /// Add text content (deduplicated)
    pub fn add_text(&mut self, text: &str) -> Result<Arc<str>, crate::error::NeuralDocFlowError> {
        let text_size = text.len();
        
        if !self.can_add(text_size) {
            return Err(crate::error::NeuralDocFlowError::MemoryError {
                message: format!("Adding {} bytes would exceed limit of {} bytes", 
                    text_size, self.size_limit)
            });
        }
        
        let cached_text = self.string_cache.lock().get_or_insert(text);
        self.current_size += text_size;
        
        Ok(cached_text)
    }
    
    /// Allocate binary data buffer
    pub fn allocate_binary(&mut self, size: usize) -> Result<BytesMut, crate::error::NeuralDocFlowError> {
        if !self.can_add(size) {
            return Err(crate::error::NeuralDocFlowError::MemoryError {
                message: format!("Allocating {} bytes would exceed limit of {} bytes", 
                    size, self.size_limit)
            });
        }
        
        let buffer = self.pool.lock().allocate(size);
        self.current_size += size;
        
        Ok(buffer)
    }
    
    /// Get current memory usage
    pub fn current_size(&self) -> usize {
        self.current_size
    }
    
    /// Get memory usage statistics
    pub fn get_stats(&self) -> OptimizedBuilderStats {
        let pool_stats = self.pool.lock().stats();
        let string_cache_stats = {
            let cache = self.string_cache.lock();
            (cache.hit_ratio(), cache.hit_count + cache.miss_count)
        };
        
        OptimizedBuilderStats {
            current_size: self.current_size,
            size_limit: self.size_limit,
            pool_stats,
            string_cache_hit_ratio: string_cache_stats.0,
            string_cache_operations: string_cache_stats.1,
            arena_allocated: self.arena.total_allocated(),
            arena_buffers: self.arena.buffer_count(),
        }
    }
    
    /// Reset builder for reuse
    pub fn reset(&mut self) {
        self.arena.reset();
        self.current_size = 0;
    }
}

/// Statistics for optimized document builder
#[derive(Debug, Clone)]
pub struct OptimizedBuilderStats {
    pub current_size: usize,
    pub size_limit: usize,
    pub pool_stats: PoolStats,
    pub string_cache_hit_ratio: f64,
    pub string_cache_operations: u64,
    pub arena_allocated: usize,
    pub arena_buffers: usize,
}

/// Memory monitor for tracking system-wide memory usage
pub struct MemoryMonitor {
    allocations: Arc<Mutex<HashMap<Uuid, usize>>>,
    total_allocated: Arc<Mutex<usize>>,
    peak_allocated: Arc<Mutex<usize>>,
    limit: usize,
}

impl MemoryMonitor {
    /// Create new memory monitor with limit
    pub fn new(limit: usize) -> Self {
        Self {
            allocations: Arc::new(Mutex::new(HashMap::new())),
            total_allocated: Arc::new(Mutex::new(0)),
            peak_allocated: Arc::new(Mutex::new(0)),
            limit,
        }
    }
    
    /// Register allocation
    pub fn allocate(&self, id: Uuid, size: usize) -> Result<(), crate::error::NeuralDocFlowError> {
        let mut total = self.total_allocated.lock();
        
        if *total + size > self.limit {
            return Err(crate::error::NeuralDocFlowError::MemoryError {
                message: format!("Allocation would exceed memory limit: {} + {} > {}", 
                    *total, size, self.limit)
            });
        }
        
        self.allocations.lock().insert(id, size);
        *total += size;
        
        let mut peak = self.peak_allocated.lock();
        *peak = (*peak).max(*total);
        
        Ok(())
    }
    
    /// Unregister allocation
    pub fn deallocate(&self, id: Uuid) {
        if let Some(size) = self.allocations.lock().remove(&id) {
            let mut total = self.total_allocated.lock();
            *total = total.saturating_sub(size);
        }
    }
    
    /// Get current memory usage
    pub fn current_usage(&self) -> usize {
        *self.total_allocated.lock()
    }
    
    /// Get peak memory usage
    pub fn peak_usage(&self) -> usize {
        *self.peak_allocated.lock()
    }
    
    /// Get memory usage ratio (0.0 to 1.0)
    pub fn usage_ratio(&self) -> f64 {
        self.current_usage() as f64 / self.limit as f64
    }
    
    /// Check if allocation would exceed limit
    pub fn would_exceed_limit(&self, additional_size: usize) -> bool {
        self.current_usage() + additional_size > self.limit
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_pool() {
        let mut pool = MemoryPool::new();
        
        // Allocate and deallocate buffers
        let buffer1 = pool.allocate(1024);
        let buffer2 = pool.allocate(4096);
        
        let stats = pool.stats();
        assert_eq!(stats.allocations, 2);
        assert!(stats.current_usage > 0);
        
        pool.deallocate(buffer1);
        pool.deallocate(buffer2);
        
        let stats = pool.stats();
        assert_eq!(stats.deallocations, 2);
    }
    
    #[test]
    fn test_string_cache() {
        let mut cache = StringCache::new(10);
        
        let s1 = cache.get_or_insert("hello");
        let s2 = cache.get_or_insert("hello");
        
        assert!(Arc::ptr_eq(&s1, &s2));
        assert!(cache.hit_ratio() > 0.0);
    }
    
    #[test]
    fn test_processing_arena() {
        let mut arena = ProcessingArena::new(1024);
        
        let ptr1 = arena.allocate(100);
        let ptr2 = arena.allocate(200);
        
        assert!(ptr1.is_some());
        assert!(ptr2.is_some());
        assert_eq!(arena.total_allocated(), 300);
        
        arena.reset();
        assert_eq!(arena.total_allocated(), 0);
    }
    
    #[test]
    fn test_memory_monitor() {
        let monitor = MemoryMonitor::new(1000);
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        
        assert!(monitor.allocate(id1, 500).is_ok());
        assert!(monitor.allocate(id2, 300).is_ok());
        assert_eq!(monitor.current_usage(), 800);
        
        // Should fail - would exceed limit
        let id3 = Uuid::new_v4();
        assert!(monitor.allocate(id3, 300).is_err());
        
        monitor.deallocate(id1);
        assert_eq!(monitor.current_usage(), 300);
    }
}
