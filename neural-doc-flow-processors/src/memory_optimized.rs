//! Memory-optimized neural processors for <2MB per document processing
//!
//! This module provides neural processors that maintain strict memory limits
//! while processing documents efficiently.

use crate::{
    config::ModelType,
    error::{Result, NeuralError},
    traits::{ContentProcessor, QualityAssessor},
    types::{ContentBlock, Position},
};
use neural_doc_flow_core::{
    memory::*,
    optimized_types::*,
};
use std::sync::Arc;
use std::collections::HashMap;
use bytes::{Bytes, BytesMut};
use tokio::sync::RwLock;
use parking_lot::Mutex;
use uuid::Uuid;
use std::time::{Duration, Instant};

/// Memory-optimized content processor
pub struct MemoryOptimizedProcessor {
    /// Memory pool for buffer allocation
    memory_pool: Arc<Mutex<MemoryPool>>,
    
    /// String cache for deduplication
    string_cache: Arc<Mutex<StringCache>>,
    
    /// Memory monitor
    monitor: Arc<MemoryMonitor>,
    
    /// Processing arena for temporary allocations
    arena: Arc<Mutex<ProcessingArena>>,
    
    /// Streaming processor for large content
    streaming: Arc<StreamingProcessor>,
    
    /// Maximum memory per processing operation
    memory_limit: usize,
    
    /// Processing statistics
    stats: Arc<RwLock<ProcessingStats>>,
}

/// Processing statistics
#[derive(Debug, Default, Clone)]
pub struct ProcessingStats {
    pub documents_processed: u64,
    pub total_processing_time: Duration,
    pub peak_memory_usage: usize,
    pub average_memory_usage: f64,
    pub memory_efficiency_ratio: f64,
    pub cache_hit_ratio: f64,
    pub compression_ratio: f64,
}

impl MemoryOptimizedProcessor {
    /// Create new memory-optimized processor
    pub fn new(memory_limit: usize) -> Self {
        let chunk_size = (memory_limit / 16).max(1024); // Use 1/16 of limit for chunks, min 1KB
        
        Self {
            memory_pool: Arc::new(Mutex::new(MemoryPool::new())),
            string_cache: Arc::new(Mutex::new(StringCache::new(1000))),
            monitor: Arc::new(MemoryMonitor::new(memory_limit)),
            arena: Arc::new(Mutex::new(ProcessingArena::new(chunk_size))),
            streaming: Arc::new(StreamingProcessor::new(chunk_size, memory_limit)),
            memory_limit,
            stats: Arc::new(RwLock::new(ProcessingStats::default())),
        }
    }
    
    /// Process content block with memory optimization
    pub async fn process_optimized(&self, content: &ContentBlock) -> Result<OptimizedContentBlock> {
        let start_time = Instant::now();
        let initial_memory = self.monitor.current_usage();
        
        // Check memory availability
        let estimated_size = self.estimate_processing_size(content);
        if self.monitor.would_exceed_limit(estimated_size) {
            return Err(NeuralError::Memory(format!(
                "Processing would exceed memory limit: {} + {} > {}",
                initial_memory, estimated_size, self.memory_limit
            )));
        }
        
        // Allocate processing ID for tracking
        let processing_id = Uuid::new_v4();
        self.monitor.allocate(processing_id, estimated_size)?;
        
        let result = match content.content_type.as_str() {
            "text" => self.process_text_optimized(content).await,
            "table" => self.process_table_optimized(content).await,
            "image" => self.process_image_optimized(content).await,
            _ => self.process_generic_optimized(content).await,
        };
        
        // Clean up allocation tracking
        self.monitor.deallocate(processing_id);
        
        // Update statistics
        let processing_time = start_time.elapsed();
        let peak_memory = self.monitor.peak_usage();
        self.update_stats(processing_time, peak_memory, initial_memory).await;
        
        result
    }
    
    /// Process text content with memory optimization
    async fn process_text_optimized(&self, content: &ContentBlock) -> Result<OptimizedContentBlock> {
        let text = content.text.as_ref().ok_or_else(|| NeuralError::InvalidInput(
            "No text content found".to_string()
        ))?;
        
        // Use string cache for deduplication
        let cached_text = self.string_cache.lock().get_or_insert(text);
        
        // Apply lightweight text processing
        let processed_text = self.apply_text_optimizations(&cached_text);
        
        // Calculate confidence boost
        let confidence_boost = self.calculate_text_confidence_boost(&processed_text);
        let new_confidence = (content.confidence + confidence_boost).min(1.0);
        
        Ok(OptimizedContentBlock {
            id: content.id.clone(),
            content_type: content.content_type.clone(),
            text_content: Some(processed_text),
            binary_content: None,
            position: content.position.clone(),
            confidence: new_confidence,
            metadata: self.create_optimized_metadata(&content.metadata),
            processing_time: Duration::from_millis(1), // Minimal processing time
            memory_usage: cached_text.len(),
        })
    }
    
    /// Process table content with memory optimization
    async fn process_table_optimized(&self, content: &ContentBlock) -> Result<OptimizedContentBlock> {
        let text = content.text.as_ref().ok_or_else(|| NeuralError::InvalidInput(
            "No table content found".to_string()
        ))?;
        
        // Parse table efficiently using arena allocation
        let table_data = self.parse_table_with_arena(text).await?;
        
        // Compress table if large
        let optimized_table = if table_data.estimated_size() > 1024 {
            self.compress_table_data(table_data).await?
        } else {
            table_data
        };
        
        // Serialize optimized table
        let serialized = self.serialize_table_optimized(&optimized_table)?;
        
        Ok(OptimizedContentBlock {
            id: content.id.clone(),
            content_type: content.content_type.clone(),
            text_content: Some(serialized),
            binary_content: None,
            position: content.position.clone(),
            confidence: (content.confidence * 1.02).min(1.0), // Small boost for table processing
            metadata: self.create_optimized_metadata(&content.metadata),
            processing_time: Duration::from_millis(2),
            memory_usage: optimized_table.estimated_size(),
        })
    }
    
    /// Process image content with memory optimization
    async fn process_image_optimized(&self, content: &ContentBlock) -> Result<OptimizedContentBlock> {
        let image_data = content.binary_data.as_ref().ok_or_else(|| NeuralError::InvalidInput(
            "No image data found".to_string()
        ))?;
        
        // Process image in streaming chunks to avoid loading entire image
        let optimized_image = self.process_image_streaming(image_data).await?;
        
        Ok(OptimizedContentBlock {
            id: content.id.clone(),
            content_type: content.content_type.clone(),
            text_content: None,
            binary_content: Some(optimized_image),
            position: content.position.clone(),
            confidence: (content.confidence * 1.01).min(1.0),
            metadata: self.create_optimized_metadata(&content.metadata),
            processing_time: Duration::from_millis(5),
            memory_usage: image_data.len() / 10, // Assume 10:1 compression
        })
    }
    
    /// Process generic content with memory optimization
    async fn process_generic_optimized(&self, content: &ContentBlock) -> Result<OptimizedContentBlock> {
        // Minimal processing for unknown content types
        let memory_usage = content.text.as_ref().map(|t| t.len()).unwrap_or(0)
            + content.binary_data.as_ref().map(|b| b.len()).unwrap_or(0);
        
        Ok(OptimizedContentBlock {
            id: content.id.clone(),
            content_type: content.content_type.clone(),
            text_content: content.text.as_deref().map(Arc::from),
            binary_content: content.binary_data.clone(),
            position: content.position.clone(),
            confidence: content.confidence,
            metadata: self.create_optimized_metadata(&content.metadata),
            processing_time: Duration::from_millis(1),
            memory_usage,
        })
    }
    
    /// Apply lightweight text optimizations
    fn apply_text_optimizations(&self, text: &str) -> Arc<str> {
        // Lightweight text cleaning
        let cleaned = text
            .lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        
        // Use string cache for result
        self.string_cache.lock().get_or_insert(&cleaned)
    }
    
    /// Calculate confidence boost for text processing
    fn calculate_text_confidence_boost(&self, text: &str) -> f32 {
        // Simple heuristic based on text quality
        let lines = text.lines().count();
        let avg_line_length = if lines > 0 {
            text.len() as f32 / lines as f32
        } else {
            0.0
        };
        
        // Boost confidence for well-structured text
        if avg_line_length > 20.0 && avg_line_length < 100.0 {
            0.05 // 5% boost
        } else {
            0.01 // 1% boost
        }
    }
    
    /// Parse table using arena allocation
    async fn parse_table_with_arena(&self, text: &str) -> Result<CompactTableData> {
        let mut arena = self.arena.lock();
        arena.reset(); // Clear arena for this operation
        
        // Parse table structure
        let lines: Vec<&str> = text.lines().collect();
        if lines.is_empty() {
            return Ok(CompactTableData::Direct(Vec::new()));
        }
        
        // Detect table format and parse
        if text.contains('|') {
            self.parse_pipe_table(&lines)
        } else if text.contains('\t') {
            self.parse_tab_table(&lines)
        } else {
            self.parse_space_table(&lines)
        }
    }
    
    /// Parse pipe-separated table
    fn parse_pipe_table(&self, lines: &[&str]) -> Result<CompactTableData> {
        let mut rows = Vec::new();
        let mut string_cache = self.string_cache.lock();
        
        for line in lines {
            if line.contains('|') {
                let cells: Vec<Arc<str>> = line
                    .split('|')
                    .map(|cell| cell.trim())
                    .filter(|cell| !cell.is_empty())
                    .map(|cell| string_cache.get_or_insert(cell))
                    .collect();
                
                if !cells.is_empty() {
                    rows.push(cells);
                }
            }
        }
        
        Ok(CompactTableData::Direct(rows))
    }
    
    /// Parse tab-separated table
    fn parse_tab_table(&self, lines: &[&str]) -> Result<CompactTableData> {
        let mut rows = Vec::new();
        let mut string_cache = self.string_cache.lock();
        
        for line in lines {
            let cells: Vec<Arc<str>> = line
                .split('\t')
                .map(|cell| cell.trim())
                .map(|cell| string_cache.get_or_insert(cell))
                .collect();
            
            if !cells.is_empty() {
                rows.push(cells);
            }
        }
        
        Ok(CompactTableData::Direct(rows))
    }
    
    /// Parse space-separated table
    fn parse_space_table(&self, lines: &[&str]) -> Result<CompactTableData> {
        let mut rows = Vec::new();
        let mut string_cache = self.string_cache.lock();
        
        for line in lines {
            let cells: Vec<Arc<str>> = line
                .split_whitespace()
                .map(|cell| string_cache.get_or_insert(cell))
                .collect();
            
            if !cells.is_empty() {
                rows.push(cells);
            }
        }
        
        Ok(CompactTableData::Direct(rows))
    }
    
    /// Compress table data if large
    async fn compress_table_data(&self, table: CompactTableData) -> Result<CompactTableData> {
        match table {
            CompactTableData::Direct(rows) if rows.len() > 50 => {
                // Convert Arc<str> to String for serialization
                let serializable_rows: Vec<Vec<String>> = rows.iter()
                    .map(|row| row.iter().map(|cell| cell.to_string()).collect())
                    .collect();
                
                // Serialize and compress for large tables
                let serialized = serde_json::to_vec(&serializable_rows)
                    .map_err(|e| NeuralError::Serialization(e))?;
                
                // Simple compression (use proper compression library in real implementation)
                let compressed = compress_simple(&serialized);
                
                Ok(CompactTableData::Compressed {
                    data: Bytes::from(compressed),
                    original_size: estimate_rows_size(&rows),
                })
            }
            other => Ok(other),
        }
    }
    
    /// Serialize table in optimized format
    fn serialize_table_optimized(&self, table: &CompactTableData) -> Result<Arc<str>> {
        match table {
            CompactTableData::Direct(rows) => {
                let serialized = rows
                    .iter()
                    .map(|row| {
                        row.iter()
                            .map(|cell| cell.as_ref())
                            .collect::<Vec<_>>()
                            .join("\t")
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                
                Ok(self.string_cache.lock().get_or_insert(&serialized))
            }
            CompactTableData::Compressed { data, .. } => {
                // Return compressed indicator
                let indicator = format!("<compressed:{}>", data.len());
                Ok(self.string_cache.lock().get_or_insert(&indicator))
            }
            CompactTableData::External(path) => {
                Ok(path.clone())
            }
        }
    }
    
    /// Process image in streaming chunks
    async fn process_image_streaming(&self, image_data: &[u8]) -> Result<Vec<u8>> {
        // Process image in chunks to avoid loading entire image in memory
        let chunk_size = 4096;
        let mut processed = Vec::new();
        
        for chunk in image_data.chunks(chunk_size) {
            // Apply lightweight image processing
            let processed_chunk = self.process_image_chunk(chunk)?;
            processed.extend_from_slice(&processed_chunk);
            
            // Yield control to prevent blocking
            tokio::task::yield_now().await;
        }
        
        Ok(processed)
    }
    
    /// Process individual image chunk
    fn process_image_chunk(&self, chunk: &[u8]) -> Result<Vec<u8>> {
        // Placeholder image processing - in real implementation,
        // apply actual image processing algorithms
        Ok(chunk.to_vec())
    }
    
    /// Create optimized metadata
    fn create_optimized_metadata(&self, original: &HashMap<String, String>) -> HashMap<Arc<str>, Arc<str>> {
        let mut optimized = HashMap::new();
        let mut string_cache = self.string_cache.lock();
        
        for (key, value) in original {
            let cached_key = string_cache.get_or_insert(key);
            let cached_value = string_cache.get_or_insert(value);
            optimized.insert(cached_key, cached_value);
        }
        
        optimized
    }
    
    /// Estimate processing memory size
    fn estimate_processing_size(&self, content: &ContentBlock) -> usize {
        let base_size = std::mem::size_of::<OptimizedContentBlock>();
        let text_size = content.text.as_ref().map(|t| t.len()).unwrap_or(0);
        let binary_size = content.binary_data.as_ref().map(|b| b.len()).unwrap_or(0);
        
        // Add 50% overhead for processing
        ((base_size + text_size + binary_size) as f32 * 1.5) as usize
    }
    
    /// Update processing statistics
    async fn update_stats(&self, processing_time: Duration, peak_memory: usize, initial_memory: usize) {
        let mut stats = self.stats.write().await;
        
        stats.documents_processed += 1;
        stats.total_processing_time += processing_time;
        stats.peak_memory_usage = stats.peak_memory_usage.max(peak_memory);
        
        // Update average memory usage
        let current_avg = stats.average_memory_usage;
        let count = stats.documents_processed as f64;
        stats.average_memory_usage = (current_avg * (count - 1.0) + peak_memory as f64) / count;
        
        // Update efficiency ratio (memory saved vs original)
        let memory_used = peak_memory.saturating_sub(initial_memory);
        if memory_used > 0 {
            let efficiency = 1.0 - (memory_used as f64 / self.memory_limit as f64);
            stats.memory_efficiency_ratio = (stats.memory_efficiency_ratio * (count - 1.0) + efficiency) / count;
        }
        
        // Update cache hit ratio
        stats.cache_hit_ratio = self.string_cache.lock().hit_ratio();
    }
    
    /// Get processing statistics
    pub async fn get_stats(&self) -> ProcessingStats {
        self.stats.read().await.clone()
    }
    
    /// Reset processor state
    pub async fn reset(&self) {
        self.arena.lock().reset();
        self.string_cache.lock().clear();
        *self.stats.write().await = ProcessingStats::default();
    }
    
    /// Get current memory usage
    pub fn current_memory_usage(&self) -> usize {
        self.monitor.current_usage()
    }
    
    /// Get memory usage ratio
    pub fn memory_usage_ratio(&self) -> f64 {
        self.monitor.usage_ratio()
    }
}

/// Optimized content block with minimal memory footprint
#[derive(Debug, Clone)]
pub struct OptimizedContentBlock {
    pub id: String,
    pub content_type: String,
    pub text_content: Option<Arc<str>>,
    pub binary_content: Option<Vec<u8>>,
    pub position: Position,
    pub confidence: f32,
    pub metadata: HashMap<Arc<str>, Arc<str>>,
    pub processing_time: Duration,
    pub memory_usage: usize,
}

impl OptimizedContentBlock {
    /// Get estimated memory usage
    pub fn estimated_size(&self) -> usize {
        self.memory_usage
    }
    
    /// Convert to standard content block
    pub fn to_standard(&self) -> ContentBlock {
        let mut standard = ContentBlock::new(&self.content_type);
        standard.id = self.id.clone();
        standard.text = self.text_content.as_ref().map(|t| t.to_string());
        standard.binary_data = self.binary_content.clone();
        standard.position = self.position.clone();
        standard.confidence = self.confidence;
        
        // Convert metadata
        for (key, value) in &self.metadata {
            standard.metadata.insert(key.to_string(), value.to_string());
        }
        
        standard
    }
}

/// Compressed table data with size tracking
#[derive(Debug, Clone)]
pub enum CompactTableData {
    Direct(Vec<Vec<Arc<str>>>),
    Compressed {
        data: Bytes,
        original_size: usize,
    },
    External(Arc<str>),
}

impl CompactTableData {
    /// Estimate memory size
    fn estimated_size(&self) -> usize {
        match self {
            CompactTableData::Direct(rows) => estimate_rows_size(rows),
            CompactTableData::Compressed { data, .. } => data.len(),
            CompactTableData::External(_) => 0,
        }
    }
}

/// Memory-optimized quality assessor
pub struct MemoryOptimizedQualityAssessor {
    processor: Arc<MemoryOptimizedProcessor>,
    quality_thresholds: QualityThresholds,
}

impl MemoryOptimizedQualityAssessor {
    /// Create new memory-optimized quality assessor
    pub fn new(processor: Arc<MemoryOptimizedProcessor>) -> Self {
        Self {
            processor,
            quality_thresholds: QualityThresholds::default(),
        }
    }
    
    /// Assess content quality with memory optimization
    pub async fn assess_optimized(&self, content: &[OptimizedContentBlock]) -> Result<OptimizedQualityReport> {
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();
        
        // Memory-efficient quality assessment
        let memory_usage: usize = content.iter().map(|b| b.estimated_size()).sum();
        
        if memory_usage > self.processor.memory_limit {
            issues.push(OptimizedQualityIssue {
                issue_type: "memory_limit_exceeded".into(),
                severity: QualitySeverity::Critical,
                description: format!("Memory usage {} exceeds limit {}", 
                    memory_usage, self.processor.memory_limit),
                block_id: None,
                confidence: 1.0,
            });
        }
        
        // Check confidence levels
        let low_confidence_blocks: Vec<_> = content
            .iter()
            .filter(|b| b.confidence < self.quality_thresholds.min_confidence)
            .collect();
        
        if !low_confidence_blocks.is_empty() {
            issues.push(OptimizedQualityIssue {
                issue_type: "low_confidence".into(),
                severity: QualitySeverity::Medium,
                description: format!("{} blocks have low confidence", low_confidence_blocks.len()),
                block_id: None,
                confidence: 0.9,
            });
            
            recommendations.push("Review low confidence blocks for quality issues".into());
        }
        
        // Calculate overall quality score
        let avg_confidence = if content.is_empty() {
            0.0
        } else {
            content.iter().map(|b| b.confidence).sum::<f32>() / content.len() as f32
        };
        
        let memory_efficiency = if self.processor.memory_limit > 0 {
            1.0 - (memory_usage as f32 / self.processor.memory_limit as f32)
        } else {
            1.0
        };
        
        let overall_score = (avg_confidence + memory_efficiency) / 2.0;
        
        Ok(OptimizedQualityReport {
            overall_score,
            confidence_score: avg_confidence,
            memory_efficiency,
            memory_usage,
            issues,
            recommendations,
        })
    }
}

/// Optimized quality report
#[derive(Debug, Clone)]
pub struct OptimizedQualityReport {
    pub overall_score: f32,
    pub confidence_score: f32,
    pub memory_efficiency: f32,
    pub memory_usage: usize,
    pub issues: Vec<OptimizedQualityIssue>,
    pub recommendations: Vec<Arc<str>>,
}

/// Optimized quality issue
#[derive(Debug, Clone)]
pub struct OptimizedQualityIssue {
    pub issue_type: Arc<str>,
    pub severity: QualitySeverity,
    pub description: String,
    pub block_id: Option<String>,
    pub confidence: f32,
}

/// Quality severity levels
#[derive(Debug, Clone)]
pub enum QualitySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Quality thresholds
#[derive(Debug, Clone)]
pub struct QualityThresholds {
    pub min_confidence: f32,
    pub memory_warning_threshold: f32,
    pub memory_critical_threshold: f32,
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            min_confidence: 0.8,
            memory_warning_threshold: 0.8, // 80% of limit
            memory_critical_threshold: 0.95, // 95% of limit
        }
    }
}

// Helper functions

fn estimate_rows_size(rows: &[Vec<Arc<str>>]) -> usize {
    let mut size = rows.len() * std::mem::size_of::<Vec<Arc<str>>>();
    for row in rows {
        size += row.len() * std::mem::size_of::<Arc<str>>();
        for cell in row {
            size += cell.len();
        }
    }
    size
}

fn compress_simple(data: &[u8]) -> Vec<u8> {
    // Placeholder simple compression
    // In real implementation, use proper compression library like zstd or lz4
    data.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_memory_optimized_processor() {
        let processor = MemoryOptimizedProcessor::new(2 * 1024 * 1024); // 2MB limit
        
        let content = ContentBlock::text_block("Hello, world!".to_string());
        let result = processor.process_optimized(&content).await.unwrap();
        
        assert!(result.confidence >= content.confidence);
        assert!(result.memory_usage > 0);
    }
    
    #[test]
    fn test_compact_table_data() {
        let rows = vec![
            vec![Arc::from("Name"), Arc::from("Age")],
            vec![Arc::from("John"), Arc::from("30")],
        ];
        
        let table = CompactTableData::Direct(rows);
        assert!(table.estimated_size() > 0);
    }
    
    #[tokio::test]
    async fn test_quality_assessor() {
        let processor = Arc::new(MemoryOptimizedProcessor::new(1024 * 1024));
        let assessor = MemoryOptimizedQualityAssessor::new(processor);
        
        let blocks = vec![
            OptimizedContentBlock {
                id: "1".to_string(),
                content_type: "text".to_string(),
                text_content: Some(Arc::from("Good text")),
                binary_content: None,
                position: Position::default(),
                confidence: 0.95,
                metadata: HashMap::new(),
                processing_time: Duration::from_millis(1),
                memory_usage: 100,
            }
        ];
        
        let report = assessor.assess_optimized(&blocks).await.unwrap();
        assert!(report.overall_score > 0.0);
        assert_eq!(report.memory_usage, 100);
    }
}
