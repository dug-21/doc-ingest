//! Async Neural Processing Pipeline
//!
//! This module implements the asynchronous neural processing pipeline
//! that orchestrates the 5-network architecture for optimal performance.

use crate::{
    config::{NeuralConfig, ModelType},
    error::{NeuralError, Result},
    neural::FannNeuralProcessor,
    types::{ContentBlock, EnhancedContent, NeuralFeatures, ProcessingMetrics},
    traits::NeuralProcessorTrait,
    domain_config::DomainConfigFactory,
    daa_integration::DaaEnhancedNeuralProcessor,
};
use futures::future::{join_all, try_join_all};
use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore};
use tracing::{debug, info, warn, error};
use uuid::Uuid;

/// Async neural processing pipeline
#[derive(Debug)]
pub struct AsyncNeuralPipeline {
    /// Neural processor
    processor: Arc<FannNeuralProcessor>,
    
    /// DAA processor (optional)
    daa_processor: Option<Arc<DaaEnhancedNeuralProcessor>>,
    
    /// Configuration
    config: NeuralConfig,
    
    /// Concurrency limiter
    semaphore: Arc<Semaphore>,
    
    /// Processing metrics
    metrics: Arc<RwLock<ProcessingMetrics>>,
}

/// Pipeline execution context
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    /// Request ID
    pub request_id: String,
    
    /// Priority level
    pub priority: Priority,
    
    /// Timeout duration
    pub timeout: std::time::Duration,
    
    /// Retry configuration
    pub retry_config: RetryConfig,
    
    /// Domain configuration
    pub domain: Option<String>,
}

/// Priority levels for processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low = 0,
    Medium = 1,
    High = 2,
    Critical = 3,
}

/// Retry configuration
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum retry attempts
    pub max_attempts: usize,
    
    /// Initial delay
    pub initial_delay: std::time::Duration,
    
    /// Backoff multiplier
    pub backoff_multiplier: f64,
    
    /// Maximum delay
    pub max_delay: std::time::Duration,
}

/// Pipeline stage for monitoring
#[derive(Debug, Clone)]
pub enum PipelineStage {
    /// Initial validation
    Validation,
    
    /// Feature extraction
    FeatureExtraction,
    
    /// Neural processing
    NeuralProcessing,
    
    /// Quality assessment
    QualityAssessment,
    
    /// Post-processing
    PostProcessing,
    
    /// Completed
    Completed,
}

/// Processing result with metadata
#[derive(Debug, Clone)]
pub struct ProcessingResult {
    /// Enhanced content
    pub content: EnhancedContent,
    
    /// Execution context
    pub context: ExecutionContext,
    
    /// Processing stages completed
    pub stages_completed: Vec<PipelineStage>,
    
    /// Total processing time
    pub total_time: std::time::Duration,
    
    /// Stage timings
    pub stage_timings: std::collections::HashMap<String, std::time::Duration>,
}

impl AsyncNeuralPipeline {
    /// Create new async neural pipeline
    pub async fn new(config: NeuralConfig) -> Result<Self> {
        info!("Initializing async neural pipeline");
        
        // Create neural processor
        let mut processor = FannNeuralProcessor::new(config.clone())?;
        processor.initialize().await?;
        
        let processor = Arc::new(processor);
        
        // Create semaphore for concurrency control
        let max_concurrent = config.processing.max_threads;
        let semaphore = Arc::new(Semaphore::new(max_concurrent));
        
        // Initialize metrics
        let metrics = Arc::new(RwLock::new(ProcessingMetrics::default()));
        
        Ok(Self {
            processor,
            daa_processor: None,
            config,
            semaphore,
            metrics,
        })
    }
    
    /// Initialize with DAA integration
    pub async fn with_daa_integration(mut self) -> Result<Self> {
        info!("Initializing DAA integration for async pipeline");
        
        let daa_processor = DaaEnhancedNeuralProcessor::new(self.config.clone()).await?;
        self.daa_processor = Some(Arc::new(daa_processor));
        
        Ok(self)
    }
    
    /// Process multiple content blocks asynchronously
    pub async fn process_batch(&self, content_blocks: Vec<ContentBlock>) -> Result<Vec<ProcessingResult>> {
        let batch_size = self.config.processing.batch_size;
        let mut results = Vec::new();
        
        // Process in batches
        for batch in content_blocks.chunks(batch_size) {
            let batch_results = self.process_batch_chunk(batch.to_vec()).await?;
            results.extend(batch_results);
        }
        
        Ok(results)
    }
    
    /// Process a single batch chunk
    async fn process_batch_chunk(&self, content_blocks: Vec<ContentBlock>) -> Result<Vec<ProcessingResult>> {
        let futures: Vec<_> = content_blocks
            .into_iter()
            .map(|block| {
                let context = ExecutionContext::default();
                self.process_single_with_context(block, context)
            })
            .collect();
        
        try_join_all(futures).await
    }
    
    /// Process single content block with context
    pub async fn process_single_with_context(
        &self,
        content_block: ContentBlock,
        context: ExecutionContext,
    ) -> Result<ProcessingResult> {
        let start_time = std::time::Instant::now();
        let mut stage_timings = std::collections::HashMap::new();
        let mut stages_completed = Vec::new();
        
        // Acquire semaphore permit
        let _permit = self.semaphore.acquire().await
            .map_err(|e| NeuralError::Inference(format!("Failed to acquire semaphore: {}", e)))?;
        
        // Apply domain configuration if specified
        let processor = if let Some(domain) = &context.domain {
            self.get_domain_processor(domain).await?
        } else {
            self.processor.clone()
        };
        
        // Stage 1: Validation
        let stage_start = std::time::Instant::now();
        self.validate_content_block(&content_block).await?;
        stages_completed.push(PipelineStage::Validation);
        stage_timings.insert("validation".to_string(), stage_start.elapsed());
        
        // Stage 2: Feature extraction
        let stage_start = std::time::Instant::now();
        let features = self.extract_features_async(&content_block).await?;
        stages_completed.push(PipelineStage::FeatureExtraction);
        stage_timings.insert("feature_extraction".to_string(), stage_start.elapsed());
        
        // Stage 3: Neural processing
        let stage_start = std::time::Instant::now();
        let enhanced_content = self.process_with_neural_networks(&content_block, &features).await?;
        stages_completed.push(PipelineStage::NeuralProcessing);
        stage_timings.insert("neural_processing".to_string(), stage_start.elapsed());
        
        // Stage 4: Quality assessment
        let stage_start = std::time::Instant::now();
        let quality_assessed = self.assess_quality_async(&enhanced_content).await?;
        stages_completed.push(PipelineStage::QualityAssessment);
        stage_timings.insert("quality_assessment".to_string(), stage_start.elapsed());
        
        // Stage 5: Post-processing
        let stage_start = std::time::Instant::now();
        let final_content = self.post_process_async(quality_assessed).await?;
        stages_completed.push(PipelineStage::PostProcessing);
        stage_timings.insert("post_processing".to_string(), stage_start.elapsed());
        
        stages_completed.push(PipelineStage::Completed);
        
        // Update metrics
        self.update_metrics(&final_content, start_time.elapsed()).await;
        
        Ok(ProcessingResult {
            content: final_content,
            context,
            stages_completed,
            total_time: start_time.elapsed(),
            stage_timings,
        })
    }
    
    /// Validate content block
    async fn validate_content_block(&self, block: &ContentBlock) -> Result<()> {
        // Basic validation
        if block.id.is_empty() {
            return Err(NeuralError::InvalidInput("Content block ID is empty".to_string()));
        }
        
        if block.content_type.is_empty() {
            return Err(NeuralError::InvalidInput("Content type is empty".to_string()));
        }
        
        // Domain-specific validation could go here
        
        Ok(())
    }
    
    /// Extract features asynchronously
    async fn extract_features_async(&self, block: &ContentBlock) -> Result<NeuralFeatures> {
        // Use SIMD-accelerated feature extraction if available
        self.processor.extract_features_simd(block)
    }
    
    /// Process with neural networks
    async fn process_with_neural_networks(&self, block: &ContentBlock, features: &NeuralFeatures) -> Result<EnhancedContent> {
        // Use DAA processor if available, otherwise use FANN processor
        if let Some(daa_processor) = &self.daa_processor {
            // Convert to document for DAA processing
            let document = self.content_block_to_document(block)?;
            daa_processor.process_document(document).await
                .map_err(|e| NeuralError::Inference(format!("DAA processing failed: {}", e)))
        } else {
            // Process with FANN processor
            self.processor.enhance_content(vec![block.clone()]).await
        }
    }
    
    /// Assess quality asynchronously
    async fn assess_quality_async(&self, content: &EnhancedContent) -> Result<EnhancedContent> {
        // Run quality assessment
        let quality_score = self.processor.assess_quality(content).await?;
        
        // Apply quality thresholds
        let min_confidence = self.config.quality.min_confidence;
        
        if quality_score.overall < min_confidence {
            warn!("Quality assessment failed: {} < {}", quality_score.overall, min_confidence);
            
            // Attempt quality improvement
            return self.improve_quality(content.clone()).await;
        }
        
        Ok(content.clone())
    }
    
    /// Improve quality of content
    async fn improve_quality(&self, mut content: EnhancedContent) -> Result<EnhancedContent> {
        // Apply quality improvement strategies
        for block in &mut content.blocks {
            if block.confidence < self.config.quality.min_confidence {
                // Reprocess low-confidence blocks
                let reprocessed = self.processor.process_content_block(block.clone()).await?;
                *block = reprocessed;
            }
        }
        
        // Recalculate overall confidence
        let total_confidence: f32 = content.blocks.iter().map(|b| b.confidence).sum();
        content.confidence = total_confidence / content.blocks.len() as f32;
        
        content.enhancements.push("quality_improvement".to_string());
        
        Ok(content)
    }
    
    /// Post-process content
    async fn post_process_async(&self, mut content: EnhancedContent) -> Result<EnhancedContent> {
        // Apply post-processing enhancements
        content.enhancements.push("async_pipeline_processing".to_string());
        
        // Apply domain-specific post-processing if needed
        
        Ok(content)
    }
    
    /// Get domain-specific processor
    async fn get_domain_processor(&self, domain: &str) -> Result<Arc<FannNeuralProcessor>> {
        // For now, return the same processor
        // In a full implementation, this would load domain-specific models
        debug!("Using domain-specific configuration for: {}", domain);
        Ok(self.processor.clone())
    }
    
    /// Convert content block to document
    fn content_block_to_document(&self, block: &ContentBlock) -> Result<neural_doc_flow_core::Document> {
        use uuid::Uuid;
        use neural_doc_flow_core::{DocumentType, DocumentSourceType, DocumentStructure};
        
        // Create a minimal document from content block
        let document_id = if let Ok(uuid) = Uuid::parse_str(&block.id) {
            uuid
        } else {
            Uuid::new_v4()
        };
        
        let content = neural_doc_flow_core::DocumentContent {
            text: block.text.clone(),
            images: Vec::new(),
            tables: Vec::new(),
            structured: std::collections::HashMap::new(),
            raw: None,
        };
        
        let metadata = neural_doc_flow_core::DocumentMetadata {
            title: block.metadata.get("title").cloned(),
            authors: block.metadata.get("author")
                .map(|s| vec![s.clone()])
                .unwrap_or_default(),
            source: block.metadata.get("source")
                .cloned()
                .unwrap_or_else(|| "neural_processor".to_string()),
            mime_type: block.metadata.get("mime_type")
                .cloned()
                .unwrap_or_else(|| "text/plain".to_string()),
            size: block.metadata.get("file_size")
                .and_then(|s| s.parse().ok()),
            language: block.metadata.get("language").cloned(),
            custom: block.metadata.iter()
                .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
                .collect(),
        };
        
        let structure = DocumentStructure {
            page_count: block.metadata.get("page_count")
                .and_then(|s| s.parse().ok()),
            sections: Vec::new(),
            toc_entries: Vec::new(),
        };
        
        let document = neural_doc_flow_core::Document {
            id: document_id,
            doc_type: DocumentType::Text,
            source_type: DocumentSourceType::Upload,
            raw_content: Vec::new(),
            metadata,
            content,
            structure,
            attachments: Vec::new(),
            processing_history: Vec::new(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        
        Ok(document)
    }
    
    /// Update processing metrics
    async fn update_metrics(&self, content: &EnhancedContent, processing_time: std::time::Duration) {
        let mut metrics = self.metrics.write().await;
        
        metrics.total_blocks_processed += content.blocks.len();
        metrics.total_processing_time += processing_time;
        metrics.average_confidence = (metrics.average_confidence + content.confidence) / 2.0;
    }
    
    /// Get processing metrics
    pub async fn get_metrics(&self) -> ProcessingMetrics {
        self.metrics.read().await.clone()
    }
    
    /// Process with timeout and retry
    pub async fn process_with_retry(
        &self,
        content_block: ContentBlock,
        context: ExecutionContext,
    ) -> Result<ProcessingResult> {
        let retry_config = context.retry_config.clone();
        let mut attempts = 0;
        let mut last_error = None;
        
        while attempts < retry_config.max_attempts {
            attempts += 1;
            
            // Apply timeout
            let processing_future = self.process_single_with_context(content_block.clone(), context.clone());
            
            match tokio::time::timeout(context.timeout, processing_future).await {
                Ok(result) => match result {
                    Ok(processing_result) => return Ok(processing_result),
                    Err(error) => {
                        last_error = Some(error);
                        if attempts < retry_config.max_attempts {
                            let delay = retry_config.initial_delay * (retry_config.backoff_multiplier.powi(attempts as i32 - 1) as u32);
                            let delay = delay.min(retry_config.max_delay);
                            tokio::time::sleep(delay).await;
                        }
                    }
                },
                Err(_timeout) => {
                    last_error = Some(NeuralError::Inference("Processing timeout".to_string()));
                    if attempts < retry_config.max_attempts {
                        let delay = retry_config.initial_delay * (retry_config.backoff_multiplier.powi(attempts as i32 - 1) as u32);
                        let delay = delay.min(retry_config.max_delay);
                        tokio::time::sleep(delay).await;
                    }
                }
            }
        }
        
        Err(last_error.unwrap_or_else(|| NeuralError::Inference("Max retry attempts exceeded".to_string())))
    }
    
    /// Shutdown the pipeline
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down async neural pipeline");
        
        // Shutdown DAA processor if available
        if let Some(daa_processor) = &self.daa_processor {
            daa_processor.shutdown().await?;
        }
        
        info!("Async neural pipeline shutdown complete");
        Ok(())
    }
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self {
            request_id: Uuid::new_v4().to_string(),
            priority: Priority::Medium,
            timeout: std::time::Duration::from_secs(30),
            retry_config: RetryConfig::default(),
            domain: None,
        }
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: std::time::Duration::from_millis(100),
            backoff_multiplier: 2.0,
            max_delay: std::time::Duration::from_secs(10),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::NeuralConfig;
    
    #[tokio::test]
    async fn test_pipeline_creation() {
        let config = NeuralConfig::default();
        let pipeline = AsyncNeuralPipeline::new(config).await;
        assert!(pipeline.is_ok());
    }
    
    #[tokio::test]
    async fn test_single_processing() {
        let config = NeuralConfig::default();
        let pipeline = AsyncNeuralPipeline::new(config).await.unwrap();
        
        let content_block = ContentBlock::new("test")
            .with_metadata("test", "value")
            .with_confidence(0.9);
        
        let context = ExecutionContext::default();
        let result = pipeline.process_single_with_context(content_block, context).await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_batch_processing() {
        let config = NeuralConfig::default();
        let pipeline = AsyncNeuralPipeline::new(config).await.unwrap();
        
        let content_blocks = vec![
            ContentBlock::new("test1"),
            ContentBlock::new("test2"),
            ContentBlock::new("test3"),
        ];
        
        let results = pipeline.process_batch(content_blocks).await;
        assert!(results.is_ok());
        assert_eq!(results.unwrap().len(), 3);
    }
    
    #[tokio::test]
    async fn test_retry_mechanism() {
        let config = NeuralConfig::default();
        let pipeline = AsyncNeuralPipeline::new(config).await.unwrap();
        
        let content_block = ContentBlock::new("test");
        let mut context = ExecutionContext::default();
        context.retry_config.max_attempts = 2;
        context.timeout = std::time::Duration::from_millis(1); // Very short timeout to trigger retry
        
        let result = pipeline.process_with_retry(content_block, context).await;
        // Should fail after retries
        assert!(result.is_err());
    }
}