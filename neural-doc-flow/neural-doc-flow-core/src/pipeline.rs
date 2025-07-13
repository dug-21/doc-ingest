//! Processing pipeline for neural document flow

use std::sync::Arc;
use std::collections::HashMap;
use tokio::time::{timeout, Duration};
use uuid::Uuid;
use tracing::{info, debug, warn, error, instrument};

use crate::coordination::{DocumentCoordinator, BatchConfig, AgentType};
use crate::types::{Document, ExtractedContent, ProcessingResult, DocumentSource as DocSource};
use crate::traits::{DocumentSource, DocumentProcessor, NeuralEnhancer, OutputFormatter, ContentValidator};
use crate::error::{CoreError, Result, ErrorContext, ContextualError};

/// Main document processing pipeline with DAA coordination
#[derive(Debug)]
pub struct DocFlow {
    coordinator: Arc<DocumentCoordinator>,
    sources: HashMap<String, Box<dyn DocumentSource>>,
    processors: Vec<Box<dyn DocumentProcessor>>,
    enhancers: Vec<Box<dyn NeuralEnhancer>>,
    formatters: HashMap<String, Box<dyn OutputFormatter>>,
    validators: Vec<Box<dyn ContentValidator>>,
    config: PipelineConfig,
}

/// Configuration for the processing pipeline
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub enable_parallel_processing: bool,
    pub enable_neural_enhancement: bool,
    pub enable_validation: bool,
    pub timeout_seconds: u32,
    pub max_concurrent_documents: usize,
    pub retry_attempts: usize,
    pub confidence_threshold: f32,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            enable_parallel_processing: true,
            enable_neural_enhancement: true,
            enable_validation: true,
            timeout_seconds: 300,
            max_concurrent_documents: 10,
            retry_attempts: 3,
            confidence_threshold: 0.8,
        }
    }
}

/// Processing pipeline that orchestrates document extraction
pub struct ProcessingPipeline {
    config: PipelineConfig,
}

impl DocFlow {
    /// Create a new DocFlow with default coordinator
    pub async fn new() -> Result<Self> {
        let coordinator = Arc::new(DocumentCoordinator::new().await?);
        Self::with_coordinator(coordinator)
    }
    
    /// Create DocFlow with existing coordinator
    pub fn with_coordinator(coordinator: Arc<DocumentCoordinator>) -> Result<Self> {
        Ok(Self {
            coordinator,
            sources: HashMap::new(),
            processors: Vec::new(),
            enhancers: Vec::new(),
            formatters: HashMap::new(),
            validators: Vec::new(),
            config: PipelineConfig::default(),
        })
    }
    
    /// Create DocFlow with custom configuration
    pub async fn with_config(config: PipelineConfig) -> Result<Self> {
        let coordinator = Arc::new(DocumentCoordinator::new().await?);
        Ok(Self {
            coordinator,
            sources: HashMap::new(),
            processors: Vec::new(),
            enhancers: Vec::new(),
            formatters: HashMap::new(),
            validators: Vec::new(),
            config,
        })
    }
    
    /// Register a document source plugin
    pub fn register_source(&mut self, source: Box<dyn DocumentSource>) {
        let name = source.name().to_string();
        info!("Registering document source: {}", name);
        self.sources.insert(name, source);
    }
    
    /// Register a document processor
    pub fn register_processor(&mut self, processor: Box<dyn DocumentProcessor>) {
        info!("Registering document processor: {}", processor.name());
        self.processors.push(processor);
    }
    
    /// Register a neural enhancer
    pub fn register_enhancer(&mut self, enhancer: Box<dyn NeuralEnhancer>) {
        info!("Registering neural enhancer: {}", enhancer.name());
        self.enhancers.push(enhancer);
    }
    
    /// Register an output formatter
    pub fn register_formatter(&mut self, formatter: Box<dyn OutputFormatter>) {
        let name = formatter.name().to_string();
        info!("Registering output formatter: {}", name);
        self.formatters.insert(name, formatter);
    }
    
    /// Register a content validator
    pub fn register_validator(&mut self, validator: Box<dyn ContentValidator>) {
        info!("Registering content validator: {}", validator.name());
        self.validators.push(validator);
    }
    
    /// Process a single document
    #[instrument(skip(self, document))]
    pub async fn process_document(&self, document: Document) -> Result<ProcessingResult> {
        info!("Processing document {}", document.id);
        
        let start_time = std::time::Instant::now();
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        
        // Find appropriate source plugin
        let source = self.find_source_for_document(&document)?;
        
        // Extract content using source plugin
        let mut extracted_content = match timeout(
            Duration::from_secs(self.config.timeout_seconds as u64),
            source.extract(&document)
        ).await {
            Ok(Ok(content)) => content,
            Ok(Err(e)) => {
                let error_msg = format!("Extraction failed: {}", e);
                error!("{}", error_msg);
                return Err(CoreError::ExtractionError(error_msg));
            }
            Err(_) => {
                let error_msg = format!("Extraction timed out after {} seconds", self.config.timeout_seconds);
                error!("{}", error_msg);
                return Err(CoreError::TimeoutError { 
                    timeout_ms: self.config.timeout_seconds as u64 * 1000 
                });
            }
        };
        
        // Apply processors
        for processor in &self.processors {
            if processor.can_process(&extracted_content) {
                debug!("Applying processor: {}", processor.name());
                match processor.process(extracted_content.clone()).await {
                    Ok(processed) => extracted_content = processed,
                    Err(e) => {
                        warnings.push(format!("Processor {} failed: {}", processor.name(), e));
                        warn!("Processor {} failed: {}", processor.name(), e);
                    }
                }
            }
        }
        
        // Apply neural enhancements if enabled
        let mut neural_enhancements = Vec::new();
        if self.config.enable_neural_enhancement {
            for enhancer in &self.enhancers {
                if enhancer.can_enhance(&extracted_content) {
                    debug!("Applying neural enhancer: {}", enhancer.name());
                    match enhancer.enhance(extracted_content.clone()).await {
                        Ok((enhanced_content, enhancements)) => {
                            extracted_content = enhanced_content;
                            neural_enhancements.extend(enhancements);
                        }
                        Err(e) => {
                            warnings.push(format!("Neural enhancer {} failed: {}", enhancer.name(), e));
                            warn!("Neural enhancer {} failed: {}", enhancer.name(), e);
                        }
                    }
                }
            }
        }
        
        // Apply validation if enabled
        if self.config.enable_validation {
            for validator in &self.validators {
                if validator.can_validate(&extracted_content) {
                    debug!("Applying validator: {}", validator.name());
                    match validator.validate(&extracted_content).await {
                        Ok(validation_result) => {
                            if !validation_result.is_valid {
                                for error in validation_result.errors {
                                    errors.push(crate::types::ProcessingError {
                                        error_type: crate::types::ErrorType::ValidationError,
                                        message: error.message,
                                        block_id: error.block_id,
                                        recoverable: true,
                                    });
                                }
                            }
                            for warning in validation_result.warnings {
                                warnings.push(warning.message);
                            }
                        }
                        Err(e) => {
                            warnings.push(format!("Validator {} failed: {}", validator.name(), e));
                            warn!("Validator {} failed: {}", validator.name(), e);
                        }
                    }
                }
            }
        }
        
        let processing_time = start_time.elapsed();
        
        let result = ProcessingResult {
            document_id: document.id,
            extracted_content,
            processing_time_ms: processing_time.as_millis() as u64,
            agent_id: None, // Will be set by coordinator if using agents
            neural_enhancements,
            errors,
            warnings,
        };
        
        info!("Document {} processed in {}ms", document.id, result.processing_time_ms);
        Ok(result)
    }
    
    /// Process multiple documents in batch with DAA coordination
    #[instrument(skip(self, documents))]
    pub async fn process_batch(&self, documents: Vec<Document>) -> Result<Vec<ProcessingResult>> {
        info!("Processing batch of {} documents", documents.len());
        
        if self.config.enable_parallel_processing && documents.len() > 1 {
            // Use DAA coordination for parallel processing
            let batch_config = BatchConfig {
                parallel_extraction: true,
                neural_enhancement: self.config.enable_neural_enhancement,
                result_aggregation: true,
                extraction_agents: std::cmp::min(documents.len(), self.config.max_concurrent_documents),
                neural_agents: if self.config.enable_neural_enhancement { 2 } else { 0 },
                processing_options: crate::coordination::ProcessingOptions {
                    enable_ocr: true,
                    extract_tables: true,
                    extract_images: false,
                    confidence_threshold: self.config.confidence_threshold,
                    timeout_seconds: self.config.timeout_seconds,
                },
            };
            
            self.coordinator.process_batch(documents, batch_config).await
        } else {
            // Sequential processing for single documents or when parallel is disabled
            let mut results = Vec::new();
            for document in documents {
                match self.process_document(document).await {
                    Ok(result) => results.push(result),
                    Err(e) => {
                        error!("Failed to process document: {}", e);
                        return Err(e);
                    }
                }
            }
            Ok(results)
        }
    }
    
    /// Find appropriate source plugin for a document
    fn find_source_for_document(&self, document: &Document) -> Result<&dyn DocumentSource> {
        for source in self.sources.values() {
            if source.can_handle(document) {
                debug!("Found source {} for document {}", source.name(), document.id);
                return Ok(source.as_ref());
            }
        }
        
        Err(CoreError::NotFound(format!(
            "No source plugin found for document {} with source {:?}",
            document.id, document.source
        )))
    }
    
    /// Get pipeline statistics
    pub async fn get_stats(&self) -> PipelineStats {
        let agent_status = self.coordinator.get_agent_status().await;
        
        PipelineStats {
            registered_sources: self.sources.len(),
            registered_processors: self.processors.len(),
            registered_enhancers: self.enhancers.len(),
            registered_formatters: self.formatters.len(),
            registered_validators: self.validators.len(),
            active_agents: agent_status.len(),
            coordinator_id: self.coordinator.id,
        }
    }
    
    /// Initialize all plugins
    pub async fn initialize(&mut self) -> Result<()> {
        info!("Initializing DocFlow pipeline");
        
        // Initialize sources
        for (name, source) in &mut self.sources {
            debug!("Initializing source: {}", name);
            if let Err(e) = source.initialize().await {
                warn!("Failed to initialize source {}: {}", name, e);
            }
        }
        
        // Initialize processors
        for processor in &mut self.processors {
            debug!("Initializing processor: {}", processor.name());
            if let Err(e) = processor.initialize().await {
                warn!("Failed to initialize processor {}: {}", processor.name(), e);
            }
        }
        
        // Initialize enhancers
        for enhancer in &mut self.enhancers {
            debug!("Initializing enhancer: {}", enhancer.name());
            if let Err(e) = enhancer.initialize().await {
                warn!("Failed to initialize enhancer {}: {}", enhancer.name(), e);
            }
        }
        
        // Initialize formatters
        for (name, formatter) in &mut self.formatters {
            debug!("Initializing formatter: {}", name);
            if let Err(e) = formatter.initialize().await {
                warn!("Failed to initialize formatter {}: {}", name, e);
            }
        }
        
        // Initialize validators
        for validator in &mut self.validators {
            debug!("Initializing validator: {}", validator.name());
            if let Err(e) = validator.initialize().await {
                warn!("Failed to initialize validator {}: {}", validator.name(), e);
            }
        }
        
        info!("DocFlow pipeline initialized successfully");
        Ok(())
    }
    
    /// Shutdown the pipeline and all components
    pub async fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down DocFlow pipeline");
        
        // Shutdown coordinator
        self.coordinator.shutdown().await?;
        
        // Shutdown all plugins
        for (name, source) in &mut self.sources {
            debug!("Shutting down source: {}", name);
            if let Err(e) = source.shutdown().await {
                warn!("Failed to shutdown source {}: {}", name, e);
            }
        }
        
        for processor in &mut self.processors {
            debug!("Shutting down processor: {}", processor.name());
            if let Err(e) = processor.shutdown().await {
                warn!("Failed to shutdown processor {}: {}", processor.name(), e);
            }
        }
        
        for enhancer in &mut self.enhancers {
            debug!("Shutting down enhancer: {}", enhancer.name());
            if let Err(e) = enhancer.shutdown().await {
                warn!("Failed to shutdown enhancer {}: {}", enhancer.name(), e);
            }
        }
        
        for (name, formatter) in &mut self.formatters {
            debug!("Shutting down formatter: {}", name);
            if let Err(e) = formatter.shutdown().await {
                warn!("Failed to shutdown formatter {}: {}", name, e);
            }
        }
        
        for validator in &mut self.validators {
            debug!("Shutting down validator: {}", validator.name());
            if let Err(e) = validator.shutdown().await {
                warn!("Failed to shutdown validator {}: {}", validator.name(), e);
            }
        }
        
        info!("DocFlow pipeline shutdown complete");
        Ok(())
    }
}

/// Pipeline statistics
#[derive(Debug)]
pub struct PipelineStats {
    pub registered_sources: usize,
    pub registered_processors: usize,
    pub registered_enhancers: usize,
    pub registered_formatters: usize,
    pub registered_validators: usize,
    pub active_agents: usize,
    pub coordinator_id: Uuid,
}

impl ProcessingPipeline {
    pub fn new() -> Self {
        Self {
            config: PipelineConfig::default(),
        }
    }
    
    pub fn with_config(config: PipelineConfig) -> Self {
        Self { config }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DocumentSource as DocSource, DocumentMetadata};
    use std::path::PathBuf;
    
    #[tokio::test]
    async fn test_docflow_creation() {
        let docflow = DocFlow::new().await;
        assert!(docflow.is_ok());
    }
    
    #[tokio::test]
    async fn test_pipeline_stats() {
        let docflow = DocFlow::new().await.unwrap();
        let stats = docflow.get_stats().await;
        assert_eq!(stats.registered_sources, 0);
        assert_eq!(stats.registered_processors, 0);
    }
}