//! Processing pipeline trait definitions

use crate::{Document, ProcessingResult, ProcessingResultData};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Core trait for processing pipelines
#[async_trait]
pub trait ProcessorPipeline: Send + Sync {
    /// Get the pipeline name/identifier
    fn name(&self) -> &str;
    
    /// Process a single document through the pipeline
    async fn process(&self, document: Document) -> ProcessingResult<ProcessingResultData<Document>>;
    
    /// Process multiple documents in batch
    async fn process_batch(&self, documents: Vec<Document>) -> Vec<ProcessingResult<ProcessingResultData<Document>>> {
        let mut results = Vec::new();
        for doc in documents {
            results.push(self.process(doc).await);
        }
        results
    }
    
    /// Validate pipeline configuration
    fn validate_config(&self) -> ProcessingResult<()>;
    
    /// Get pipeline capabilities
    fn capabilities(&self) -> PipelineCapabilities;
    
    /// Get current pipeline status
    async fn status(&self) -> PipelineStatus;
    
    /// Shutdown the pipeline gracefully
    async fn shutdown(&self) -> ProcessingResult<()>;
}

/// Individual processor trait
#[async_trait]
pub trait Processor: Send + Sync {
    /// Get processor name
    fn name(&self) -> &str;
    
    /// Get processor version
    fn version(&self) -> &str;
    
    /// Process a document
    async fn process(&self, document: Document, context: &ProcessingContext) -> ProcessingResult<Document>;
    
    /// Check if processor can handle the document
    fn can_process(&self, document: &Document) -> bool;
    
    /// Get processor configuration schema
    fn config_schema(&self) -> serde_json::Value;
    
    /// Initialize processor with configuration
    async fn initialize(&mut self, config: &ProcessorConfig) -> ProcessingResult<()>;
    
    /// Cleanup processor resources
    async fn cleanup(&self) -> ProcessingResult<()>;
}

/// Pipeline capabilities description
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineCapabilities {
    /// Supported input formats
    pub input_formats: Vec<String>,
    
    /// Supported output formats
    pub output_formats: Vec<String>,
    
    /// Maximum concurrent documents
    pub max_concurrency: Option<usize>,
    
    /// Average processing time per document (seconds)
    pub avg_processing_time: Option<f64>,
    
    /// Memory requirements (MB)
    pub memory_requirements: Option<u64>,
    
    /// GPU requirements
    pub requires_gpu: bool,
    
    /// Features supported
    pub features: Vec<ProcessingFeature>,
}

/// Processing features available
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingFeature {
    /// Text extraction
    TextExtraction,
    
    /// Image extraction and processing
    ImageProcessing,
    
    /// Table detection and extraction
    TableExtraction,
    
    /// Neural content analysis
    NeuralAnalysis,
    
    /// OCR processing
    OpticalCharacterRecognition,
    
    /// Language detection
    LanguageDetection,
    
    /// Sentiment analysis
    SentimentAnalysis,
    
    /// Entity recognition
    EntityRecognition,
    
    /// Custom feature
    Custom(String),
}

/// Current pipeline status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStatus {
    /// Pipeline state
    pub state: PipelineState,
    
    /// Currently processing documents
    pub active_documents: usize,
    
    /// Documents in queue
    pub queued_documents: usize,
    
    /// Total documents processed
    pub total_processed: u64,
    
    /// Pipeline uptime in seconds
    pub uptime_seconds: u64,
    
    /// Last error (if any)
    pub last_error: Option<String>,
    
    /// Performance metrics
    pub metrics: PipelineMetrics,
}

/// Pipeline execution state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PipelineState {
    /// Pipeline is initializing
    Initializing,
    
    /// Pipeline is ready to process
    Ready,
    
    /// Pipeline is actively processing
    Processing,
    
    /// Pipeline is paused
    Paused,
    
    /// Pipeline has encountered an error
    Error,
    
    /// Pipeline is shutting down
    Shutdown,
}

/// Performance metrics for pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineMetrics {
    /// Average processing time (ms)
    pub avg_processing_time_ms: f64,
    
    /// Throughput (documents per second)
    pub throughput_docs_per_sec: f64,
    
    /// Error rate (0.0 to 1.0)
    pub error_rate: f64,
    
    /// Memory usage (MB)
    pub memory_usage_mb: f64,
    
    /// CPU usage percentage
    pub cpu_usage_percent: f64,
    
    /// GPU usage percentage (if applicable)
    pub gpu_usage_percent: Option<f64>,
}

/// Processing context shared between processors
#[derive(Debug, Clone)]
pub struct ProcessingContext {
    /// Session identifier
    pub session_id: Uuid,
    
    /// Processing parameters
    pub parameters: HashMap<String, serde_json::Value>,
    
    /// Shared state between processors
    pub shared_state: HashMap<String, serde_json::Value>,
    
    /// Processing start time
    pub start_time: chrono::DateTime<chrono::Utc>,
    
    /// Timeout for processing
    pub timeout: Option<std::time::Duration>,
}

/// Processor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorConfig {
    /// Processor name
    pub name: String,
    
    /// Processor-specific parameters
    pub parameters: HashMap<String, serde_json::Value>,
    
    /// Processing priority (0-10, higher = more important)
    pub priority: u8,
    
    /// Enable processor
    pub enabled: bool,
    
    /// Processor dependencies (must run after these)
    pub dependencies: Vec<String>,
    
    /// Maximum execution time (seconds)
    pub timeout_seconds: Option<u64>,
    
    /// Retry configuration
    pub retry_config: Option<RetryConfig>,
}

/// Retry configuration for processors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retries
    pub max_retries: u32,
    
    /// Initial delay between retries (milliseconds)
    pub initial_delay_ms: u64,
    
    /// Backoff multiplier for delays
    pub backoff_multiplier: f64,
    
    /// Maximum delay between retries (milliseconds)
    pub max_delay_ms: u64,
}

/// Builder for creating processing pipelines
pub struct PipelineBuilder {
    #[allow(dead_code)] // Used in future implementations
    pub(crate) name: String,
    pub(crate) processors: Vec<Box<dyn Processor>>,
    pub(crate) config: HashMap<String, serde_json::Value>,
}

impl PipelineBuilder {
    /// Create a new pipeline builder
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            processors: Vec::new(),
            config: HashMap::new(),
        }
    }
    
    /// Add a processor to the pipeline
    pub fn add_processor(mut self, processor: Box<dyn Processor>) -> Self {
        self.processors.push(processor);
        self
    }
    
    /// Set a configuration parameter
    pub fn with_config(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.config.insert(key.into(), value);
        self
    }
    
    /// Build the pipeline (placeholder - would return actual implementation)
    pub async fn build(self) -> ProcessingResult<Box<dyn ProcessorPipeline>> {
        // This would create the actual pipeline implementation
        todo!("Pipeline implementation not yet available")
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay_ms: 1000,
            backoff_multiplier: 2.0,
            max_delay_ms: 30000,
        }
    }
}

impl ProcessingContext {
    /// Create a new processing context
    pub fn new(session_id: Uuid) -> Self {
        Self {
            session_id,
            parameters: HashMap::new(),
            shared_state: HashMap::new(),
            start_time: chrono::Utc::now(),
            timeout: None,
        }
    }
    
    /// Set a parameter in the context
    pub fn set_parameter(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.parameters.insert(key.into(), value);
    }
    
    /// Get a parameter from the context
    pub fn get_parameter(&self, key: &str) -> Option<&serde_json::Value> {
        self.parameters.get(key)
    }
    
    /// Set shared state
    pub fn set_shared_state(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.shared_state.insert(key.into(), value);
    }
    
    /// Get shared state
    pub fn get_shared_state(&self, key: &str) -> Option<&serde_json::Value> {
        self.shared_state.get(key)
    }
}