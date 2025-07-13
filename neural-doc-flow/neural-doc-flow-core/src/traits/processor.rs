//! Processing pipeline traits for neural enhancement
//!
//! This module defines traits for building and executing document processing
//! pipelines with neural enhancement capabilities.

use async_trait::async_trait;
use std::collections::HashMap;
use crate::{
    PipelineError, ExtractedDocument, ProcessingResult, PipelineConfig, 
    StageConfig, NeuralDocFlowResult, ProcessingTask, TaskPriority
};

/// Core trait for document processing pipelines
/// 
/// This trait defines the interface for processing pipelines that can
/// enhance extracted documents through multiple stages of processing.
/// 
/// # Example Implementation
/// 
/// ```rust,no_run
/// use async_trait::async_trait;
/// use neural_doc_flow_core::{ProcessorPipeline, ExtractedDocument, ProcessingResult, PipelineError, PipelineConfig};
/// 
/// struct MyPipeline {
///     config: PipelineConfig,
/// }
/// 
/// #[async_trait]
/// impl ProcessorPipeline for MyPipeline {
///     async fn process(&self, document: ExtractedDocument) -> Result<ProcessingResult, PipelineError> {
///         // Processing logic
///         Ok(ProcessingResult {
///             stage_id: "my_pipeline".to_string(),
///             document,
///             metadata: Default::default(),
///             quality_metrics: Default::default(),
///         })
///     }
///     
///     async fn process_batch(&self, documents: Vec<ExtractedDocument>) -> Result<Vec<ProcessingResult>, PipelineError> {
///         let mut results = Vec::new();
///         for doc in documents {
///             results.push(self.process(doc).await?);
///         }
///         Ok(results)
///     }
///     
///     fn pipeline_id(&self) -> &str { "my_pipeline" }
///     fn name(&self) -> &str { "My Processing Pipeline" }
///     fn version(&self) -> &str { "1.0.0" }
///     
///     async fn initialize(&mut self, config: PipelineConfig) -> Result<(), PipelineError> {
///         self.config = config;
///         Ok(())
///     }
///     
///     async fn cleanup(&mut self) -> Result<(), PipelineError> {
///         Ok(())
///     }
/// }
/// ```
#[async_trait]
pub trait ProcessorPipeline: Send + Sync {
    /// Process a single document through the pipeline
    /// 
    /// This is the main processing method that takes an extracted document
    /// and applies neural enhancement and other processing stages.
    /// 
    /// # Parameters
    /// - `document`: The extracted document to process
    /// 
    /// # Returns
    /// - `Ok(ProcessingResult)` with enhanced document and metadata
    /// - `Err(_)` if processing failed
    async fn process(&self, document: ExtractedDocument) -> Result<ProcessingResult, PipelineError>;

    /// Process multiple documents in batch
    /// 
    /// This method processes multiple documents together, potentially
    /// with optimizations for batch processing.
    /// 
    /// # Parameters
    /// - `documents`: Vector of documents to process
    /// 
    /// # Returns
    /// - `Ok(Vec<ProcessingResult>)` with results for all documents
    /// - `Err(_)` if batch processing failed
    async fn process_batch(&self, documents: Vec<ExtractedDocument>) -> Result<Vec<ProcessingResult>, PipelineError>;

    /// Unique identifier for this pipeline
    fn pipeline_id(&self) -> &str;

    /// Human-readable name of this pipeline
    fn name(&self) -> &str;

    /// Pipeline version
    fn version(&self) -> &str;

    /// Initialize the pipeline with configuration
    /// 
    /// Called once during pipeline setup with the pipeline's configuration.
    /// Use this to set up processing stages, load models, and prepare resources.
    /// 
    /// # Parameters
    /// - `config`: Pipeline configuration
    /// 
    /// # Returns
    /// - `Ok(())` if initialization succeeded
    /// - `Err(_)` if initialization failed
    async fn initialize(&mut self, config: PipelineConfig) -> Result<(), PipelineError>;

    /// Clean up pipeline resources
    /// 
    /// Called when the pipeline is being unloaded. Use this to clean up
    /// resources, unload models, stop background tasks, etc.
    async fn cleanup(&mut self) -> Result<(), PipelineError>;

    /// Get supported document types
    /// 
    /// Returns the document types (source IDs) that this pipeline can process.
    /// Default implementation supports all document types.
    fn supported_document_types(&self) -> Vec<String> {
        vec!["*".to_string()] // Support all by default
    }

    /// Get pipeline capabilities
    /// 
    /// Returns information about what this pipeline can do.
    fn capabilities(&self) -> PipelineCapabilities {
        PipelineCapabilities::default()
    }

    /// Get pipeline performance metrics
    /// 
    /// Returns performance statistics for this pipeline.
    fn metrics(&self) -> PipelineMetrics {
        PipelineMetrics::default()
    }

    /// Check if pipeline supports streaming
    /// 
    /// Returns true if this pipeline can process documents as they arrive.
    fn supports_streaming(&self) -> bool {
        false
    }

    /// Check if pipeline supports parallel processing
    /// 
    /// Returns true if this pipeline can safely process multiple documents
    /// in parallel.
    fn supports_parallel(&self) -> bool {
        true
    }

    /// Validate a document before processing
    /// 
    /// Optional validation step before processing. Default implementation
    /// always returns Ok.
    async fn validate_document(&self, _document: &ExtractedDocument) -> Result<(), PipelineError> {
        Ok(())
    }

    /// Estimate processing time
    /// 
    /// Provides an estimate of how long processing will take.
    /// Default implementation returns None.
    fn estimate_processing_time(&self, _document: &ExtractedDocument) -> Option<std::time::Duration> {
        None
    }
}

/// Pipeline capabilities information
#[derive(Debug, Clone, Default)]
pub struct PipelineCapabilities {
    /// Can enhance text quality
    pub text_enhancement: bool,
    /// Can improve layout detection
    pub layout_enhancement: bool,
    /// Can enhance table extraction
    pub table_enhancement: bool,
    /// Can improve image processing
    pub image_enhancement: bool,
    /// Can perform semantic analysis
    pub semantic_analysis: bool,
    /// Can extract entities
    pub entity_extraction: bool,
    /// Can perform classification
    pub document_classification: bool,
    /// Can generate summaries
    pub summarization: bool,
    /// Can translate content
    pub translation: bool,
    /// Supported languages
    pub supported_languages: Vec<String>,
    /// Maximum document size (bytes)
    pub max_document_size: Option<usize>,
    /// Supported quality levels
    pub quality_levels: Vec<String>,
}

/// Pipeline performance metrics
#[derive(Debug, Clone, Default)]
pub struct PipelineMetrics {
    /// Total documents processed
    pub documents_processed: u64,
    /// Total processing time
    pub total_processing_time: std::time::Duration,
    /// Average processing time per document
    pub average_processing_time: std::time::Duration,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f32,
    /// Average quality improvement (0.0 to 1.0)
    pub average_quality_improvement: f32,
    /// Peak memory usage (bytes)
    pub peak_memory_usage: usize,
    /// Throughput (documents per second)
    pub throughput: f32,
}

/// Trait for individual processing stages
/// 
/// This trait defines the interface for individual stages within a
/// processing pipeline. Each stage can perform specific transformations
/// or enhancements on the document.
#[async_trait]
pub trait ProcessingStage: Send + Sync {
    /// Execute this stage on a document
    /// 
    /// # Parameters
    /// - `document`: The document to process
    /// - `context`: Processing context from previous stages
    /// 
    /// # Returns
    /// - `Ok(ExtractedDocument)` with stage modifications
    /// - `Err(_)` if stage processing failed
    async fn execute(
        &self,
        document: ExtractedDocument,
        context: &ProcessingContext,
    ) -> Result<ExtractedDocument, PipelineError>;

    /// Stage identifier
    fn stage_id(&self) -> &str;

    /// Stage name
    fn name(&self) -> &str;

    /// Stage dependencies
    /// 
    /// Returns the IDs of stages that must be executed before this stage.
    fn dependencies(&self) -> Vec<String> {
        Vec::new()
    }

    /// Initialize the stage
    async fn initialize(&mut self, config: StageConfig) -> Result<(), PipelineError>;

    /// Clean up stage resources
    async fn cleanup(&mut self) -> Result<(), PipelineError>;

    /// Check if stage can process the document
    fn can_process(&self, document: &ExtractedDocument) -> bool {
        true
    }

    /// Estimate processing time for this stage
    fn estimate_time(&self, _document: &ExtractedDocument) -> Option<std::time::Duration> {
        None
    }
}

/// Processing context shared between stages
#[derive(Debug, Clone, Default)]
pub struct ProcessingContext {
    /// Stage-specific data
    pub stage_data: HashMap<String, serde_json::Value>,
    /// Global processing metadata
    pub metadata: HashMap<String, String>,
    /// Processing start time
    pub start_time: chrono::DateTime<chrono::Utc>,
    /// Processing deadline
    pub deadline: Option<chrono::DateTime<chrono::Utc>>,
    /// Processing priority
    pub priority: TaskPriority,
}

impl ProcessingContext {
    /// Create new processing context
    pub fn new() -> Self {
        Self {
            start_time: chrono::Utc::now(),
            ..Default::default()
        }
    }

    /// Add stage data
    pub fn set_stage_data(&mut self, stage_id: &str, data: serde_json::Value) {
        self.stage_data.insert(stage_id.to_string(), data);
    }

    /// Get stage data
    pub fn get_stage_data(&self, stage_id: &str) -> Option<&serde_json::Value> {
        self.stage_data.get(stage_id)
    }

    /// Add metadata
    pub fn set_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }

    /// Get metadata
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(|s| s.as_str())
    }

    /// Check if processing is near deadline
    pub fn is_near_deadline(&self) -> bool {
        if let Some(deadline) = self.deadline {
            let now = chrono::Utc::now();
            let remaining = deadline.signed_duration_since(now);
            remaining.num_seconds() < 30 // Less than 30 seconds remaining
        } else {
            false
        }
    }

    /// Get elapsed processing time
    pub fn elapsed_time(&self) -> chrono::Duration {
        chrono::Utc::now().signed_duration_since(self.start_time)
    }
}

/// Trait for managing processing pipelines
/// 
/// This trait provides higher-level pipeline management capabilities,
/// including pipeline discovery, scheduling, and resource management.
#[async_trait]
pub trait PipelineManager: Send + Sync {
    /// Get available pipelines
    async fn list_pipelines(&self) -> Result<Vec<String>, PipelineError>;

    /// Get a specific pipeline
    async fn get_pipeline(&self, id: &str) -> Result<Box<dyn ProcessorPipeline>, PipelineError>;

    /// Create a new pipeline from configuration
    async fn create_pipeline(&self, config: PipelineConfig) -> Result<Box<dyn ProcessorPipeline>, PipelineError>;

    /// Submit a processing task
    async fn submit_task(&self, task: ProcessingTask) -> Result<String, PipelineError>;

    /// Get task status
    async fn get_task_status(&self, task_id: &str) -> Result<TaskStatus, PipelineError>;

    /// Cancel a task
    async fn cancel_task(&self, task_id: &str) -> Result<(), PipelineError>;

    /// Get system resources usage
    fn get_resource_usage(&self) -> ResourceUsage;
}

/// Task status information
#[derive(Debug, Clone)]
pub enum TaskStatus {
    /// Task is queued for processing
    Queued {
        position: usize,
        estimated_start: Option<chrono::DateTime<chrono::Utc>>,
    },
    /// Task is currently being processed
    Processing {
        stage: String,
        progress: f32, // 0.0 to 1.0
        estimated_completion: Option<chrono::DateTime<chrono::Utc>>,
    },
    /// Task completed successfully
    Completed {
        result: ProcessingResult,
        completion_time: chrono::DateTime<chrono::Utc>,
    },
    /// Task failed
    Failed {
        error: String,
        failure_time: chrono::DateTime<chrono::Utc>,
    },
    /// Task was cancelled
    Cancelled {
        cancellation_time: chrono::DateTime<chrono::Utc>,
    },
}

/// System resource usage information
#[derive(Debug, Clone, Default)]
pub struct ResourceUsage {
    /// CPU usage percentage (0.0 to 100.0)
    pub cpu_usage: f32,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// GPU usage percentage (0.0 to 100.0)
    pub gpu_usage: f32,
    /// GPU memory usage in bytes
    pub gpu_memory_usage: usize,
    /// Number of active tasks
    pub active_tasks: usize,
    /// Number of queued tasks
    pub queued_tasks: usize,
    /// Average task processing time
    pub average_task_time: std::time::Duration,
}

/// Trait for pipeline optimization
/// 
/// This trait provides methods for optimizing pipeline performance
/// and resource usage.
pub trait PipelineOptimizer: Send + Sync {
    /// Analyze pipeline performance
    fn analyze_performance(&self, metrics: &[PipelineMetrics]) -> PerformanceAnalysis;

    /// Suggest optimizations
    fn suggest_optimizations(&self, analysis: &PerformanceAnalysis) -> Vec<OptimizationSuggestion>;

    /// Apply optimization automatically
    fn apply_optimization(&self, suggestion: &OptimizationSuggestion) -> NeuralDocFlowResult<()>;
}

/// Performance analysis results
#[derive(Debug, Clone)]
pub struct PerformanceAnalysis {
    /// Overall performance score (0.0 to 1.0)
    pub performance_score: f32,
    /// Identified bottlenecks
    pub bottlenecks: Vec<Bottleneck>,
    /// Resource utilization
    pub resource_utilization: ResourceUsage,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Performance bottleneck information
#[derive(Debug, Clone)]
pub struct Bottleneck {
    /// Stage or component with bottleneck
    pub component: String,
    /// Severity (0.0 to 1.0)
    pub severity: f32,
    /// Description of the issue
    pub description: String,
    /// Suggested solutions
    pub solutions: Vec<String>,
}

/// Optimization suggestion
#[derive(Debug, Clone)]
pub struct OptimizationSuggestion {
    /// Type of optimization
    pub optimization_type: OptimizationType,
    /// Description
    pub description: String,
    /// Expected impact (0.0 to 1.0)
    pub expected_impact: f32,
    /// Implementation complexity (0.0 to 1.0)
    pub complexity: f32,
    /// Configuration changes required
    pub config_changes: HashMap<String, serde_json::Value>,
}

/// Types of optimizations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizationType {
    /// Increase parallelism
    IncreaseParallelism,
    /// Adjust memory allocation
    MemoryOptimization,
    /// Change processing order
    ProcessingOrder,
    /// Enable caching
    Caching,
    /// Model optimization
    ModelOptimization,
    /// Hardware acceleration
    HardwareAcceleration,
    /// Custom optimization
    Custom(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    struct TestPipeline {
        config: Option<PipelineConfig>,
    }

    impl TestPipeline {
        fn new() -> Self {
            Self { config: None }
        }
    }

    #[async_trait]
    impl ProcessorPipeline for TestPipeline {
        async fn process(&self, document: ExtractedDocument) -> Result<ProcessingResult, PipelineError> {
            Ok(ProcessingResult {
                stage_id: "test_pipeline".to_string(),
                document,
                metadata: crate::ProcessingMetadata {
                    timestamp: chrono::Utc::now(),
                    duration: std::time::Duration::from_millis(10),
                    model_versions: HashMap::new(),
                    parameters: HashMap::new(),
                },
                quality_metrics: crate::QualityMetrics {
                    overall_confidence: 0.95,
                    text_quality: 0.9,
                    layout_quality: 0.85,
                    table_quality: 0.8,
                    error_rate: 0.05,
                    processing_speed: 100.0,
                },
            })
        }

        async fn process_batch(&self, documents: Vec<ExtractedDocument>) -> Result<Vec<ProcessingResult>, PipelineError> {
            let mut results = Vec::new();
            for doc in documents {
                results.push(self.process(doc).await?);
            }
            Ok(results)
        }

        fn pipeline_id(&self) -> &str { "test_pipeline" }
        fn name(&self) -> &str { "Test Pipeline" }
        fn version(&self) -> &str { "1.0.0" }

        async fn initialize(&mut self, config: PipelineConfig) -> Result<(), PipelineError> {
            self.config = Some(config);
            Ok(())
        }

        async fn cleanup(&mut self) -> Result<(), PipelineError> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_processor_pipeline_trait() {
        let pipeline = TestPipeline::new();
        assert_eq!(pipeline.pipeline_id(), "test_pipeline");
        assert_eq!(pipeline.name(), "Test Pipeline");

        let doc = ExtractedDocument {
            id: "test".to_string(),
            source_id: "test_source".to_string(),
            metadata: DocumentMetadata {
                title: Some("Test Document".to_string()),
                author: None,
                created_date: None,
                modified_date: None,
                page_count: 1,
                language: Some("en".to_string()),
                keywords: vec![],
                custom_metadata: HashMap::new(),
            },
            content: vec![],
            structure: DocumentStructure {
                sections: vec![],
                hierarchy: vec![],
                table_of_contents: vec![],
            },
            confidence: 1.0,
            metrics: ExtractionMetrics {
                extraction_time: std::time::Duration::from_millis(10),
                pages_processed: 1,
                blocks_extracted: 0,
                memory_used: 1024,
            },
        };

        let result = pipeline.process(doc).await.unwrap();
        assert_eq!(result.stage_id, "test_pipeline");
    }

    #[test]
    fn test_processing_context() {
        let mut context = ProcessingContext::new();
        
        context.set_metadata("test_key", "test_value");
        assert_eq!(context.get_metadata("test_key"), Some("test_value"));

        context.set_stage_data("stage1", serde_json::json!({"data": "value"}));
        assert!(context.get_stage_data("stage1").is_some());

        assert!(!context.is_near_deadline()); // No deadline set
    }

    #[test]
    fn test_optimization_types() {
        assert_eq!(
            OptimizationType::IncreaseParallelism,
            OptimizationType::IncreaseParallelism
        );
        assert_ne!(
            OptimizationType::IncreaseParallelism,
            OptimizationType::MemoryOptimization
        );
    }
}