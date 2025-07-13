//! Task definitions for DAA coordination

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use std::collections::HashMap;

use crate::types::{Document, ExtractedContent, ProcessingResult};
use crate::traits::{ValidationResult, FormattedOutput, FormatOptions};
use crate::coordination::{BatchConfig, ProcessingOptions};

/// Tasks that can be executed by DAA agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DocumentTask {
    /// Process a single document
    ProcessDocument {
        document: Document,
        options: ProcessingOptions,
    },
    
    /// Coordinate batch processing of multiple documents
    CoordinateBatch {
        documents: Vec<Document>,
        config: BatchConfig,
    },
    
    /// Extract content from a document (specialized extraction)
    ExtractContent {
        document: Document,
        extraction_type: ExtractionTaskType,
        options: ExtractionOptions,
    },
    
    /// Validate extracted content
    ValidateContent {
        content: ExtractedContent,
    },
    
    /// Enhance content using neural models
    EnhanceContent {
        content: ExtractedContent,
        enhancement_type: EnhancementTaskType,
        model_config: NeuralModelConfig,
    },
    
    /// Format output for delivery
    FormatOutput {
        content: ExtractedContent,
        options: FormatOptions,
    },
    
    /// Aggregate results from multiple agents
    AggregateResults {
        results: Vec<ProcessingResult>,
        aggregation_strategy: AggregationStrategy,
    },
    
    /// Perform consensus validation
    ConsensusValidation {
        candidates: Vec<ExtractedContent>,
        validation_criteria: ValidationCriteria,
    },
    
    /// Train neural model
    TrainModel {
        training_data: TrainingDataSet,
        model_config: NeuralModelConfig,
        target_accuracy: f32,
    },
    
    /// Monitor agent performance
    MonitorPerformance {
        agent_ids: Vec<Uuid>,
        metrics: Vec<PerformanceMetric>,
    },
}

/// Result of task execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskResult {
    /// Document processing completed
    ProcessingComplete(ProcessingResult),
    
    /// Batch coordination completed
    BatchCoordinated(Vec<ProcessingResult>),
    
    /// Content extraction completed
    ExtractionComplete(ExtractedContent),
    
    /// Validation completed
    ValidationComplete(ValidationResult),
    
    /// Neural enhancement completed
    EnhancementComplete {
        enhanced_content: ExtractedContent,
        confidence_improvement: f32,
        processing_time_ms: u32,
    },
    
    /// Output formatting completed
    FormattingComplete(FormattedOutput),
    
    /// Result aggregation completed
    AggregationComplete(ProcessingResult),
    
    /// Consensus reached
    ConsensusReached {
        consensus_content: ExtractedContent,
        confidence: f32,
        participating_agents: Vec<Uuid>,
    },
    
    /// Model training completed
    TrainingComplete {
        model_id: String,
        final_accuracy: f32,
        training_time_ms: u64,
        iterations: u32,
    },
    
    /// Performance monitoring completed
    MonitoringComplete {
        agent_metrics: HashMap<Uuid, AgentPerformanceReport>,
        system_health: SystemHealthReport,
    },
    
    /// Task failed
    TaskFailed {
        error_message: String,
        error_code: String,
        recoverable: bool,
    },
}

/// Status of a task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskStatus {
    Pending,
    Assigned(Uuid), // Agent ID
    InProgress,
    Completed,
    Failed(String),
    Cancelled,
}

/// Types of extraction tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExtractionTaskType {
    FullDocument,
    TextOnly,
    TablesOnly,
    ImagesOnly,
    MetadataOnly,
    StructureOnly,
    SelectiveExtraction(Vec<String>), // Specify which elements to extract
}

/// Options for extraction tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionOptions {
    pub enable_ocr: bool,
    pub ocr_language: Option<String>,
    pub extract_fonts: bool,
    pub extract_colors: bool,
    pub preserve_formatting: bool,
    pub detect_language: bool,
    pub confidence_threshold: f32,
    pub timeout_seconds: u32,
    pub custom_extractors: Vec<String>,
}

impl Default for ExtractionOptions {
    fn default() -> Self {
        Self {
            enable_ocr: true,
            ocr_language: None,
            extract_fonts: false,
            extract_colors: false,
            preserve_formatting: true,
            detect_language: true,
            confidence_threshold: 0.8,
            timeout_seconds: 120,
            custom_extractors: vec![],
        }
    }
}

/// Types of neural enhancement tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnhancementTaskType {
    TextCorrection,
    LayoutAnalysis,
    TableStructureDetection,
    ImageAnalysis,
    LanguageDetection,
    QualityAssessment,
    ErrorCorrection,
    ConfidenceScoring,
    SemanticAnalysis,
    ContentClassification,
}

/// Configuration for neural models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralModelConfig {
    pub model_type: String,
    pub model_version: String,
    pub use_pretrained: bool,
    pub enable_simd: bool,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub regularization: f32,
    pub max_iterations: u32,
    pub convergence_threshold: f32,
    pub custom_parameters: HashMap<String, serde_json::Value>,
}

impl Default for NeuralModelConfig {
    fn default() -> Self {
        Self {
            model_type: "feedforward".to_string(),
            model_version: "latest".to_string(),
            use_pretrained: true,
            enable_simd: true,
            batch_size: 32,
            learning_rate: 0.001,
            regularization: 0.01,
            max_iterations: 1000,
            convergence_threshold: 0.001,
            custom_parameters: HashMap::new(),
        }
    }
}

/// Strategies for aggregating results from multiple agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationStrategy {
    /// Take result with highest confidence
    HighestConfidence,
    
    /// Average results from all agents
    Average,
    
    /// Use majority voting for conflicts
    MajorityVote,
    
    /// Weight results by agent performance history
    WeightedByPerformance,
    
    /// Use consensus algorithm
    Consensus {
        min_agreement: f32,
        max_iterations: u32,
    },
    
    /// Custom aggregation logic
    Custom(String),
}

/// Criteria for validation during consensus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationCriteria {
    pub min_confidence: f32,
    pub max_disagreement: f32,
    pub required_validators: usize,
    pub validation_rules: Vec<ValidationRule>,
    pub timeout_seconds: u32,
}

/// Validation rule for consensus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    pub rule_id: String,
    pub description: String,
    pub weight: f32,
    pub critical: bool,
}

/// Training dataset for neural models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingDataSet {
    pub dataset_id: String,
    pub training_examples: Vec<TrainingExample>,
    pub validation_examples: Vec<TrainingExample>,
    pub metadata: HashMap<String, String>,
}

/// Single training example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    pub input: ExtractedContent,
    pub expected_output: ExtractedContent,
    pub weight: f32,
    pub metadata: HashMap<String, String>,
}

/// Performance metrics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceMetric {
    ProcessingTime,
    AccuracyScore,
    ThroughputDocumentsPerSecond,
    MemoryUsage,
    CpuUsage,
    ErrorRate,
    ConfidenceScore,
    NetworkLatency,
    TaskQueueLength,
    AgentUtilization,
}

/// Performance report for an individual agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentPerformanceReport {
    pub agent_id: Uuid,
    pub uptime_seconds: u64,
    pub tasks_completed: u32,
    pub tasks_failed: u32,
    pub average_processing_time_ms: f32,
    pub average_accuracy: f32,
    pub memory_usage_mb: f32,
    pub cpu_usage_percent: f32,
    pub error_rate: f32,
    pub throughput_docs_per_hour: f32,
    pub last_activity: chrono::DateTime<chrono::Utc>,
    pub specializations: Vec<String>,
    pub performance_trend: PerformanceTrend,
}

/// System health report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealthReport {
    pub total_agents: usize,
    pub active_agents: usize,
    pub failed_agents: usize,
    pub pending_tasks: usize,
    pub completed_tasks_last_hour: u32,
    pub system_memory_usage_mb: f32,
    pub system_cpu_usage_percent: f32,
    pub network_latency_ms: f32,
    pub coordination_overhead_percent: f32,
    pub neural_model_accuracy: f32,
    pub overall_health_score: f32,
}

/// Performance trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceTrend {
    Improving,
    Stable,
    Declining,
    Volatile,
    Unknown,
}

impl DocumentTask {
    /// Get task priority based on type and content
    pub fn priority(&self) -> TaskPriority {
        match self {
            DocumentTask::ProcessDocument { .. } => TaskPriority::Normal,
            DocumentTask::CoordinateBatch { .. } => TaskPriority::High,
            DocumentTask::ExtractContent { .. } => TaskPriority::Normal,
            DocumentTask::ValidateContent { .. } => TaskPriority::Low,
            DocumentTask::EnhanceContent { .. } => TaskPriority::Normal,
            DocumentTask::FormatOutput { .. } => TaskPriority::Low,
            DocumentTask::AggregateResults { .. } => TaskPriority::High,
            DocumentTask::ConsensusValidation { .. } => TaskPriority::High,
            DocumentTask::TrainModel { .. } => TaskPriority::Low,
            DocumentTask::MonitorPerformance { .. } => TaskPriority::Low,
        }
    }
    
    /// Get estimated processing time for the task
    pub fn estimated_duration_seconds(&self) -> u32 {
        match self {
            DocumentTask::ProcessDocument { .. } => 30,
            DocumentTask::CoordinateBatch { documents, .. } => documents.len() as u32 * 5,
            DocumentTask::ExtractContent { .. } => 15,
            DocumentTask::ValidateContent { .. } => 5,
            DocumentTask::EnhanceContent { .. } => 20,
            DocumentTask::FormatOutput { .. } => 2,
            DocumentTask::AggregateResults { results, .. } => results.len() as u32,
            DocumentTask::ConsensusValidation { candidates, .. } => candidates.len() as u32 * 3,
            DocumentTask::TrainModel { .. } => 3600, // 1 hour
            DocumentTask::MonitorPerformance { agent_ids, .. } => agent_ids.len() as u32,
        }
    }
    
    /// Check if task can be parallelized
    pub fn is_parallelizable(&self) -> bool {
        matches!(
            self,
            DocumentTask::CoordinateBatch { .. } |
            DocumentTask::AggregateResults { .. } |
            DocumentTask::ConsensusValidation { .. } |
            DocumentTask::MonitorPerformance { .. }
        )
    }
}

/// Task priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TaskPriority {
    Low,
    Normal,
    High,
    Critical,
}

impl Default for ValidationCriteria {
    fn default() -> Self {
        Self {
            min_confidence: 0.8,
            max_disagreement: 0.2,
            required_validators: 3,
            validation_rules: vec![],
            timeout_seconds: 60,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DocumentSource, DocumentMetadata};
    use std::path::PathBuf;
    
    #[test]
    fn test_task_priority() {
        let task = DocumentTask::ProcessDocument {
            document: Document::new(DocumentSource::File { 
                path: PathBuf::from("test.pdf") 
            }),
            options: ProcessingOptions::default(),
        };
        
        assert_eq!(task.priority(), TaskPriority::Normal);
        assert!(!task.is_parallelizable());
    }
    
    #[test]
    fn test_batch_task() {
        let task = DocumentTask::CoordinateBatch {
            documents: vec![
                Document::new(DocumentSource::File { path: PathBuf::from("test1.pdf") }),
                Document::new(DocumentSource::File { path: PathBuf::from("test2.pdf") }),
            ],
            config: BatchConfig::default(),
        };
        
        assert_eq!(task.priority(), TaskPriority::High);
        assert!(task.is_parallelizable());
        assert_eq!(task.estimated_duration_seconds(), 10); // 2 documents * 5 seconds
    }
}