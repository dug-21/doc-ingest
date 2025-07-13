//! Neural processing trait definitions

use crate::{Document, NeuralResult};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Core trait for neural processors
#[async_trait]
pub trait NeuralProcessor: Send + Sync {
    /// Get processor name
    fn name(&self) -> &str;
    
    /// Get processor version
    fn version(&self) -> &str;
    
    /// Process document with neural analysis
    async fn process(&self, document: &Document, config: &NeuralConfig) -> NeuralResult<NeuralAnalysisResult>;
    
    /// Process multiple documents in batch
    async fn process_batch(&self, documents: &[Document], config: &NeuralConfig) -> Vec<NeuralResult<NeuralAnalysisResult>> {
        let mut results = Vec::new();
        for doc in documents {
            results.push(self.process(doc, config).await);
        }
        results
    }
    
    /// Train the neural model with new data
    async fn train(&mut self, training_data: &[TrainingExample], config: &TrainingConfig) -> NeuralResult<TrainingResult>;
    
    /// Evaluate model performance
    async fn evaluate(&self, test_data: &[TestExample]) -> NeuralResult<EvaluationResult>;
    
    /// Get model capabilities
    fn capabilities(&self) -> NeuralCapabilities;
    
    /// Load pre-trained model
    async fn load_model(&mut self, model_path: &str) -> NeuralResult<()>;
    
    /// Save current model
    async fn save_model(&self, model_path: &str) -> NeuralResult<()>;
    
    /// Get model metadata
    fn model_metadata(&self) -> Option<ModelMetadata>;
}

/// Configuration for neural processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralConfig {
    /// Analysis types to perform
    pub analysis_types: Vec<AnalysisType>,
    
    /// Model parameters
    pub model_params: HashMap<String, serde_json::Value>,
    
    /// Confidence threshold (0.0 to 1.0)
    pub confidence_threshold: f64,
    
    /// Maximum processing time (seconds)
    pub timeout_seconds: Option<u64>,
    
    /// Enable GPU acceleration
    pub use_gpu: bool,
    
    /// Batch size for processing
    pub batch_size: Option<usize>,
    
    /// Custom parameters
    pub custom: HashMap<String, serde_json::Value>,
}

/// Types of neural analysis
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AnalysisType {
    /// Content classification
    Classification,
    
    /// Entity recognition
    EntityRecognition,
    
    /// Sentiment analysis
    SentimentAnalysis,
    
    /// Topic modeling
    TopicModeling,
    
    /// Summarization
    Summarization,
    
    /// Question answering
    QuestionAnswering,
    
    /// Language detection
    LanguageDetection,
    
    /// Content embedding generation
    EmbeddingGeneration,
    
    /// Image analysis
    ImageAnalysis,
    
    /// Custom analysis
    Custom(String),
}

/// Result of neural analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralAnalysisResult {
    /// Document ID
    pub document_id: uuid::Uuid,
    
    /// Analysis results by type
    pub analyses: HashMap<AnalysisType, AnalysisResult>,
    
    /// Overall confidence score
    pub confidence: f64,
    
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    
    /// Model used for analysis
    pub model_info: ModelInfo,
    
    /// Any warnings or issues
    pub warnings: Vec<String>,
}

/// Individual analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    /// Analysis type
    pub analysis_type: AnalysisType,
    
    /// Analysis results
    pub results: serde_json::Value,
    
    /// Confidence score for this analysis
    pub confidence: f64,
    
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model name
    pub name: String,
    
    /// Model version
    pub version: String,
    
    /// Model architecture
    pub architecture: String,
    
    /// Model size in parameters
    pub parameter_count: Option<u64>,
    
    /// Training data information
    pub training_data: Option<String>,
    
    /// Model capabilities
    pub capabilities: Vec<String>,
}

/// Training example for model training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    /// Input data
    pub input: TrainingInput,
    
    /// Expected output
    pub expected_output: serde_json::Value,
    
    /// Example weight (importance)
    pub weight: Option<f64>,
    
    /// Example metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Training input data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrainingInput {
    /// Text input
    Text(String),
    
    /// Document input
    Document(Document),
    
    /// Image input
    Image(Vec<u8>),
    
    /// Structured data input
    Structured(serde_json::Value),
    
    /// Multiple inputs
    Multiple(Vec<TrainingInput>),
}

/// Configuration for model training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f64,
    
    /// Number of epochs
    pub epochs: u32,
    
    /// Batch size
    pub batch_size: usize,
    
    /// Validation split (0.0 to 1.0)
    pub validation_split: f64,
    
    /// Early stopping patience
    pub early_stopping_patience: Option<u32>,
    
    /// Model save frequency
    pub save_frequency: Option<u32>,
    
    /// Enable mixed precision training
    pub mixed_precision: bool,
    
    /// Custom training parameters
    pub custom: HashMap<String, serde_json::Value>,
}

/// Training result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    /// Training was successful
    pub success: bool,
    
    /// Final training loss
    pub final_loss: f64,
    
    /// Final validation loss
    pub validation_loss: Option<f64>,
    
    /// Training accuracy
    pub accuracy: Option<f64>,
    
    /// Number of epochs completed
    pub epochs_completed: u32,
    
    /// Training time in seconds
    pub training_time_seconds: u64,
    
    /// Model path (if saved)
    pub model_path: Option<String>,
    
    /// Training metrics history
    pub metrics_history: Vec<TrainingMetrics>,
}

/// Training metrics for an epoch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Epoch number
    pub epoch: u32,
    
    /// Training loss
    pub loss: f64,
    
    /// Validation loss
    pub validation_loss: Option<f64>,
    
    /// Accuracy
    pub accuracy: Option<f64>,
    
    /// Learning rate used
    pub learning_rate: f64,
    
    /// Epoch duration in seconds
    pub duration_seconds: f64,
}

/// Test example for model evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestExample {
    /// Input data
    pub input: TrainingInput,
    
    /// Ground truth output
    pub ground_truth: serde_json::Value,
    
    /// Example identifier
    pub id: Option<String>,
}

/// Evaluation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResult {
    /// Overall accuracy
    pub accuracy: f64,
    
    /// Precision score
    pub precision: Option<f64>,
    
    /// Recall score
    pub recall: Option<f64>,
    
    /// F1 score
    pub f1_score: Option<f64>,
    
    /// Confusion matrix (if applicable)
    pub confusion_matrix: Option<Vec<Vec<u32>>>,
    
    /// Per-class metrics
    pub class_metrics: HashMap<String, ClassificationMetrics>,
    
    /// Evaluation time in seconds
    pub evaluation_time_seconds: f64,
    
    /// Number of test examples
    pub test_examples_count: usize,
}

/// Classification metrics for individual classes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationMetrics {
    /// Precision for this class
    pub precision: f64,
    
    /// Recall for this class
    pub recall: f64,
    
    /// F1 score for this class
    pub f1_score: f64,
    
    /// Support (number of examples)
    pub support: u32,
}

/// Neural processor capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralCapabilities {
    /// Supported analysis types
    pub supported_analyses: Vec<AnalysisType>,
    
    /// Supported input types
    pub input_types: Vec<String>,
    
    /// Requires GPU
    pub requires_gpu: bool,
    
    /// Supports training
    pub supports_training: bool,
    
    /// Supports fine-tuning
    pub supports_fine_tuning: bool,
    
    /// Maximum input size (tokens/pixels)
    pub max_input_size: Option<usize>,
    
    /// Memory requirements (MB)
    pub memory_requirements: Option<u64>,
    
    /// Supported languages
    pub supported_languages: Vec<String>,
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model identifier
    pub id: String,
    
    /// Model name
    pub name: String,
    
    /// Model version
    pub version: String,
    
    /// Creation date
    pub created_at: chrono::DateTime<chrono::Utc>,
    
    /// Last modified date
    pub modified_at: chrono::DateTime<chrono::Utc>,
    
    /// Model author
    pub author: Option<String>,
    
    /// Model description
    pub description: Option<String>,
    
    /// Model license
    pub license: Option<String>,
    
    /// Model tags
    pub tags: Vec<String>,
    
    /// Model performance metrics
    pub performance: Option<EvaluationResult>,
    
    /// Custom metadata
    pub custom: HashMap<String, serde_json::Value>,
}

/// Factory trait for creating neural processors
pub trait NeuralProcessorFactory: Send + Sync {
    /// Create a new neural processor
    fn create_processor(&self, config: &NeuralProcessorConfig) -> NeuralResult<Box<dyn NeuralProcessor>>;
    
    /// Get processor type
    fn processor_type(&self) -> &'static str;
    
    /// Get configuration schema
    fn config_schema(&self) -> serde_json::Value;
}

/// Configuration for creating neural processors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralProcessorConfig {
    /// Processor type
    pub processor_type: String,
    
    /// Model configuration
    pub model_config: ModelConfig,
    
    /// Hardware configuration
    pub hardware_config: HardwareConfig,
    
    /// Custom parameters
    pub custom: HashMap<String, serde_json::Value>,
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model name or path
    pub model_name: String,
    
    /// Model parameters
    pub parameters: HashMap<String, serde_json::Value>,
    
    /// Use pre-trained weights
    pub use_pretrained: bool,
    
    /// Model architecture overrides
    pub architecture_overrides: Option<HashMap<String, serde_json::Value>>,
}

/// Hardware configuration for neural processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    /// Use GPU if available
    pub use_gpu: bool,
    
    /// Specific GPU device ID
    pub gpu_device_id: Option<u32>,
    
    /// Number of CPU threads
    pub cpu_threads: Option<usize>,
    
    /// Memory limit (MB)
    pub memory_limit_mb: Option<u64>,
    
    /// Enable mixed precision
    pub mixed_precision: bool,
}

impl Default for NeuralConfig {
    fn default() -> Self {
        Self {
            analysis_types: vec![AnalysisType::Classification, AnalysisType::EntityRecognition],
            model_params: HashMap::new(),
            confidence_threshold: 0.5,
            timeout_seconds: Some(300),
            use_gpu: true,
            batch_size: Some(32),
            custom: HashMap::new(),
        }
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            epochs: 10,
            batch_size: 32,
            validation_split: 0.2,
            early_stopping_patience: Some(5),
            save_frequency: Some(1),
            mixed_precision: false,
            custom: HashMap::new(),
        }
    }
}

impl Default for HardwareConfig {
    fn default() -> Self {
        Self {
            use_gpu: true,
            gpu_device_id: None,
            cpu_threads: None,
            memory_limit_mb: None,
            mixed_precision: false,
        }
    }
}