//! Core traits for neural document processing plugins

use async_trait::async_trait;
use std::collections::HashMap;
use uuid::Uuid;

use crate::types::{Document, ExtractedContent, ProcessingResult, NeuralEnhancement};
use crate::error::{CoreError, Result};

/// Trait for document source plugins
/// 
/// This trait defines the interface for extracting content from various document sources.
/// Sources can include PDF files, DOCX documents, HTML pages, images, and custom formats.
#[async_trait]
pub trait DocumentSource: Send + Sync {
    /// Extract content from a document
    async fn extract(&self, document: &Document) -> Result<ExtractedContent>;
    
    /// Check if this source can handle the given document
    fn can_handle(&self, document: &Document) -> bool;
    
    /// Get supported MIME types
    fn supported_mime_types(&self) -> Vec<&'static str>;
    
    /// Get source plugin name
    fn name(&self) -> &'static str;
    
    /// Get source plugin version
    fn version(&self) -> &'static str;
    
    /// Initialize the source plugin
    async fn initialize(&mut self) -> Result<()> {
        Ok(())
    }
    
    /// Shutdown the source plugin
    async fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Trait for document processors
/// 
/// Document processors transform extracted content, applying various enhancements
/// and validations to improve accuracy and structure.
#[async_trait]
pub trait DocumentProcessor: Send + Sync {
    /// Process extracted content
    async fn process(&self, content: ExtractedContent) -> Result<ExtractedContent>;
    
    /// Get processor name
    fn name(&self) -> &'static str;
    
    /// Get processor version
    fn version(&self) -> &'static str;
    
    /// Check if processor can handle the content type
    fn can_process(&self, content: &ExtractedContent) -> bool;
    
    /// Get processing priority (higher values process first)
    fn priority(&self) -> i32 {
        0
    }
    
    /// Initialize the processor
    async fn initialize(&mut self) -> Result<()> {
        Ok(())
    }
    
    /// Shutdown the processor
    async fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Trait for neural enhancement plugins
/// 
/// Neural enhancers use AI models (via ruv-FANN) to improve extraction accuracy,
/// correct errors, and enhance content understanding.
#[async_trait]
pub trait NeuralEnhancer: Send + Sync {
    /// Enhance extracted content using neural models
    async fn enhance(&self, content: ExtractedContent) -> Result<(ExtractedContent, Vec<NeuralEnhancement>)>;
    
    /// Get enhancer name
    fn name(&self) -> &'static str;
    
    /// Get enhancer version
    fn version(&self) -> &'static str;
    
    /// Get model configuration
    fn model_config(&self) -> ModelConfig;
    
    /// Check if enhancer can handle the content type
    fn can_enhance(&self, content: &ExtractedContent) -> bool;
    
    /// Train the neural model with new data
    async fn train(&mut self, training_data: Vec<TrainingExample>) -> Result<()>;
    
    /// Load pre-trained model
    async fn load_model(&mut self, model_path: &str) -> Result<()>;
    
    /// Save trained model
    async fn save_model(&self, model_path: &str) -> Result<()>;
    
    /// Initialize the neural enhancer
    async fn initialize(&mut self) -> Result<()> {
        Ok(())
    }
    
    /// Shutdown the enhancer
    async fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Trait for output formatters
/// 
/// Formatters transform processed content into user-specified output formats
/// such as JSON, XML, CSV, or custom schemas.
#[async_trait]
pub trait OutputFormatter: Send + Sync {
    /// Format processed content
    async fn format(&self, content: &ExtractedContent, options: &FormatOptions) -> Result<FormattedOutput>;
    
    /// Get formatter name
    fn name(&self) -> &'static str;
    
    /// Get supported output formats
    fn supported_formats(&self) -> Vec<OutputFormat>;
    
    /// Validate format options
    fn validate_options(&self, options: &FormatOptions) -> Result<()>;
    
    /// Initialize the formatter
    async fn initialize(&mut self) -> Result<()> {
        Ok(())
    }
    
    /// Shutdown the formatter
    async fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Trait for validation plugins
/// 
/// Validators check the quality and accuracy of extracted content,
/// providing confidence scores and identifying potential errors.
#[async_trait]
pub trait ContentValidator: Send + Sync {
    /// Validate extracted content
    async fn validate(&self, content: &ExtractedContent) -> Result<ValidationResult>;
    
    /// Get validator name
    fn name(&self) -> &'static str;
    
    /// Get validation rules
    fn validation_rules(&self) -> Vec<ValidationRule>;
    
    /// Check if validator can handle the content type
    fn can_validate(&self, content: &ExtractedContent) -> bool;
    
    /// Initialize the validator
    async fn initialize(&mut self) -> Result<()> {
        Ok(())
    }
    
    /// Shutdown the validator
    async fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Configuration for neural models
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_type: ModelType,
    pub input_size: usize,
    pub hidden_layers: Vec<usize>,
    pub output_size: usize,
    pub learning_rate: f32,
    pub activation_function: ActivationFunction,
    pub training_algorithm: TrainingAlgorithm,
    pub use_simd: bool,
}

/// Type of neural model
#[derive(Debug, Clone)]
pub enum ModelType {
    FeedForward,
    Recurrent,
    Convolutional,
    Transformer,
}

/// Activation function for neural networks
#[derive(Debug, Clone)]
pub enum ActivationFunction {
    Sigmoid,
    Tanh,
    ReLU,
    LeakyReLU,
    Softmax,
}

/// Training algorithm for neural networks
#[derive(Debug, Clone)]
pub enum TrainingAlgorithm {
    Backpropagation,
    GeneticAlgorithm,
    AdaGrad,
    Adam,
}

/// Training example for neural enhancement
#[derive(Debug, Clone)]
pub struct TrainingExample {
    pub input: ExtractedContent,
    pub expected_output: ExtractedContent,
    pub weight: f32,
}

/// Options for output formatting
#[derive(Debug, Clone)]
pub struct FormatOptions {
    pub format: OutputFormat,
    pub schema: Option<String>,
    pub custom_fields: HashMap<String, String>,
    pub include_metadata: bool,
    pub include_confidence: bool,
    pub pretty_print: bool,
}

/// Supported output formats
#[derive(Debug, Clone)]
pub enum OutputFormat {
    Json,
    Xml,
    Csv,
    Yaml,
    Html,
    Markdown,
    Custom(String),
}

/// Formatted output result
#[derive(Debug, Clone)]
pub struct FormattedOutput {
    pub format: OutputFormat,
    pub data: Vec<u8>,
    pub metadata: HashMap<String, String>,
}

/// Result of content validation
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub confidence_score: f32,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
    pub suggestions: Vec<ValidationSuggestion>,
}

/// Validation rule
#[derive(Debug, Clone)]
pub struct ValidationRule {
    pub rule_id: String,
    pub description: String,
    pub severity: ValidationSeverity,
    pub rule_type: ValidationRuleType,
}

/// Validation error
#[derive(Debug, Clone)]
pub struct ValidationError {
    pub error_id: String,
    pub message: String,
    pub block_id: Option<Uuid>,
    pub severity: ValidationSeverity,
    pub suggestions: Vec<String>,
}

/// Validation warning
#[derive(Debug, Clone)]
pub struct ValidationWarning {
    pub warning_id: String,
    pub message: String,
    pub block_id: Option<Uuid>,
}

/// Validation suggestion
#[derive(Debug, Clone)]
pub struct ValidationSuggestion {
    pub suggestion_id: String,
    pub message: String,
    pub block_id: Option<Uuid>,
    pub confidence: f32,
}

/// Severity of validation issues
#[derive(Debug, Clone)]
pub enum ValidationSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Type of validation rule
#[derive(Debug, Clone)]
pub enum ValidationRuleType {
    Structure,
    Content,
    Format,
    Accuracy,
    Completeness,
    Consistency,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_type: ModelType::FeedForward,
            input_size: 100,
            hidden_layers: vec![50, 25],
            output_size: 10,
            learning_rate: 0.001,
            activation_function: ActivationFunction::ReLU,
            training_algorithm: TrainingAlgorithm::Adam,
            use_simd: true,
        }
    }
}

impl Default for FormatOptions {
    fn default() -> Self {
        Self {
            format: OutputFormat::Json,
            schema: None,
            custom_fields: HashMap::new(),
            include_metadata: true,
            include_confidence: true,
            pretty_print: true,
        }
    }
}