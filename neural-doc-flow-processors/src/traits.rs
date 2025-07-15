//! Traits for neural document flow processors

use crate::{
    error::Result,
    types::{ContentBlock, EnhancedContent, NeuralFeatures, ConfidenceScore},
};
use async_trait::async_trait;
use std::path::Path;

/// Main trait for neural processors
#[async_trait]
pub trait NeuralProcessorTrait: Send + Sync {
    /// Enhance content using neural processing
    async fn enhance_content(&self, content: Vec<ContentBlock>) -> Result<EnhancedContent>;

    /// Process a single content block
    async fn process_content_block(&self, block: ContentBlock) -> Result<ContentBlock>;

    /// Analyze document layout using neural networks
    async fn analyze_layout(&self, features: NeuralFeatures) -> Result<LayoutAnalysis>;

    /// Detect tables in document content
    async fn detect_tables(&self, features: NeuralFeatures) -> Result<Vec<TableRegion>>;

    /// Assess content quality using neural networks
    async fn assess_quality(&self, content: &EnhancedContent) -> Result<ConfidenceScore>;
}

/// Trait for loading and managing neural models
#[async_trait]
pub trait ModelLoader: Send + Sync {
    /// Load a model from file
    async fn load_model(&self, path: &Path, model_id: &str) -> Result<()>;

    /// Save a model to file
    async fn save_model(&self, model_id: &str, path: &Path) -> Result<()>;

    /// Check if a model is loaded
    fn is_model_loaded(&self, model_id: &str) -> bool;

    /// Unload a model from memory
    async fn unload_model(&self, model_id: &str) -> Result<()>;

    /// List all loaded models
    fn list_loaded_models(&self) -> Vec<String>;

    /// Get model metadata
    fn get_model_metadata(&self, model_id: &str) -> Result<ModelMetadata>;
}

/// Trait for content processing components
#[async_trait]
pub trait ContentProcessor: Send + Sync {
    /// Process content and return enhanced version
    async fn process(&self, content: &ContentBlock) -> Result<ContentBlock>;

    /// Get the processor name
    fn name(&self) -> &str;

    /// Get processor version
    fn version(&self) -> &str;

    /// Check if processor can handle this content type
    fn can_process(&self, content_type: &str) -> bool;

    /// Get supported content types
    fn supported_types(&self) -> Vec<String>;
}

/// Trait for feature extraction
pub trait FeatureExtractor: Send + Sync {
    /// Extract features from content block
    fn extract_features(&self, block: &ContentBlock) -> Result<NeuralFeatures>;

    /// Get feature vector size
    fn feature_size(&self) -> usize;

    /// Get feature extractor type
    fn extractor_type(&self) -> &str;
}

/// Trait for model training
#[async_trait]
pub trait ModelTrainer: Send + Sync {
    /// Train a model with given data
    async fn train(&mut self, training_data: &TrainingData) -> Result<TrainingResults>;

    /// Validate a model
    async fn validate(&self, validation_data: &ValidationData) -> Result<ValidationResults>;

    /// Fine-tune an existing model
    async fn fine_tune(&mut self, model_id: &str, data: &TrainingData) -> Result<TrainingResults>;

    /// Get training progress
    fn get_training_progress(&self) -> Option<TrainingProgress>;
}

/// Trait for neural network inference
#[async_trait]
pub trait NeuralInference: Send + Sync {
    /// Run inference on input data
    async fn infer(&self, input: &[f32]) -> Result<Vec<f32>>;

    /// Run batch inference
    async fn infer_batch(&self, inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>>;

    /// Get model input size
    fn input_size(&self) -> usize;

    /// Get model output size
    fn output_size(&self) -> usize;

    /// Get inference statistics
    fn get_inference_stats(&self) -> InferenceStats;
}

/// Trait for quality assessment
#[async_trait]
pub trait QualityAssessor: Send + Sync {
    /// Assess quality of processed content
    async fn assess(&self, content: &EnhancedContent) -> Result<QualityReport>;

    /// Check if content meets quality thresholds
    fn meets_quality_threshold(&self, content: &EnhancedContent, threshold: f32) -> bool;

    /// Suggest improvements for low quality content
    fn suggest_improvements(&self, content: &EnhancedContent) -> Vec<QualityImprovement>;
}

/// Trait for caching neural results
#[async_trait]
pub trait NeuralCache: Send + Sync {
    /// Store processing result in cache
    async fn store(&self, key: &str, result: &EnhancedContent) -> Result<()>;

    /// Retrieve result from cache
    async fn retrieve(&self, key: &str) -> Result<Option<EnhancedContent>>;

    /// Check if key exists in cache
    async fn contains(&self, key: &str) -> bool;

    /// Clear cache
    async fn clear(&self) -> Result<()>;

    /// Get cache statistics
    fn get_stats(&self) -> CacheStats;
}

/// Trait for neural model optimization
pub trait ModelOptimizer: Send + Sync {
    /// Optimize model for inference speed
    fn optimize_for_speed(&self, model_id: &str) -> Result<()>;

    /// Optimize model for memory usage
    fn optimize_for_memory(&self, model_id: &str) -> Result<()>;

    /// Quantize model weights
    fn quantize(&self, model_id: &str, bits: u8) -> Result<()>;

    /// Prune model weights
    fn prune(&self, model_id: &str, threshold: f32) -> Result<()>;

    /// Compress model
    fn compress(&self, model_id: &str) -> Result<()>;
}

/// Trait for performance monitoring
pub trait PerformanceMonitor: Send + Sync {
    /// Record processing time
    fn record_processing_time(&self, component: &str, duration: std::time::Duration);

    /// Record memory usage
    fn record_memory_usage(&self, component: &str, bytes: usize);

    /// Record inference time
    fn record_inference_time(&self, model_id: &str, duration: std::time::Duration);

    /// Get performance metrics
    fn get_metrics(&self) -> PerformanceMetrics;

    /// Reset metrics
    fn reset_metrics(&self);
}

// Supporting types for traits

/// Model metadata information
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub id: String,
    pub name: String,
    pub version: String,
    pub model_type: String,
    pub size_bytes: usize,
    pub input_size: usize,
    pub output_size: usize,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub description: String,
}

/// Training data for model training
#[derive(Debug, Clone)]
pub struct TrainingData {
    pub inputs: Vec<Vec<f32>>,
    pub outputs: Vec<Vec<f32>>,
    pub metadata: std::collections::HashMap<String, String>,
}

impl TrainingData {
    pub fn new() -> Self {
        Self {
            inputs: Vec::new(),
            outputs: Vec::new(),
            metadata: std::collections::HashMap::new(),
        }
    }

    pub fn add_sample(&mut self, input: Vec<f32>, output: Vec<f32>) {
        self.inputs.push(input);
        self.outputs.push(output);
    }

    pub fn size(&self) -> usize {
        self.inputs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty()
    }
}

/// Validation data for model validation
#[derive(Debug, Clone)]
pub struct ValidationData {
    pub inputs: Vec<Vec<f32>>,
    pub expected_outputs: Vec<Vec<f32>>,
    pub metadata: std::collections::HashMap<String, String>,
}

/// Training results
#[derive(Debug, Clone)]
pub struct TrainingResults {
    pub model_id: String,
    pub final_error: f32,
    pub epochs_completed: u32,
    pub training_time: std::time::Duration,
    pub validation_accuracy: Option<f32>,
}

/// Validation results
#[derive(Debug, Clone)]
pub struct ValidationResults {
    pub model_id: String,
    pub accuracy: f32,
    pub mean_squared_error: f32,
    pub predictions: Vec<Vec<f32>>,
    pub confusion_matrix: Option<Vec<Vec<usize>>>,
}

/// Training progress information
#[derive(Debug, Clone)]
pub struct TrainingProgress {
    pub current_epoch: u32,
    pub total_epochs: u32,
    pub current_error: f32,
    pub target_error: f32,
    pub estimated_time_remaining: std::time::Duration,
}

/// Inference statistics
#[derive(Debug, Clone)]
pub struct InferenceStats {
    pub total_inferences: usize,
    pub average_time: std::time::Duration,
    pub total_time: std::time::Duration,
    pub errors: usize,
}

impl Default for InferenceStats {
    fn default() -> Self {
        Self {
            total_inferences: 0,
            average_time: std::time::Duration::from_millis(0),
            total_time: std::time::Duration::from_millis(0),
            errors: 0,
        }
    }
}

/// Layout analysis results
#[derive(Debug, Clone)]
pub struct LayoutAnalysis {
    pub document_structure: DocumentStructure,
    pub confidence: f32,
    pub regions: Vec<LayoutRegion>,
    pub reading_order: Vec<String>,
}

/// Document structure information
#[derive(Debug, Clone)]
pub struct DocumentStructure {
    pub sections: Vec<String>,
    pub hierarchy_level: usize,
    pub reading_order: Vec<usize>,
}

/// Layout region information
#[derive(Debug, Clone)]
pub struct LayoutRegion {
    pub region_type: String,
    pub position: (f32, f32, f32, f32), // x, y, width, height
    pub confidence: f32,
    pub content_blocks: Vec<String>,
}

/// Table region information
#[derive(Debug, Clone)]
pub struct TableRegion {
    pub confidence: f32,
    pub rows: usize,
    pub columns: usize,
    pub position: (f32, f32, f32, f32), // x, y, width, height
    pub cells: Vec<Vec<String>>,
}

/// Quality report
#[derive(Debug, Clone)]
pub struct QualityReport {
    pub overall_score: f32,
    pub text_quality: f32,
    pub layout_quality: f32,
    pub table_quality: f32,
    pub issues: Vec<QualityIssue>,
    pub recommendations: Vec<String>,
}

/// Quality issue
#[derive(Debug, Clone)]
pub struct QualityIssue {
    pub issue_type: String,
    pub severity: QualitySeverity,
    pub description: String,
    pub block_id: Option<String>,
    pub confidence: f32,
}

/// Quality issue severity
#[derive(Debug, Clone)]
pub enum QualitySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Quality improvement suggestion
#[derive(Debug, Clone)]
pub struct QualityImprovement {
    pub improvement_type: String,
    pub description: String,
    pub expected_impact: f32,
    pub block_id: Option<String>,
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub total_entries: usize,
    pub hit_rate: f32,
    pub memory_usage: usize,
    pub eviction_count: usize,
}

/// Performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub processing_times: std::collections::HashMap<String, std::time::Duration>,
    pub memory_usage: std::collections::HashMap<String, usize>,
    pub inference_times: std::collections::HashMap<String, std::time::Duration>,
    pub throughput: f32,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            processing_times: std::collections::HashMap::new(),
            memory_usage: std::collections::HashMap::new(),
            inference_times: std::collections::HashMap::new(),
            throughput: 0.0,
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_data() {
        let mut data = TrainingData::new();
        assert!(data.is_empty());

        data.add_sample(vec![1.0, 2.0], vec![0.5]);
        assert_eq!(data.size(), 1);
        assert!(!data.is_empty());
    }

    #[test]
    fn test_inference_stats() {
        let stats = InferenceStats::default();
        assert_eq!(stats.total_inferences, 0);
        assert_eq!(stats.errors, 0);
    }

    #[test]
    fn test_model_metadata() {
        let metadata = ModelMetadata {
            id: "test_model".to_string(),
            name: "Test Model".to_string(),
            version: "1.0.0".to_string(),
            model_type: "text".to_string(),
            size_bytes: 1024,
            input_size: 64,
            output_size: 8,
            created_at: chrono::Utc::now(),
            description: "A test model".to_string(),
        };

        assert_eq!(metadata.id, "test_model");
        assert_eq!(metadata.input_size, 64);
        assert_eq!(metadata.output_size, 8);
    }
}