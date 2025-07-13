//! Neural processing module
//! 
//! This module contains the core neural processing implementations
//! using ruv-FANN for document enhancement and analysis.

pub mod engine;

use crate::{
    config::ModelType,
    error::{NeuralError, Result},
    traits::{NeuralProcessorTrait, NeuralInference, FeatureExtractor},
    types::{ContentBlock, NeuralFeatures, ConfidenceScore},
};
use ruv_fann::Fann;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

pub use engine::NeuralEngine;

/// Main neural processor implementation
#[derive(Debug)]
pub struct NeuralProcessor {
    /// Active neural networks
    networks: HashMap<ModelType, Arc<Mutex<Fann>>>,
    
    /// Feature extractors for different content types
    extractors: HashMap<String, Box<dyn FeatureExtractor>>,
    
    /// Performance metrics
    inference_stats: Arc<Mutex<HashMap<ModelType, InferenceMetrics>>>,
}

impl NeuralProcessor {
    /// Create a new neural processor
    pub fn new() -> Self {
        Self {
            networks: HashMap::new(),
            extractors: HashMap::new(),
            inference_stats: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Add a neural network for a specific model type
    pub fn add_network(&mut self, model_type: ModelType, network: Fann) {
        self.networks.insert(model_type, Arc::new(Mutex::new(network)));
    }

    /// Add a feature extractor
    pub fn add_extractor(&mut self, content_type: String, extractor: Box<dyn FeatureExtractor>) {
        self.extractors.insert(content_type, extractor);
    }

    /// Get inference statistics
    pub fn get_inference_stats(&self) -> HashMap<ModelType, InferenceMetrics> {
        self.inference_stats.lock().unwrap().clone()
    }

    /// Reset inference statistics
    pub fn reset_stats(&self) {
        self.inference_stats.lock().unwrap().clear();
    }
}

#[async_trait::async_trait]
impl NeuralInference for NeuralProcessor {
    async fn infer(&self, input: &[f32]) -> Result<Vec<f32>> {
        // This would need to specify which model to use
        // For now, just return an error since we need model specification
        Err(NeuralError::InvalidInput("Model type not specified".to_string()))
    }

    async fn infer_batch(&self, inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::new();
        for input in inputs {
            let result = self.infer(input).await?;
            results.push(result);
        }
        Ok(results)
    }

    fn input_size(&self) -> usize {
        // Return size for a default model or error
        64 // Default size
    }

    fn output_size(&self) -> usize {
        // Return size for a default model or error
        8 // Default size
    }

    fn get_inference_stats(&self) -> crate::traits::InferenceStats {
        crate::traits::InferenceStats::default()
    }
}

/// Inference metrics for tracking performance
#[derive(Debug, Clone)]
pub struct InferenceMetrics {
    pub total_inferences: usize,
    pub total_time: std::time::Duration,
    pub average_time: std::time::Duration,
    pub errors: usize,
    pub last_inference: Option<std::time::Instant>,
}

impl Default for InferenceMetrics {
    fn default() -> Self {
        Self {
            total_inferences: 0,
            total_time: std::time::Duration::from_millis(0),
            average_time: std::time::Duration::from_millis(0),
            errors: 0,
            last_inference: None,
        }
    }
}

impl InferenceMetrics {
    /// Record a successful inference
    pub fn record_inference(&mut self, duration: std::time::Duration) {
        self.total_inferences += 1;
        self.total_time += duration;
        self.average_time = self.total_time / self.total_inferences as u32;
        self.last_inference = Some(std::time::Instant::now());
    }

    /// Record an inference error
    pub fn record_error(&mut self) {
        self.errors += 1;
    }

    /// Get error rate
    pub fn error_rate(&self) -> f32 {
        if self.total_inferences == 0 {
            0.0
        } else {
            self.errors as f32 / self.total_inferences as f32
        }
    }
}

/// Network types supported by the neural processor
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkType {
    /// Standard feedforward network
    Standard,
    /// Sparse connected network
    Sparse,
    /// Shortcut connected network
    Shortcut,
    /// Cascade correlation network
    Cascade,
}

impl std::fmt::Display for NetworkType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NetworkType::Standard => write!(f, "standard"),
            NetworkType::Sparse => write!(f, "sparse"),
            NetworkType::Shortcut => write!(f, "shortcut"),
            NetworkType::Cascade => write!(f, "cascade"),
        }
    }
}

/// Neural network configuration for creating new networks
#[derive(Debug, Clone)]
pub struct NetworkConfig {
    /// Network type
    pub network_type: NetworkType,
    
    /// Layer sizes (including input and output)
    pub layers: Vec<u32>,
    
    /// Connection rate for sparse networks (0.0 to 1.0)
    pub connection_rate: Option<f32>,
    
    /// Activation function for hidden layers
    pub activation_function: ActivationFunction,
    
    /// Training algorithm
    pub training_algorithm: TrainingAlgorithm,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            network_type: NetworkType::Standard,
            layers: vec![64, 128, 64, 8],
            connection_rate: None,
            activation_function: ActivationFunction::Sigmoid,
            training_algorithm: TrainingAlgorithm::Rprop,
        }
    }
}

/// Activation functions available in ruv-FANN
#[derive(Debug, Clone, Copy)]
pub enum ActivationFunction {
    Linear,
    Threshold,
    ThresholdSymmetric,
    Sigmoid,
    SigmoidStepwise,
    SigmoidSymmetric,
    SigmoidSymmetricStepwise,
    Gaussian,
    GaussianSymmetric,
    Elliot,
    ElliotSymmetric,
    LinearPiece,
    LinearPieceSymmetric,
    SinSymmetric,
    CosSymmetric,
    Sin,
    Cos,
}

impl From<ActivationFunction> for ruv_fann::ActivationFunc {
    fn from(af: ActivationFunction) -> Self {
        match af {
            ActivationFunction::Linear => ruv_fann::ActivationFunc::Linear,
            ActivationFunction::Threshold => ruv_fann::ActivationFunc::Threshold,
            ActivationFunction::ThresholdSymmetric => ruv_fann::ActivationFunc::ThresholdSymmetric,
            ActivationFunction::Sigmoid => ruv_fann::ActivationFunc::Sigmoid,
            ActivationFunction::SigmoidStepwise => ruv_fann::ActivationFunc::SigmoidStepwise,
            ActivationFunction::SigmoidSymmetric => ruv_fann::ActivationFunc::SigmoidSymmetric,
            ActivationFunction::SigmoidSymmetricStepwise => ruv_fann::ActivationFunc::SigmoidSymmetricStepwise,
            ActivationFunction::Gaussian => ruv_fann::ActivationFunc::Gaussian,
            ActivationFunction::GaussianSymmetric => ruv_fann::ActivationFunc::GaussianSymmetric,
            ActivationFunction::Elliot => ruv_fann::ActivationFunc::Elliot,
            ActivationFunction::ElliotSymmetric => ruv_fann::ActivationFunc::ElliotSymmetric,
            ActivationFunction::LinearPiece => ruv_fann::ActivationFunc::LinearPiece,
            ActivationFunction::LinearPieceSymmetric => ruv_fann::ActivationFunc::LinearPieceSymmetric,
            ActivationFunction::SinSymmetric => ruv_fann::ActivationFunc::SinSymmetric,
            ActivationFunction::CosSymmetric => ruv_fann::ActivationFunc::CosSymmetric,
            ActivationFunction::Sin => ruv_fann::ActivationFunc::Sin,
            ActivationFunction::Cos => ruv_fann::ActivationFunc::Cos,
        }
    }
}

/// Training algorithms available in ruv-FANN
#[derive(Debug, Clone, Copy)]
pub enum TrainingAlgorithm {
    Incremental,
    Batch,
    Rprop,
    Quickprop,
    Sarprop,
}

impl From<TrainingAlgorithm> for ruv_fann::TrainingAlgorithm {
    fn from(ta: TrainingAlgorithm) -> Self {
        match ta {
            TrainingAlgorithm::Incremental => ruv_fann::TrainingAlgorithm::Incremental,
            TrainingAlgorithm::Batch => ruv_fann::TrainingAlgorithm::Batch,
            TrainingAlgorithm::Rprop => ruv_fann::TrainingAlgorithm::Rprop,
            TrainingAlgorithm::Quickprop => ruv_fann::TrainingAlgorithm::Quickprop,
            TrainingAlgorithm::Sarprop => ruv_fann::TrainingAlgorithm::Sarprop,
        }
    }
}

/// Neural network factory for creating new networks
pub struct NetworkFactory;

impl NetworkFactory {
    /// Create a new neural network based on configuration
    pub fn create_network(config: &NetworkConfig) -> Result<Fann> {
        let network = match config.network_type {
            NetworkType::Standard => {
                Fann::new_standard(&config.layers)
                    .map_err(|e| NeuralError::NetworkCreation(format!("Standard network: {}", e)))?
            }
            NetworkType::Sparse => {
                let connection_rate = config.connection_rate.unwrap_or(0.8);
                Fann::new_sparse(connection_rate, &config.layers)
                    .map_err(|e| NeuralError::NetworkCreation(format!("Sparse network: {}", e)))?
            }
            NetworkType::Shortcut => {
                Fann::new_shortcut(&config.layers)
                    .map_err(|e| NeuralError::NetworkCreation(format!("Shortcut network: {}", e)))?
            }
            NetworkType::Cascade => {
                // Cascade networks are created differently in ruv-FANN
                // For now, create a standard network and note this limitation
                Fann::new_standard(&config.layers)
                    .map_err(|e| NeuralError::NetworkCreation(format!("Cascade network (fallback to standard): {}", e)))?
            }
        };

        Ok(network)
    }

    /// Create a network optimized for text processing
    pub fn create_text_network() -> Result<Fann> {
        let config = NetworkConfig {
            network_type: NetworkType::Standard,
            layers: vec![64, 128, 64, 32, 8],
            activation_function: ActivationFunction::Sigmoid,
            training_algorithm: TrainingAlgorithm::Rprop,
            ..Default::default()
        };
        Self::create_network(&config)
    }

    /// Create a network optimized for layout analysis
    pub fn create_layout_network() -> Result<Fann> {
        let config = NetworkConfig {
            network_type: NetworkType::Shortcut,
            layers: vec![128, 256, 128, 64, 16],
            activation_function: ActivationFunction::Sigmoid,
            training_algorithm: TrainingAlgorithm::Rprop,
            ..Default::default()
        };
        Self::create_network(&config)
    }

    /// Create a network optimized for table detection
    pub fn create_table_network() -> Result<Fann> {
        let config = NetworkConfig {
            network_type: NetworkType::Standard,
            layers: vec![32, 64, 32, 16, 4],
            activation_function: ActivationFunction::Sigmoid,
            training_algorithm: TrainingAlgorithm::Rprop,
            ..Default::default()
        };
        Self::create_network(&config)
    }

    /// Create a network optimized for quality assessment
    pub fn create_quality_network() -> Result<Fann> {
        let config = NetworkConfig {
            network_type: NetworkType::Standard,
            layers: vec![16, 32, 16, 8, 4],
            activation_function: ActivationFunction::Sigmoid,
            training_algorithm: TrainingAlgorithm::Rprop,
            ..Default::default()
        };
        Self::create_network(&config)
    }
}

/// Model inference helper functions
pub mod inference {
    use super::*;

    /// Run inference on a neural network with error handling
    pub async fn run_inference(
        network: Arc<Mutex<Fann>>,
        input: &[f32],
        model_type: ModelType,
        stats: Arc<Mutex<HashMap<ModelType, InferenceMetrics>>>,
    ) -> Result<Vec<f32>> {
        let start_time = std::time::Instant::now();

        let result = tokio::task::spawn_blocking(move || {
            let mut net = network.lock().unwrap();
            net.run(input)
        }).await;

        let duration = start_time.elapsed();

        // Update statistics
        {
            let mut stats_map = stats.lock().unwrap();
            let metrics = stats_map.entry(model_type).or_insert_with(InferenceMetrics::default);
            
            match &result {
                Ok(_) => metrics.record_inference(duration),
                Err(_) => metrics.record_error(),
            }
        }

        result
            .map_err(|e| NeuralError::Inference(format!("Task join error: {}", e)))?
            .map_err(|e| NeuralError::Inference(format!("ruv-FANN error: {}", e)))
    }

    /// Run batch inference with parallel processing
    pub async fn run_batch_inference(
        network: Arc<Mutex<Fann>>,
        inputs: &[Vec<f32>],
        model_type: ModelType,
        stats: Arc<Mutex<HashMap<ModelType, InferenceMetrics>>>,
    ) -> Result<Vec<Vec<f32>>> {
        use futures::future::try_join_all;

        let futures: Vec<_> = inputs
            .iter()
            .map(|input| {
                run_inference(
                    Arc::clone(&network),
                    input,
                    model_type.clone(),
                    Arc::clone(&stats),
                )
            })
            .collect();

        try_join_all(futures).await
    }
}

/// Utility functions for neural processing
pub mod utils {
    use super::*;

    /// Normalize input data to 0-1 range
    pub fn normalize_input(data: &mut [f32]) {
        if data.is_empty() {
            return;
        }

        let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        
        if max_val > min_val {
            let range = max_val - min_val;
            for value in data.iter_mut() {
                *value = (*value - min_val) / range;
            }
        }
    }

    /// Apply softmax to output probabilities
    pub fn softmax(output: &mut [f32]) {
        if output.is_empty() {
            return;
        }

        // Find maximum value for numerical stability
        let max_val = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        
        // Compute exponentials
        for value in output.iter_mut() {
            *value = (*value - max_val).exp();
        }

        // Normalize
        let sum: f32 = output.iter().sum();
        if sum > 0.0 {
            for value in output.iter_mut() {
                *value /= sum;
            }
        }
    }

    /// Convert confidence scores to quality assessment
    pub fn confidence_to_quality(confidence: f32) -> QualityLevel {
        match confidence {
            c if c >= 0.9 => QualityLevel::Excellent,
            c if c >= 0.8 => QualityLevel::Good,
            c if c >= 0.7 => QualityLevel::Fair,
            c if c >= 0.6 => QualityLevel::Poor,
            _ => QualityLevel::VeryPoor,
        }
    }

    /// Quality levels based on confidence scores
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum QualityLevel {
        Excellent,
        Good,
        Fair,
        Poor,
        VeryPoor,
    }

    impl std::fmt::Display for QualityLevel {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                QualityLevel::Excellent => write!(f, "excellent"),
                QualityLevel::Good => write!(f, "good"),
                QualityLevel::Fair => write!(f, "fair"),
                QualityLevel::Poor => write!(f, "poor"),
                QualityLevel::VeryPoor => write!(f, "very_poor"),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neural_processor_creation() {
        let processor = NeuralProcessor::new();
        assert_eq!(processor.networks.len(), 0);
        assert_eq!(processor.extractors.len(), 0);
    }

    #[test]
    fn test_network_factory() {
        let text_network = NetworkFactory::create_text_network();
        assert!(text_network.is_ok());

        let layout_network = NetworkFactory::create_layout_network();
        assert!(layout_network.is_ok());

        let table_network = NetworkFactory::create_table_network();
        assert!(table_network.is_ok());

        let quality_network = NetworkFactory::create_quality_network();
        assert!(quality_network.is_ok());
    }

    #[test]
    fn test_inference_metrics() {
        let mut metrics = InferenceMetrics::default();
        assert_eq!(metrics.total_inferences, 0);
        assert_eq!(metrics.error_rate(), 0.0);

        metrics.record_inference(std::time::Duration::from_millis(100));
        assert_eq!(metrics.total_inferences, 1);
        assert_eq!(metrics.average_time, std::time::Duration::from_millis(100));

        metrics.record_error();
        assert_eq!(metrics.errors, 1);
        assert_eq!(metrics.error_rate(), 0.5);
    }

    #[test]
    fn test_utils_normalization() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        utils::normalize_input(&mut data);
        
        assert!((data[0] - 0.0).abs() < f32::EPSILON);
        assert!((data[4] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_utils_softmax() {
        let mut output = vec![1.0, 2.0, 3.0];
        utils::softmax(&mut output);
        
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < f32::EPSILON);
        assert!(output[2] > output[1] && output[1] > output[0]);
    }

    #[test]
    fn test_confidence_to_quality() {
        assert_eq!(utils::confidence_to_quality(0.95), utils::QualityLevel::Excellent);
        assert_eq!(utils::confidence_to_quality(0.85), utils::QualityLevel::Good);
        assert_eq!(utils::confidence_to_quality(0.75), utils::QualityLevel::Fair);
        assert_eq!(utils::confidence_to_quality(0.65), utils::QualityLevel::Poor);
        assert_eq!(utils::confidence_to_quality(0.5), utils::QualityLevel::VeryPoor);
    }
}