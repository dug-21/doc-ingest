//! Base traits and structures for neural security models

use neural_doc_flow_core::ProcessingError;
use ruv_fann::{Network, NetworkBuilder, ActivationFunction, TrainingAlgorithm};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Instant;
use std::fs;
use std::sync::Mutex;

/// Base trait for all neural security models
pub trait NeuralSecurityModel: Send + Sync {
    /// Get model name
    fn name(&self) -> &str;
    
    /// Get model version
    fn version(&self) -> &str;
    
    /// Get input size
    fn input_size(&self) -> usize;
    
    /// Get output size
    fn output_size(&self) -> usize;
    
    /// Predict on input features
    fn predict(&self, features: &[f32]) -> Result<Vec<f32>, ProcessingError>;
    
    /// Train the model
    fn train(&mut self, data: &TrainingData) -> Result<TrainingResult, ProcessingError>;
    
    /// Save model to disk
    fn save(&self, path: &Path) -> Result<(), ProcessingError>;
    
    /// Load model from disk
    fn load(&mut self, path: &Path) -> Result<(), ProcessingError>;
    
    /// Get model metrics
    fn get_metrics(&self) -> ModelMetrics;
    
    /// Validate model performance
    fn validate(&self, test_data: &TrainingData) -> Result<ValidationResult, ProcessingError>;
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub version: String,
    pub input_size: usize,
    pub output_size: usize,
    pub hidden_layers: Vec<usize>,
    pub activation_function: String,
    pub learning_rate: f32,
    pub momentum: f32,
    pub target_error: f32,
    pub max_epochs: u32,
    pub simd_enabled: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            name: "unnamed".to_string(),
            version: "1.0.0".to_string(),
            input_size: 128,
            output_size: 1,
            hidden_layers: vec![64, 32],
            activation_function: "sigmoid".to_string(),
            learning_rate: 0.01,
            momentum: 0.9,
            target_error: 0.001,
            max_epochs: 10000,
            simd_enabled: true,
        }
    }
}

/// Training data container
#[derive(Debug, Clone)]
pub struct TrainingData {
    pub inputs: Vec<Vec<f32>>,
    pub outputs: Vec<Vec<f32>>,
    pub labels: Option<Vec<String>>,
}

impl TrainingData {
    /// Create new training data
    pub fn new(inputs: Vec<Vec<f32>>, outputs: Vec<Vec<f32>>) -> Self {
        Self {
            inputs,
            outputs,
            labels: None,
        }
    }
    
    /// Add labels to training data
    pub fn with_labels(mut self, labels: Vec<String>) -> Self {
        self.labels = Some(labels);
        self
    }
    
    /// Split into train and test sets
    pub fn split(&self, test_ratio: f32) -> (TrainingData, TrainingData) {
        let split_idx = ((self.inputs.len() as f32) * (1.0 - test_ratio)) as usize;
        
        let train_data = TrainingData {
            inputs: self.inputs[..split_idx].to_vec(),
            outputs: self.outputs[..split_idx].to_vec(),
            labels: self.labels.as_ref().map(|l| l[..split_idx].to_vec()),
        };
        
        let test_data = TrainingData {
            inputs: self.inputs[split_idx..].to_vec(),
            outputs: self.outputs[split_idx..].to_vec(),
            labels: self.labels.as_ref().map(|l| l[split_idx..].to_vec()),
        };
        
        (train_data, test_data)
    }
}

/// Training result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    pub epochs_trained: u32,
    pub final_error: f32,
    pub training_time_ms: u64,
    pub converged: bool,
}

/// Validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub accuracy: f32,
    pub precision: f32,
    pub recall: f32,
    pub f1_score: f32,
    pub confusion_matrix: Option<Vec<Vec<u32>>>,
}

/// Model performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    pub model_name: String,
    pub accuracy: f32,
    pub inference_time_us: u64,
    pub model_size_bytes: usize,
    pub total_predictions: u64,
    pub last_training: Option<TrainingResult>,
}

/// Base neural model implementation using ruv-FANN
pub struct BaseNeuralModel {
    pub config: ModelConfig,
    pub network: Mutex<Network<f32>>,
    pub metrics: Mutex<ModelMetrics>,
}

impl BaseNeuralModel {
    /// Create a new neural model
    pub fn new(config: ModelConfig) -> Result<Self, ProcessingError> {
        // Build layer sizes
        let mut layers = vec![config.input_size];
        layers.extend(&config.hidden_layers);
        layers.push(config.output_size);
        
        // Create network using NetworkBuilder
        let network = NetworkBuilder::new()
            .layers_from_sizes(&layers)
            .build();
        
        let metrics = ModelMetrics {
            model_name: config.name.clone(),
            accuracy: 0.0,
            inference_time_us: 0,
            model_size_bytes: 0,
            total_predictions: 0,
            last_training: None,
        };
        
        Ok(Self {
            config,
            network: Mutex::new(network),
            metrics: Mutex::new(metrics),
        })
    }
    
    /// Parse activation function from string
    fn parse_activation(name: &str) -> ActivationFunction {
        match name.to_lowercase().as_str() {
            "sigmoid" => ActivationFunction::Sigmoid,
            "tanh" => ActivationFunction::Tanh,
            "relu" => ActivationFunction::ReLU,
            "linear" => ActivationFunction::Linear,
            _ => ActivationFunction::Sigmoid,
        }
    }
    
    /// Run prediction with timing
    pub fn predict_timed(&self, features: &[f32]) -> Result<Vec<f32>, ProcessingError> {
        if features.len() != self.config.input_size {
            return Err(ProcessingError::InvalidInput(
                format!("Expected {} features, got {}", self.config.input_size, features.len())
            ));
        }
        
        let start = Instant::now();
        let output = self.network.lock().unwrap().run(features);
        
        let mut metrics = self.metrics.lock().unwrap();
        metrics.inference_time_us = start.elapsed().as_micros() as u64;
        metrics.total_predictions += 1;
        
        Ok(output)
    }
    
    /// Train the network
    pub fn train_network(&mut self, data: &TrainingData) -> Result<TrainingResult, ProcessingError> {
        let start = Instant::now();
        
        // Train network using available API
        self.network.lock().unwrap().train(
            &data.inputs,
            &data.outputs,
            self.config.learning_rate,
            self.config.max_epochs as usize,
        ).map_err(|e| ProcessingError::Training(e.to_string()))?;
        
        let final_error = 0.0; // TODO: Calculate actual error when MSE is available
        let converged = true; // TODO: Implement proper convergence check
        
        let result = TrainingResult {
            epochs_trained: self.config.max_epochs,
            final_error,
            training_time_ms: start.elapsed().as_millis() as u64,
            converged,
        };
        
        self.metrics.lock().unwrap().last_training = Some(result.clone());
        
        Ok(result)
    }
    
    /// Calculate validation metrics
    pub fn validate_network(&self, test_data: &TrainingData) -> Result<ValidationResult, ProcessingError> {
        let mut correct = 0;
        let mut true_positives = 0;
        let mut false_positives = 0;
        let mut false_negatives = 0;
        let mut true_negatives = 0;
        
        for (input, expected) in test_data.inputs.iter().zip(&test_data.outputs) {
            let output = self.network.lock().unwrap().run(input);
            
            // For binary classification
            if self.config.output_size == 1 {
                let predicted = output[0] > 0.5;
                let actual = expected[0] > 0.5;
                
                if predicted == actual {
                    correct += 1;
                }
                
                match (predicted, actual) {
                    (true, true) => true_positives += 1,
                    (true, false) => false_positives += 1,
                    (false, true) => false_negatives += 1,
                    (false, false) => true_negatives += 1,
                }
            }
        }
        
        let total = test_data.inputs.len() as f32;
        let accuracy = correct as f32 / total;
        
        let precision = if true_positives + false_positives > 0 {
            true_positives as f32 / (true_positives + false_positives) as f32
        } else {
            0.0
        };
        
        let recall = if true_positives + false_negatives > 0 {
            true_positives as f32 / (true_positives + false_negatives) as f32
        } else {
            0.0
        };
        
        let f1_score = if precision + recall > 0.0 {
            2.0 * (precision * recall) / (precision + recall)
        } else {
            0.0
        };
        
        Ok(ValidationResult {
            accuracy,
            precision,
            recall,
            f1_score,
            confusion_matrix: Some(vec![
                vec![true_negatives, false_positives],
                vec![false_negatives, true_positives],
            ]),
        })
    }

    /// Save network to file (placeholder implementation)
    pub fn save_network(&self, _path: &Path) -> Result<(), ProcessingError> {
        // TODO: Implement proper save using ruv_fann I/O
        Ok(())
    }

    /// Load network from file (placeholder implementation)  
    pub fn load_network(&mut self, _path: &Path) -> Result<(), ProcessingError> {
        // TODO: Implement proper load using ruv_fann I/O
        Ok(())
    }
}