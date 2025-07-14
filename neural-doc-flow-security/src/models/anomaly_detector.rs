//! Anomaly detection neural model using autoencoder architecture

use crate::models::base::{BaseNeuralModel, ModelConfig, ModelMetrics, NeuralSecurityModel, TrainingData, TrainingResult, ValidationResult};
use neural_doc_flow_core::ProcessingError;
use std::path::{Path, PathBuf};

/// Neural model for anomaly detection
/// Uses autoencoder to detect unusual document patterns
pub struct AnomalyDetectorModel {
    base_model: BaseNeuralModel,
    model_path: PathBuf,
    anomaly_threshold: f32,
}

impl AnomalyDetectorModel {
    /// Create a new anomaly detector model
    pub fn new() -> Result<Self, ProcessingError> {
        let config = ModelConfig {
            name: "AnomalyDetector".to_string(),
            version: "1.0.0".to_string(),
            input_size: 128,  // Input features
            output_size: 128, // Autoencoder - same as input
            hidden_layers: vec![64, 32, 16, 32, 64], // Bottleneck architecture
            activation_function: "tanh".to_string(),
            learning_rate: 0.01,
            momentum: 0.9,
            target_error: 0.005,
            max_epochs: 20000,
            simd_enabled: true,
        };
        
        let base_model = BaseNeuralModel::new(config)?;
        
        Ok(Self {
            base_model,
            model_path: PathBuf::from("models/anomaly_detector.fann"),
            anomaly_threshold: 0.15, // Reconstruction error threshold
        })
    }
    
    /// Load or create model
    pub fn load_or_create(model_dir: &Path) -> Result<Self, ProcessingError> {
        let mut model = Self::new()?;
        let model_path = model_dir.join("anomaly_detector.fann");
        model.model_path = model_path.clone();
        
        if model_path.exists() {
            model.load(&model_path)?;
        }
        
        Ok(model)
    }
    
    /// Extract features for anomaly detection
    pub fn extract_features(raw_features: &crate::SecurityFeatures) -> Vec<f32> {
        let mut features = Vec::with_capacity(128);
        
        // Statistical features
        features.push((raw_features.file_size as f32).ln() / 20.0);
        features.push(raw_features.header_entropy / 8.0);
        features.push((raw_features.stream_count as f32) / 50.0);
        features.push(raw_features.obfuscation_score);
        
        // Structural anomalies
        features.push((raw_features.embedded_files.len() as f32) / 20.0);
        features.push((raw_features.url_count as f32) / 50.0);
        features.push((raw_features.suspicious_keywords.len() as f32) / 30.0);
        
        // Binary features
        features.push(if raw_features.javascript_present { 1.0 } else { 0.0 });
        features.push(if raw_features.header_entropy > 7.5 { 1.0 } else { 0.0 });
        features.push(if raw_features.file_size > 50_000_000 { 1.0 } else { 0.0 });
        features.push(if raw_features.file_size < 1000 { 1.0 } else { 0.0 });
        
        // Ratio features
        let embedded_ratio = raw_features.embedded_files.len() as f32 / 
            (raw_features.stream_count as f32).max(1.0);
        features.push(embedded_ratio);
        
        let keyword_density = raw_features.suspicious_keywords.len() as f32 / 
            (raw_features.file_size as f32 / 1000.0).max(1.0);
        features.push(keyword_density.min(1.0));
        
        // Entropy deviations
        let normal_entropy = 5.5;
        let entropy_deviation = (raw_features.header_entropy - normal_entropy).abs() / 3.0;
        features.push(entropy_deviation.min(1.0));
        
        // File type anomalies
        let executable_count = raw_features.embedded_files.iter()
            .filter(|f| {
                let ft = f.file_type.to_lowercase();
                ft.contains("exe") || ft.contains("dll") || ft.contains("scr")
            }).count();
        features.push((executable_count as f32) / 5.0);
        
        // Pattern complexity
        let unique_keywords = raw_features.suspicious_keywords.len();
        let pattern_complexity = if unique_keywords > 0 {
            (unique_keywords as f32).ln() / 3.0
        } else {
            0.0
        };
        features.push(pattern_complexity.min(1.0));
        
        // Size distribution features
        if !raw_features.embedded_files.is_empty() {
            let sizes: Vec<f32> = raw_features.embedded_files.iter()
                .map(|f| f.size as f32)
                .collect();
            
            let mean_size = sizes.iter().sum::<f32>() / sizes.len() as f32;
            let variance = sizes.iter()
                .map(|&s| (s - mean_size).powi(2))
                .sum::<f32>() / sizes.len() as f32;
            
            features.push((mean_size.ln() / 20.0).min(1.0));
            features.push((variance.sqrt() / mean_size).min(1.0));
        } else {
            features.push(0.0);
            features.push(0.0);
        }
        
        // Behavioral anomaly indicators
        let high_risk_combination = 
            raw_features.javascript_present && 
            raw_features.obfuscation_score > 0.5 &&
            raw_features.url_count > 10;
        features.push(if high_risk_combination { 1.0 } else { 0.0 });
        
        // Pad to expected size
        while features.len() < 128 {
            features.push(0.0);
        }
        
        features
    }
    
    /// Detect anomalies by measuring reconstruction error
    pub fn detect(&self, features: &[f32]) -> Result<f32, ProcessingError> {
        let reconstruction = self.base_model.network.lock().unwrap().run(features);
        
        // Calculate mean squared error between input and reconstruction
        let mse = features.iter().zip(reconstruction.iter())
            .map(|(&input, &output)| (input - output).powi(2))
            .sum::<f32>() / features.len() as f32;
        
        // Normalize to 0-1 range
        let anomaly_score = (mse / self.anomaly_threshold).min(1.0);
        
        Ok(anomaly_score)
    }
    
    /// Get detailed anomaly analysis
    pub fn analyze_anomaly(&self, features: &[f32]) -> Result<AnomalyAnalysis, ProcessingError> {
        let reconstruction = self.base_model.network.lock().unwrap().run(features);
        
        // Calculate per-feature reconstruction errors
        let feature_errors: Vec<f32> = features.iter().zip(reconstruction.iter())
            .map(|(&input, &output)| (input - output).abs())
            .collect();
        
        // Find most anomalous features
        let mut indexed_errors: Vec<(usize, f32)> = feature_errors.iter()
            .enumerate()
            .map(|(i, &e)| (i, e))
            .collect();
        indexed_errors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        let top_anomalies: Vec<(usize, f32)> = indexed_errors.iter()
            .take(10)
            .cloned()
            .collect();
        
        let total_error: f32 = feature_errors.iter().sum();
        let mean_error = total_error / feature_errors.len() as f32;
        let anomaly_score = (mean_error / self.anomaly_threshold).min(1.0);
        
        Ok(AnomalyAnalysis {
            anomaly_score,
            is_anomalous: anomaly_score > 0.5,
            reconstruction_error: mean_error,
            top_anomalous_features: top_anomalies,
            confidence: 1.0 - (mean_error - self.anomaly_threshold).abs() / self.anomaly_threshold,
        })
    }
    
    /// Advanced anomaly detection with statistical analysis
    pub fn detect_advanced_anomalies(&self, features: &[f32], raw_features: &crate::SecurityFeatures) -> Result<AdvancedAnomalyResult, ProcessingError> {
        let basic_analysis = self.analyze_anomaly(features)?;
        
        // Additional statistical anomaly indicators
        let size_anomaly = self.detect_size_anomaly(raw_features);
        let entropy_anomaly = self.detect_entropy_anomaly(raw_features);
        let structure_anomaly = self.detect_structure_anomaly(raw_features);
        
        // Combined anomaly score
        let combined_score = (basic_analysis.anomaly_score * 0.5 + 
                            size_anomaly * 0.2 + 
                            entropy_anomaly * 0.2 + 
                            structure_anomaly * 0.1).min(1.0);
        
        let severity = match combined_score {
            s if s > 0.9 => AnomalySeverity::Critical,
            s if s > 0.7 => AnomalySeverity::High,
            s if s > 0.5 => AnomalySeverity::Medium,
            s if s > 0.3 => AnomalySeverity::Low,
            _ => AnomalySeverity::Normal,
        };
        
        Ok(AdvancedAnomalyResult {
            basic_analysis,
            combined_score,
            size_anomaly_score: size_anomaly,
            entropy_anomaly_score: entropy_anomaly,
            structure_anomaly_score: structure_anomaly,
            severity,
            anomaly_indicators: self.get_anomaly_indicators(raw_features),
        })
    }
    
    fn detect_size_anomaly(&self, features: &crate::SecurityFeatures) -> f32 {
        // Detect unusual file sizes
        match features.file_size {
            size if size > 100_000_000 => 0.8, // Very large files
            size if size < 100 => 0.6,         // Very small files
            size if size > 50_000_000 => 0.4,  // Large files
            _ => 0.0,
        }
    }
    
    fn detect_entropy_anomaly(&self, features: &crate::SecurityFeatures) -> f32 {
        // Detect unusual entropy patterns
        match features.header_entropy {
            entropy if entropy > 7.8 => 0.9,   // Very high entropy (encrypted/compressed)
            entropy if entropy < 1.0 => 0.7,   // Very low entropy (repetitive)
            entropy if entropy > 7.5 => 0.5,   // High entropy
            entropy if entropy < 2.0 => 0.4,   // Low entropy
            _ => 0.0,
        }
    }
    
    fn detect_structure_anomaly(&self, features: &crate::SecurityFeatures) -> f32 {
        let mut score: f32 = 0.0;
        
        // Too many embedded files
        if features.embedded_files.len() > 20 {
            score += 0.4;
        }
        
        // Too many URLs
        if features.url_count > 50 {
            score += 0.3;
        }
        
        // High obfuscation
        if features.obfuscation_score > 0.8 {
            score += 0.5;
        }
        
        // Many suspicious keywords
        if features.suspicious_keywords.len() > 30 {
            score += 0.3;
        }
        
        score.min(1.0)
    }
    
    fn get_anomaly_indicators(&self, features: &crate::SecurityFeatures) -> Vec<String> {
        let mut indicators = Vec::new();
        
        if features.file_size > 100_000_000 {
            indicators.push("Extremely large file size".to_string());
        }
        
        if features.header_entropy > 7.8 {
            indicators.push("Very high entropy content".to_string());
        }
        
        if features.embedded_files.len() > 20 {
            indicators.push("Excessive embedded files".to_string());
        }
        
        if features.obfuscation_score > 0.8 {
            indicators.push("Heavy content obfuscation".to_string());
        }
        
        if features.url_count > 50 {
            indicators.push("Excessive URL references".to_string());
        }
        
        indicators
    }
}

impl NeuralSecurityModel for AnomalyDetectorModel {
    fn name(&self) -> &str {
        &self.base_model.config.name
    }
    
    fn version(&self) -> &str {
        &self.base_model.config.version
    }
    
    fn input_size(&self) -> usize {
        self.base_model.config.input_size
    }
    
    fn output_size(&self) -> usize {
        self.base_model.config.output_size
    }
    
    fn predict(&self, features: &[f32]) -> Result<Vec<f32>, ProcessingError> {
        Ok(self.base_model.network.lock().unwrap().run(features))
    }
    
    fn train(&mut self, data: &TrainingData) -> Result<TrainingResult, ProcessingError> {
        // For autoencoder, output should be same as input
        let autoencoder_data = TrainingData::new(
            data.inputs.clone(),
            data.inputs.clone(), // Train to reconstruct input
        );
        
        self.base_model.train_network(&autoencoder_data)
    }
    
    fn save(&self, path: &Path) -> Result<(), ProcessingError> {
        self.base_model.save_network(path)
    }
    
    fn load(&mut self, path: &Path) -> Result<(), ProcessingError> {
        self.base_model.load_network(path)
    }
    
    fn get_metrics(&self) -> ModelMetrics {
        self.base_model.metrics.lock().unwrap().clone()
    }
    
    fn validate(&self, test_data: &TrainingData) -> Result<ValidationResult, ProcessingError> {
        let mut total_error = 0.0;
        let mut anomaly_count = 0;
        
        for input in &test_data.inputs {
            let reconstruction = self.predict(input)?;
            
            let mse: f32 = input.iter().zip(reconstruction.iter())
                .map(|(&i, &r)| (i - r).powi(2))
                .sum::<f32>() / input.len() as f32;
            
            total_error += mse;
            
            if mse > self.anomaly_threshold {
                anomaly_count += 1;
            }
        }
        
        let mean_reconstruction_error = total_error / test_data.inputs.len() as f32;
        let anomaly_rate = anomaly_count as f32 / test_data.inputs.len() as f32;
        
        // For anomaly detection, we measure reconstruction quality
        let accuracy = 1.0 - mean_reconstruction_error.min(1.0);
        
        Ok(ValidationResult {
            accuracy,
            precision: 1.0 - anomaly_rate, // Normal detection precision
            recall: accuracy, // Ability to reconstruct normal patterns
            f1_score: 2.0 * accuracy * (1.0 - anomaly_rate) / (accuracy + 1.0 - anomaly_rate),
            confusion_matrix: None,
        })
    }
    
}

/// Detailed anomaly analysis result
#[derive(Debug, Clone)]
pub struct AnomalyAnalysis {
    pub anomaly_score: f32,
    pub is_anomalous: bool,
    pub reconstruction_error: f32,
    pub top_anomalous_features: Vec<(usize, f32)>,
    pub confidence: f32,
}

/// Advanced anomaly detection result
#[derive(Debug, Clone)]
pub struct AdvancedAnomalyResult {
    pub basic_analysis: AnomalyAnalysis,
    pub combined_score: f32,
    pub size_anomaly_score: f32,
    pub entropy_anomaly_score: f32,
    pub structure_anomaly_score: f32,
    pub severity: AnomalySeverity,
    pub anomaly_indicators: Vec<String>,
}

/// Anomaly severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnomalySeverity {
    Normal,
    Low,
    Medium,
    High,
    Critical,
}