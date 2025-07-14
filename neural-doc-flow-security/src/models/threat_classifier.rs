//! Multi-class threat classification neural model (5 categories)

use crate::models::base::{BaseNeuralModel, ModelConfig, ModelMetrics, NeuralSecurityModel, TrainingData, TrainingResult, ValidationResult};
use neural_doc_flow_core::ProcessingError;
use std::path::{Path, PathBuf};

/// Neural model for threat classification
/// Classifies threats into 5 main categories
pub struct ThreatClassifierModel {
    base_model: BaseNeuralModel,
    model_path: PathBuf,
    threat_categories: Vec<String>,
}

impl ThreatClassifierModel {
    /// Create a new threat classifier model
    pub fn new() -> Result<Self, ProcessingError> {
        let config = ModelConfig {
            name: "ThreatClassifier".to_string(),
            version: "1.0.0".to_string(),
            input_size: 192,  // Optimized feature set for classification
            output_size: 5,   // 5 threat categories
            hidden_layers: vec![256, 128, 64], // Moderate depth for multi-class
            activation_function: "tanh".to_string(),
            learning_rate: 0.005,
            momentum: 0.9,
            target_error: 0.001,
            max_epochs: 30000,
            simd_enabled: true,
        };
        
        let base_model = BaseNeuralModel::new(config)?;
        
        let threat_categories = vec![
            "JavaScript/ActiveX Exploits".to_string(),
            "Embedded Malware".to_string(),
            "Buffer Overflow/Memory Exploits".to_string(),
            "Phishing/Social Engineering".to_string(),
            "Data Exfiltration".to_string(),
        ];
        
        Ok(Self {
            base_model,
            model_path: PathBuf::from("models/threat_classifier.fann"),
            threat_categories,
        })
    }
    
    /// Load or create model
    pub fn load_or_create(model_dir: &Path) -> Result<Self, ProcessingError> {
        let mut model = Self::new()?;
        let model_path = model_dir.join("threat_classifier.fann");
        model.model_path = model_path.clone();
        
        if model_path.exists() {
            model.load(&model_path)?;
        }
        
        Ok(model)
    }
    
    /// Extract features optimized for threat classification
    pub fn extract_features(raw_features: &crate::SecurityFeatures) -> Vec<f32> {
        let mut features = Vec::with_capacity(192);
        
        // JavaScript/ActiveX indicators
        features.push(if raw_features.javascript_present { 1.0 } else { 0.0 });
        let js_keywords = ["eval", "document.write", "ActiveXObject", "WScript"];
        for keyword in &js_keywords {
            let count = raw_features.suspicious_keywords.iter()
                .filter(|k| k.contains(keyword)).count();
            features.push((count as f32).min(1.0));
        }
        
        // Embedded malware indicators
        features.push((raw_features.embedded_files.len() as f32) / 10.0);
        let executable_types = ["exe", "dll", "scr", "bat", "cmd"];
        for ext in &executable_types {
            let count = raw_features.embedded_files.iter()
                .filter(|f| f.file_type.to_lowercase().contains(ext)).count();
            features.push((count as f32).min(1.0));
        }
        
        // Memory exploit indicators
        features.push(raw_features.header_entropy / 8.0);
        features.push(if raw_features.header_entropy > 7.8 { 1.0 } else { 0.0 });
        let memory_keywords = ["buffer", "overflow", "heap", "stack", "shellcode"];
        for keyword in &memory_keywords {
            let found = raw_features.suspicious_keywords.iter()
                .any(|k| k.to_lowercase().contains(keyword));
            features.push(if found { 1.0 } else { 0.0 });
        }
        
        // Phishing indicators
        features.push((raw_features.url_count as f32) / 20.0);
        let phishing_keywords = ["password", "login", "account", "verify", "suspended"];
        for keyword in &phishing_keywords {
            let found = raw_features.suspicious_keywords.iter()
                .any(|k| k.to_lowercase().contains(keyword));
            features.push(if found { 1.0 } else { 0.0 });
        }
        
        // Data exfiltration indicators
        features.push(raw_features.obfuscation_score);
        let exfil_keywords = ["upload", "post", "send", "transmit", "exfiltrate"];
        for keyword in &exfil_keywords {
            let found = raw_features.suspicious_keywords.iter()
                .any(|k| k.to_lowercase().contains(keyword));
            features.push(if found { 1.0 } else { 0.0 });
        }
        
        // Cross-category features
        features.push((raw_features.file_size as f32).ln() / 20.0);
        features.push((raw_features.stream_count as f32) / 20.0);
        
        // Statistical combinations
        let js_and_urls = if raw_features.javascript_present && raw_features.url_count > 5 { 1.0 } else { 0.0 };
        features.push(js_and_urls);
        
        let embedded_and_obfuscated = if !raw_features.embedded_files.is_empty() && 
            raw_features.obfuscation_score > 0.5 { 1.0 } else { 0.0 };
        features.push(embedded_and_obfuscated);
        
        // Pattern density features
        let pattern_density = raw_features.suspicious_keywords.len() as f32 / 
            (raw_features.file_size as f32 / 10000.0).max(1.0);
        features.push(pattern_density.min(1.0));
        
        // Pad to expected size
        while features.len() < 192 {
            features.push(0.0);
        }
        
        features
    }
    
    /// Classify threat into categories
    pub fn classify(&self, features: &[f32]) -> Result<Vec<(String, f32)>, ProcessingError> {
        let output = self.base_model.network.lock().unwrap().run(features);
        
        // Apply softmax normalization
        let max_val = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = output.iter().map(|&x| (x - max_val).exp()).sum();
        let probabilities: Vec<f32> = output.iter()
            .map(|&x| ((x - max_val).exp() / exp_sum))
            .collect();
        
        // Return categories with probabilities above threshold
        let mut results = Vec::new();
        for (i, &prob) in probabilities.iter().enumerate() {
            if prob > 0.1 {  // 10% threshold
                results.push((self.threat_categories[i].clone(), prob));
            }
        }
        
        // Sort by probability descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        Ok(results)
    }
    
    /// Get primary threat category
    pub fn get_primary_threat(&self, features: &[f32]) -> Result<ThreatClassification, ProcessingError> {
        let classifications = self.classify(features)?;
        
        if let Some((category, confidence)) = classifications.first() {
            Ok(ThreatClassification {
                primary_category: category.clone(),
                confidence: *confidence,
                all_categories: classifications.clone(),
                severity_score: Some(*confidence),
                recommended_action: Some("Monitor".to_string()),
            })
        } else {
            Ok(ThreatClassification {
                primary_category: "Unknown".to_string(),
                confidence: 0.0,
                all_categories: vec![],
                severity_score: Some(0.0),
                recommended_action: Some("Allow".to_string()),
            })
        }
    }
    
    /// Enhanced threat classification with contextual analysis
    pub fn analyze_threat_context(&self, features: &[f32], raw_features: &crate::SecurityFeatures) -> Result<ThreatClassification, ProcessingError> {
        let mut classification = self.get_primary_threat(features)?;
        
        // Add contextual confidence adjustments
        if raw_features.javascript_present && classification.primary_category.contains("JavaScript") {
            classification.confidence = (classification.confidence * 1.2).min(1.0);
        }
        
        if raw_features.obfuscation_score > 0.8 {
            classification.confidence = (classification.confidence * 1.1).min(1.0);
        }
        
        // Add threat severity based on combinations
        if raw_features.embedded_files.len() > 5 && raw_features.suspicious_keywords.len() > 10 {
            classification.confidence = (classification.confidence * 1.15).min(1.0);
        }
        
        Ok(classification)
    }
}

impl NeuralSecurityModel for ThreatClassifierModel {
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
        self.base_model.train_network(data)
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
        // Custom validation for multi-class classification
        let mut correct = 0;
        let mut confusion_matrix = vec![vec![0u32; 5]; 5];
        
        for (input, expected) in test_data.inputs.iter().zip(&test_data.outputs) {
            let output = self.predict(input)?;
            
            // Get predicted and actual class indices
            let predicted_idx = output.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
                
            let actual_idx = expected.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            
            if predicted_idx == actual_idx {
                correct += 1;
            }
            
            confusion_matrix[actual_idx][predicted_idx] += 1;
        }
        
        let total = test_data.inputs.len() as f32;
        let accuracy = correct as f32 / total;
        
        // Calculate per-class metrics
        let mut precision_sum = 0.0;
        let mut recall_sum = 0.0;
        
        for i in 0..5 {
            let true_positives = confusion_matrix[i][i] as f32;
            let false_positives: f32 = (0..5).filter(|&j| j != i)
                .map(|j| confusion_matrix[j][i] as f32)
                .sum();
            let false_negatives: f32 = (0..5).filter(|&j| j != i)
                .map(|j| confusion_matrix[i][j] as f32)
                .sum();
            
            if true_positives + false_positives > 0.0 {
                precision_sum += true_positives / (true_positives + false_positives);
            }
            
            if true_positives + false_negatives > 0.0 {
                recall_sum += true_positives / (true_positives + false_negatives);
            }
        }
        
        let precision = precision_sum / 5.0;
        let recall = recall_sum / 5.0;
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
            confusion_matrix: Some(confusion_matrix),
        })
    }
    
}

/// Threat classification result
#[derive(Debug, Clone)]
pub struct ThreatClassification {
    pub primary_category: String,
    pub confidence: f32,
    pub all_categories: Vec<(String, f32)>,
    pub severity_score: Option<f32>,
    pub recommended_action: Option<String>,
}

impl ThreatClassification {
    /// Calculate overall threat severity
    pub fn calculate_severity(&self) -> f32 {
        // Weight by both confidence and number of detected categories
        let category_weight = self.all_categories.len() as f32 / 5.0;
        let confidence_weight = self.confidence;
        
        (category_weight * 0.3 + confidence_weight * 0.7).min(1.0)
    }
    
    /// Get recommended security action
    pub fn get_recommended_action(&self) -> String {
        let severity = self.calculate_severity();
        
        match severity {
            s if s > 0.9 => "BLOCK - Critical threat detected".to_string(),
            s if s > 0.7 => "QUARANTINE - High risk threat".to_string(),
            s if s > 0.5 => "SANITIZE - Medium risk threat".to_string(),
            s if s > 0.3 => "MONITOR - Low risk threat".to_string(),
            _ => "ALLOW - Minimal threat".to_string(),
        }
    }
}