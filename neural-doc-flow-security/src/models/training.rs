//! Training utilities and data generation for security models

use crate::models::base::{TrainingData, TrainingResult, NeuralSecurityModel};
use crate::SecurityFeatures;
use neural_doc_flow_core::ProcessingError;
use rand::{thread_rng, Rng, distributions::Uniform};
use std::path::Path;

/// Training configuration for security models
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub batch_size: usize,
    pub validation_split: f32,
    pub early_stopping_patience: u32,
    pub learning_rate_decay: f32,
    pub augmentation_factor: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            validation_split: 0.2,
            early_stopping_patience: 10,
            learning_rate_decay: 0.95,
            augmentation_factor: 2.0,
        }
    }
}

/// Model trainer for security neural networks
pub struct ModelTrainer {
    config: TrainingConfig,
}

impl ModelTrainer {
    /// Create a new model trainer
    pub fn new(config: TrainingConfig) -> Self {
        Self { config }
    }
    
    /// Train a model with the given data
    pub fn train_model<M: NeuralSecurityModel>(
        &self,
        model: &mut M,
        training_data: TrainingData,
    ) -> Result<TrainingResult, ProcessingError> {
        // Split data into train and validation sets
        let (train_data, val_data) = training_data.split(self.config.validation_split);
        
        // Augment training data
        let augmented_data = self.augment_training_data(&train_data);
        
        // Train the model
        let result = model.train(&augmented_data)?;
        
        // Validate on validation set
        let validation = model.validate(&val_data)?;
        
        println!("Training complete - Accuracy: {:.2}%, F1: {:.3}", 
                 validation.accuracy * 100.0, validation.f1_score);
        
        Ok(result)
    }
    
    /// Augment training data to improve model robustness
    fn augment_training_data(&self, data: &TrainingData) -> TrainingData {
        let mut augmented_inputs = data.inputs.clone();
        let mut augmented_outputs = data.outputs.clone();
        let mut rng = thread_rng();
        
        let augmentation_count = (data.inputs.len() as f32 * self.config.augmentation_factor) as usize;
        
        for _ in 0..augmentation_count {
            let idx = rng.gen_range(0..data.inputs.len());
            let original_input = &data.inputs[idx];
            let original_output = &data.outputs[idx];
            
            // Apply random augmentation
            let augmented_input = self.augment_features(original_input, &mut rng);
            
            augmented_inputs.push(augmented_input);
            augmented_outputs.push(original_output.clone());
        }
        
        TrainingData::new(augmented_inputs, augmented_outputs)
    }
    
    /// Apply augmentation to feature vector
    fn augment_features(&self, features: &[f32], rng: &mut impl Rng) -> Vec<f32> {
        let noise_dist = Uniform::new(-0.05, 0.05);
        
        features.iter()
            .map(|&f| {
                let noise = rng.sample(noise_dist);
                (f + noise).max(0.0).min(1.0)
            })
            .collect()
    }
}

/// Generate training data for malware detection
pub fn generate_malware_training_data(sample_count: usize) -> TrainingData {
    let mut rng = thread_rng();
    let mut inputs = Vec::with_capacity(sample_count);
    let mut outputs = Vec::with_capacity(sample_count);
    
    for i in 0..sample_count {
        let is_malware = i % 2 == 0;  // Balanced dataset
        
        let features = if is_malware {
            generate_malware_features(&mut rng)
        } else {
            generate_benign_features(&mut rng)
        };
        
        inputs.push(features);
        outputs.push(vec![if is_malware { 1.0 } else { 0.0 }]);
    }
    
    TrainingData::new(inputs, outputs)
}

/// Generate training data for threat classification
pub fn generate_threat_classification_data(sample_count: usize) -> TrainingData {
    let mut rng = thread_rng();
    let mut inputs = Vec::with_capacity(sample_count);
    let mut outputs = Vec::with_capacity(sample_count);
    
    let threat_types = 5;
    
    for i in 0..sample_count {
        let threat_type = i % threat_types;
        let features = generate_threat_features(threat_type, &mut rng);
        
        let mut output = vec![0.0; threat_types];
        output[threat_type] = 1.0;
        
        inputs.push(features);
        outputs.push(output);
    }
    
    TrainingData::new(inputs, outputs)
}

/// Generate training data for anomaly detection
pub fn generate_anomaly_training_data(sample_count: usize) -> TrainingData {
    let mut rng = thread_rng();
    let mut inputs = Vec::with_capacity(sample_count);
    let mut outputs = Vec::with_capacity(sample_count);
    
    // For anomaly detection, we train on normal data only
    for _ in 0..sample_count {
        let features = generate_normal_document_features(&mut rng);
        inputs.push(features.clone());
        outputs.push(features);  // Autoencoder - output equals input
    }
    
    TrainingData::new(inputs, outputs)
}

/// Generate training data for behavioral analysis
pub fn generate_behavioral_training_data(sample_count: usize) -> TrainingData {
    let mut rng = thread_rng();
    let mut inputs = Vec::with_capacity(sample_count);
    let mut outputs = Vec::with_capacity(sample_count);
    
    let behavior_types = 8;
    
    for _ in 0..sample_count {
        let features = generate_behavioral_features(&mut rng);
        
        // Multi-label - multiple behaviors can be present
        let mut output = vec![0.0; behavior_types];
        let behavior_count = rng.gen_range(0..4);
        for _ in 0..behavior_count {
            let behavior_idx = rng.gen_range(0..behavior_types);
            output[behavior_idx] = 1.0;
        }
        
        inputs.push(features);
        outputs.push(output);
    }
    
    TrainingData::new(inputs, outputs)
}

/// Generate training data for exploit detection
pub fn generate_exploit_training_data(sample_count: usize) -> TrainingData {
    let mut rng = thread_rng();
    let mut inputs = Vec::with_capacity(sample_count);
    let mut outputs = Vec::with_capacity(sample_count);
    
    for i in 0..sample_count {
        let has_exploit = i % 3 < 2;  // 66% positive samples for better detection
        
        let features = if has_exploit {
            generate_exploit_features(&mut rng)
        } else {
            generate_benign_features(&mut rng)
        };
        
        // 64-dimensional output for exploit encoding
        let mut output = vec![0.0; 64];
        if has_exploit {
            // Generate exploit signature pattern
            for j in 0..16 {
                output[j] = rng.gen_range(0.7..1.0);
            }
        }
        
        inputs.push(features);
        outputs.push(output);
    }
    
    TrainingData::new(inputs, outputs)
}

// Feature generation helpers

fn generate_malware_features(rng: &mut impl Rng) -> Vec<f32> {
    let mut features = vec![0.0; 256];
    
    // High entropy
    features[1] = rng.gen_range(0.8..1.0);
    // JavaScript present
    features[3] = 1.0;
    // Embedded files
    features[4] = rng.gen_range(0.3..0.8);
    // Suspicious keywords
    features[5] = rng.gen_range(0.5..1.0);
    // Obfuscation
    features[7] = rng.gen_range(0.6..1.0);
    
    // Add noise to other features
    for i in 8..256 {
        features[i] = rng.gen_range(0.0..0.3);
    }
    
    features
}

fn generate_benign_features(rng: &mut impl Rng) -> Vec<f32> {
    let mut features = vec![0.0; 256];
    
    // Normal entropy
    features[1] = rng.gen_range(0.4..0.7);
    // No JavaScript
    features[3] = 0.0;
    // Few embedded files
    features[4] = rng.gen_range(0.0..0.2);
    // No suspicious keywords
    features[5] = 0.0;
    // Low obfuscation
    features[7] = rng.gen_range(0.0..0.2);
    
    // Add noise to other features
    for i in 8..256 {
        features[i] = rng.gen_range(0.0..0.1);
    }
    
    features
}

fn generate_threat_features(threat_type: usize, rng: &mut impl Rng) -> Vec<f32> {
    let mut features = vec![0.0; 192];
    
    match threat_type {
        0 => { // JavaScript/ActiveX
            features[0] = 1.0;
            for i in 1..5 {
                features[i] = rng.gen_range(0.5..1.0);
            }
        }
        1 => { // Embedded Malware
            features[5] = rng.gen_range(0.5..1.0);
            for i in 6..10 {
                features[i] = rng.gen_range(0.3..0.8);
            }
        }
        2 => { // Buffer Overflow
            features[10] = rng.gen_range(0.8..1.0);
            features[11] = 1.0;
            for i in 12..16 {
                features[i] = rng.gen_range(0.4..0.9);
            }
        }
        3 => { // Phishing
            features[16] = rng.gen_range(0.7..1.0);
            for i in 17..21 {
                features[i] = rng.gen_range(0.5..1.0);
            }
        }
        4 => { // Data Exfiltration
            features[21] = rng.gen_range(0.8..1.0);
            for i in 22..26 {
                features[i] = rng.gen_range(0.4..0.9);
            }
        }
        _ => {}
    }
    
    // Add noise
    for i in 30..192 {
        features[i] = rng.gen_range(0.0..0.2);
    }
    
    features
}

fn generate_normal_document_features(rng: &mut impl Rng) -> Vec<f32> {
    let mut features = vec![0.0; 128];
    
    // Normal file size (log-normalized)
    features[0] = rng.gen_range(0.3..0.7);
    // Normal entropy
    features[1] = rng.gen_range(0.5..0.7);
    // Few streams
    features[2] = rng.gen_range(0.0..0.3);
    // Low obfuscation
    features[3] = rng.gen_range(0.0..0.2);
    
    // Normal patterns
    for i in 4..20 {
        features[i] = rng.gen_range(0.0..0.3);
    }
    
    features
}

fn generate_behavioral_features(rng: &mut impl Rng) -> Vec<f32> {
    let mut features = vec![0.0; 160];
    
    // Random behavioral indicators
    for i in 0..40 {
        if rng.gen_bool(0.3) {
            features[i] = rng.gen_range(0.5..1.0);
        }
    }
    
    // Add correlations
    for i in 40..160 {
        features[i] = rng.gen_range(0.0..0.4);
    }
    
    features
}

fn generate_exploit_features(rng: &mut impl Rng) -> Vec<f32> {
    let mut features = vec![0.0; 256];
    
    // Shellcode patterns
    for i in 0..8 {
        if rng.gen_bool(0.6) {
            features[i] = 1.0;
        }
    }
    
    // High entropy for encoded exploit
    features[9] = rng.gen_range(0.8..1.0);
    
    // Exploit kit signatures
    for i in 20..25 {
        if rng.gen_bool(0.4) {
            features[i] = 1.0;
        }
    }
    
    // Add exploit-specific patterns
    for i in 50..100 {
        if rng.gen_bool(0.3) {
            features[i] = rng.gen_range(0.5..1.0);
        }
    }
    
    features
}

/// Batch training function for all security models
pub fn train_all_security_models(
    models_dir: &Path,
    sample_count: usize,
) -> Result<(), ProcessingError> {
    use std::fs;
    
    // Create models directory if it doesn't exist
    fs::create_dir_all(models_dir)
        .map_err(|e| ProcessingError::Training(format!("Failed to create models directory: {}", e)))?;
    
    let trainer = ModelTrainer::new(TrainingConfig::default());
    let mut training_reports = Vec::new();
    
    println!("🚀 Starting comprehensive training of all 5 neural security models...");
    println!("📊 Sample count per model: {}", sample_count);
    println!("📁 Models directory: {:?}", models_dir);
    
    // Training Malware Detector with enhanced data
    println!("\n🔍 Training Malware Detector Model (>99.5% accuracy target)...");
    let malware_data = generate_enhanced_malware_training_data(sample_count * 2); // More data for higher accuracy
    let mut malware_model = crate::models::MalwareDetectorModel::new()?;
    let malware_result = trainer.train_model(&mut malware_model, malware_data)?;
    malware_model.save(&models_dir.join("malware_detector.fann"))?;
    training_reports.push(("MalwareDetector".to_string(), malware_result));
    
    // Training Threat Classifier
    println!("\n🎯 Training Threat Classifier Model (5 categories)...");
    let threat_data = generate_enhanced_threat_classification_data(sample_count);
    let mut threat_model = crate::models::ThreatClassifierModel::new()?;
    let threat_result = trainer.train_model(&mut threat_model, threat_data)?;
    threat_model.save(&models_dir.join("threat_classifier.fann"))?;
    training_reports.push(("ThreatClassifier".to_string(), threat_result));
    
    // Training Anomaly Detector
    println!("\n🚨 Training Anomaly Detector Model (autoencoder)...");
    let anomaly_data = generate_enhanced_anomaly_training_data(sample_count);
    let mut anomaly_model = crate::models::AnomalyDetectorModel::new()?;
    let anomaly_result = trainer.train_model(&mut anomaly_model, anomaly_data)?;
    anomaly_model.save(&models_dir.join("anomaly_detector.fann"))?;
    training_reports.push(("AnomalyDetector".to_string(), anomaly_result));
    
    // Training Behavioral Analyzer
    println!("\n🧠 Training Behavioral Analyzer Model (8 behavior types)...");
    let behavioral_data = generate_enhanced_behavioral_training_data(sample_count);
    let mut behavioral_model = crate::models::BehavioralAnalyzerModel::new()?;
    let behavioral_result = trainer.train_model(&mut behavioral_model, behavioral_data)?;
    behavioral_model.save(&models_dir.join("behavioral_analyzer.fann"))?;
    training_reports.push(("BehavioralAnalyzer".to_string(), behavioral_result));
    
    // Training Exploit Detector
    println!("\n💥 Training Exploit Detector Model (signature + zero-day)...");
    let exploit_data = generate_enhanced_exploit_training_data(sample_count);
    let mut exploit_model = crate::models::ExploitDetectorModel::new()?;
    let exploit_result = trainer.train_model(&mut exploit_model, exploit_data)?;
    exploit_model.save(&models_dir.join("exploit_detector.fann"))?;
    training_reports.push(("ExploitDetector".to_string(), exploit_result));
    
    // Generate comprehensive training report
    generate_training_report(models_dir, &training_reports)?;
    
    println!("\n✅ All 5 neural security models trained successfully!");
    println!("📊 Training report saved to: {:?}/training_report.json", models_dir);
    
    Ok(())
}

/// Generate enhanced malware training data with realistic patterns
pub fn generate_enhanced_malware_training_data(sample_count: usize) -> TrainingData {
    let mut rng = thread_rng();
    let mut inputs = Vec::with_capacity(sample_count);
    let mut outputs = Vec::with_capacity(sample_count);
    
    // 60% malware, 40% benign for better accuracy
    let malware_count = (sample_count as f32 * 0.6) as usize;
    
    for i in 0..sample_count {
        let is_malware = i < malware_count;
        
        let features = if is_malware {
            generate_realistic_malware_features(&mut rng, i % 5) // 5 different malware types
        } else {
            generate_realistic_benign_features(&mut rng, i % 3) // 3 different benign types
        };
        
        inputs.push(features);
        outputs.push(vec![if is_malware { 1.0 } else { 0.0 }]);
    }
    
    TrainingData::new(inputs, outputs)
}

/// Generate enhanced threat classification data with more realistic patterns
pub fn generate_enhanced_threat_classification_data(sample_count: usize) -> TrainingData {
    let mut rng = thread_rng();
    let mut inputs = Vec::with_capacity(sample_count);
    let mut outputs = Vec::with_capacity(sample_count);
    
    let threat_types = 5;
    
    for i in 0..sample_count {
        let threat_type = i % threat_types;
        let features = generate_realistic_threat_features(threat_type, &mut rng);
        
        let mut output = vec![0.0; threat_types];
        output[threat_type] = 1.0;
        
        // Add some noise to create more realistic multi-class scenarios
        if rng.gen_bool(0.1) { // 10% chance of secondary threat
            let secondary_type = (threat_type + 1 + rng.gen_range(0..threat_types-1)) % threat_types;
            output[secondary_type] = 0.3;
        }
        
        inputs.push(features);
        outputs.push(output);
    }
    
    TrainingData::new(inputs, outputs)
}

/// Generate enhanced anomaly training data
pub fn generate_enhanced_anomaly_training_data(sample_count: usize) -> TrainingData {
    let mut rng = thread_rng();
    let mut inputs = Vec::with_capacity(sample_count);
    let mut outputs = Vec::with_capacity(sample_count);
    
    // Train mostly on normal data with some anomalies
    let normal_count = (sample_count as f32 * 0.9) as usize;
    
    for i in 0..sample_count {
        let features = if i < normal_count {
            generate_realistic_normal_document_features(&mut rng)
        } else {
            generate_anomalous_document_features(&mut rng)
        };
        
        inputs.push(features.clone());
        outputs.push(features); // Autoencoder - output equals input
    }
    
    TrainingData::new(inputs, outputs)
}

/// Generate enhanced behavioral training data
pub fn generate_enhanced_behavioral_training_data(sample_count: usize) -> TrainingData {
    let mut rng = thread_rng();
    let mut inputs = Vec::with_capacity(sample_count);
    let mut outputs = Vec::with_capacity(sample_count);
    
    let behavior_types = 8;
    
    for _ in 0..sample_count {
        let features = generate_realistic_behavioral_features(&mut rng);
        
        // Multi-label - realistic behavior combinations
        let mut output = vec![0.0; behavior_types];
        let primary_behavior = rng.gen_range(0..behavior_types);
        output[primary_behavior] = 1.0;
        
        // Add correlated behaviors
        match primary_behavior {
            0 => { // Dropper - likely to have downloader
                if rng.gen_bool(0.7) { output[1] = 0.8; }
            },
            2 => { // Code injection - likely to have evasion
                if rng.gen_bool(0.6) { output[6] = 0.7; }
            },
            4 => { // Data collection - likely to have network
                if rng.gen_bool(0.8) { output[5] = 0.9; }
            },
            _ => {
                // Random secondary behaviors
                let secondary_count = rng.gen_range(0..3);
                for _ in 0..secondary_count {
                    let secondary_idx = rng.gen_range(0..behavior_types);
                    if secondary_idx != primary_behavior {
                        output[secondary_idx] = rng.gen_range(0.3..0.7);
                    }
                }
            }
        }
        
        inputs.push(features);
        outputs.push(output);
    }
    
    TrainingData::new(inputs, outputs)
}

/// Generate enhanced exploit training data
pub fn generate_enhanced_exploit_training_data(sample_count: usize) -> TrainingData {
    let mut rng = thread_rng();
    let mut inputs = Vec::with_capacity(sample_count);
    let mut outputs = Vec::with_capacity(sample_count);
    
    // 70% positive samples for better exploit detection
    let exploit_count = (sample_count as f32 * 0.7) as usize;
    
    for i in 0..sample_count {
        let has_exploit = i < exploit_count;
        
        let features = if has_exploit {
            generate_realistic_exploit_features(&mut rng, i % 6) // 6 different exploit types
        } else {
            generate_realistic_benign_features(&mut rng, i % 3)
        };
        
        // 64-dimensional output for exploit encoding
        let mut output = vec![0.0; 64];
        if has_exploit {
            // Generate realistic exploit signature pattern
            let exploit_type = i % 6;
            match exploit_type {
                0 => { // CVE-2017-11882 pattern
                    for j in 0..8 {
                        output[j] = rng.gen_range(0.8..1.0);
                    }
                },
                1 => { // Flash exploit pattern
                    for j in 8..16 {
                        output[j] = rng.gen_range(0.7..0.9);
                    }
                },
                2 => { // Heap spray pattern
                    for j in 16..24 {
                        output[j] = rng.gen_range(0.6..0.8);
                    }
                },
                3 => { // ROP chain pattern
                    for j in 24..32 {
                        output[j] = rng.gen_range(0.7..0.9);
                    }
                },
                4 => { // Zero-day pattern
                    for j in 32..40 {
                        output[j] = rng.gen_range(0.9..1.0);
                    }
                },
                _ => { // Generic exploit pattern
                    for j in 40..48 {
                        output[j] = rng.gen_range(0.5..0.7);
                    }
                }
            }
        }
        
        inputs.push(features);
        outputs.push(output);
    }
    
    TrainingData::new(inputs, outputs)
}

// Enhanced feature generation functions

fn generate_realistic_malware_features(rng: &mut impl Rng, malware_type: usize) -> Vec<f32> {
    let mut features = vec![0.0; 256];
    
    match malware_type {
        0 => { // Banking trojan
            features[1] = rng.gen_range(0.7..0.9); // High entropy
            features[3] = 1.0; // JavaScript present
            features[5] = rng.gen_range(0.8..1.0); // Many suspicious keywords
            features[7] = rng.gen_range(0.8..1.0); // High obfuscation
            features[6] = rng.gen_range(0.5..0.8); // URLs for C&C
        },
        1 => { // Ransomware
            features[1] = rng.gen_range(0.9..1.0); // Very high entropy
            features[4] = rng.gen_range(0.6..0.9); // Embedded encryption modules
            features[5] = rng.gen_range(0.7..1.0); // Crypto-related keywords
            features[7] = rng.gen_range(0.6..0.8); // Moderate obfuscation
        },
        2 => { // APT malware
            features[1] = rng.gen_range(0.8..1.0); // High entropy
            features[4] = rng.gen_range(0.3..0.6); // Few but sophisticated embedded files
            features[7] = rng.gen_range(0.9..1.0); // Very high obfuscation
            features[8] = rng.gen_range(0.8..1.0); // Advanced evasion techniques
        },
        3 => { // Botnet malware
            features[6] = rng.gen_range(0.7..1.0); // Many URLs for C&C
            features[5] = rng.gen_range(0.6..0.8); // Network-related keywords
            features[7] = rng.gen_range(0.5..0.7); // Moderate obfuscation
        },
        _ => { // Generic malware
            features[1] = rng.gen_range(0.6..0.8); // Moderate entropy
            features[3] = if rng.gen_bool(0.7) { 1.0 } else { 0.0 }; // JavaScript sometimes
            features[5] = rng.gen_range(0.4..0.7); // Some suspicious keywords
            features[7] = rng.gen_range(0.4..0.6); // Low to moderate obfuscation
        }
    }
    
    // Add noise to remaining features
    for i in 9..256 {
        features[i] = rng.gen_range(0.0..0.3);
    }
    
    features
}

fn generate_realistic_benign_features(rng: &mut impl Rng, benign_type: usize) -> Vec<f32> {
    let mut features = vec![0.0; 256];
    
    match benign_type {
        0 => { // Normal office document
            features[1] = rng.gen_range(0.4..0.6); // Normal entropy
            features[4] = rng.gen_range(0.0..0.2); // Few embedded files
            features[5] = 0.0; // No suspicious keywords
            features[7] = rng.gen_range(0.0..0.1); // No obfuscation
        },
        1 => { // Web page with scripts
            features[1] = rng.gen_range(0.5..0.7); // Slightly higher entropy
            features[3] = 1.0; // JavaScript present (legitimate)
            features[6] = rng.gen_range(0.1..0.3); // Some URLs
            features[7] = rng.gen_range(0.0..0.2); // Minimal obfuscation
        },
        _ => { // PDF document
            features[1] = rng.gen_range(0.3..0.5); // Lower entropy
            features[4] = rng.gen_range(0.0..0.1); // Minimal embedded content
            features[5] = 0.0; // No suspicious keywords
            features[7] = 0.0; // No obfuscation
        }
    }
    
    // Add minimal noise
    for i in 8..256 {
        features[i] = rng.gen_range(0.0..0.1);
    }
    
    features
}

fn generate_realistic_threat_features(threat_type: usize, rng: &mut impl Rng) -> Vec<f32> {
    let mut features = vec![0.0; 192];
    
    match threat_type {
        0 => { // JavaScript/ActiveX Exploits
            features[0] = 1.0; // JavaScript present
            for i in 1..5 {
                features[i] = rng.gen_range(0.7..1.0); // High JS exploit indicators
            }
            features[20] = rng.gen_range(0.5..0.8); // Some obfuscation
        },
        1 => { // Embedded Malware
            features[5] = rng.gen_range(0.7..1.0); // Many embedded files
            for i in 6..10 {
                features[i] = rng.gen_range(0.5..0.9); // Executable types
            }
            features[25] = rng.gen_range(0.6..0.9); // High entropy
        },
        2 => { // Buffer Overflow/Memory Exploits
            features[10] = rng.gen_range(0.8..1.0); // High entropy
            features[11] = 1.0; // Memory exploit indicators
            for i in 12..16 {
                features[i] = rng.gen_range(0.6..0.9); // Memory-related keywords
            }
        },
        3 => { // Phishing/Social Engineering
            features[16] = rng.gen_range(0.6..1.0); // URLs
            for i in 17..21 {
                features[i] = rng.gen_range(0.7..1.0); // Phishing keywords
            }
            features[30] = rng.gen_range(0.3..0.6); // Moderate obfuscation
        },
        4 => { // Data Exfiltration
            features[21] = rng.gen_range(0.8..1.0); // High obfuscation
            for i in 22..26 {
                features[i] = rng.gen_range(0.5..0.9); // Exfiltration keywords
            }
            features[35] = rng.gen_range(0.4..0.7); // Network indicators
        },
        _ => {}
    }
    
    // Add background noise
    for i in 40..192 {
        features[i] = rng.gen_range(0.0..0.2);
    }
    
    features
}

fn generate_realistic_normal_document_features(rng: &mut impl Rng) -> Vec<f32> {
    let mut features = vec![0.0; 128];
    
    // Realistic normal document characteristics
    features[0] = rng.gen_range(0.3..0.6); // Normal file size
    features[1] = rng.gen_range(0.5..0.7); // Normal entropy
    features[2] = rng.gen_range(0.0..0.2); // Few streams
    features[3] = rng.gen_range(0.0..0.1); // Low obfuscation
    
    // Add natural variation
    for i in 4..20 {
        features[i] = rng.gen_range(0.0..0.3);
    }
    
    features
}

fn generate_anomalous_document_features(rng: &mut impl Rng) -> Vec<f32> {
    let mut features = vec![0.0; 128];
    
    // Anomalous characteristics
    features[0] = if rng.gen_bool(0.5) { 
        rng.gen_range(0.9..1.0) // Very large
    } else { 
        rng.gen_range(0.0..0.1) // Very small
    };
    features[1] = if rng.gen_bool(0.7) { 
        rng.gen_range(0.9..1.0) // Very high entropy
    } else { 
        rng.gen_range(0.0..0.2) // Very low entropy
    };
    features[2] = rng.gen_range(0.7..1.0); // Many streams
    features[3] = rng.gen_range(0.8..1.0); // High obfuscation
    
    // Add anomalous patterns
    for i in 4..20 {
        if rng.gen_bool(0.3) {
            features[i] = rng.gen_range(0.7..1.0); // Anomalous spikes
        } else {
            features[i] = rng.gen_range(0.0..0.2);
        }
    }
    
    features
}

fn generate_realistic_behavioral_features(rng: &mut impl Rng) -> Vec<f32> {
    let mut features = vec![0.0; 160];
    
    // Generate realistic behavioral patterns
    for i in 0..40 {
        if rng.gen_bool(0.3) {
            features[i] = rng.gen_range(0.4..0.9);
        }
    }
    
    // Add behavioral correlations
    for i in 40..80 {
        features[i] = rng.gen_range(0.0..0.5);
    }
    
    // Cross-behavior features
    for i in 80..160 {
        features[i] = rng.gen_range(0.0..0.3);
    }
    
    features
}

fn generate_realistic_exploit_features(rng: &mut impl Rng, exploit_type: usize) -> Vec<f32> {
    let mut features = vec![0.0; 256];
    
    match exploit_type {
        0 => { // Equation Editor exploit
            features[8] = 1.0; // Large file indicator
            features[9] = rng.gen_range(0.8..1.0); // High entropy
            for i in 12..16 { // ROP gadgets
                features[i] = rng.gen_range(0.7..0.9);
            }
        },
        1 => { // Flash exploit
            features[20] = 1.0; // JavaScript present
            for i in 50..60 { // Flash-specific patterns
                if rng.gen_bool(0.7) {
                    features[i] = rng.gen_range(0.6..0.9);
                }
            }
        },
        2 => { // PDF exploit
            features[70] = 1.0; // PDF indicators
            for i in 71..78 {
                features[i] = rng.gen_range(0.5..0.8);
            }
        },
        3 => { // Office exploit
            features[80] = 1.0; // Office indicators
            for i in 81..88 {
                features[i] = rng.gen_range(0.6..0.9);
            }
        },
        4 => { // Zero-day exploit
            features[9] = rng.gen_range(0.9..1.0); // Very high entropy
            features[25] = rng.gen_range(0.9..1.0); // High obfuscation
            // Unusual patterns
            for i in 200..210 {
                features[i] = rng.gen_range(0.8..1.0);
            }
        },
        _ => { // Generic exploit
            for i in 50..100 {
                if rng.gen_bool(0.3) {
                    features[i] = rng.gen_range(0.5..0.8);
                }
            }
        }
    }
    
    features
}

/// Generate comprehensive training report
fn generate_training_report(models_dir: &Path, reports: &[(String, TrainingResult)]) -> Result<(), ProcessingError> {
    use chrono::Utc;
    use serde_json::json;
    
    let report = json!({
        "training_session": {
            "timestamp": Utc::now(),
            "models_trained": reports.len(),
            "training_summary": "Comprehensive neural security model training session"
        },
        "model_results": reports.iter().map(|(name, result)| {
            json!({
                "model_name": name,
                "epochs_trained": result.epochs_trained,
                "final_error": result.final_error,
                "training_time_ms": result.training_time_ms,
                "converged": result.converged,
                "training_status": if result.converged { "SUCCESS" } else { "PARTIAL" }
            })
        }).collect::<Vec<_>>(),
        "performance_summary": {
            "total_training_time_ms": reports.iter().map(|(_, r)| r.training_time_ms).sum::<u64>(),
            "average_final_error": reports.iter().map(|(_, r)| r.final_error).sum::<f32>() / reports.len() as f32,
            "convergence_rate": reports.iter().filter(|(_, r)| r.converged).count() as f32 / reports.len() as f32
        },
        "model_specifications": {
            "malware_detector": {
                "input_size": 256,
                "output_size": 1,
                "target_accuracy": ">99.5%",
                "architecture": "Deep feedforward with class balancing"
            },
            "threat_classifier": {
                "input_size": 192,
                "output_size": 5,
                "categories": ["JavaScript/ActiveX", "Embedded Malware", "Buffer Overflow", "Phishing", "Data Exfiltration"],
                "architecture": "Multi-class classification with softmax"
            },
            "anomaly_detector": {
                "input_size": 128,
                "output_size": 128,
                "type": "Autoencoder",
                "architecture": "Bottleneck autoencoder for reconstruction"
            },
            "behavioral_analyzer": {
                "input_size": 160,
                "output_size": 8,
                "behaviors": ["Dropper", "Downloader", "Code Injection", "Persistence", "Data Collection", "Network", "Evasion", "Privilege Escalation"],
                "architecture": "Multi-label classification"
            },
            "exploit_detector": {
                "input_size": 256,
                "output_size": 64,
                "type": "Signature encoding + zero-day detection",
                "architecture": "Deep encoder with signature database"
            }
        }
    });
    
    let report_path = models_dir.join("training_report.json");
    std::fs::write(&report_path, serde_json::to_string_pretty(&report)?)
        .map_err(|e| ProcessingError::SerializationError(e.to_string()))?;
    
    println!("📊 Training report generated: {:?}", report_path);
    
    Ok(())
}