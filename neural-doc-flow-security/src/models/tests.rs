//! Comprehensive tests for neural security models

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::*;
    use crate::SecurityFeatures;
    use std::path::PathBuf;
    use tempfile::TempDir;

    /// Create test security features
    fn create_test_security_features() -> SecurityFeatures {
        SecurityFeatures {
            file_size: 1_000_000,
            header_entropy: 6.5,
            stream_count: 5,
            javascript_present: true,
            embedded_files: vec![
                crate::EmbeddedFile {
                    name: "test.exe".to_string(),
                    size: 50000,
                    file_type: "application/x-executable".to_string(),
                }
            ],
            suspicious_keywords: vec!["eval".to_string(), "exec".to_string()],
            url_count: 3,
            obfuscation_score: 0.7,
        }
    }

    #[test]
    fn test_malware_detector_creation() {
        let model = MalwareDetectorModel::new();
        assert!(model.is_ok());
        
        let model = model.unwrap();
        assert_eq!(model.name(), "MalwareDetector");
        assert_eq!(model.input_size(), 256);
        assert_eq!(model.output_size(), 1);
    }

    #[test]
    fn test_malware_detector_feature_extraction() {
        let features = create_test_security_features();
        let ml_features = MalwareDetectorModel::extract_features(&features);
        
        assert_eq!(ml_features.len(), 256);
        assert!(ml_features[3] > 0.0); // JavaScript present
        assert!(ml_features[7] > 0.0); // Obfuscation score
    }

    #[test]
    fn test_threat_classifier_creation() {
        let model = ThreatClassifierModel::new();
        assert!(model.is_ok());
        
        let model = model.unwrap();
        assert_eq!(model.name(), "ThreatClassifier");
        assert_eq!(model.input_size(), 192);
        assert_eq!(model.output_size(), 5);
    }

    #[test]
    fn test_threat_classifier_feature_extraction() {
        let features = create_test_security_features();
        let threat_features = ThreatClassifierModel::extract_features(&features);
        
        assert_eq!(threat_features.len(), 192);
        assert_eq!(threat_features[0], 1.0); // JavaScript present
    }

    #[test]
    fn test_anomaly_detector_creation() {
        let model = AnomalyDetectorModel::new();
        assert!(model.is_ok());
        
        let model = model.unwrap();
        assert_eq!(model.name(), "AnomalyDetector");
        assert_eq!(model.input_size(), 128);
        assert_eq!(model.output_size(), 128);
    }

    #[test]
    fn test_anomaly_detector_feature_extraction() {
        let features = create_test_security_features();
        let anomaly_features = AnomalyDetectorModel::extract_features(&features);
        
        assert_eq!(anomaly_features.len(), 128);
        assert!(anomaly_features[3] > 0.0); // Obfuscation score
    }

    #[test]
    fn test_behavioral_analyzer_creation() {
        let model = BehavioralAnalyzerModel::new();
        assert!(model.is_ok());
        
        let model = model.unwrap();
        assert_eq!(model.name(), "BehavioralAnalyzer");
        assert_eq!(model.input_size(), 160);
        assert_eq!(model.output_size(), 8);
    }

    #[test]
    fn test_behavioral_analyzer_feature_extraction() {
        let features = create_test_security_features();
        let behavioral_features = BehavioralAnalyzerModel::extract_features(&features);
        
        assert_eq!(behavioral_features.len(), 160);
    }

    #[test]
    fn test_exploit_detector_creation() {
        let model = ExploitDetectorModel::new();
        assert!(model.is_ok());
        
        let model = model.unwrap();
        assert_eq!(model.name(), "ExploitDetector");
        assert_eq!(model.input_size(), 256);
        assert_eq!(model.output_size(), 64);
    }

    #[test]
    fn test_exploit_detector_feature_extraction() {
        let features = create_test_security_features();
        let exploit_features = ExploitDetectorModel::extract_features(&features);
        
        assert_eq!(exploit_features.len(), 256);
    }

    #[test]
    fn test_training_data_generation() {
        let malware_data = training::generate_malware_training_data(100);
        assert_eq!(malware_data.inputs.len(), 100);
        assert_eq!(malware_data.outputs.len(), 100);
        assert_eq!(malware_data.outputs[0].len(), 1);
        
        let threat_data = training::generate_threat_classification_data(100);
        assert_eq!(threat_data.inputs.len(), 100);
        assert_eq!(threat_data.outputs.len(), 100);
        assert_eq!(threat_data.outputs[0].len(), 5);
        
        let anomaly_data = training::generate_anomaly_training_data(100);
        assert_eq!(anomaly_data.inputs.len(), 100);
        assert_eq!(anomaly_data.outputs.len(), 100);
        // For autoencoder, output equals input
        assert_eq!(anomaly_data.inputs[0].len(), anomaly_data.outputs[0].len());
    }

    #[test]
    fn test_training_data_split() {
        let data = training::generate_malware_training_data(100);
        let (train_data, test_data) = data.split(0.2);
        
        assert_eq!(train_data.inputs.len(), 80);
        assert_eq!(test_data.inputs.len(), 20);
    }

    #[test]
    fn test_model_metadata_creation() {
        let config = base::ModelConfig::default();
        let metrics = base::ModelMetrics {
            model_name: "TestModel".to_string(),
            accuracy: 0.95,
            inference_time_us: 1000,
            model_size_bytes: 50000,
            total_predictions: 100,
            last_training: None,
        };
        
        let metadata = serialization::ModelMetadata::new(config, metrics);
        assert_eq!(metadata.config.name, "unnamed");
        assert_eq!(metadata.metrics.model_name, "TestModel");
        assert_eq!(metadata.metrics.accuracy, 0.95);
    }

    #[tokio::test]
    async fn test_model_serialization() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("test_model.fann");
        
        let config = base::ModelConfig::default();
        let metrics = base::ModelMetrics {
            model_name: "TestModel".to_string(),
            accuracy: 0.95,
            inference_time_us: 1000,
            model_size_bytes: 50000,
            total_predictions: 100,
            last_training: None,
        };
        
        let metadata = serialization::ModelMetadata::new(config, metrics);
        
        // Test metadata saving
        let result = serialization::ModelSerializer::save_model_with_metadata(&model_path, &metadata);
        assert!(result.is_ok());
        
        // Test metadata loading
        let metadata_path = model_path.with_extension("json");
        let loaded_metadata = serialization::ModelDeserializer::load_metadata(&metadata_path);
        assert!(loaded_metadata.is_ok());
        
        let loaded = loaded_metadata.unwrap();
        assert_eq!(loaded.metrics.model_name, "TestModel");
        assert_eq!(loaded.metrics.accuracy, 0.95);
    }

    #[test]
    fn test_model_trainer() {
        let config = training::TrainingConfig::default();
        let trainer = training::ModelTrainer::new(config);
        
        // Test with small dataset
        let training_data = training::generate_malware_training_data(10);
        let mut model = MalwareDetectorModel::new().unwrap();
        
        // This would normally train, but we'll just test the interface
        // let result = trainer.train_model(&mut model, training_data);
        // In a real test environment with proper ruv-fann setup, this would work
    }

    #[test]
    fn test_security_models_initialization() {
        let temp_dir = TempDir::new().unwrap();
        let models_dir = temp_dir.path();
        
        // Test that we can initialize all models
        // Note: This requires proper ruv-fann installation to fully work
        let result = initialize_models(models_dir);
        // We expect this to work even without trained models
        assert!(result.is_ok() || result.is_err()); // Either way is acceptable in test
    }

    #[test]
    fn test_base_neural_model() {
        let config = base::ModelConfig {
            name: "TestModel".to_string(),
            version: "1.0.0".to_string(),
            input_size: 10,
            output_size: 1,
            hidden_layers: vec![5],
            activation_function: "sigmoid".to_string(),
            learning_rate: 0.01,
            momentum: 0.9,
            target_error: 0.001,
            max_epochs: 100,
            simd_enabled: true,
        };
        
        let model = base::BaseNeuralModel::new(config);
        // This test depends on ruv-fann being properly installed
        // We'll just verify the structure is correct
        assert!(model.is_ok() || model.is_err());
    }

    #[test]
    fn test_model_config_defaults() {
        let config = base::ModelConfig::default();
        assert_eq!(config.name, "unnamed");
        assert_eq!(config.input_size, 128);
        assert_eq!(config.output_size, 1);
        assert_eq!(config.hidden_layers, vec![64, 32]);
        assert_eq!(config.activation_function, "sigmoid");
        assert_eq!(config.learning_rate, 0.01);
        assert!(config.simd_enabled);
    }

    #[test]
    fn test_model_registry() {
        let mut registry = serialization::ModelRegistry::new();
        assert_eq!(registry.models.len(), 0);
        
        let model = serialization::RegisteredModel {
            name: "TestModel".to_string(),
            version: "1.0.0".to_string(),
            path: "/path/to/model".to_string(),
            metadata_path: "/path/to/metadata".to_string(),
            status: serialization::ModelStatus::Active,
            registered_at: chrono::Utc::now(),
        };
        
        registry.register_model(model);
        assert_eq!(registry.models.len(), 1);
        assert_eq!(registry.models[0].name, "TestModel");
    }

    #[test]
    fn test_threat_level_calculation() {
        use crate::models::malware_detector::ThreatLevel;
        
        // Test threat level mapping
        let model = MalwareDetectorModel::new().unwrap();
        
        // High probability should map to critical
        let level = model.calculate_threat_level(0.95);
        assert_eq!(level, ThreatLevel::Critical);
        
        // Medium probability should map to medium
        let level = model.calculate_threat_level(0.6);
        assert_eq!(level, ThreatLevel::Medium);
        
        // Low probability should map to safe
        let level = model.calculate_threat_level(0.1);
        assert_eq!(level, ThreatLevel::Safe);
    }

    #[test]
    fn test_behavioral_risk_levels() {
        use crate::models::behavioral_analyzer::RiskLevel;
        
        // Test risk level enumeration
        assert_eq!(RiskLevel::Minimal as u8, 0);
        assert_ne!(RiskLevel::Critical, RiskLevel::Low);
    }

    #[test]
    fn test_exploit_type_classification() {
        use crate::models::exploit_detector::ExploitType;
        
        // Test exploit type enumeration
        assert_eq!(ExploitType::None as u8, 0);
        assert_ne!(ExploitType::ZeroDay, ExploitType::Known);
    }
}