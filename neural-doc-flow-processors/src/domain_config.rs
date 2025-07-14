//! Domain-specific neural configurations
//!
//! This module provides specialized neural network configurations for different document domains
//! such as legal, medical, financial, and technical documents.

use crate::{
    config::{NeuralConfig, ModelConfig, ModelType, NetworkArchitecture, NetworkType, TrainingAlgorithm, OptimizationConfig},
    error::{NeuralError, Result},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Domain-specific neural configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainConfig {
    /// Domain name
    pub domain: String,
    
    /// Domain-specific neural configuration
    pub neural_config: NeuralConfig,
    
    /// Domain-specific feature extractors
    pub feature_extractors: HashMap<String, FeatureExtractorConfig>,
    
    /// Domain-specific quality thresholds
    pub quality_thresholds: QualityThresholds,
    
    /// Domain-specific training data paths
    pub training_data_paths: HashMap<ModelType, PathBuf>,
    
    /// Domain-specific model architectures
    pub model_architectures: HashMap<ModelType, NetworkArchitecture>,
}

/// Feature extractor configuration for domain-specific processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureExtractorConfig {
    /// Feature vector size
    pub feature_size: usize,
    
    /// Domain-specific feature weights
    pub feature_weights: Vec<f32>,
    
    /// Feature normalization parameters
    pub normalization: NormalizationConfig,
    
    /// Feature-specific processing options
    pub options: HashMap<String, serde_json::Value>,
}

/// Feature normalization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationConfig {
    /// Normalization method
    pub method: NormalizationMethod,
    
    /// Min-max normalization parameters
    pub min_values: Option<Vec<f32>>,
    pub max_values: Option<Vec<f32>>,
    
    /// Z-score normalization parameters
    pub mean_values: Option<Vec<f32>>,
    pub std_values: Option<Vec<f32>>,
}

/// Normalization methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NormalizationMethod {
    None,
    MinMax,
    ZScore,
    Robust,
    Custom(String),
}

/// Domain-specific quality thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityThresholds {
    /// Minimum confidence for accepting results
    pub min_confidence: f32,
    
    /// Text extraction accuracy threshold
    pub text_accuracy: f32,
    
    /// Layout analysis accuracy threshold
    pub layout_accuracy: f32,
    
    /// Table detection accuracy threshold
    pub table_accuracy: f32,
    
    /// Image processing accuracy threshold
    pub image_accuracy: f32,
    
    /// Domain-specific quality metrics
    pub domain_specific: HashMap<String, f32>,
}

/// Domain configuration factory
pub struct DomainConfigFactory;

impl DomainConfigFactory {
    /// Create configuration for legal document processing
    pub fn legal_domain() -> DomainConfig {
        let mut neural_config = NeuralConfig::default();
        
        // Add legal-specific models
        neural_config.models.push(ModelConfig {
            model_type: ModelType::Custom("legal_classifier".to_string()),
            path: "models/legal/legal_classifier.fann".to_string(),
            version: "1.0.0".to_string(),
            architecture: NetworkArchitecture {
                layers: vec![128, 256, 128, 64, 16],
                network_type: NetworkType::Standard,
            },
            training_algorithm: Some(TrainingAlgorithm::Rprop),
            settings: HashMap::new(),
            preload: true,
            optimization: OptimizationConfig::default(),
        });
        
        neural_config.models.push(ModelConfig {
            model_type: ModelType::Custom("legal_entity_extractor".to_string()),
            path: "models/legal/entity_extractor.fann".to_string(),
            version: "1.0.0".to_string(),
            architecture: NetworkArchitecture {
                layers: vec![64, 128, 64, 32, 8],
                network_type: NetworkType::Shortcut,
            },
            training_algorithm: Some(TrainingAlgorithm::Rprop),
            settings: HashMap::new(),
            preload: true,
            optimization: OptimizationConfig::default(),
        });
        
        // Legal-specific feature extractors
        let mut feature_extractors = HashMap::new();
        
        feature_extractors.insert("legal_text".to_string(), FeatureExtractorConfig {
            feature_size: 128,
            feature_weights: vec![1.0; 128],
            normalization: NormalizationConfig {
                method: NormalizationMethod::ZScore,
                min_values: None,
                max_values: None,
                mean_values: Some(vec![0.0; 128]),
                std_values: Some(vec![1.0; 128]),
            },
            options: {
                let mut opts = HashMap::new();
                opts.insert("extract_citations".to_string(), serde_json::Value::Bool(true));
                opts.insert("extract_case_numbers".to_string(), serde_json::Value::Bool(true));
                opts.insert("extract_legal_entities".to_string(), serde_json::Value::Bool(true));
                opts
            },
        });
        
        feature_extractors.insert("legal_structure".to_string(), FeatureExtractorConfig {
            feature_size: 64,
            feature_weights: vec![1.0; 64],
            normalization: NormalizationConfig {
                method: NormalizationMethod::MinMax,
                min_values: Some(vec![0.0; 64]),
                max_values: Some(vec![1.0; 64]),
                mean_values: None,
                std_values: None,
            },
            options: {
                let mut opts = HashMap::new();
                opts.insert("detect_sections".to_string(), serde_json::Value::Bool(true));
                opts.insert("detect_paragraphs".to_string(), serde_json::Value::Bool(true));
                opts
            },
        });
        
        // Legal-specific quality thresholds
        let quality_thresholds = QualityThresholds {
            min_confidence: 0.95, // Higher threshold for legal documents
            text_accuracy: 0.99,
            layout_accuracy: 0.95,
            table_accuracy: 0.90,
            image_accuracy: 0.85,
            domain_specific: {
                let mut domain_metrics = HashMap::new();
                domain_metrics.insert("citation_accuracy".to_string(), 0.98);
                domain_metrics.insert("case_number_accuracy".to_string(), 0.99);
                domain_metrics.insert("legal_entity_accuracy".to_string(), 0.95);
                domain_metrics
            },
        };
        
        // Training data paths
        let mut training_data_paths = HashMap::new();
        training_data_paths.insert(ModelType::Text, PathBuf::from("training_data/legal/text_samples.json"));
        training_data_paths.insert(ModelType::Layout, PathBuf::from("training_data/legal/layout_samples.json"));
        training_data_paths.insert(ModelType::Table, PathBuf::from("training_data/legal/table_samples.json"));
        
        // Model architectures
        let mut model_architectures = HashMap::new();
        model_architectures.insert(ModelType::Text, NetworkArchitecture {
            layers: vec![128, 256, 128, 64, 32],
            network_type: NetworkType::Standard,
        });
        model_architectures.insert(ModelType::Layout, NetworkArchitecture {
            layers: vec![64, 128, 64, 32, 16],
            network_type: NetworkType::Shortcut,
        });
        
        DomainConfig {
            domain: "legal".to_string(),
            neural_config,
            feature_extractors,
            quality_thresholds,
            training_data_paths,
            model_architectures,
        }
    }
    
    /// Create configuration for medical document processing
    pub fn medical_domain() -> DomainConfig {
        let mut neural_config = NeuralConfig::default();
        
        // Add medical-specific models
        neural_config.models.push(ModelConfig {
            model_type: ModelType::Custom("medical_classifier".to_string()),
            path: "models/medical/medical_classifier.fann".to_string(),
            version: "1.0.0".to_string(),
            architecture: NetworkArchitecture {
                layers: vec![96, 192, 96, 48, 12],
                network_type: NetworkType::Standard,
            },
            training_algorithm: Some(TrainingAlgorithm::Rprop),
            settings: HashMap::new(),
            preload: true,
            optimization: OptimizationConfig::default(),
        });
        
        neural_config.models.push(ModelConfig {
            model_type: ModelType::Custom("medical_entity_extractor".to_string()),
            path: "models/medical/entity_extractor.fann".to_string(),
            version: "1.0.0".to_string(),
            architecture: NetworkArchitecture {
                layers: vec![64, 128, 64, 32, 8],
                network_type: NetworkType::Shortcut,
            },
            training_algorithm: Some(TrainingAlgorithm::Rprop),
            settings: HashMap::new(),
            preload: true,
            optimization: OptimizationConfig::default(),
        });
        
        // Medical-specific feature extractors
        let mut feature_extractors = HashMap::new();
        
        feature_extractors.insert("medical_text".to_string(), FeatureExtractorConfig {
            feature_size: 96,
            feature_weights: vec![1.0; 96],
            normalization: NormalizationConfig {
                method: NormalizationMethod::ZScore,
                min_values: None,
                max_values: None,
                mean_values: Some(vec![0.0; 96]),
                std_values: Some(vec![1.0; 96]),
            },
            options: {
                let mut opts = HashMap::new();
                opts.insert("extract_medications".to_string(), serde_json::Value::Bool(true));
                opts.insert("extract_diagnoses".to_string(), serde_json::Value::Bool(true));
                opts.insert("extract_procedures".to_string(), serde_json::Value::Bool(true));
                opts.insert("extract_dates".to_string(), serde_json::Value::Bool(true));
                opts
            },
        });
        
        feature_extractors.insert("medical_structure".to_string(), FeatureExtractorConfig {
            feature_size: 48,
            feature_weights: vec![1.0; 48],
            normalization: NormalizationConfig {
                method: NormalizationMethod::MinMax,
                min_values: Some(vec![0.0; 48]),
                max_values: Some(vec![1.0; 48]),
                mean_values: None,
                std_values: None,
            },
            options: {
                let mut opts = HashMap::new();
                opts.insert("detect_sections".to_string(), serde_json::Value::Bool(true));
                opts.insert("detect_forms".to_string(), serde_json::Value::Bool(true));
                opts
            },
        });
        
        // Medical-specific quality thresholds
        let quality_thresholds = QualityThresholds {
            min_confidence: 0.98, // Very high threshold for medical documents
            text_accuracy: 0.995,
            layout_accuracy: 0.95,
            table_accuracy: 0.92,
            image_accuracy: 0.88,
            domain_specific: {
                let mut domain_metrics = HashMap::new();
                domain_metrics.insert("medication_accuracy".to_string(), 0.99);
                domain_metrics.insert("diagnosis_accuracy".to_string(), 0.98);
                domain_metrics.insert("procedure_accuracy".to_string(), 0.97);
                domain_metrics.insert("date_accuracy".to_string(), 0.995);
                domain_metrics
            },
        };
        
        // Training data paths
        let mut training_data_paths = HashMap::new();
        training_data_paths.insert(ModelType::Text, PathBuf::from("training_data/medical/text_samples.json"));
        training_data_paths.insert(ModelType::Layout, PathBuf::from("training_data/medical/layout_samples.json"));
        training_data_paths.insert(ModelType::Table, PathBuf::from("training_data/medical/table_samples.json"));
        
        // Model architectures
        let mut model_architectures = HashMap::new();
        model_architectures.insert(ModelType::Text, NetworkArchitecture {
            layers: vec![96, 192, 96, 48, 24],
            network_type: NetworkType::Standard,
        });
        model_architectures.insert(ModelType::Layout, NetworkArchitecture {
            layers: vec![48, 96, 48, 24, 12],
            network_type: NetworkType::Shortcut,
        });
        
        DomainConfig {
            domain: "medical".to_string(),
            neural_config,
            feature_extractors,
            quality_thresholds,
            training_data_paths,
            model_architectures,
        }
    }
    
    /// Create configuration for financial document processing
    pub fn financial_domain() -> DomainConfig {
        let mut neural_config = NeuralConfig::default();
        
        // Add financial-specific models
        neural_config.models.push(ModelConfig {
            model_type: ModelType::Custom("financial_classifier".to_string()),
            path: "models/financial/financial_classifier.fann".to_string(),
            version: "1.0.0".to_string(),
            architecture: NetworkArchitecture {
                layers: vec![80, 160, 80, 40, 10],
                network_type: NetworkType::Standard,
            },
            training_algorithm: Some(TrainingAlgorithm::Rprop),
            settings: HashMap::new(),
            preload: true,
            optimization: OptimizationConfig::default(),
        });
        
        neural_config.models.push(ModelConfig {
            model_type: ModelType::Custom("financial_entity_extractor".to_string()),
            path: "models/financial/entity_extractor.fann".to_string(),
            version: "1.0.0".to_string(),
            architecture: NetworkArchitecture {
                layers: vec![64, 128, 64, 32, 8],
                network_type: NetworkType::Shortcut,
            },
            training_algorithm: Some(TrainingAlgorithm::Rprop),
            settings: HashMap::new(),
            preload: true,
            optimization: OptimizationConfig::default(),
        });
        
        // Financial-specific feature extractors
        let mut feature_extractors = HashMap::new();
        
        feature_extractors.insert("financial_text".to_string(), FeatureExtractorConfig {
            feature_size: 80,
            feature_weights: vec![1.0; 80],
            normalization: NormalizationConfig {
                method: NormalizationMethod::Robust,
                min_values: None,
                max_values: None,
                mean_values: Some(vec![0.0; 80]),
                std_values: Some(vec![1.0; 80]),
            },
            options: {
                let mut opts = HashMap::new();
                opts.insert("extract_amounts".to_string(), serde_json::Value::Bool(true));
                opts.insert("extract_dates".to_string(), serde_json::Value::Bool(true));
                opts.insert("extract_account_numbers".to_string(), serde_json::Value::Bool(true));
                opts.insert("extract_transaction_ids".to_string(), serde_json::Value::Bool(true));
                opts
            },
        });
        
        feature_extractors.insert("financial_tables".to_string(), FeatureExtractorConfig {
            feature_size: 40,
            feature_weights: vec![1.0; 40],
            normalization: NormalizationConfig {
                method: NormalizationMethod::MinMax,
                min_values: Some(vec![0.0; 40]),
                max_values: Some(vec![1.0; 40]),
                mean_values: None,
                std_values: None,
            },
            options: {
                let mut opts = HashMap::new();
                opts.insert("detect_balance_sheets".to_string(), serde_json::Value::Bool(true));
                opts.insert("detect_income_statements".to_string(), serde_json::Value::Bool(true));
                opts.insert("detect_cash_flows".to_string(), serde_json::Value::Bool(true));
                opts
            },
        });
        
        // Financial-specific quality thresholds
        let quality_thresholds = QualityThresholds {
            min_confidence: 0.96, // High threshold for financial documents
            text_accuracy: 0.98,
            layout_accuracy: 0.94,
            table_accuracy: 0.95, // Very high for financial tables
            image_accuracy: 0.87,
            domain_specific: {
                let mut domain_metrics = HashMap::new();
                domain_metrics.insert("amount_accuracy".to_string(), 0.995);
                domain_metrics.insert("date_accuracy".to_string(), 0.99);
                domain_metrics.insert("account_accuracy".to_string(), 0.98);
                domain_metrics.insert("transaction_accuracy".to_string(), 0.97);
                domain_metrics
            },
        };
        
        // Training data paths
        let mut training_data_paths = HashMap::new();
        training_data_paths.insert(ModelType::Text, PathBuf::from("training_data/financial/text_samples.json"));
        training_data_paths.insert(ModelType::Layout, PathBuf::from("training_data/financial/layout_samples.json"));
        training_data_paths.insert(ModelType::Table, PathBuf::from("training_data/financial/table_samples.json"));
        
        // Model architectures
        let mut model_architectures = HashMap::new();
        model_architectures.insert(ModelType::Text, NetworkArchitecture {
            layers: vec![80, 160, 80, 40, 20],
            network_type: NetworkType::Standard,
        });
        model_architectures.insert(ModelType::Table, NetworkArchitecture {
            layers: vec![40, 80, 40, 20, 10],
            network_type: NetworkType::Standard,
        });
        
        DomainConfig {
            domain: "financial".to_string(),
            neural_config,
            feature_extractors,
            quality_thresholds,
            training_data_paths,
            model_architectures,
        }
    }
    
    /// Create configuration for technical document processing
    pub fn technical_domain() -> DomainConfig {
        let mut neural_config = NeuralConfig::default();
        
        // Add technical-specific models
        neural_config.models.push(ModelConfig {
            model_type: ModelType::Custom("technical_classifier".to_string()),
            path: "models/technical/technical_classifier.fann".to_string(),
            version: "1.0.0".to_string(),
            architecture: NetworkArchitecture {
                layers: vec![112, 224, 112, 56, 14],
                network_type: NetworkType::Standard,
            },
            training_algorithm: Some(TrainingAlgorithm::Rprop),
            settings: HashMap::new(),
            preload: true,
            optimization: OptimizationConfig::default(),
        });
        
        // Technical-specific feature extractors
        let mut feature_extractors = HashMap::new();
        
        feature_extractors.insert("technical_text".to_string(), FeatureExtractorConfig {
            feature_size: 112,
            feature_weights: vec![1.0; 112],
            normalization: NormalizationConfig {
                method: NormalizationMethod::ZScore,
                min_values: None,
                max_values: None,
                mean_values: Some(vec![0.0; 112]),
                std_values: Some(vec![1.0; 112]),
            },
            options: {
                let mut opts = HashMap::new();
                opts.insert("extract_code_blocks".to_string(), serde_json::Value::Bool(true));
                opts.insert("extract_formulas".to_string(), serde_json::Value::Bool(true));
                opts.insert("extract_references".to_string(), serde_json::Value::Bool(true));
                opts.insert("extract_figures".to_string(), serde_json::Value::Bool(true));
                opts
            },
        });
        
        feature_extractors.insert("technical_diagrams".to_string(), FeatureExtractorConfig {
            feature_size: 56,
            feature_weights: vec![1.0; 56],
            normalization: NormalizationConfig {
                method: NormalizationMethod::MinMax,
                min_values: Some(vec![0.0; 56]),
                max_values: Some(vec![1.0; 56]),
                mean_values: None,
                std_values: None,
            },
            options: {
                let mut opts = HashMap::new();
                opts.insert("detect_flowcharts".to_string(), serde_json::Value::Bool(true));
                opts.insert("detect_schematics".to_string(), serde_json::Value::Bool(true));
                opts.insert("detect_graphs".to_string(), serde_json::Value::Bool(true));
                opts
            },
        });
        
        // Technical-specific quality thresholds
        let quality_thresholds = QualityThresholds {
            min_confidence: 0.92, // Moderate threshold for technical documents
            text_accuracy: 0.95,
            layout_accuracy: 0.93,
            table_accuracy: 0.90,
            image_accuracy: 0.88,
            domain_specific: {
                let mut domain_metrics = HashMap::new();
                domain_metrics.insert("code_accuracy".to_string(), 0.97);
                domain_metrics.insert("formula_accuracy".to_string(), 0.95);
                domain_metrics.insert("reference_accuracy".to_string(), 0.92);
                domain_metrics.insert("figure_accuracy".to_string(), 0.90);
                domain_metrics
            },
        };
        
        // Training data paths
        let mut training_data_paths = HashMap::new();
        training_data_paths.insert(ModelType::Text, PathBuf::from("training_data/technical/text_samples.json"));
        training_data_paths.insert(ModelType::Layout, PathBuf::from("training_data/technical/layout_samples.json"));
        training_data_paths.insert(ModelType::Image, PathBuf::from("training_data/technical/image_samples.json"));
        
        // Model architectures
        let mut model_architectures = HashMap::new();
        model_architectures.insert(ModelType::Text, NetworkArchitecture {
            layers: vec![112, 224, 112, 56, 28],
            network_type: NetworkType::Standard,
        });
        model_architectures.insert(ModelType::Image, NetworkArchitecture {
            layers: vec![256, 512, 256, 128, 64],
            network_type: NetworkType::Shortcut,
        });
        
        DomainConfig {
            domain: "technical".to_string(),
            neural_config,
            feature_extractors,
            quality_thresholds,
            training_data_paths,
            model_architectures,
        }
    }
    
    /// Get domain configuration by name
    pub fn get_domain_config(domain: &str) -> Result<DomainConfig> {
        match domain.to_lowercase().as_str() {
            "legal" => Ok(Self::legal_domain()),
            "medical" => Ok(Self::medical_domain()),
            "financial" => Ok(Self::financial_domain()),
            "technical" => Ok(Self::technical_domain()),
            _ => Err(NeuralError::Configuration(format!("Unknown domain: {}", domain))),
        }
    }
    
    /// List all available domains
    pub fn available_domains() -> Vec<String> {
        vec![
            "legal".to_string(),
            "medical".to_string(),
            "financial".to_string(),
            "technical".to_string(),
        ]
    }
}

impl Default for NormalizationConfig {
    fn default() -> Self {
        Self {
            method: NormalizationMethod::MinMax,
            min_values: None,
            max_values: None,
            mean_values: None,
            std_values: None,
        }
    }
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            min_confidence: 0.8,
            text_accuracy: 0.95,
            layout_accuracy: 0.9,
            table_accuracy: 0.85,
            image_accuracy: 0.8,
            domain_specific: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_domain_config_creation() {
        let legal_config = DomainConfigFactory::legal_domain();
        assert_eq!(legal_config.domain, "legal");
        assert!(legal_config.neural_config.models.len() > 4); // Should have default + legal models
        assert!(legal_config.quality_thresholds.min_confidence > 0.9);
        
        let medical_config = DomainConfigFactory::medical_domain();
        assert_eq!(medical_config.domain, "medical");
        assert!(medical_config.quality_thresholds.min_confidence > 0.95);
        
        let financial_config = DomainConfigFactory::financial_domain();
        assert_eq!(financial_config.domain, "financial");
        assert!(financial_config.quality_thresholds.table_accuracy > 0.9);
        
        let technical_config = DomainConfigFactory::technical_domain();
        assert_eq!(technical_config.domain, "technical");
        assert!(technical_config.feature_extractors.contains_key("technical_text"));
    }
    
    #[test]
    fn test_get_domain_config() {
        let legal_config = DomainConfigFactory::get_domain_config("legal").unwrap();
        assert_eq!(legal_config.domain, "legal");
        
        let medical_config = DomainConfigFactory::get_domain_config("MEDICAL").unwrap();
        assert_eq!(medical_config.domain, "medical");
        
        let result = DomainConfigFactory::get_domain_config("unknown");
        assert!(result.is_err());
    }
    
    #[test]
    fn test_available_domains() {
        let domains = DomainConfigFactory::available_domains();
        assert_eq!(domains.len(), 4);
        assert!(domains.contains(&"legal".to_string()));
        assert!(domains.contains(&"medical".to_string()));
        assert!(domains.contains(&"financial".to_string()));
        assert!(domains.contains(&"technical".to_string()));
    }
}