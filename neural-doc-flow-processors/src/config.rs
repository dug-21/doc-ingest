//! Configuration types for neural document flow processors

use crate::error::{NeuralError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Main configuration for neural processing engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralConfig {
    /// Path to model files directory
    pub model_path: PathBuf,

    /// Models to load on initialization
    pub models: Vec<ModelConfig>,

    /// Default activation function for hidden layers
    pub default_activation: ActivationFunction,

    /// Processing configuration
    pub processing: ProcessingConfig,

    /// Performance configuration
    pub performance: PerformanceConfig,

    /// Quality thresholds
    pub quality: QualityConfig,

    /// Sparse network connection rate (0.0 to 1.0)
    pub sparse_connection_rate: f32,

    /// Enable SIMD acceleration where available
    pub enable_simd: bool,

    /// Memory management settings
    pub memory: MemoryConfig,

    /// Logging configuration
    pub logging: LoggingConfig,

    /// Feature extraction settings
    pub features: FeatureConfig,
}

impl Default for NeuralConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("models"),
            models: vec![
                ModelConfig::text_model(),
                ModelConfig::layout_model(),
                ModelConfig::table_model(),
                ModelConfig::quality_model(),
            ],
            default_activation: ActivationFunction::Sigmoid,
            processing: ProcessingConfig::default(),
            performance: PerformanceConfig::default(),
            quality: QualityConfig::default(),
            sparse_connection_rate: 0.8,
            enable_simd: true,
            memory: MemoryConfig::default(),
            logging: LoggingConfig::default(),
            features: FeatureConfig::default(),
        }
    }
}

/// Configuration for individual neural models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Type of model (text, layout, table, etc.)
    pub model_type: ModelType,

    /// Path to model file
    pub path: String,

    /// Model version
    pub version: String,

    /// Network architecture
    pub architecture: NetworkArchitecture,

    /// Training algorithm to use
    pub training_algorithm: Option<TrainingAlgorithm>,

    /// Model-specific settings
    pub settings: HashMap<String, serde_json::Value>,

    /// Whether to preload this model
    pub preload: bool,

    /// Model optimization settings
    pub optimization: OptimizationConfig,
}

impl ModelConfig {
    /// Create configuration for text processing model
    pub fn text_model() -> Self {
        Self {
            model_type: ModelType::Text,
            path: "models/text_enhancement.fann".to_string(),
            version: "1.0.0".to_string(),
            architecture: NetworkArchitecture {
                layers: vec![64, 128, 64, 32, 8],
                network_type: NetworkType::Standard,
            },
            training_algorithm: Some(TrainingAlgorithm::Rprop),
            settings: HashMap::new(),
            preload: true,
            optimization: OptimizationConfig::default(),
        }
    }

    /// Create configuration for layout analysis model
    pub fn layout_model() -> Self {
        Self {
            model_type: ModelType::Layout,
            path: "models/layout_analysis.fann".to_string(),
            version: "1.0.0".to_string(),
            architecture: NetworkArchitecture {
                layers: vec![128, 256, 128, 64, 16],
                network_type: NetworkType::Shortcut,
            },
            training_algorithm: Some(TrainingAlgorithm::Rprop),
            settings: HashMap::new(),
            preload: true,
            optimization: OptimizationConfig::default(),
        }
    }

    /// Create configuration for table detection model
    pub fn table_model() -> Self {
        Self {
            model_type: ModelType::Table,
            path: "models/table_detection.fann".to_string(),
            version: "1.0.0".to_string(),
            architecture: NetworkArchitecture {
                layers: vec![32, 64, 32, 16, 4],
                network_type: NetworkType::Standard,
            },
            training_algorithm: Some(TrainingAlgorithm::Rprop),
            settings: HashMap::new(),
            preload: true,
            optimization: OptimizationConfig::default(),
        }
    }

    /// Create configuration for quality assessment model
    pub fn quality_model() -> Self {
        Self {
            model_type: ModelType::Quality,
            path: "models/quality_assessment.fann".to_string(),
            version: "1.0.0".to_string(),
            architecture: NetworkArchitecture {
                layers: vec![16, 32, 16, 8, 4],
                network_type: NetworkType::Standard,
            },
            training_algorithm: Some(TrainingAlgorithm::Rprop),
            settings: HashMap::new(),
            preload: true,
            optimization: OptimizationConfig::default(),
        }
    }
}

/// Processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    /// Maximum parallel threads for processing
    pub max_threads: usize,

    /// Batch size for processing multiple items
    pub batch_size: usize,

    /// Timeout for individual processing operations
    pub timeout_ms: u64,

    /// Enable caching of processed results
    pub enable_caching: bool,

    /// Cache size limit (number of items)
    pub cache_size: usize,

    /// Processing modes to enable
    pub enabled_modes: Vec<ProcessingMode>,
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            max_threads: num_cpus::get(),
            batch_size: 32,
            timeout_ms: 5000,
            enable_caching: true,
            cache_size: 1000,
            enabled_modes: vec![
                ProcessingMode::TextEnhancement,
                ProcessingMode::LayoutAnalysis,
                ProcessingMode::TableDetection,
                ProcessingMode::QualityAssessment,
            ],
        }
    }
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Enable memory mapping for large models
    pub use_memory_mapping: bool,

    /// Model compression level (0-9)
    pub compression_level: u8,

    /// Inference batch size
    pub inference_batch_size: usize,

    /// Maximum memory usage in MB
    pub max_memory_mb: usize,

    /// Optimization target
    pub optimization_target: OptimizationTarget,

    /// Enable model quantization
    pub enable_quantization: bool,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            use_memory_mapping: true,
            compression_level: 6,
            inference_batch_size: 16,
            max_memory_mb: 1024, // 1GB
            optimization_target: OptimizationTarget::Balanced,
            enable_quantization: false,
        }
    }
}

/// Quality assessment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityConfig {
    /// Minimum confidence threshold for accepting results
    pub min_confidence: f32,

    /// Text extraction accuracy threshold
    pub text_accuracy_threshold: f32,

    /// Layout analysis accuracy threshold
    pub layout_accuracy_threshold: f32,

    /// Table detection accuracy threshold
    pub table_accuracy_threshold: f32,

    /// Whether to enable quality auto-correction
    pub enable_auto_correction: bool,

    /// Number of quality validation passes
    pub validation_passes: usize,
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.8,
            text_accuracy_threshold: 0.95,
            layout_accuracy_threshold: 0.9,
            table_accuracy_threshold: 0.85,
            enable_auto_correction: true,
            validation_passes: 2,
        }
    }
}

/// Memory management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Memory pool size for neural operations
    pub pool_size_mb: usize,

    /// Enable memory pool for performance
    pub enable_pooling: bool,

    /// Garbage collection threshold
    pub gc_threshold_mb: usize,

    /// Model cache size
    pub model_cache_size: usize,

    /// Feature cache size
    pub feature_cache_size: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            pool_size_mb: 512,
            enable_pooling: true,
            gc_threshold_mb: 256,
            model_cache_size: 10,
            feature_cache_size: 1000,
        }
    }
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level for neural operations
    pub level: LogLevel,

    /// Enable performance metrics logging
    pub enable_metrics: bool,

    /// Enable detailed debug logging
    pub enable_debug: bool,

    /// Log file path (optional)
    pub log_file: Option<PathBuf>,

    /// Maximum log file size in MB
    pub max_log_size_mb: usize,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
            enable_metrics: true,
            enable_debug: false,
            log_file: None,
            max_log_size_mb: 100,
        }
    }
}

/// Feature extraction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureConfig {
    /// Text feature extraction settings
    pub text: TextFeatureConfig,

    /// Layout feature extraction settings
    pub layout: LayoutFeatureConfig,

    /// Table feature extraction settings
    pub table: TableFeatureConfig,

    /// Image feature extraction settings
    pub image: ImageFeatureConfig,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            text: TextFeatureConfig::default(),
            layout: LayoutFeatureConfig::default(),
            table: TableFeatureConfig::default(),
            image: ImageFeatureConfig::default(),
        }
    }
}

/// Text feature extraction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextFeatureConfig {
    /// Maximum text length to analyze
    pub max_length: usize,

    /// Enable language detection
    pub enable_language_detection: bool,

    /// Enable sentiment analysis
    pub enable_sentiment: bool,

    /// Feature vector size
    pub feature_size: usize,
}

impl Default for TextFeatureConfig {
    fn default() -> Self {
        Self {
            max_length: 10000,
            enable_language_detection: true,
            enable_sentiment: false,
            feature_size: 64,
        }
    }
}

/// Layout feature extraction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutFeatureConfig {
    /// Grid resolution for layout analysis
    pub grid_resolution: (usize, usize),

    /// Enable geometric features
    pub enable_geometric: bool,

    /// Enable spatial relationships
    pub enable_spatial: bool,

    /// Feature vector size
    pub feature_size: usize,
}

impl Default for LayoutFeatureConfig {
    fn default() -> Self {
        Self {
            grid_resolution: (32, 32),
            enable_geometric: true,
            enable_spatial: true,
            feature_size: 128,
        }
    }
}

/// Table feature extraction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableFeatureConfig {
    /// Minimum table size to detect
    pub min_table_size: (usize, usize),

    /// Table structure features
    pub enable_structure_features: bool,

    /// Content-based features
    pub enable_content_features: bool,

    /// Feature vector size
    pub feature_size: usize,
}

impl Default for TableFeatureConfig {
    fn default() -> Self {
        Self {
            min_table_size: (2, 2),
            enable_structure_features: true,
            enable_content_features: true,
            feature_size: 32,
        }
    }
}

/// Image feature extraction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageFeatureConfig {
    /// Maximum image resolution to process
    pub max_resolution: (usize, usize),

    /// Enable histogram features
    pub enable_histogram: bool,

    /// Enable edge detection features
    pub enable_edges: bool,

    /// Feature vector size
    pub feature_size: usize,
}

impl Default for ImageFeatureConfig {
    fn default() -> Self {
        Self {
            max_resolution: (1024, 1024),
            enable_histogram: true,
            enable_edges: true,
            feature_size: 256,
        }
    }
}

/// Network architecture definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkArchitecture {
    /// Layer sizes (including input and output)
    pub layers: Vec<u32>,

    /// Network type
    pub network_type: NetworkType,
}

/// Model optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Enable model pruning
    pub enable_pruning: bool,

    /// Pruning threshold
    pub pruning_threshold: f32,

    /// Enable weight quantization
    pub enable_quantization: bool,

    /// Quantization bits (8, 16)
    pub quantization_bits: u8,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            enable_pruning: false,
            pruning_threshold: 0.01,
            enable_quantization: false,
            quantization_bits: 16,
        }
    }
}

// Enums for configuration options

/// Model types supported by the neural engine
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelType {
    Text,
    Layout,
    Table,
    Image,
    Quality,
    Custom(String),
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelType::Text => write!(f, "text"),
            ModelType::Layout => write!(f, "layout"),
            ModelType::Table => write!(f, "table"),
            ModelType::Image => write!(f, "image"),
            ModelType::Quality => write!(f, "quality"),
            ModelType::Custom(name) => write!(f, "custom_{}", name),
        }
    }
}

/// Neural network types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkType {
    Standard,
    Sparse,
    Shortcut,
    Cascade,
}

/// Activation functions supported by ruv-FANN
#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// Training algorithms supported by ruv-FANN
#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// Processing modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingMode {
    TextEnhancement,
    LayoutAnalysis,
    TableDetection,
    ImageProcessing,
    QualityAssessment,
    Custom(String),
}

/// Optimization targets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationTarget {
    Speed,
    Accuracy,
    Memory,
    Balanced,
}

/// Log levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl NeuralConfig {
    /// Load configuration from file
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| NeuralError::Configuration(format!("Failed to read config file: {}", e)))?;
        
        let config: NeuralConfig = if path.as_ref().extension().and_then(|s| s.to_str()) == Some("toml") {
            toml::from_str(&content)
                .map_err(|e| NeuralError::Configuration(format!("Failed to parse TOML config: {}", e)))?
        } else {
            serde_json::from_str(&content)
                .map_err(|e| NeuralError::Configuration(format!("Failed to parse JSON config: {}", e)))?
        };
        
        config.validate()?;
        Ok(config)
    }

    /// Save configuration to file
    pub fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        self.validate()?;
        
        let content = if path.as_ref().extension().and_then(|s| s.to_str()) == Some("toml") {
            toml::to_string_pretty(self)
                .map_err(|e| NeuralError::Configuration(format!("Failed to serialize to TOML: {}", e)))?
        } else {
            serde_json::to_string_pretty(self)?
        };
        
        std::fs::write(path, content)
            .map_err(|e| NeuralError::Configuration(format!("Failed to write config file: {}", e)))?;
        
        Ok(())
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        // Validate model paths
        for model in &self.models {
            if model.path.is_empty() {
                return Err(NeuralError::Configuration(
                    format!("Model {} has empty path", model.model_type)
                ));
            }
        }

        // Validate thresholds
        if self.quality.min_confidence < 0.0 || self.quality.min_confidence > 1.0 {
            return Err(NeuralError::Configuration(
                "min_confidence must be between 0.0 and 1.0".to_string()
            ));
        }

        if self.sparse_connection_rate < 0.0 || self.sparse_connection_rate > 1.0 {
            return Err(NeuralError::Configuration(
                "sparse_connection_rate must be between 0.0 and 1.0".to_string()
            ));
        }

        // Validate performance settings
        if self.performance.max_memory_mb == 0 {
            return Err(NeuralError::Configuration(
                "max_memory_mb must be greater than 0".to_string()
            ));
        }

        Ok(())
    }

    /// Create configuration for specific domain
    pub fn for_domain(domain: &str) -> Self {
        let mut config = Self::default();
        
        match domain {
            "legal" => {
                config.models.push(ModelConfig {
                    model_type: ModelType::Custom("legal_classifier".to_string()),
                    path: "models/legal/classifier.fann".to_string(),
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
            }
            "medical" => {
                config.models.push(ModelConfig {
                    model_type: ModelType::Custom("medical_classifier".to_string()),
                    path: "models/medical/classifier.fann".to_string(),
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
            }
            "financial" => {
                config.models.push(ModelConfig {
                    model_type: ModelType::Custom("financial_classifier".to_string()),
                    path: "models/financial/classifier.fann".to_string(),
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
            }
            _ => {} // Use default configuration
        }
        
        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_config() {
        let config = NeuralConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.models.len(), 4);
        assert!(config.enable_simd);
    }

    #[test]
    fn test_model_config_creation() {
        let text_config = ModelConfig::text_model();
        assert_eq!(text_config.model_type, ModelType::Text);
        assert!(text_config.preload);
        assert_eq!(text_config.architecture.layers.len(), 5);
    }

    #[test]
    fn test_config_validation() {
        let mut config = NeuralConfig::default();
        config.quality.min_confidence = 1.5; // Invalid
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_serialization() {
        let config = NeuralConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: NeuralConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.models.len(), deserialized.models.len());
    }

    #[test]
    fn test_domain_config() {
        let legal_config = NeuralConfig::for_domain("legal");
        assert_eq!(legal_config.models.len(), 5); // 4 default + 1 legal
        
        let has_legal_model = legal_config.models.iter().any(|m| {
            matches!(m.model_type, ModelType::Custom(ref name) if name == "legal_classifier")
        });
        assert!(has_legal_model);
    }

    #[test]
    fn test_config_file_operations() {
        let config = NeuralConfig::default();
        
        // Test JSON format
        let temp_file = NamedTempFile::new().unwrap();
        let json_path = temp_file.path().with_extension("json");
        
        config.save_to_file(&json_path).unwrap();
        let loaded_config = NeuralConfig::from_file(&json_path).unwrap();
        assert_eq!(config.models.len(), loaded_config.models.len());
    }
}