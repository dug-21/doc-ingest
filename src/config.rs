//! Configuration management for doc-ingest
//!
//! This module handles all configuration aspects including defaults,
//! environment variables, and configuration file loading.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use neural_doc_flow_coordination::CoordinationConfig;

/// Ingestion system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestConfig {
    pub daa_config: CoordinationConfig,
    pub max_workers: usize,
    pub processing_timeout: Duration,
}

impl Default for IngestConfig {
    fn default() -> Self {
        Self {
            daa_config: CoordinationConfig::default(),
            max_workers: 8,
            processing_timeout: Duration::from_secs(300),
        }
    }
}

/// Main configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Core engine configuration
    pub core: CoreConfig,
    
    /// DAA coordination configuration
    pub daa: DaaConfig,
    
    /// Neural processing configuration
    pub neural: NeuralConfig,
    
    /// Source plugin configuration
    pub sources: SourcesConfig,
    
    /// Performance and optimization settings
    pub performance: PerformanceConfig,
    
    /// Security settings
    pub security: SecurityConfig,
    
    /// Logging and monitoring configuration
    pub monitoring: MonitoringConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            core: CoreConfig::default(),
            daa: DaaConfig::default(),
            neural: NeuralConfig::default(),
            sources: SourcesConfig::default(),
            performance: PerformanceConfig::default(),
            security: SecurityConfig::default(),
            monitoring: MonitoringConfig::default(),
        }
    }
}

impl Config {
    /// Load configuration from file
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| NeuralDocFlowError::config(format!("Failed to read config file: {}", e)))?;
        
        if content.trim_start().starts_with('{') {
            // JSON format
            serde_json::from_str(&content)
                .map_err(|e| NeuralDocFlowError::config(format!("Invalid JSON config: {}", e)))
        } else {
            // YAML format
            serde_yaml::from_str(&content)
                .map_err(|e| NeuralDocFlowError::config(format!("Invalid YAML config: {}", e)))
        }
    }

    /// Load configuration from environment variables
    pub fn from_env() -> Self {
        let mut config = Self::default();
        
        // Core settings
        if let Ok(workers) = std::env::var("NEURALDOCFLOW_WORKERS") {
            if let Ok(workers) = workers.parse() {
                config.core.worker_threads = workers;
            }
        }
        
        if let Ok(timeout) = std::env::var("NEURALDOCFLOW_TIMEOUT_MS") {
            if let Ok(timeout) = timeout.parse() {
                config.core.default_timeout = Duration::from_millis(timeout);
            }
        }

        // DAA settings
        if let Ok(agents) = std::env::var("NEURALDOCFLOW_MAX_AGENTS") {
            if let Ok(agents) = agents.parse() {
                config.daa.max_agents = agents;
            }
        }

        // Neural settings
        if let Ok(enabled) = std::env::var("NEURALDOCFLOW_NEURAL_ENABLED") {
            config.neural.enabled = enabled.to_lowercase() == "true";
        }

        // Performance settings
        if let Ok(memory) = std::env::var("NEURALDOCFLOW_MAX_MEMORY_MB") {
            if let Ok(memory) = memory.parse::<usize>() {
                config.performance.max_memory_usage = memory * 1024 * 1024;
            }
        }

        config
    }

    /// Merge with another configuration (other takes precedence)
    pub fn merge(mut self, other: Config) -> Self {
        self.core = other.core;
        self.daa = other.daa;
        self.neural = other.neural;
        self.sources = other.sources;
        self.performance = other.performance;
        self.security = other.security;
        self.monitoring = other.monitoring;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        // Validate core settings
        if self.core.worker_threads == 0 {
            return Err(NeuralDocFlowError::config("worker_threads must be > 0"));
        }

        if self.core.default_timeout.as_millis() == 0 {
            return Err(NeuralDocFlowError::config("default_timeout must be > 0"));
        }

        // Validate DAA settings
        if self.daa.max_agents == 0 {
            return Err(NeuralDocFlowError::config("max_agents must be > 0"));
        }

        // Validate performance settings
        if self.performance.max_memory_usage < 1024 * 1024 {
            return Err(NeuralDocFlowError::config("max_memory_usage must be >= 1MB"));
        }

        // Validate source directories exist
        for dir in &self.sources.plugin_directories {
            if !dir.exists() {
                return Err(NeuralDocFlowError::config(format!(
                    "Plugin directory does not exist: {:?}", dir
                )));
            }
        }

        Ok(())
    }
}

/// Core engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreConfig {
    /// Number of worker threads
    pub worker_threads: usize,
    
    /// Default operation timeout
    pub default_timeout: Duration,
    
    /// Maximum concurrent documents
    pub max_concurrent_documents: usize,
    
    /// Enable detailed metrics collection
    pub enable_metrics: bool,
    
    /// Temporary directory for processing
    pub temp_directory: PathBuf,
}

impl Default for CoreConfig {
    fn default() -> Self {
        Self {
            worker_threads: num_cpus::get(),
            default_timeout: Duration::from_secs(30),
            max_concurrent_documents: 100,
            enable_metrics: true,
            temp_directory: std::env::temp_dir().join("neuraldocflow"),
        }
    }
}

/// DAA coordination configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaaConfig {
    /// Maximum number of agents
    pub max_agents: usize,
    
    /// Agent coordination timeout
    pub coordination_timeout: Duration,
    
    /// Message queue size
    pub message_queue_size: usize,
    
    /// Enable consensus validation
    pub enable_consensus: bool,
    
    /// Consensus threshold (0.0 to 1.0)
    pub consensus_threshold: f32,
    
    /// Agent health check interval
    pub health_check_interval: Duration,
}

impl Default for DaaConfig {
    fn default() -> Self {
        Self {
            max_agents: 16,
            coordination_timeout: Duration::from_secs(5),
            message_queue_size: 1000,
            enable_consensus: true,
            consensus_threshold: 0.8,
            health_check_interval: Duration::from_secs(10),
        }
    }
}

/// Neural processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralConfig {
    /// Enable neural processing
    pub enabled: bool,
    
    /// Model directory path
    pub model_directory: PathBuf,
    
    /// Maximum models to keep in memory
    pub max_loaded_models: usize,
    
    /// Model loading timeout
    pub model_load_timeout: Duration,
    
    /// Neural processing settings
    pub processing: NeuralProcessingConfig,
    
    /// Model-specific configurations
    pub models: HashMap<String, ModelConfig>,
}

impl Default for NeuralConfig {
    fn default() -> Self {
        let mut models = HashMap::new();
        models.insert("layout_analyzer".to_string(), ModelConfig::default());
        models.insert("text_enhancer".to_string(), ModelConfig::default());
        models.insert("table_detector".to_string(), ModelConfig::default());
        
        Self {
            enabled: true,
            model_directory: PathBuf::from("./models"),
            max_loaded_models: 5,
            model_load_timeout: Duration::from_secs(30),
            processing: NeuralProcessingConfig::default(),
            models,
        }
    }
}

/// Neural processing settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralProcessingConfig {
    /// Batch size for neural inference
    pub batch_size: usize,
    
    /// Enable GPU acceleration
    pub enable_gpu: bool,
    
    /// Number of inference threads
    pub inference_threads: usize,
    
    /// Memory limit for neural processing
    pub memory_limit: usize,
}

impl Default for NeuralProcessingConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            enable_gpu: false,
            inference_threads: 2,
            memory_limit: 2 * 1024 * 1024 * 1024, // 2GB
        }
    }
}

/// Model-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model file path
    pub model_path: Option<PathBuf>,
    
    /// Model weight file path
    pub weights_path: Option<PathBuf>,
    
    /// Input preprocessing configuration
    pub preprocessing: PreprocessingConfig,
    
    /// Output postprocessing configuration
    pub postprocessing: PostprocessingConfig,
    
    /// Model-specific parameters
    pub parameters: HashMap<String, serde_json::Value>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_path: None,
            weights_path: None,
            preprocessing: PreprocessingConfig::default(),
            postprocessing: PostprocessingConfig::default(),
            parameters: HashMap::new(),
        }
    }
}

/// Preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    /// Enable normalization
    pub normalize: bool,
    
    /// Image resize dimensions
    pub resize_dimensions: Option<(u32, u32)>,
    
    /// Text tokenization settings
    pub tokenization: TokenizationConfig,
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            normalize: true,
            resize_dimensions: Some((224, 224)),
            tokenization: TokenizationConfig::default(),
        }
    }
}

/// Tokenization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizationConfig {
    /// Maximum sequence length
    pub max_length: usize,
    
    /// Vocabulary size
    pub vocab_size: usize,
    
    /// Enable subword tokenization
    pub subword_tokenization: bool,
}

impl Default for TokenizationConfig {
    fn default() -> Self {
        Self {
            max_length: 512,
            vocab_size: 30000,
            subword_tokenization: true,
        }
    }
}

/// Postprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostprocessingConfig {
    /// Confidence threshold for predictions
    pub confidence_threshold: f32,
    
    /// Apply non-maximum suppression
    pub apply_nms: bool,
    
    /// NMS threshold
    pub nms_threshold: f32,
}

impl Default for PostprocessingConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.5,
            apply_nms: true,
            nms_threshold: 0.3,
        }
    }
}

/// Source plugin configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourcesConfig {
    /// Plugin directories to search
    pub plugin_directories: Vec<PathBuf>,
    
    /// Enable hot reloading of plugins
    pub enable_hot_reload: bool,
    
    /// Plugin loading timeout
    pub plugin_load_timeout: Duration,
    
    /// Maximum plugin memory usage
    pub max_plugin_memory: usize,
    
    /// Source-specific configurations
    pub source_configs: HashMap<String, SourceConfig>,
}

impl Default for SourcesConfig {
    fn default() -> Self {
        let mut source_configs = HashMap::new();
        source_configs.insert("pdf".to_string(), SourceConfig::pdf_default());
        source_configs.insert("docx".to_string(), SourceConfig::docx_default());
        
        Self {
            plugin_directories: vec![
                PathBuf::from("./plugins"),
                PathBuf::from("/usr/local/lib/neuraldocflow/plugins"),
            ],
            enable_hot_reload: false,
            plugin_load_timeout: Duration::from_secs(10),
            max_plugin_memory: 500 * 1024 * 1024, // 500MB
            source_configs,
        }
    }
}

/// Source-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceConfig {
    /// Source enabled
    pub enabled: bool,
    
    /// Source priority (higher = preferred)
    pub priority: u32,
    
    /// Source timeout
    pub timeout: Duration,
    
    /// Memory limit for source
    pub memory_limit: Option<usize>,
    
    /// Thread pool size
    pub thread_pool_size: Option<usize>,
    
    /// Retry configuration
    pub retry: RetryConfig,
    
    /// Source-specific settings
    pub settings: serde_json::Value,
}

impl SourceConfig {
    /// Default PDF source configuration
    pub fn pdf_default() -> Self {
        Self {
            enabled: true,
            priority: 100,
            timeout: Duration::from_secs(30),
            memory_limit: Some(100 * 1024 * 1024),
            thread_pool_size: Some(4),
            retry: RetryConfig::default(),
            settings: serde_json::json!({
                "max_file_size": 100 * 1024 * 1024,
                "enable_ocr": false,
                "extract_tables": true,
                "extract_images": true
            }),
        }
    }

    /// Default DOCX source configuration
    pub fn docx_default() -> Self {
        Self {
            enabled: true,
            priority: 90,
            timeout: Duration::from_secs(20),
            memory_limit: Some(50 * 1024 * 1024),
            thread_pool_size: Some(2),
            retry: RetryConfig::default(),
            settings: serde_json::json!({
                "max_file_size": 50 * 1024 * 1024,
                "extract_comments": true,
                "extract_track_changes": false,
                "preserve_formatting": true
            }),
        }
    }
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum retry attempts
    pub max_attempts: usize,
    
    /// Base delay between retries
    pub base_delay: Duration,
    
    /// Maximum delay between retries
    pub max_delay: Duration,
    
    /// Exponential backoff multiplier
    pub backoff_multiplier: f32,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(5),
            backoff_multiplier: 2.0,
        }
    }
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Maximum memory usage
    pub max_memory_usage: usize,
    
    /// Enable memory mapping for large files
    pub enable_memory_mapping: bool,
    
    /// Buffer pool settings
    pub buffer_pool: BufferPoolConfig,
    
    /// Cache settings
    pub cache: CacheConfig,
    
    /// Profiling settings
    pub profiling: ProfilingConfig,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            max_memory_usage: 4 * 1024 * 1024 * 1024, // 4GB
            enable_memory_mapping: true,
            buffer_pool: BufferPoolConfig::default(),
            cache: CacheConfig::default(),
            profiling: ProfilingConfig::default(),
        }
    }
}

/// Buffer pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferPoolConfig {
    /// Initial buffer count
    pub initial_buffers: usize,
    
    /// Maximum buffer count
    pub max_buffers: usize,
    
    /// Buffer size
    pub buffer_size: usize,
}

impl Default for BufferPoolConfig {
    fn default() -> Self {
        Self {
            initial_buffers: 16,
            max_buffers: 256,
            buffer_size: 64 * 1024, // 64KB
        }
    }
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Enable caching
    pub enabled: bool,
    
    /// Cache size limit
    pub max_size: usize,
    
    /// Cache TTL
    pub ttl: Duration,
    
    /// Cache cleanup interval
    pub cleanup_interval: Duration,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_size: 1024 * 1024 * 1024, // 1GB
            ttl: Duration::from_secs(3600), // 1 hour
            cleanup_interval: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// Profiling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingConfig {
    /// Enable profiling
    pub enabled: bool,
    
    /// Sampling rate
    pub sampling_rate: f32,
    
    /// Profile output directory
    pub output_directory: PathBuf,
}

impl Default for ProfilingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            sampling_rate: 0.01, // 1%
            output_directory: PathBuf::from("./profiles"),
        }
    }
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable security checks
    pub enabled: bool,
    
    /// Maximum file size to process
    pub max_file_size: usize,
    
    /// Allowed MIME types
    pub allowed_mime_types: Vec<String>,
    
    /// Sandbox configuration
    pub sandbox: SandboxConfig,
    
    /// Input validation settings
    pub validation: ValidationConfig,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_file_size: 1024 * 1024 * 1024, // 1GB
            allowed_mime_types: vec![
                "application/pdf".to_string(),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document".to_string(),
                "text/html".to_string(),
                "image/png".to_string(),
                "image/jpeg".to_string(),
            ],
            sandbox: SandboxConfig::default(),
            validation: ValidationConfig::default(),
        }
    }
}

/// Sandbox configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxConfig {
    /// Enable sandboxing
    pub enabled: bool,
    
    /// Sandbox memory limit
    pub memory_limit: usize,
    
    /// Sandbox time limit
    pub time_limit: Duration,
    
    /// Allowed system calls
    pub allowed_syscalls: Vec<String>,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            memory_limit: 100 * 1024 * 1024, // 100MB
            time_limit: Duration::from_secs(30),
            allowed_syscalls: vec![
                "read".to_string(),
                "write".to_string(),
                "open".to_string(),
                "close".to_string(),
            ],
        }
    }
}

/// Input validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Enable strict validation
    pub strict_mode: bool,
    
    /// Maximum nesting depth
    pub max_nesting_depth: usize,
    
    /// Maximum field count
    pub max_field_count: usize,
    
    /// Maximum string length
    pub max_string_length: usize,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            strict_mode: true,
            max_nesting_depth: 100,
            max_field_count: 10000,
            max_string_length: 10 * 1024 * 1024, // 10MB
        }
    }
}

/// Monitoring and logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Logging configuration
    pub logging: LoggingConfig,
    
    /// Metrics configuration
    pub metrics: MetricsConfig,
    
    /// Tracing configuration
    pub tracing: TracingConfig,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            logging: LoggingConfig::default(),
            metrics: MetricsConfig::default(),
            tracing: TracingConfig::default(),
        }
    }
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level
    pub level: String,
    
    /// Log output format
    pub format: String,
    
    /// Log file path
    pub file_path: Option<PathBuf>,
    
    /// Enable structured logging
    pub structured: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            format: "pretty".to_string(),
            file_path: None,
            structured: false,
        }
    }
}

/// Metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Enable metrics collection
    pub enabled: bool,
    
    /// Metrics endpoint
    pub endpoint: String,
    
    /// Collection interval
    pub collection_interval: Duration,
    
    /// Retention period
    pub retention_period: Duration,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            endpoint: "127.0.0.1:9090".to_string(),
            collection_interval: Duration::from_secs(10),
            retention_period: Duration::from_secs(24 * 3600), // 24 hours
        }
    }
}

/// Tracing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracingConfig {
    /// Enable distributed tracing
    pub enabled: bool,
    
    /// Tracing endpoint
    pub endpoint: String,
    
    /// Sampling rate
    pub sampling_rate: f32,
    
    /// Service name
    pub service_name: String,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoint: "http://localhost:14268/api/traces".to_string(),
            sampling_rate: 0.1,
            service_name: "neuraldocflow".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert!(config.core.worker_threads > 0);
        assert!(config.daa.max_agents > 0);
        assert!(config.neural.enabled);
        assert!(config.security.enabled);
    }

    #[test]
    fn test_config_validation() {
        let mut config = Config::default();
        assert!(config.validate().is_ok());

        config.core.worker_threads = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_merge() {
        let mut config1 = Config::default();
        config1.core.worker_threads = 4;

        let mut config2 = Config::default();
        config2.core.worker_threads = 8;

        let merged = config1.merge(config2);
        assert_eq!(merged.core.worker_threads, 8);
    }

    #[test]
    fn test_config_serialization() {
        let config = Config::default();
        
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: Config = serde_json::from_str(&json).unwrap();
        
        assert_eq!(config.core.worker_threads, deserialized.core.worker_threads);
    }

    #[test]
    fn test_config_from_file() {
        let config_data = r#"
        {
            "core": {
                "worker_threads": 16,
                "default_timeout": {"secs": 60, "nanos": 0},
                "max_concurrent_documents": 200,
                "enable_metrics": true,
                "temp_directory": "/tmp/neuraldocflow"
            }
        }
        "#;

        let mut temp_file = NamedTempFile::new().unwrap();
        std::io::Write::write_all(temp_file.as_file_mut(), config_data.as_bytes()).unwrap();

        let config = Config::from_file(temp_file.path()).unwrap();
        assert_eq!(config.core.worker_threads, 16);
        assert_eq!(config.core.max_concurrent_documents, 200);
    }

    #[test]
    fn test_source_config_defaults() {
        let pdf_config = SourceConfig::pdf_default();
        assert!(pdf_config.enabled);
        assert_eq!(pdf_config.priority, 100);

        let docx_config = SourceConfig::docx_default();
        assert!(docx_config.enabled);
        assert_eq!(docx_config.priority, 90);
    }

    #[test]
    fn test_retry_config() {
        let retry = RetryConfig::default();
        assert_eq!(retry.max_attempts, 3);
        assert_eq!(retry.backoff_multiplier, 2.0);
    }
}