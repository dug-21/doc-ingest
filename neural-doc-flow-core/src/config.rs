//! Configuration structures for the neural document flow system
//!
//! This module provides centralized configuration management for all components

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Main configuration for the neural document flow system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralDocFlowConfig {
    /// System-wide settings
    pub system: SystemConfig,
    
    /// Source configurations
    pub sources: SourcesConfig,
    
    /// Processing pipeline configuration
    pub pipeline: PipelineConfig,
    
    /// Neural processing configuration
    pub neural: NeuralConfig,
    
    /// Output formatting configuration
    pub output: OutputConfig,
    
    /// Security configuration
    pub security: SecurityConfig,
    
    /// Monitoring and metrics configuration
    pub monitoring: MonitoringConfig,
    
    /// Custom extensions
    pub extensions: HashMap<String, serde_json::Value>,
}

/// System-wide configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    /// Maximum concurrent documents
    pub max_concurrent_documents: usize,
    
    /// Global timeout in seconds
    pub global_timeout_seconds: u64,
    
    /// Enable debug mode
    pub debug_mode: bool,
    
    /// Working directory for temporary files
    pub working_directory: PathBuf,
    
    /// Cache configuration
    pub cache: CacheConfig,
    
    /// Resource limits
    pub resource_limits: ResourceLimits,
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Enable caching
    pub enabled: bool,
    
    /// Cache directory
    pub directory: PathBuf,
    
    /// Maximum cache size in MB
    pub max_size_mb: u64,
    
    /// Cache TTL in seconds
    pub ttl_seconds: u64,
    
    /// Cache eviction policy
    pub eviction_policy: CacheEvictionPolicy,
}

/// Cache eviction policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheEvictionPolicy {
    /// Least Recently Used
    LRU,
    
    /// Least Frequently Used
    LFU,
    
    /// First In First Out
    FIFO,
    
    /// Time-based expiration only
    TimeBasedOnly,
}

/// Resource limits configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum memory usage in MB
    pub max_memory_mb: u64,
    
    /// Maximum CPU threads
    pub max_cpu_threads: usize,
    
    /// Maximum file size in MB
    pub max_file_size_mb: u64,
    
    /// Maximum processing time per document in seconds
    pub max_processing_time_seconds: u64,
}

/// Sources configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourcesConfig {
    /// Enabled source types
    pub enabled_sources: Vec<String>,
    
    /// Source-specific configurations
    pub source_configs: HashMap<String, SourceTypeConfig>,
    
    /// Default source options
    pub default_options: SourceDefaultOptions,
}

/// Configuration for specific source types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceTypeConfig {
    /// Source type name
    pub source_type: String,
    
    /// Enable this source
    pub enabled: bool,
    
    /// Priority (higher = preferred)
    pub priority: u32,
    
    /// Source-specific parameters
    pub parameters: HashMap<String, serde_json::Value>,
}

/// Default options for sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceDefaultOptions {
    /// Enable parallel loading
    pub parallel_loading: bool,
    
    /// Batch size for parallel loading
    pub batch_size: usize,
    
    /// Retry configuration
    pub retry_config: RetryConfig,
    
    /// Validation strictness
    pub validation_level: ValidationLevel,
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum retry attempts
    pub max_attempts: u32,
    
    /// Initial delay in milliseconds
    pub initial_delay_ms: u64,
    
    /// Backoff multiplier
    pub backoff_multiplier: f64,
    
    /// Maximum delay in milliseconds
    pub max_delay_ms: u64,
    
    /// Jitter factor (0.0 to 1.0)
    pub jitter_factor: f64,
}

/// Validation strictness levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationLevel {
    /// Minimal validation
    Minimal,
    
    /// Standard validation
    Standard,
    
    /// Strict validation
    Strict,
    
    /// Custom validation rules
    Custom(String),
}

/// Pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Pipeline execution mode
    pub execution_mode: ExecutionMode,
    
    /// Processor chain configuration
    pub processors: Vec<ProcessorConfig>,
    
    /// Error handling strategy
    pub error_handling: ErrorHandlingStrategy,
    
    /// Performance optimization settings
    pub optimization: OptimizationConfig,
}

/// Pipeline execution modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionMode {
    /// Sequential processing
    Sequential,
    
    /// Parallel processing where possible
    Parallel,
    
    /// Adaptive mode based on system load
    Adaptive,
    
    /// Custom execution strategy
    Custom(String),
}

/// Processor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorConfig {
    /// Processor name
    pub name: String,
    
    /// Processor type
    pub processor_type: String,
    
    /// Enable this processor
    pub enabled: bool,
    
    /// Processing priority
    pub priority: u32,
    
    /// Processor-specific parameters
    pub parameters: HashMap<String, serde_json::Value>,
    
    /// Dependencies on other processors
    pub dependencies: Vec<String>,
    
    /// Timeout override for this processor
    pub timeout_seconds: Option<u64>,
}

/// Error handling strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorHandlingStrategy {
    /// Stop on first error
    FailFast,
    
    /// Continue processing, collect all errors
    ContinueOnError,
    
    /// Retry failed processors
    RetryOnError(RetryConfig),
    
    /// Custom error handling
    Custom(String),
}

/// Optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    
    /// Enable GPU acceleration
    pub enable_gpu: bool,
    
    /// Memory pool size in MB
    pub memory_pool_size_mb: u64,
    
    /// Thread pool size
    pub thread_pool_size: usize,
    
    /// Enable result caching
    pub enable_result_caching: bool,
    
    /// Batch processing configuration
    pub batch_config: BatchConfig,
}

/// Batch processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Enable batch processing
    pub enabled: bool,
    
    /// Minimum batch size
    pub min_batch_size: usize,
    
    /// Maximum batch size
    pub max_batch_size: usize,
    
    /// Batch timeout in milliseconds
    pub batch_timeout_ms: u64,
    
    /// Adaptive batching
    pub adaptive_batching: bool,
}

/// Neural processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralConfig {
    /// Enable neural processing
    pub enabled: bool,
    
    /// Neural backend to use
    pub backend: NeuralBackend,
    
    /// Model configurations
    pub models: HashMap<String, ModelConfig>,
    
    /// Default neural processing options
    pub default_options: NeuralOptions,
    
    /// Hardware configuration
    pub hardware: HardwareConfig,
}

/// Neural processing backends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NeuralBackend {
    /// Candle backend
    Candle,
    
    /// FANN backend
    Fann,
    
    /// ONNX Runtime
    OnnxRuntime,
    
    /// Custom backend
    Custom(String),
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model name
    pub name: String,
    
    /// Model path or URL
    pub path: String,
    
    /// Model type
    pub model_type: String,
    
    /// Enable this model
    pub enabled: bool,
    
    /// Model-specific parameters
    pub parameters: HashMap<String, serde_json::Value>,
    
    /// Preprocessing configuration
    pub preprocessing: Option<PreprocessingConfig>,
}

/// Preprocessing configuration for models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    /// Normalization parameters
    pub normalization: Option<NormalizationConfig>,
    
    /// Tokenization settings
    pub tokenization: Option<TokenizationConfig>,
    
    /// Image preprocessing
    pub image_preprocessing: Option<ImagePreprocessingConfig>,
}

/// Normalization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationConfig {
    /// Mean values
    pub mean: Vec<f32>,
    
    /// Standard deviation values
    pub std: Vec<f32>,
    
    /// Min-max scaling
    pub min_max: Option<(f32, f32)>,
}

/// Tokenization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizationConfig {
    /// Tokenizer type
    pub tokenizer_type: String,
    
    /// Vocabulary path
    pub vocab_path: Option<String>,
    
    /// Maximum sequence length
    pub max_length: usize,
    
    /// Padding strategy
    pub padding: PaddingStrategy,
}

/// Padding strategies for tokenization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PaddingStrategy {
    /// No padding
    None,
    
    /// Pad to maximum length
    MaxLength,
    
    /// Pad to longest in batch
    Longest,
    
    /// Custom padding
    Custom(String),
}

/// Image preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImagePreprocessingConfig {
    /// Target image size
    pub target_size: (u32, u32),
    
    /// Resize mode
    pub resize_mode: ResizeMode,
    
    /// Color mode
    pub color_mode: ColorMode,
    
    /// Data augmentation
    pub augmentation: Option<AugmentationConfig>,
}

/// Image resize modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResizeMode {
    /// Stretch to fit
    Stretch,
    
    /// Maintain aspect ratio, pad
    PadToFit,
    
    /// Maintain aspect ratio, crop
    CropToFit,
    
    /// Custom resize
    Custom(String),
}

/// Color modes for images
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColorMode {
    /// RGB color
    RGB,
    
    /// RGBA with alpha
    RGBA,
    
    /// Grayscale
    Grayscale,
    
    /// Custom color mode
    Custom(String),
}

/// Data augmentation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentationConfig {
    /// Random rotation range
    pub rotation_range: Option<f32>,
    
    /// Random zoom range
    pub zoom_range: Option<(f32, f32)>,
    
    /// Horizontal flip probability
    pub horizontal_flip: Option<f32>,
    
    /// Vertical flip probability
    pub vertical_flip: Option<f32>,
    
    /// Brightness adjustment range
    pub brightness_range: Option<(f32, f32)>,
}

/// Neural processing options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralOptions {
    /// Confidence threshold
    pub confidence_threshold: f64,
    
    /// Batch size for inference
    pub batch_size: usize,
    
    /// Enable mixed precision
    pub mixed_precision: bool,
    
    /// Enable model quantization
    pub quantization: Option<QuantizationConfig>,
}

/// Quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Quantization mode
    pub mode: QuantizationMode,
    
    /// Quantization bits
    pub bits: u8,
    
    /// Calibration dataset size
    pub calibration_size: Option<usize>,
}

/// Quantization modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizationMode {
    /// Dynamic quantization
    Dynamic,
    
    /// Static quantization
    Static,
    
    /// Quantization-aware training
    QAT,
    
    /// Custom quantization
    Custom(String),
}

/// Hardware configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    /// Use GPU if available
    pub use_gpu: bool,
    
    /// GPU device IDs to use
    pub gpu_devices: Vec<u32>,
    
    /// CPU thread count
    pub cpu_threads: Option<usize>,
    
    /// Memory limit per model in MB
    pub memory_limit_mb: Option<u64>,
    
    /// Enable hardware-specific optimizations
    pub hardware_optimizations: bool,
}

/// Output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Default output format
    pub default_format: String,
    
    /// Output directory
    pub output_directory: PathBuf,
    
    /// Formatter configurations
    pub formatters: HashMap<String, FormatterConfig>,
    
    /// Template configuration
    pub templates: TemplateConfig,
}

/// Formatter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormatterConfig {
    /// Formatter name
    pub name: String,
    
    /// Enable this formatter
    pub enabled: bool,
    
    /// Formatter-specific options
    pub options: HashMap<String, serde_json::Value>,
    
    /// Quality settings
    pub quality: QualitySettings,
}

/// Quality settings for output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualitySettings {
    /// Compression level (0-9)
    pub compression_level: Option<u8>,
    
    /// DPI for image outputs
    pub dpi: Option<u32>,
    
    /// Color depth
    pub color_depth: Option<u8>,
    
    /// Enable lossy compression
    pub lossy_compression: bool,
}

/// Template configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateConfig {
    /// Template directory
    pub template_directory: PathBuf,
    
    /// Default template
    pub default_template: Option<String>,
    
    /// Template engine
    pub template_engine: String,
    
    /// Custom template functions
    pub custom_functions: Vec<String>,
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Enable monitoring
    pub enabled: bool,
    
    /// Metrics collection interval in seconds
    pub metrics_interval_seconds: u64,
    
    /// Metrics export configuration
    pub export: MetricsExportConfig,
    
    /// Alerting configuration
    pub alerting: AlertingConfig,
    
    /// Logging configuration
    pub logging: LoggingConfig,
}

/// Metrics export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsExportConfig {
    /// Export format
    pub format: MetricsFormat,
    
    /// Export destination
    pub destination: MetricsDestination,
    
    /// Export interval in seconds
    pub interval_seconds: u64,
    
    /// Metrics to export
    pub metrics: Vec<String>,
}

/// Metrics formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricsFormat {
    /// Prometheus format
    Prometheus,
    
    /// JSON format
    Json,
    
    /// CSV format
    Csv,
    
    /// Custom format
    Custom(String),
}

/// Metrics destinations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricsDestination {
    /// File output
    File(PathBuf),
    
    /// HTTP endpoint
    Http(String),
    
    /// StatsD server
    StatsD(String),
    
    /// Custom destination
    Custom(String),
}

/// Alerting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertingConfig {
    /// Enable alerting
    pub enabled: bool,
    
    /// Alert rules
    pub rules: Vec<AlertRule>,
    
    /// Alert destinations
    pub destinations: Vec<AlertDestination>,
}

/// Alert rule definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    /// Rule name
    pub name: String,
    
    /// Metric to monitor
    pub metric: String,
    
    /// Alert condition
    pub condition: AlertCondition,
    
    /// Alert severity
    pub severity: AlertSeverity,
    
    /// Cool-down period in seconds
    pub cooldown_seconds: u64,
}

/// Alert conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertCondition {
    /// Greater than threshold
    GreaterThan(f64),
    
    /// Less than threshold
    LessThan(f64),
    
    /// Outside range
    OutsideRange(f64, f64),
    
    /// Rate of change
    RateOfChange(f64),
    
    /// Custom condition
    Custom(String),
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// Informational
    Info,
    
    /// Warning
    Warning,
    
    /// Error
    Error,
    
    /// Critical
    Critical,
}

/// Alert destinations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertDestination {
    /// Email notification
    Email(String),
    
    /// Webhook
    Webhook(String),
    
    /// Log file
    LogFile(PathBuf),
    
    /// Custom destination
    Custom(String),
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level
    pub level: LogLevel,
    
    /// Log format
    pub format: LogFormat,
    
    /// Log outputs
    pub outputs: Vec<LogOutput>,
    
    /// Log rotation configuration
    pub rotation: Option<LogRotationConfig>,
}

/// Log levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    /// Trace level
    Trace,
    
    /// Debug level
    Debug,
    
    /// Info level
    Info,
    
    /// Warning level
    Warn,
    
    /// Error level
    Error,
}

/// Log formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogFormat {
    /// Plain text
    Text,
    
    /// JSON format
    Json,
    
    /// Compact format
    Compact,
    
    /// Custom format
    Custom(String),
}

/// Log output destinations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogOutput {
    /// Standard output
    Stdout,
    
    /// Standard error
    Stderr,
    
    /// File output
    File(PathBuf),
    
    /// Syslog
    Syslog,
    
    /// Custom output
    Custom(String),
}

/// Log rotation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogRotationConfig {
    /// Maximum file size in MB
    pub max_size_mb: u64,
    
    /// Maximum number of files
    pub max_files: u32,
    
    /// Rotation period
    pub rotation_period: RotationPeriod,
}

/// Log rotation periods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RotationPeriod {
    /// Hourly rotation
    Hourly,
    
    /// Daily rotation
    Daily,
    
    /// Weekly rotation
    Weekly,
    
    /// Size-based only
    SizeOnly,
}

/// Security configuration for the system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable security scanning
    pub enabled: bool,
    
    /// Security scanning mode
    pub scan_mode: SecurityScanMode,
    
    /// Threat detection configuration
    pub threat_detection: ThreatDetectionConfig,
    
    /// Sandbox configuration
    pub sandboxing: SandboxConfig,
    
    /// Audit configuration
    pub audit: AuditConfig,
    
    /// Security policies
    pub policies: SecurityPolicies,
}

/// Security scanning modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityScanMode {
    /// Disabled - no security scanning
    Disabled,
    
    /// Basic - heuristic-based scanning only
    Basic,
    
    /// Standard - neural + heuristic scanning
    Standard,
    
    /// Comprehensive - full neural analysis with behavioral detection
    Comprehensive,
    
    /// Custom - user-defined security scanning rules
    Custom(String),
}

/// Threat detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatDetectionConfig {
    /// Enable malware detection
    pub malware_detection: bool,
    
    /// Enable behavioral analysis
    pub behavioral_analysis: bool,
    
    /// Enable anomaly detection
    pub anomaly_detection: bool,
    
    /// Threat confidence threshold (0.0 to 1.0)
    pub confidence_threshold: f32,
    
    /// Neural model configuration for threat detection
    pub neural_models: HashMap<String, ModelConfig>,
    
    /// Custom threat patterns
    pub custom_patterns: Vec<ThreatPattern>,
}

/// Custom threat pattern definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatPattern {
    /// Pattern name
    pub name: String,
    
    /// Pattern type (regex, substring, neural, etc.)
    pub pattern_type: String,
    
    /// Pattern rule
    pub rule: String,
    
    /// Threat severity for this pattern
    pub severity: ThreatSeverity,
    
    /// Action to take when pattern matches
    pub action: SecurityAction,
}

/// Threat severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThreatSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Security actions to take
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecurityAction {
    /// Allow processing to continue
    Allow,
    
    /// Sanitize content and continue
    Sanitize,
    
    /// Quarantine document for review
    Quarantine,
    
    /// Block document processing
    Block,
    
    /// Custom action
    Custom(String),
}

/// Sandbox configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxConfig {
    /// Enable sandboxing for plugin execution
    pub enabled: bool,
    
    /// Sandbox type (process, container, etc.)
    pub sandbox_type: SandboxType,
    
    /// Resource limits for sandboxed execution
    pub resource_limits: SandboxResourceLimits,
    
    /// Network access policy
    pub network_policy: NetworkPolicy,
    
    /// Filesystem access policy
    pub filesystem_policy: FilesystemPolicy,
}

/// Sandbox implementation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SandboxType {
    /// Process-based sandboxing
    Process,
    
    /// Container-based sandboxing
    Container,
    
    /// Virtual machine sandboxing
    VM,
    
    /// Custom sandbox implementation
    Custom(String),
}

/// Sandbox resource limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxResourceLimits {
    /// Maximum memory in MB
    pub max_memory_mb: u64,
    
    /// Maximum CPU percent
    pub max_cpu_percent: f32,
    
    /// Maximum execution time in seconds
    pub max_execution_time_seconds: u64,
    
    /// Maximum file operations per second
    pub max_file_ops_per_second: u32,
}

/// Network access policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkPolicy {
    /// No network access allowed
    None,
    
    /// Limited access to specific hosts
    Restricted(Vec<String>),
    
    /// Full network access
    Full,
    
    /// Custom network policy
    Custom(String),
}

/// Filesystem access policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilesystemPolicy {
    /// No filesystem access
    None,
    
    /// Read-only access to specific paths
    ReadOnly(Vec<String>),
    
    /// Read-write access to specific paths
    ReadWrite(Vec<String>),
    
    /// Full filesystem access
    Full,
    
    /// Custom filesystem policy
    Custom(String),
}

/// Audit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    /// Enable security audit logging
    pub enabled: bool,
    
    /// Audit log level
    pub log_level: AuditLogLevel,
    
    /// Audit log destination
    pub log_destination: AuditLogDestination,
    
    /// Log retention period in days
    pub retention_days: u32,
    
    /// Include sensitive data in logs
    pub include_sensitive_data: bool,
    
    /// Audit event types to log
    pub logged_events: Vec<AuditEventType>,
}

/// Audit log levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditLogLevel {
    /// Log all security events
    All,
    
    /// Log threats and errors only
    ThreatsAndErrors,
    
    /// Log errors only
    ErrorsOnly,
    
    /// Custom log level
    Custom(Vec<String>),
}

/// Audit log destinations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditLogDestination {
    /// File-based logging
    File(PathBuf),
    
    /// Syslog
    Syslog,
    
    /// Remote logging server
    Remote(String),
    
    /// Custom destination
    Custom(String),
}

/// Types of security events to audit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditEventType {
    /// Security scan started
    ScanStart,
    
    /// Security scan completed
    ScanComplete,
    
    /// Threat detected
    ThreatDetected,
    
    /// Document quarantined
    DocumentQuarantined,
    
    /// Document blocked
    DocumentBlocked,
    
    /// Plugin sandboxed
    PluginSandboxed,
    
    /// Security policy violation
    PolicyViolation,
    
    /// Custom event type
    Custom(String),
}

/// Security policies configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityPolicies {
    /// Maximum file size for processing (in MB)
    pub max_file_size_mb: u64,
    
    /// Allowed file types (MIME types)
    pub allowed_file_types: Vec<String>,
    
    /// Blocked file types (MIME types)
    pub blocked_file_types: Vec<String>,
    
    /// Require secure source validation
    pub require_source_validation: bool,
    
    /// Enable content encryption at rest
    pub encrypt_content_at_rest: bool,
    
    /// Security action overrides for specific conditions
    pub action_overrides: HashMap<String, SecurityAction>,
}

// Default implementations

impl Default for NeuralDocFlowConfig {
    fn default() -> Self {
        Self {
            system: SystemConfig::default(),
            sources: SourcesConfig::default(),
            pipeline: PipelineConfig::default(),
            neural: NeuralConfig::default(),
            output: OutputConfig::default(),
            security: SecurityConfig::default(),
            monitoring: MonitoringConfig::default(),
            extensions: HashMap::new(),
        }
    }
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            max_concurrent_documents: 100,
            global_timeout_seconds: 300,
            debug_mode: false,
            working_directory: PathBuf::from("/tmp/neural-doc-flow"),
            cache: CacheConfig::default(),
            resource_limits: ResourceLimits::default(),
        }
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            directory: PathBuf::from("/tmp/neural-doc-flow/cache"),
            max_size_mb: 1024,
            ttl_seconds: 3600,
            eviction_policy: CacheEvictionPolicy::LRU,
        }
    }
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_mb: 4096,
            max_cpu_threads: 4, // Default to 4 threads
            max_file_size_mb: 100,
            max_processing_time_seconds: 300,
        }
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay_ms: 1000,
            backoff_multiplier: 2.0,
            max_delay_ms: 60000,
            jitter_factor: 0.1,
        }
    }
}

impl Default for HardwareConfig {
    fn default() -> Self {
        Self {
            use_gpu: true,
            gpu_devices: vec![0],
            cpu_threads: None,
            memory_limit_mb: None,
            hardware_optimizations: true,
        }
    }
}

impl Default for SourcesConfig {
    fn default() -> Self {
        Self {
            enabled_sources: vec!["filesystem".to_string(), "http".to_string()],
            source_configs: HashMap::new(),
            default_options: SourceDefaultOptions::default(),
        }
    }
}

impl Default for SourceDefaultOptions {
    fn default() -> Self {
        Self {
            parallel_loading: true,
            batch_size: 10,
            retry_config: RetryConfig::default(),
            validation_level: ValidationLevel::Standard,
        }
    }
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            execution_mode: ExecutionMode::Adaptive,
            processors: Vec::new(),
            error_handling: ErrorHandlingStrategy::ContinueOnError,
            optimization: OptimizationConfig::default(),
        }
    }
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            enable_simd: true,
            enable_gpu: true,
            memory_pool_size_mb: 1024,
            thread_pool_size: 4,
            enable_result_caching: true,
            batch_config: BatchConfig::default(),
        }
    }
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_batch_size: 1,
            max_batch_size: 100,
            batch_timeout_ms: 5000,
            adaptive_batching: true,
        }
    }
}

impl Default for NeuralConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            backend: NeuralBackend::Candle,
            models: HashMap::new(),
            default_options: NeuralOptions::default(),
            hardware: HardwareConfig::default(),
        }
    }
}

impl Default for NeuralOptions {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.5,
            batch_size: 32,
            mixed_precision: false,
            quantization: None,
        }
    }
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            default_format: "json".to_string(),
            output_directory: PathBuf::from("./output"),
            formatters: HashMap::new(),
            templates: TemplateConfig::default(),
        }
    }
}

impl Default for TemplateConfig {
    fn default() -> Self {
        Self {
            template_directory: PathBuf::from("./templates"),
            default_template: None,
            template_engine: "handlebars".to_string(),
            custom_functions: Vec::new(),
        }
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            metrics_interval_seconds: 60,
            export: MetricsExportConfig::default(),
            alerting: AlertingConfig::default(),
            logging: LoggingConfig::default(),
        }
    }
}

impl Default for MetricsExportConfig {
    fn default() -> Self {
        Self {
            format: MetricsFormat::Json,
            destination: MetricsDestination::File(PathBuf::from("./metrics.json")),
            interval_seconds: 300,
            metrics: vec!["throughput".to_string(), "latency".to_string(), "errors".to_string()],
        }
    }
}

impl Default for AlertingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            rules: Vec::new(),
            destinations: Vec::new(),
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
            format: LogFormat::Text,
            outputs: vec![LogOutput::Stdout],
            rotation: None,
        }
    }
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            scan_mode: SecurityScanMode::Standard,
            threat_detection: ThreatDetectionConfig::default(),
            sandboxing: SandboxConfig::default(),
            audit: AuditConfig::default(),
            policies: SecurityPolicies::default(),
        }
    }
}

impl Default for ThreatDetectionConfig {
    fn default() -> Self {
        Self {
            malware_detection: true,
            behavioral_analysis: true,
            anomaly_detection: true,
            confidence_threshold: 0.7,
            neural_models: HashMap::new(),
            custom_patterns: Vec::new(),
        }
    }
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sandbox_type: SandboxType::Process,
            resource_limits: SandboxResourceLimits::default(),
            network_policy: NetworkPolicy::None,
            filesystem_policy: FilesystemPolicy::ReadOnly(vec!["/tmp".to_string()]),
        }
    }
}

impl Default for SandboxResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_mb: 1024,
            max_cpu_percent: 50.0,
            max_execution_time_seconds: 300,
            max_file_ops_per_second: 100,
        }
    }
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            log_level: AuditLogLevel::ThreatsAndErrors,
            log_destination: AuditLogDestination::File(PathBuf::from("./security-audit.log")),
            retention_days: 90,
            include_sensitive_data: false,
            logged_events: vec![
                AuditEventType::ThreatDetected,
                AuditEventType::DocumentQuarantined,
                AuditEventType::DocumentBlocked,
                AuditEventType::PolicyViolation,
            ],
        }
    }
}

impl Default for SecurityPolicies {
    fn default() -> Self {
        Self {
            max_file_size_mb: 100,
            allowed_file_types: vec![
                "application/pdf".to_string(),
                "text/plain".to_string(),
                "text/html".to_string(),
                "application/json".to_string(),
                "application/xml".to_string(),
                "text/xml".to_string(),
                "text/markdown".to_string(),
            ],
            blocked_file_types: vec![
                "application/x-executable".to_string(),
                "application/x-msdos-program".to_string(),
                "application/x-msdownload".to_string(),
            ],
            require_source_validation: true,
            encrypt_content_at_rest: false,
            action_overrides: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_config() {
        let config = NeuralDocFlowConfig::default();
        assert_eq!(config.system.max_concurrent_documents, 100);
        assert!(config.system.cache.enabled);
        assert_eq!(config.system.resource_limits.max_cpu_threads, 4);
    }
    
    #[test]
    fn test_config_serialization() {
        let config = NeuralDocFlowConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: NeuralDocFlowConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.system.max_concurrent_documents, deserialized.system.max_concurrent_documents);
    }
}