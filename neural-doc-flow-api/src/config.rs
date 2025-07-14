//! Server configuration management

use serde::{Deserialize, Serialize};
use std::net::IpAddr;
use clap::Parser;

/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize, Parser)]
#[command(author, version, about, long_about = None)]
pub struct ServerConfig {
    /// Server host address
    #[arg(long, env = "HOST", default_value = "0.0.0.0")]
    pub host: IpAddr,

    /// Server port
    #[arg(short, long, env = "PORT", default_value = "8080")]
    pub port: u16,

    /// Database URL
    #[arg(long, env = "DATABASE_URL", default_value = "sqlite:neural_doc_flow.db")]
    pub database_url: String,

    /// JWT secret key
    #[arg(long, env = "JWT_SECRET")]
    pub jwt_secret: String,

    /// Maximum file size in bytes (default: 100MB)
    #[arg(long, env = "MAX_FILE_SIZE", default_value = "104857600")]
    pub max_file_size: u64,

    /// Request timeout in seconds
    #[arg(long, env = "REQUEST_TIMEOUT", default_value = "300")]
    pub request_timeout: u64,

    /// Rate limit: requests per minute per IP
    #[arg(long, env = "RATE_LIMIT_RPM", default_value = "100")]
    pub rate_limit_rpm: u32,

    /// Neural processing configuration
    #[command(flatten)]
    pub neural: NeuralConfig,

    /// Security configuration
    #[command(flatten)]
    pub security: SecurityConfig,

    /// Monitoring configuration
    #[command(flatten)]
    pub monitoring: MonitoringConfig,

    /// Background job configuration
    #[command(flatten)]
    pub jobs: JobConfig,
}

/// Neural processing configuration
#[derive(Debug, Clone, Serialize, Deserialize, Parser)]
pub struct NeuralConfig {
    /// Enable neural enhancement
    #[arg(long, env = "NEURAL_ENABLED", default_value = "true")]
    pub enabled: bool,

    /// Neural model path
    #[arg(long, env = "NEURAL_MODEL_PATH", default_value = "./models")]
    pub model_path: String,

    /// Maximum neural processing threads
    #[arg(long, env = "NEURAL_MAX_THREADS", default_value = "4")]
    pub max_threads: usize,

    /// Neural processing timeout in seconds
    #[arg(long, env = "NEURAL_TIMEOUT", default_value = "120")]
    pub timeout: u64,

    /// Enable GPU acceleration if available
    #[arg(long, env = "NEURAL_GPU_ENABLED", default_value = "false")]
    pub gpu_enabled: bool,
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize, Parser)]
pub struct SecurityConfig {
    /// Enable security scanning
    #[arg(long, env = "SECURITY_ENABLED", default_value = "true")]
    pub enabled: bool,

    /// Security scanning level (0-3)
    #[arg(long, env = "SECURITY_LEVEL", default_value = "2")]
    pub level: u8,

    /// Enable sandboxing for document processing
    #[arg(long, env = "SECURITY_SANDBOX", default_value = "true")]
    pub sandbox_enabled: bool,

    /// Security scan timeout in seconds
    #[arg(long, env = "SECURITY_TIMEOUT", default_value = "30")]
    pub timeout: u64,

    /// Maximum quarantine time for suspicious files (hours)
    #[arg(long, env = "SECURITY_QUARANTINE_HOURS", default_value = "24")]
    pub quarantine_hours: u64,
}

/// Monitoring and metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize, Parser)]
pub struct MonitoringConfig {
    /// Enable Prometheus metrics
    #[arg(long, env = "METRICS_ENABLED", default_value = "true")]
    pub metrics_enabled: bool,

    /// Metrics collection interval in seconds
    #[arg(long, env = "METRICS_INTERVAL", default_value = "60")]
    pub metrics_interval: u64,

    /// Enable detailed tracing
    #[arg(long, env = "TRACING_ENABLED", default_value = "true")]
    pub tracing_enabled: bool,

    /// Log level
    #[arg(long, env = "LOG_LEVEL", default_value = "info")]
    pub log_level: String,

    /// Enable performance profiling
    #[arg(long, env = "PROFILING_ENABLED", default_value = "false")]
    pub profiling_enabled: bool,
}

/// Background job processing configuration
#[derive(Debug, Clone, Serialize, Deserialize, Parser)]
pub struct JobConfig {
    /// Number of job worker threads
    #[arg(long, env = "JOB_WORKERS", default_value = "4")]
    pub workers: usize,

    /// Job queue capacity
    #[arg(long, env = "JOB_QUEUE_SIZE", default_value = "1000")]
    pub queue_size: usize,

    /// Job retry attempts
    #[arg(long, env = "JOB_RETRY_ATTEMPTS", default_value = "3")]
    pub retry_attempts: u32,

    /// Job timeout in seconds
    #[arg(long, env = "JOB_TIMEOUT", default_value = "600")]
    pub timeout: u64,

    /// Job cleanup interval in seconds
    #[arg(long, env = "JOB_CLEANUP_INTERVAL", default_value = "3600")]
    pub cleanup_interval: u64,

    /// Keep completed jobs for this many seconds
    #[arg(long, env = "JOB_RETENTION", default_value = "86400")]
    pub retention_seconds: u64,
}

impl ServerConfig {
    /// Load configuration from environment and command line
    pub fn load() -> Result<Self, Box<dyn std::error::Error>> {
        let config = Self::parse();
        
        // Validate configuration
        config.validate()?;
        
        Ok(config)
    }

    /// Load configuration from file
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let settings = config::Config::builder()
            .add_source(config::File::with_name(path))
            .add_source(config::Environment::with_prefix("NEURAL_DOC_FLOW"))
            .build()?;

        let mut config: Self = settings.try_deserialize()?;
        config.validate()?;
        
        Ok(config)
    }

    /// Validate configuration values
    fn validate(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Validate port range
        if self.port == 0 {
            return Err("Port cannot be 0".into());
        }

        // Validate JWT secret
        if self.jwt_secret.is_empty() {
            return Err("JWT secret is required".into());
        }

        // Validate max file size
        if self.max_file_size == 0 {
            return Err("Max file size must be greater than 0".into());
        }

        // Validate rate limiting
        if self.rate_limit_rpm == 0 {
            return Err("Rate limit must be greater than 0".into());
        }

        // Validate neural config
        if self.neural.enabled && self.neural.max_threads == 0 {
            return Err("Neural max threads must be greater than 0 when neural processing is enabled".into());
        }

        // Validate security config
        if self.security.level > 3 {
            return Err("Security level must be between 0-3".into());
        }

        // Validate job config
        if self.jobs.workers == 0 {
            return Err("Job workers must be greater than 0".into());
        }

        if self.jobs.queue_size == 0 {
            return Err("Job queue size must be greater than 0".into());
        }

        Ok(())
    }

    /// Get database configuration for SQLx
    pub fn database_options(&self) -> sqlx::sqlite::SqliteConnectOptions {
        use sqlx::sqlite::SqliteConnectOptions;
        use std::str::FromStr;

        SqliteConnectOptions::from_str(&self.database_url)
            .unwrap_or_else(|_| SqliteConnectOptions::new().filename("neural_doc_flow.db"))
            .create_if_missing(true)
            .journal_mode(sqlx::sqlite::SqliteJournalMode::Wal)
            .foreign_keys(true)
    }

    /// Get processing configuration for neural document flow
    pub fn processing_config(&self) -> neural_doc_flow_core::ProcessingConfig {
        let mut config = neural_doc_flow_core::ProcessingConfig::default();
        
        config.set_neural_enabled(self.neural.enabled);
        config.set_max_file_size(self.max_file_size);
        config.set_timeout_ms((self.request_timeout * 1000) as u32);
        config.set_security_level(self.security.level);
        
        // Add output formats
        config.add_output_format("text".to_string()).ok();
        config.add_output_format("json".to_string()).ok();
        config.add_output_format("markdown".to_string()).ok();
        
        // Set custom options
        config.set_custom_option("api_mode".to_string(), "true".to_string());
        config.set_custom_option("max_threads".to_string(), self.neural.max_threads.to_string());
        
        config
    }

    /// Check if development mode is enabled
    pub fn is_development(&self) -> bool {
        self.log_level.to_lowercase() == "debug" || 
        std::env::var("RUST_ENV").unwrap_or_default() == "development"
    }

    /// Get CORS configuration
    pub fn cors_origins(&self) -> Vec<String> {
        if self.is_development() {
            vec![
                "http://localhost:3000".to_string(),
                "http://localhost:8080".to_string(),
                "http://127.0.0.1:3000".to_string(),
                "http://127.0.0.1:8080".to_string(),
            ]
        } else {
            std::env::var("CORS_ORIGINS")
                .unwrap_or_default()
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".parse().unwrap(),
            port: 8080,
            database_url: "sqlite:neural_doc_flow.db".to_string(),
            jwt_secret: "your-secret-key-here".to_string(),
            max_file_size: 100 * 1024 * 1024, // 100MB
            request_timeout: 300,
            rate_limit_rpm: 100,
            neural: NeuralConfig::default(),
            security: SecurityConfig::default(),
            monitoring: MonitoringConfig::default(),
            jobs: JobConfig::default(),
        }
    }
}

impl Default for NeuralConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            model_path: "./models".to_string(),
            max_threads: 4,
            timeout: 120,
            gpu_enabled: false,
        }
    }
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            level: 2,
            sandbox_enabled: true,
            timeout: 30,
            quarantine_hours: 24,
        }
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            metrics_enabled: true,
            metrics_interval: 60,
            tracing_enabled: true,
            log_level: "info".to_string(),
            profiling_enabled: false,
        }
    }
}

impl Default for JobConfig {
    fn default() -> Self {
        Self {
            workers: 4,
            queue_size: 1000,
            retry_attempts: 3,
            timeout: 600,
            cleanup_interval: 3600,
            retention_seconds: 86400,
        }
    }
}