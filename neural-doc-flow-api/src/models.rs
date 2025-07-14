//! API data models and request/response types

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use validator::Validate;
use std::collections::HashMap;

/// Document processing request
#[derive(Debug, Serialize, Deserialize, ToSchema, Validate)]
pub struct ProcessRequest {
    /// Base64 encoded document content or file URL
    #[validate(length(min = 1, message = "Content cannot be empty"))]
    pub content: String,
    
    /// Original filename
    #[validate(length(min = 1, max = 255, message = "Filename must be 1-255 characters"))]
    pub filename: String,
    
    /// MIME type of the document
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_type: Option<String>,
    
    /// Processing configuration options
    #[serde(default)]
    pub options: ProcessingOptions,
    
    /// Whether to return results immediately or process asynchronously
    #[serde(default)]
    pub async_processing: bool,
    
    /// Client-provided request ID for tracking
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_request_id: Option<String>,
}

/// Processing configuration options
#[derive(Debug, Serialize, Deserialize, ToSchema, Validate)]
pub struct ProcessingOptions {
    /// Enable neural enhancement
    #[serde(default = "default_neural_enabled")]
    pub neural_enhancement: bool,
    
    /// Security scanning level (0-3)
    #[validate(range(min = 0, max = 3, message = "Security level must be 0-3"))]
    #[serde(default = "default_security_level")]
    pub security_level: u8,
    
    /// Output formats to generate
    #[serde(default = "default_output_formats")]
    pub output_formats: Vec<String>,
    
    /// Custom processing parameters
    #[serde(default)]
    pub custom_parameters: HashMap<String, String>,
    
    /// Processing timeout in seconds
    #[validate(range(min = 1, max = 600, message = "Timeout must be 1-600 seconds"))]
    #[serde(default = "default_timeout")]
    pub timeout_seconds: u32,
    
    /// Extract metadata only (skip content processing)
    #[serde(default)]
    pub metadata_only: bool,
}

/// Document processing response
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ProcessResponse {
    /// Job ID for tracking processing status
    pub job_id: String,
    
    /// Processing status
    pub status: JobStatus,
    
    /// Processing result (if completed synchronously)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<ProcessingResult>,
    
    /// Estimated completion time for async jobs
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_completion: Option<chrono::DateTime<chrono::Utc>>,
    
    /// Job submission timestamp
    pub submitted_at: chrono::DateTime<chrono::Utc>,
    
    /// Processing warnings
    #[serde(default)]
    pub warnings: Vec<String>,
}

/// Batch processing request
#[derive(Debug, Serialize, Deserialize, ToSchema, Validate)]
pub struct BatchRequest {
    /// List of documents to process
    #[validate(length(min = 1, max = 100, message = "Batch must contain 1-100 documents"))]
    pub documents: Vec<BatchDocument>,
    
    /// Common processing options for all documents
    #[serde(default)]
    pub options: ProcessingOptions,
    
    /// Batch processing configuration
    #[serde(default)]
    pub batch_config: BatchConfig,
}

/// Document in a batch request
#[derive(Debug, Serialize, Deserialize, ToSchema, Validate)]
pub struct BatchDocument {
    /// Unique identifier within the batch
    #[validate(length(min = 1, max = 50, message = "Document ID must be 1-50 characters"))]
    pub id: String,
    
    /// Base64 encoded content
    #[validate(length(min = 1, message = "Content cannot be empty"))]
    pub content: String,
    
    /// Original filename
    #[validate(length(min = 1, max = 255, message = "Filename must be 1-255 characters"))]
    pub filename: String,
    
    /// MIME type
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_type: Option<String>,
    
    /// Document-specific options (overrides batch options)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<ProcessingOptions>,
}

/// Batch processing configuration
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct BatchConfig {
    /// Maximum concurrent processing jobs
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent: u32,
    
    /// Whether to stop on first error
    #[serde(default)]
    pub fail_fast: bool,
    
    /// Whether to preserve document order in results
    #[serde(default = "default_preserve_order")]
    pub preserve_order: bool,
    
    /// Batch timeout in seconds
    #[serde(default = "default_batch_timeout")]
    pub timeout_seconds: u32,
}

/// Batch processing response
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct BatchResponse {
    /// Batch job ID
    pub batch_id: String,
    
    /// Overall batch status
    pub status: JobStatus,
    
    /// Individual document job IDs
    pub document_jobs: HashMap<String, String>,
    
    /// Batch processing statistics
    pub statistics: BatchStatistics,
    
    /// Batch submission timestamp
    pub submitted_at: chrono::DateTime<chrono::Utc>,
    
    /// Estimated completion time
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_completion: Option<chrono::DateTime<chrono::Utc>>,
}

/// Job status enumeration
#[derive(Debug, Serialize, Deserialize, ToSchema, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum JobStatus {
    /// Job is queued for processing
    Queued,
    /// Job is currently being processed
    Processing,
    /// Job completed successfully
    Completed,
    /// Job failed with error
    Failed,
    /// Job was cancelled
    Cancelled,
    /// Job timed out
    Timeout,
}

/// Status check response
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct StatusResponse {
    /// Job ID
    pub job_id: String,
    
    /// Current status
    pub status: JobStatus,
    
    /// Processing progress (0-100)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress: Option<u8>,
    
    /// Status message
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    
    /// Job creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    
    /// Last updated timestamp
    pub updated_at: chrono::DateTime<chrono::Utc>,
    
    /// Processing duration (if completed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub processing_duration_ms: Option<u64>,
    
    /// Error details (if failed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Processing result response
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ResultResponse {
    /// Job ID
    pub job_id: String,
    
    /// Processing result
    pub result: ProcessingResult,
    
    /// Result metadata
    pub metadata: ResultMetadata,
}

/// Document processing result
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ProcessingResult {
    /// Processing success status
    pub success: bool,
    
    /// Extracted text content
    pub content: String,
    
    /// Document metadata
    pub document_metadata: HashMap<String, String>,
    
    /// Generated outputs in different formats
    pub outputs: HashMap<String, OutputData>,
    
    /// Security scan results
    #[serde(skip_serializing_if = "Option::is_none")]
    pub security_results: Option<SecurityResults>,
    
    /// Processing warnings
    #[serde(default)]
    pub warnings: Vec<String>,
    
    /// Processing statistics
    pub statistics: ProcessingStatistics,
}

/// Output data for different formats
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct OutputData {
    /// Output format type
    pub format: String,
    
    /// Output content (base64 encoded for binary formats)
    pub content: String,
    
    /// Content type/MIME type
    pub content_type: String,
    
    /// Output size in bytes
    pub size: u64,
    
    /// Additional format-specific metadata
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

/// Security scan results
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct SecurityResults {
    /// Overall safety assessment
    pub is_safe: bool,
    
    /// Threat level (0-10)
    pub threat_level: u8,
    
    /// Detected threats
    pub detected_threats: Vec<String>,
    
    /// Security scan duration
    pub scan_duration_ms: u32,
    
    /// Detailed scan results
    pub scan_details: HashMap<String, String>,
}

/// Processing statistics
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ProcessingStatistics {
    /// Total processing time in milliseconds
    pub processing_time_ms: u64,
    
    /// Neural processing time (if enabled)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub neural_time_ms: Option<u64>,
    
    /// Security scanning time
    #[serde(skip_serializing_if = "Option::is_none")]
    pub security_time_ms: Option<u32>,
    
    /// Memory usage during processing
    pub memory_usage_bytes: u64,
    
    /// Number of pages processed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pages_processed: Option<u32>,
    
    /// Character count
    #[serde(skip_serializing_if = "Option::is_none")]
    pub character_count: Option<u64>,
}

/// Result metadata
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ResultMetadata {
    /// Result generation timestamp
    pub generated_at: chrono::DateTime<chrono::Utc>,
    
    /// API version used
    pub api_version: String,
    
    /// Processing pipeline version
    pub pipeline_version: String,
    
    /// Result expiration time
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<chrono::DateTime<chrono::Utc>>,
    
    /// Result caching information
    pub cache_info: CacheInfo,
}

/// Cache information
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct CacheInfo {
    /// Whether result was served from cache
    pub from_cache: bool,
    
    /// Cache key used
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_key: Option<String>,
    
    /// Cache hit timestamp
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_hit_at: Option<chrono::DateTime<chrono::Utc>>,
}

/// Batch processing statistics
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct BatchStatistics {
    /// Total number of documents
    pub total_documents: u32,
    
    /// Number of completed documents
    pub completed_documents: u32,
    
    /// Number of failed documents
    pub failed_documents: u32,
    
    /// Number of pending documents
    pub pending_documents: u32,
    
    /// Overall processing progress (0-100)
    pub progress: u8,
    
    /// Average processing time per document
    #[serde(skip_serializing_if = "Option::is_none")]
    pub average_processing_time_ms: Option<u64>,
}

/// Health check response
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct HealthResponse {
    /// Overall health status
    pub status: String,
    
    /// Service uptime in seconds
    pub uptime_seconds: u64,
    
    /// API version
    pub version: String,
    
    /// System timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    
    /// Detailed service statuses
    pub services: HashMap<String, ServiceHealth>,
    
    /// System metrics
    pub metrics: SystemMetrics,
}

/// Individual service health
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ServiceHealth {
    /// Service status
    pub status: String,
    
    /// Last check timestamp
    pub last_check: chrono::DateTime<chrono::Utc>,
    
    /// Additional service details
    #[serde(default)]
    pub details: HashMap<String, String>,
}

/// System metrics
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct SystemMetrics {
    /// Memory usage in bytes
    pub memory_usage_bytes: u64,
    
    /// CPU usage percentage
    pub cpu_usage_percent: f64,
    
    /// Active jobs count
    pub active_jobs: u32,
    
    /// Total processed documents
    pub total_processed: u64,
    
    /// Cache size
    pub cache_size: u32,
}

/// Authentication request
#[derive(Debug, Serialize, Deserialize, ToSchema, Validate)]
pub struct LoginRequest {
    /// Username or email
    #[validate(length(min = 1, max = 255, message = "Username/email required"))]
    pub username: String,
    
    /// Password
    #[validate(length(min = 8, message = "Password must be at least 8 characters"))]
    pub password: String,
}

/// Authentication response
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct LoginResponse {
    /// JWT access token
    pub access_token: String,
    
    /// Token type (always "Bearer")
    pub token_type: String,
    
    /// Token expiration time
    pub expires_at: chrono::DateTime<chrono::Utc>,
    
    /// User information
    pub user: UserInfo,
}

/// User registration request
#[derive(Debug, Serialize, Deserialize, ToSchema, Validate)]
pub struct RegisterRequest {
    /// Username
    #[validate(length(min = 3, max = 50, message = "Username must be 3-50 characters"))]
    pub username: String,
    
    /// Email address
    #[validate(email(message = "Invalid email address"))]
    pub email: String,
    
    /// Password
    #[validate(length(min = 8, message = "Password must be at least 8 characters"))]
    pub password: String,
    
    /// Full name
    #[validate(length(min = 1, max = 100, message = "Name must be 1-100 characters"))]
    pub full_name: String,
}

/// User information
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct UserInfo {
    /// User ID
    pub id: String,
    
    /// Username
    pub username: String,
    
    /// Email address
    pub email: String,
    
    /// Full name
    pub full_name: String,
    
    /// User role
    pub role: String,
    
    /// Account creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    
    /// Last login timestamp
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_login: Option<chrono::DateTime<chrono::Utc>>,
}

/// Error response
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ErrorResponse {
    /// Error type
    pub error: String,
    
    /// Error message
    pub message: String,
    
    /// Optional error details
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
    
    /// Request ID for tracing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    
    /// Error timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

// Default value functions
fn default_neural_enabled() -> bool { true }
fn default_security_level() -> u8 { 2 }
fn default_output_formats() -> Vec<String> { vec!["text".to_string(), "json".to_string()] }
fn default_timeout() -> u32 { 300 }
fn default_max_concurrent() -> u32 { 4 }
fn default_preserve_order() -> bool { true }
fn default_batch_timeout() -> u32 { 1800 }

impl Default for ProcessingOptions {
    fn default() -> Self {
        Self {
            neural_enhancement: default_neural_enabled(),
            security_level: default_security_level(),
            output_formats: default_output_formats(),
            custom_parameters: HashMap::new(),
            timeout_seconds: default_timeout(),
            metadata_only: false,
        }
    }
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_concurrent: default_max_concurrent(),
            fail_fast: false,
            preserve_order: default_preserve_order(),
            timeout_seconds: default_batch_timeout(),
        }
    }
}