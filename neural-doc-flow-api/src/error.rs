//! Error handling for the REST API

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use utoipa::ToSchema;

/// API error types
#[derive(Error, Debug)]
pub enum ApiError {
    #[error("Invalid request: {message}")]
    BadRequest { message: String },

    #[error("Authentication failed: {message}")]
    Unauthorized { message: String },

    #[error("Access denied: {message}")]
    Forbidden { message: String },

    #[error("Resource not found: {message}")]
    NotFound { message: String },

    #[error("Request too large: {message}")]
    PayloadTooLarge { message: String },

    #[error("Rate limit exceeded")]
    RateLimited,

    #[error("Validation error: {message}")]
    ValidationError { message: String },

    #[error("Processing error: {message}")]
    ProcessingError { message: String },

    #[error("Security error: {message}")]
    SecurityError { message: String },

    #[error("Database error: {message}")]
    DatabaseError { message: String },

    #[error("Internal server error: {message}")]
    InternalError { message: String },

    #[error("Service unavailable: {message}")]
    ServiceUnavailable { message: String },

    #[error("Request timeout")]
    Timeout,
}

/// Error response structure
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ErrorResponse {
    /// Error type identifier
    pub error: String,
    /// Human-readable error message
    pub message: String,
    /// Optional error details
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
    /// Request ID for tracing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    /// Timestamp of the error
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// API result type alias
pub type ApiResult<T> = Result<T, ApiError>;

impl ApiError {
    /// Get the HTTP status code for this error
    pub fn status_code(&self) -> StatusCode {
        match self {
            ApiError::BadRequest { .. } => StatusCode::BAD_REQUEST,
            ApiError::Unauthorized { .. } => StatusCode::UNAUTHORIZED,
            ApiError::Forbidden { .. } => StatusCode::FORBIDDEN,
            ApiError::NotFound { .. } => StatusCode::NOT_FOUND,
            ApiError::PayloadTooLarge { .. } => StatusCode::PAYLOAD_TOO_LARGE,
            ApiError::RateLimited => StatusCode::TOO_MANY_REQUESTS,
            ApiError::ValidationError { .. } => StatusCode::UNPROCESSABLE_ENTITY,
            ApiError::ProcessingError { .. } => StatusCode::UNPROCESSABLE_ENTITY,
            ApiError::SecurityError { .. } => StatusCode::FORBIDDEN,
            ApiError::DatabaseError { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ApiError::InternalError { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            ApiError::ServiceUnavailable { .. } => StatusCode::SERVICE_UNAVAILABLE,
            ApiError::Timeout => StatusCode::REQUEST_TIMEOUT,
        }
    }

    /// Get the error type identifier
    pub fn error_type(&self) -> &'static str {
        match self {
            ApiError::BadRequest { .. } => "bad_request",
            ApiError::Unauthorized { .. } => "unauthorized",
            ApiError::Forbidden { .. } => "forbidden",
            ApiError::NotFound { .. } => "not_found",
            ApiError::PayloadTooLarge { .. } => "payload_too_large",
            ApiError::RateLimited => "rate_limited",
            ApiError::ValidationError { .. } => "validation_error",
            ApiError::ProcessingError { .. } => "processing_error",
            ApiError::SecurityError { .. } => "security_error",
            ApiError::DatabaseError { .. } => "database_error",
            ApiError::InternalError { .. } => "internal_error",
            ApiError::ServiceUnavailable { .. } => "service_unavailable",
            ApiError::Timeout => "timeout",
        }
    }

    /// Create error response
    pub fn into_response(self) -> ErrorResponse {
        ErrorResponse {
            error: self.error_type().to_string(),
            message: self.to_string(),
            details: None,
            request_id: None, // This would be set by middleware
            timestamp: chrono::Utc::now(),
        }
    }

    /// Create error response with details
    pub fn with_details(self, details: serde_json::Value) -> ErrorResponse {
        let mut response = self.into_response();
        response.details = Some(details);
        response
    }

    /// Create error response with request ID
    pub fn with_request_id(self, request_id: String) -> ErrorResponse {
        let mut response = self.into_response();
        response.request_id = Some(request_id);
        response
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let response = self.into_response();
        
        // Log the error
        match status {
            StatusCode::INTERNAL_SERVER_ERROR | StatusCode::SERVICE_UNAVAILABLE => {
                tracing::error!("API Error: {}", self);
            }
            StatusCode::BAD_REQUEST | StatusCode::UNPROCESSABLE_ENTITY => {
                tracing::warn!("API Error: {}", self);
            }
            _ => {
                tracing::debug!("API Error: {}", self);
            }
        }

        (status, Json(response)).into_response()
    }
}

// Conversion from various error types
impl From<neural_doc_flow_core::error::ProcessingError> for ApiError {
    fn from(error: neural_doc_flow_core::error::ProcessingError) -> Self {
        ApiError::ProcessingError {
            message: error.to_string(),
        }
    }
}

impl From<sqlx::Error> for ApiError {
    fn from(error: sqlx::Error) -> Self {
        match error {
            sqlx::Error::RowNotFound => ApiError::NotFound {
                message: "Resource not found".to_string(),
            },
            _ => ApiError::DatabaseError {
                message: error.to_string(),
            },
        }
    }
}

impl From<serde_json::Error> for ApiError {
    fn from(error: serde_json::Error) -> Self {
        ApiError::BadRequest {
            message: format!("JSON parsing error: {}", error),
        }
    }
}

impl From<validator::ValidationErrors> for ApiError {
    fn from(errors: validator::ValidationErrors) -> Self {
        let mut messages = Vec::new();
        for (field, field_errors) in errors.field_errors() {
            for error in field_errors {
                let message = error.message
                    .as_ref()
                    .map(|m| m.to_string())
                    .unwrap_or_else(|| format!("Invalid value for field '{}'", field));
                messages.push(format!("{}: {}", field, message));
            }
        }
        
        ApiError::ValidationError {
            message: messages.join(", "),
        }
    }
}

impl From<jsonwebtoken::errors::Error> for ApiError {
    fn from(error: jsonwebtoken::errors::Error) -> Self {
        match error.kind() {
            jsonwebtoken::errors::ErrorKind::ExpiredSignature => ApiError::Unauthorized {
                message: "Token has expired".to_string(),
            },
            jsonwebtoken::errors::ErrorKind::InvalidToken => ApiError::Unauthorized {
                message: "Invalid token".to_string(),
            },
            _ => ApiError::Unauthorized {
                message: "Authentication failed".to_string(),
            },
        }
    }
}

impl From<tokio::time::error::Elapsed> for ApiError {
    fn from(_: tokio::time::error::Elapsed) -> Self {
        ApiError::Timeout
    }
}

/// Helper macros for creating specific error types
#[macro_export]
macro_rules! bad_request {
    ($msg:expr) => {
        ApiError::BadRequest {
            message: $msg.to_string(),
        }
    };
}

#[macro_export]
macro_rules! unauthorized {
    ($msg:expr) => {
        ApiError::Unauthorized {
            message: $msg.to_string(),
        }
    };
}

#[macro_export]
macro_rules! forbidden {
    ($msg:expr) => {
        ApiError::Forbidden {
            message: $msg.to_string(),
        }
    };
}

#[macro_export]
macro_rules! not_found {
    ($msg:expr) => {
        ApiError::NotFound {
            message: $msg.to_string(),
        }
    };
}

#[macro_export]
macro_rules! internal_error {
    ($msg:expr) => {
        ApiError::InternalError {
            message: $msg.to_string(),
        }
    };
}

/// Error handler middleware result
pub type MiddlewareResult<T> = Result<T, Response>;

/// Convert API errors to middleware-compatible responses
pub fn handle_error(error: ApiError) -> Response {
    error.into_response()
}

/// Create a standardized error response
pub fn error_response(
    error_type: &str,
    message: &str,
    status: StatusCode,
) -> Response {
    let response = ErrorResponse {
        error: error_type.to_string(),
        message: message.to_string(),
        details: None,
        request_id: None,
        timestamp: chrono::Utc::now(),
    };

    (status, Json(response)).into_response()
}

/// Custom error types for specific use cases
#[derive(Error, Debug)]
pub enum AuthError {
    #[error("Invalid credentials")]
    InvalidCredentials,
    #[error("Token expired")]
    TokenExpired,
    #[error("Token invalid")]
    TokenInvalid,
    #[error("Insufficient permissions")]
    InsufficientPermissions,
}

impl From<AuthError> for ApiError {
    fn from(error: AuthError) -> Self {
        match error {
            AuthError::InvalidCredentials => ApiError::Unauthorized {
                message: "Invalid credentials provided".to_string(),
            },
            AuthError::TokenExpired => ApiError::Unauthorized {
                message: "Authentication token has expired".to_string(),
            },
            AuthError::TokenInvalid => ApiError::Unauthorized {
                message: "Invalid authentication token".to_string(),
            },
            AuthError::InsufficientPermissions => ApiError::Forbidden {
                message: "Insufficient permissions for this operation".to_string(),
            },
        }
    }
}

#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("File too large: {size} bytes (max: {max} bytes)")]
    FileTooLarge { size: u64, max: u64 },
    #[error("Unsupported file type: {file_type}")]
    UnsupportedFileType { file_type: String },
    #[error("Invalid file format")]
    InvalidFileFormat,
    #[error("Missing required field: {field}")]
    MissingField { field: String },
}

impl From<ValidationError> for ApiError {
    fn from(error: ValidationError) -> Self {
        match error {
            ValidationError::FileTooLarge { .. } => ApiError::PayloadTooLarge {
                message: error.to_string(),
            },
            _ => ApiError::ValidationError {
                message: error.to_string(),
            },
        }
    }
}