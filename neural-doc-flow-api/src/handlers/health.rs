//! Health check handlers

use axum::{extract::State, Json};
use std::sync::Arc;

use crate::state::AppState;
use crate::models::{HealthResponse, ServiceHealth, SystemMetrics};
use crate::error::ApiResult;

/// Health check endpoint
#[utoipa::path(
    get,
    path = "/health",
    tag = "Health",
    summary = "Health check",
    description = "Check the health status of the API and its dependencies",
    responses(
        (status = 200, description = "Service is healthy", body = HealthResponse),
        (status = 503, description = "Service is unhealthy", body = ErrorResponse)
    )
)]
pub async fn health_check(
    State(state): State<Arc<AppState>>,
) -> ApiResult<Json<HealthResponse>> {
    // Update health status
    state.update_health_status().await;
    
    let health = state.health_status.read().await;
    let stats = state.get_statistics().await?;
    
    let mut services = std::collections::HashMap::new();
    
    // Database health
    services.insert("database".to_string(), ServiceHealth {
        status: match health.database_status {
            crate::state::ServiceStatus::Healthy => "healthy".to_string(),
            crate::state::ServiceStatus::Degraded => "degraded".to_string(),
            crate::state::ServiceStatus::Unhealthy => "unhealthy".to_string(),
        },
        last_check: health.last_check,
        details: std::collections::HashMap::new(),
    });

    // Processor health
    services.insert("processor".to_string(), ServiceHealth {
        status: match health.processor_status {
            crate::state::ServiceStatus::Healthy => "healthy".to_string(),
            crate::state::ServiceStatus::Degraded => "degraded".to_string(),
            crate::state::ServiceStatus::Unhealthy => "unhealthy".to_string(),
        },
        last_check: health.last_check,
        details: std::collections::HashMap::new(),
    });

    // Coordinator health
    services.insert("coordinator".to_string(), ServiceHealth {
        status: match health.coordinator_status {
            crate::state::ServiceStatus::Healthy => "healthy".to_string(),
            crate::state::ServiceStatus::Degraded => "degraded".to_string(),
            crate::state::ServiceStatus::Unhealthy => "unhealthy".to_string(),
        },
        last_check: health.last_check,
        details: std::collections::HashMap::new(),
    });

    let response = HealthResponse {
        status: match health.overall_status {
            crate::state::ServiceStatus::Healthy => "healthy".to_string(),
            crate::state::ServiceStatus::Degraded => "degraded".to_string(),
            crate::state::ServiceStatus::Unhealthy => "unhealthy".to_string(),
        },
        uptime_seconds: health.uptime.as_secs(),
        version: health.version.clone(),
        timestamp: chrono::Utc::now(),
        services,
        metrics: SystemMetrics {
            memory_usage_bytes: stats.memory_usage,
            cpu_usage_percent: stats.cpu_usage,
            active_jobs: stats.active_jobs as u32,
            total_processed: stats.processor_stats.documents_processed,
            cache_size: stats.cache_size as u32,
        },
    };

    Ok(Json(response))
}

/// Readiness check endpoint (for Kubernetes)
pub async fn readiness_check(
    State(state): State<Arc<AppState>>,
) -> ApiResult<Json<serde_json::Value>> {
    state.update_health_status().await;
    
    let health = state.health_status.read().await;
    
    match health.overall_status {
        crate::state::ServiceStatus::Healthy => {
            Ok(Json(serde_json::json!({
                "status": "ready",
                "timestamp": chrono::Utc::now()
            })))
        }
        _ => {
            Err(crate::error::ApiError::ServiceUnavailable {
                message: "Service not ready".to_string(),
            })
        }
    }
}