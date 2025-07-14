//! Job status handlers

use axum::{extract::{State, Path}, Json};
use std::sync::Arc;

use crate::state::AppState;
use crate::models::StatusResponse;
use crate::error::ApiResult;

/// Get job status
#[utoipa::path(
    get,
    path = "/api/v1/status/{job_id}",
    tag = "Status",
    summary = "Get job status",
    description = "Get the current status of a processing job",
    params(
        ("job_id" = String, Path, description = "Job ID to check status for")
    ),
    responses(
        (status = 200, description = "Job status retrieved", body = StatusResponse),
        (status = 404, description = "Job not found", body = ErrorResponse)
    ),
    security(
        ("bearer_auth" = [])
    )
)]
pub async fn get_status(
    State(_state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
) -> ApiResult<Json<StatusResponse>> {
    // TODO: Implement job status lookup
    Ok(Json(StatusResponse {
        job_id,
        status: crate::models::JobStatus::Processing,
        progress: Some(50),
        message: Some("Processing document...".to_string()),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        processing_duration_ms: None,
        error: None,
    }))
}