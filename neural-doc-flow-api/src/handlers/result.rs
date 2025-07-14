//! Job result handlers

use axum::{extract::{State, Path}, Json};
use std::sync::Arc;

use crate::state::AppState;
use crate::models::ResultResponse;
use crate::error::ApiResult;

/// Get job result
#[utoipa::path(
    get,
    path = "/api/v1/result/{job_id}",
    tag = "Status",
    summary = "Get job result",
    description = "Get the result of a completed processing job",
    params(
        ("job_id" = String, Path, description = "Job ID to get result for")
    ),
    responses(
        (status = 200, description = "Job result retrieved", body = ResultResponse),
        (status = 404, description = "Job not found", body = ErrorResponse),
        (status = 202, description = "Job not completed yet", body = ErrorResponse)
    ),
    security(
        ("bearer_auth" = [])
    )
)]
pub async fn get_result(
    State(_state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
) -> ApiResult<Json<ResultResponse>> {
    // TODO: Implement job result retrieval
    Err(crate::error::ApiError::NotFound {
        message: "Job not found or not completed".to_string(),
    })
}