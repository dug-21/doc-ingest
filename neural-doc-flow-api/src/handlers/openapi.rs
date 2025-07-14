//! OpenAPI specification handler

use axum::Json;
use crate::ApiDoc;
use utoipa::OpenApi;

/// OpenAPI specification endpoint
pub async fn openapi_spec() -> Json<utoipa::openapi::OpenApi> {
    Json(ApiDoc::openapi())
}