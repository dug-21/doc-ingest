//! API route definitions

use axum::{routing::{get, post}, Router};
use std::sync::Arc;

use crate::state::AppState;
use crate::handlers;

/// Create API routes
pub fn api_routes() -> Router<Arc<AppState>> {
    Router::new()
        // Authentication routes
        .route("/auth/login", post(handlers::auth::login))
        .route("/auth/register", post(handlers::auth::register))
        
        // Processing routes
        .route("/process", post(handlers::process::process_document))
        .route("/batch", post(handlers::process::process_batch))
        
        // Status and result routes
        .route("/status/:job_id", get(handlers::status::get_status))
        .route("/result/:job_id", get(handlers::result::get_result))
}