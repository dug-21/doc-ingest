//! Metrics middleware

use axum::{
    http::Request,
    middleware::Next,
    response::Response,
};

/// Metrics collection middleware
pub async fn metrics_middleware<B>(
    request: Request<B>,
    next: Next<B>,
) -> Response {
    let start = std::time::Instant::now();
    let method = request.method().clone();
    let path = request.uri().path().to_string();

    let response = next.run(request).await;

    let duration = start.elapsed();
    let status = response.status();

    // TODO: Record metrics
    tracing::debug!(
        method = %method,
        path = %path,
        status = %status,
        duration_ms = duration.as_millis(),
        "Metrics recorded"
    );

    response
}