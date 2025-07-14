//! Logging middleware

use axum::{
    http::Request,
    middleware::Next,
    response::Response,
};

/// Request logging middleware
pub async fn logging_middleware<B>(
    request: Request<B>,
    next: Next<B>,
) -> Response {
    let method = request.method().clone();
    let uri = request.uri().clone();
    let start = std::time::Instant::now();

    let response = next.run(request).await;

    let duration = start.elapsed();
    let status = response.status();

    tracing::info!(
        method = %method,
        uri = %uri,
        status = %status,
        duration_ms = duration.as_millis(),
        "HTTP request"
    );

    response
}