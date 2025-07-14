//! Rate limiting middleware

use axum::{
    extract::{State, ConnectInfo},
    http::Request,
    middleware::Next,
    response::Response,
};
use std::{sync::Arc, net::SocketAddr};

use crate::state::AppState;

/// Rate limiting middleware
pub async fn rate_limit_middleware<B>(
    State(state): State<Arc<AppState>>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    request: Request<B>,
    next: Next<B>,
) -> Result<Response, Response> {
    let ip = addr.ip().to_string();
    let rate_limiter = state.get_rate_limiter(&ip);

    // Check rate limit
    if rate_limiter.check_key(&ip).is_err() {
        return Err(crate::error::handle_error(
            crate::error::ApiError::RateLimited
        ));
    }

    Ok(next.run(request).await)
}