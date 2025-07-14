//! Authentication middleware

use axum::{
    extract::State,
    http::{Request, header::AUTHORIZATION},
    middleware::Next,
    response::Response,
};
use std::sync::Arc;

use crate::state::AppState;
use crate::auth::{AuthManager, Claims};

/// Authentication middleware
pub async fn auth_middleware<B>(
    State(state): State<Arc<AppState>>,
    mut request: Request<B>,
    next: Next<B>,
) -> Result<Response, Response> {
    // Skip auth for public endpoints
    let path = request.uri().path();
    if is_public_endpoint(path) {
        return Ok(next.run(request).await);
    }

    // Extract authorization header
    let auth_header = request
        .headers()
        .get(AUTHORIZATION)
        .and_then(|header| header.to_str().ok());

    if let Some(auth_header) = auth_header {
        if let Some(token) = AuthManager::extract_token(auth_header) {
            match state.auth.validate_token(token) {
                Ok(claims) => {
                    // Add claims to request extensions
                    request.extensions_mut().insert(claims);
                    return Ok(next.run(request).await);
                }
                Err(_) => {
                    return Err(crate::error::handle_error(
                        crate::error::ApiError::Unauthorized {
                            message: "Invalid token".to_string(),
                        }
                    ));
                }
            }
        }
    }

    Err(crate::error::handle_error(
        crate::error::ApiError::Unauthorized {
            message: "Missing or invalid authorization header".to_string(),
        }
    ))
}

/// Check if endpoint is public (doesn't require authentication)
fn is_public_endpoint(path: &str) -> bool {
    matches!(path, 
        "/health" | "/ready" | "/metrics" | "/docs" | "/docs/" | 
        "/openapi.json" | "/api/v1/auth/login" | "/api/v1/auth/register" |
        path if path.starts_with("/docs/") || path.starts_with("/_app/")
    )
}

/// Extract claims from request
impl<B> axum::extract::FromRequestParts<B> for Claims
where
    B: Send + Sync,
{
    type Rejection = crate::error::ApiError;

    async fn from_request_parts(
        parts: &mut axum::http::request::Parts,
        _state: &B,
    ) -> Result<Self, Self::Rejection> {
        parts
            .extensions
            .get::<Claims>()
            .cloned()
            .ok_or_else(|| crate::error::ApiError::Unauthorized {
                message: "Missing authentication".to_string(),
            })
    }
}