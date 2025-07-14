//! Authentication handlers

use axum::{extract::State, Json};
use std::sync::Arc;
use validator::Validate;

use crate::state::AppState;
use crate::models::{LoginRequest, LoginResponse, RegisterRequest, UserInfo};
use crate::error::ApiResult;

/// User login
#[utoipa::path(
    post,
    path = "/api/v1/auth/login",
    tag = "Authentication",
    summary = "User login",
    description = "Authenticate user and return JWT token",
    request_body = LoginRequest,
    responses(
        (status = 200, description = "Login successful", body = LoginResponse),
        (status = 401, description = "Invalid credentials", body = ErrorResponse),
        (status = 422, description = "Validation error", body = ErrorResponse)
    )
)]
pub async fn login(
    State(state): State<Arc<AppState>>,
    Json(request): Json<LoginRequest>,
) -> ApiResult<Json<LoginResponse>> {
    request.validate()?;
    let response = state.auth.login(request).await?;
    Ok(Json(response))
}

/// User registration
#[utoipa::path(
    post,
    path = "/api/v1/auth/register",
    tag = "Authentication",
    summary = "User registration",
    description = "Register a new user account",
    request_body = RegisterRequest,
    responses(
        (status = 201, description = "User registered successfully", body = UserInfo),
        (status = 400, description = "User already exists", body = ErrorResponse),
        (status = 422, description = "Validation error", body = ErrorResponse)
    )
)]
pub async fn register(
    State(state): State<Arc<AppState>>,
    Json(request): Json<RegisterRequest>,
) -> ApiResult<Json<UserInfo>> {
    request.validate()?;
    let user = state.auth.register(request).await?;
    Ok(Json(user))
}