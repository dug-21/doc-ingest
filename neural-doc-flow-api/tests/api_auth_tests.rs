mod common;

use common::*;
use neural_doc_flow_api::models::{
    ApiKeyCreate, ApiKeyResponse, AuthResponse, LoginRequest, RegisterRequest,
};

#[tokio::test]
async fn test_user_registration() {
    let (app, _state) = create_test_app().await;

    let register = RegisterRequest {
        email: "test@example.com".to_string(),
        password: "SecurePassword123!".to_string(),
        name: "Test User".to_string(),
    };

    let response = TestRequest::post("/api/v1/auth/register")
        .json(&register)
        .send(app)
        .await;

    response.assert_created();

    let auth: AuthResponse = response.json();
    assert!(!auth.token.is_empty());
    assert_eq!(auth.user.email, "test@example.com");
}

#[tokio::test]
async fn test_user_login() {
    let (app, _state) = create_test_app().await;

    // First register a user
    let register = RegisterRequest {
        email: "login@example.com".to_string(),
        password: "SecurePassword123!".to_string(),
        name: "Login Test".to_string(),
    };

    TestRequest::post("/api/v1/auth/register")
        .json(&register)
        .send(app.clone())
        .await
        .assert_created();

    // Now try to login
    let login = LoginRequest {
        email: "login@example.com".to_string(),
        password: "SecurePassword123!".to_string(),
    };

    let response = TestRequest::post("/api/v1/auth/login")
        .json(&login)
        .send(app)
        .await;

    response.assert_ok();

    let auth: AuthResponse = response.json();
    assert!(!auth.token.is_empty());
    assert_eq!(auth.user.email, "login@example.com");
}

#[tokio::test]
async fn test_invalid_login() {
    let (app, _state) = create_test_app().await;

    let login = LoginRequest {
        email: "nonexistent@example.com".to_string(),
        password: "WrongPassword".to_string(),
    };

    let response = TestRequest::post("/api/v1/auth/login")
        .json(&login)
        .send(app)
        .await;

    response.assert_unauthorized();
}

#[tokio::test]
async fn test_jwt_authentication() {
    let (app, state) = create_test_app().await;

    // Create a test token
    let token = MockAuth::create_test_token(&state, "test-user-id", "test@example.com");

    // Try to access a protected endpoint
    let response = TestRequest::get("/api/v1/auth/me")
        .bearer_auth(&token)
        .send(app)
        .await;

    response.assert_ok();
}

#[tokio::test]
async fn test_invalid_jwt() {
    let (app, _state) = create_test_app().await;

    let response = TestRequest::get("/api/v1/auth/me")
        .bearer_auth("invalid.jwt.token")
        .send(app)
        .await;

    response.assert_unauthorized();
}

#[tokio::test]
async fn test_expired_jwt() {
    let (app, state) = create_test_app().await;

    // Create an expired token
    let claims = neural_doc_flow_api::auth::Claims {
        sub: "test-user".to_string(),
        email: "test@example.com".to_string(),
        exp: (chrono::Utc::now() - chrono::Duration::hours(1)).timestamp() as usize,
        iat: (chrono::Utc::now() - chrono::Duration::hours(2)).timestamp() as usize,
    };

    let token = neural_doc_flow_api::auth::JwtAuth::encode_token(&claims, &state.config.jwt_secret).unwrap();

    let response = TestRequest::get("/api/v1/auth/me")
        .bearer_auth(&token)
        .send(app)
        .await;

    response.assert_unauthorized();
}

#[tokio::test]
async fn test_api_key_creation() {
    let (app, state) = create_test_app().await;

    // First need to authenticate
    let token = MockAuth::create_test_token(&state, "test-user", "test@example.com");

    let api_key_request = ApiKeyCreate {
        name: "Test API Key".to_string(),
        expires_in_days: Some(30),
    };

    let response = TestRequest::post("/api/v1/auth/api-keys")
        .bearer_auth(&token)
        .json(&api_key_request)
        .send(app)
        .await;

    response.assert_created();

    let api_key: ApiKeyResponse = response.json();
    assert_eq!(api_key.name, "Test API Key");
    assert!(!api_key.key.is_empty());
}

#[tokio::test]
async fn test_api_key_authentication() {
    let (app, state) = create_test_app().await;

    // Create an API key
    let api_key = MockAuth::create_test_api_key(&state, "test-key").await;

    // Use the API key to access a protected endpoint
    let response = TestRequest::get("/api/v1/documents")
        .api_key(&api_key.key)
        .send(app)
        .await;

    response.assert_ok();
}

#[tokio::test]
async fn test_invalid_api_key() {
    let (app, _state) = create_test_app().await;

    let response = TestRequest::get("/api/v1/documents")
        .api_key("invalid-api-key")
        .send(app)
        .await;

    response.assert_unauthorized();
}

#[tokio::test]
async fn test_deactivated_api_key() {
    let (app, state) = create_test_app().await;

    // Create and then deactivate an API key
    let mut api_key = MockAuth::create_test_api_key(&state, "deactivated-key").await;
    
    // Deactivate the key
    sqlx::query!(
        "UPDATE api_keys SET is_active = false WHERE id = ?",
        api_key.id
    )
    .execute(&state.db)
    .await
    .unwrap();

    api_key.is_active = false;

    // Try to use the deactivated key
    let response = TestRequest::get("/api/v1/documents")
        .api_key(&api_key.key)
        .send(app)
        .await;

    response.assert_unauthorized();
}

#[tokio::test]
async fn test_api_key_listing() {
    let (app, state) = create_test_app().await;

    // Authenticate
    let token = MockAuth::create_test_token(&state, "test-user", "test@example.com");

    // Create some API keys
    MockAuth::create_test_api_key(&state, "key1").await;
    MockAuth::create_test_api_key(&state, "key2").await;

    let response = TestRequest::get("/api/v1/auth/api-keys")
        .bearer_auth(&token)
        .send(app)
        .await;

    response.assert_ok();

    let keys: Vec<serde_json::Value> = response.json();
    assert!(keys.len() >= 2);
}

#[tokio::test]
async fn test_api_key_revocation() {
    let (app, state) = create_test_app().await;

    // Authenticate
    let token = MockAuth::create_test_token(&state, "test-user", "test@example.com");

    // Create an API key
    let api_key = MockAuth::create_test_api_key(&state, "to-revoke").await;

    // Revoke the key
    let response = TestRequest::delete(&format!("/api/v1/auth/api-keys/{}", api_key.id))
        .bearer_auth(&token)
        .send(app.clone())
        .await;

    response.assert_ok();

    // Try to use the revoked key
    let response = TestRequest::get("/api/v1/documents")
        .api_key(&api_key.key)
        .send(app)
        .await;

    response.assert_unauthorized();
}

#[tokio::test]
async fn test_password_requirements() {
    let (app, _state) = create_test_app().await;

    // Test weak password
    let register = RegisterRequest {
        email: "weak@example.com".to_string(),
        password: "weak".to_string(),
        name: "Weak Password".to_string(),
    };

    let response = TestRequest::post("/api/v1/auth/register")
        .json(&register)
        .send(app)
        .await;

    response.assert_bad_request();
}

#[tokio::test]
async fn test_duplicate_email_registration() {
    let (app, _state) = create_test_app().await;

    let register = RegisterRequest {
        email: "duplicate@example.com".to_string(),
        password: "SecurePassword123!".to_string(),
        name: "First User".to_string(),
    };

    // First registration should succeed
    TestRequest::post("/api/v1/auth/register")
        .json(&register)
        .send(app.clone())
        .await
        .assert_created();

    // Second registration with same email should fail
    let response = TestRequest::post("/api/v1/auth/register")
        .json(&register)
        .send(app)
        .await;

    response.assert_bad_request();
}

#[tokio::test]
async fn test_missing_auth_header() {
    let (app, _state) = create_test_app().await;

    // Try to access protected endpoint without auth
    let response = TestRequest::get("/api/v1/auth/me")
        .send(app)
        .await;

    response.assert_unauthorized();
}