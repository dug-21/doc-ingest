mod common;

use common::*;
use neural_doc_flow_api::models::HealthResponse;

#[tokio::test]
async fn test_health_check() {
    let (app, _state) = create_test_app().await;

    let response = TestRequest::get("/api/v1/health")
        .send(app)
        .await;

    response.assert_ok();
    
    let health: HealthResponse = response.json();
    assert_eq!(health.status, "healthy");
    assert!(health.version.len() > 0);
}

#[tokio::test]
async fn test_cors_headers() {
    let (app, _state) = create_test_app().await;

    let response = TestRequest::get("/api/v1/health")
        .header("Origin", "http://localhost:3000")
        .send(app)
        .await;

    response
        .assert_ok()
        .assert_header("access-control-allow-origin", "*");
}

#[tokio::test]
async fn test_404_not_found() {
    let (app, _state) = create_test_app().await;

    let response = TestRequest::get("/api/v1/nonexistent")
        .send(app)
        .await;

    response.assert_not_found();
}

#[tokio::test]
async fn test_method_not_allowed() {
    let (app, _state) = create_test_app().await;

    // Try to POST to a GET-only endpoint
    let response = TestRequest::post("/api/v1/health")
        .send(app)
        .await;

    assert_eq!(response.status, axum::http::StatusCode::METHOD_NOT_ALLOWED);
}

#[tokio::test]
async fn test_request_id_header() {
    let (app, _state) = create_test_app().await;

    let response = TestRequest::get("/api/v1/health")
        .send(app)
        .await;

    // Check that X-Request-Id header is present
    assert!(response.headers.contains_key("x-request-id"));
}

#[tokio::test]
async fn test_json_content_type() {
    let (app, _state) = create_test_app().await;

    let response = TestRequest::get("/api/v1/health")
        .send(app)
        .await;

    response.assert_header("content-type", "application/json");
}

#[tokio::test]
async fn test_metrics_endpoint() {
    let (app, _state) = create_test_app().await;

    let response = TestRequest::get("/metrics")
        .send(app)
        .await;

    response.assert_ok();
    
    let body = response.text();
    assert!(body.contains("# HELP"));
    assert!(body.contains("# TYPE"));
}

#[tokio::test]
async fn test_openapi_spec() {
    let (app, _state) = create_test_app().await;

    let response = TestRequest::get("/api/v1/openapi.json")
        .send(app)
        .await;

    response
        .assert_ok()
        .assert_header("content-type", "application/json");

    let spec: serde_json::Value = response.json();
    assert_eq!(spec["openapi"], "3.0.0");
    assert!(spec["info"]["title"].is_string());
    assert!(spec["paths"].is_object());
}

#[tokio::test]
async fn test_rate_limiting() {
    let (app, state) = create_test_app().await;

    // Create an API key for testing
    let api_key = MockAuth::create_test_api_key(&state, "rate-limit-test").await;

    // Make requests up to the limit
    for _ in 0..10 {
        let response = TestRequest::get("/api/v1/health")
            .api_key(&api_key.key)
            .send(app.clone())
            .await;
        
        response.assert_ok();
    }

    // The next request should be rate limited
    let response = TestRequest::get("/api/v1/health")
        .api_key(&api_key.key)
        .send(app)
        .await;

    assert_eq!(response.status, axum::http::StatusCode::TOO_MANY_REQUESTS);
}

#[tokio::test]
async fn test_large_payload_rejection() {
    let (app, _state) = create_test_app().await;

    // Create a payload larger than the configured limit
    let large_payload = "x".repeat(11 * 1024 * 1024); // 11MB

    let response = TestRequest::post("/api/v1/documents")
        .header("content-type", "text/plain")
        .header("content-length", large_payload.len().to_string())
        .send(app)
        .await;

    // Should reject before even trying to authenticate
    assert_eq!(response.status, axum::http::StatusCode::PAYLOAD_TOO_LARGE);
}

#[tokio::test]
async fn test_service_info() {
    let (app, _state) = create_test_app().await;

    let response = TestRequest::get("/api/v1/info")
        .send(app)
        .await;

    response.assert_ok();

    let info: serde_json::Value = response.json();
    assert!(info["service"].is_string());
    assert!(info["version"].is_string());
    assert!(info["build_time"].is_string());
}