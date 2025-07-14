use axum::{
    body::Body,
    http::{header, Method, Request, StatusCode},
    Router,
};
use neural_doc_flow_api::{
    auth::{ApiKey, Claims, JwtAuth},
    config::Config,
    handlers,
    models::{ApiKeyResponse, AuthResponse, LoginRequest},
    state::AppState,
};
use serde::de::DeserializeOwned;
use sqlx::SqlitePool;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower::ServiceExt;
use uuid::Uuid;

/// Creates a test application with mock state
pub async fn create_test_app() -> (Router, Arc<AppState>) {
    // Create in-memory SQLite database
    let pool = SqlitePool::connect(":memory:")
        .await
        .expect("Failed to create test database");

    // Run migrations
    sqlx::migrate!("./migrations")
        .run(&pool)
        .await
        .expect("Failed to run migrations");

    // Create test config
    let config = Config {
        server_address: "127.0.0.1:0".to_string(),
        database_url: ":memory:".to_string(),
        jwt_secret: "test-secret-key-for-testing-only".to_string(),
        rate_limit_requests_per_second: 10,
        enable_metrics: true,
        enable_openapi: true,
        log_level: "debug".to_string(),
        max_upload_size: 10 * 1024 * 1024, // 10MB
        processing_timeout: 300,
        cors_allowed_origins: vec!["*".to_string()],
        retention_days: 30,
    };

    // Create app state
    let state = Arc::new(AppState::new(config, pool).await);

    // Build router
    let app = handlers::routes(state.clone());

    (app, state)
}

/// Creates a test request builder
pub struct TestRequest {
    method: Method,
    uri: String,
    headers: Vec<(String, String)>,
    body: Option<String>,
}

impl TestRequest {
    pub fn get(uri: impl Into<String>) -> Self {
        Self {
            method: Method::GET,
            uri: uri.into(),
            headers: vec![],
            body: None,
        }
    }

    pub fn post(uri: impl Into<String>) -> Self {
        Self {
            method: Method::POST,
            uri: uri.into(),
            headers: vec![],
            body: None,
        }
    }

    pub fn put(uri: impl Into<String>) -> Self {
        Self {
            method: Method::PUT,
            uri: uri.into(),
            headers: vec![],
            body: None,
        }
    }

    pub fn delete(uri: impl Into<String>) -> Self {
        Self {
            method: Method::DELETE,
            uri: uri.into(),
            headers: vec![],
            body: None,
        }
    }

    pub fn header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((key.into(), value.into()));
        self
    }

    pub fn json<T: serde::Serialize>(mut self, body: &T) -> Self {
        self.body = Some(serde_json::to_string(body).unwrap());
        self.headers.push((
            header::CONTENT_TYPE.to_string(),
            "application/json".to_string(),
        ));
        self
    }

    pub fn bearer_auth(self, token: &str) -> Self {
        self.header(header::AUTHORIZATION, format!("Bearer {}", token))
    }

    pub fn api_key(self, key: &str) -> Self {
        self.header("X-API-Key", key)
    }

    pub async fn send(self, app: Router) -> TestResponse {
        let mut request = Request::builder()
            .method(self.method)
            .uri(self.uri);

        for (key, value) in self.headers {
            request = request.header(key, value);
        }

        let request = if let Some(body) = self.body {
            request.body(Body::from(body)).unwrap()
        } else {
            request.body(Body::empty()).unwrap()
        };

        let response = app.oneshot(request).await.unwrap();

        TestResponse::from_response(response).await
    }
}

/// Test response wrapper with assertion helpers
pub struct TestResponse {
    pub status: StatusCode,
    pub headers: axum::http::HeaderMap,
    pub body: bytes::Bytes,
}

impl TestResponse {
    async fn from_response(response: axum::http::Response<Body>) -> Self {
        let (parts, body) = response.into_parts();
        let body = axum::body::to_bytes(body, usize::MAX).await.unwrap();

        Self {
            status: parts.status,
            headers: parts.headers,
            body,
        }
    }

    pub fn assert_status(&self, expected: StatusCode) -> &Self {
        assert_eq!(self.status, expected, "Unexpected status code");
        self
    }

    pub fn assert_ok(&self) -> &Self {
        self.assert_status(StatusCode::OK)
    }

    pub fn assert_created(&self) -> &Self {
        self.assert_status(StatusCode::CREATED)
    }

    pub fn assert_unauthorized(&self) -> &Self {
        self.assert_status(StatusCode::UNAUTHORIZED)
    }

    pub fn assert_bad_request(&self) -> &Self {
        self.assert_status(StatusCode::BAD_REQUEST)
    }

    pub fn assert_not_found(&self) -> &Self {
        self.assert_status(StatusCode::NOT_FOUND)
    }

    pub fn json<T: DeserializeOwned>(&self) -> T {
        serde_json::from_slice(&self.body).expect("Failed to deserialize response body")
    }

    pub fn text(&self) -> String {
        String::from_utf8(self.body.to_vec()).expect("Response body is not valid UTF-8")
    }

    pub fn assert_header(&self, key: impl AsRef<str>, expected: impl AsRef<str>) -> &Self {
        let value = self
            .headers
            .get(key.as_ref())
            .expect(&format!("Header {} not found", key.as_ref()));
        
        assert_eq!(
            value.to_str().unwrap(),
            expected.as_ref(),
            "Header {} has unexpected value",
            key.as_ref()
        );
        self
    }
}

/// Mock authentication helpers
pub struct MockAuth;

impl MockAuth {
    /// Create a test JWT token
    pub fn create_test_token(state: &AppState, user_id: &str, email: &str) -> String {
        let claims = Claims {
            sub: user_id.to_string(),
            email: email.to_string(),
            exp: (chrono::Utc::now() + chrono::Duration::hours(1)).timestamp() as usize,
            iat: chrono::Utc::now().timestamp() as usize,
        };

        JwtAuth::encode_token(&claims, &state.config.jwt_secret).unwrap()
    }

    /// Create a test API key
    pub async fn create_test_api_key(state: &AppState, name: &str) -> ApiKey {
        let key = ApiKey {
            id: Uuid::new_v4().to_string(),
            name: name.to_string(),
            key: format!("test_key_{}", Uuid::new_v4()),
            created_at: chrono::Utc::now(),
            last_used: None,
            expires_at: None,
            is_active: true,
        };

        // Insert into database
        sqlx::query!(
            r#"
            INSERT INTO api_keys (id, name, key, created_at, is_active)
            VALUES (?1, ?2, ?3, ?4, ?5)
            "#,
            key.id,
            key.name,
            key.key,
            key.created_at,
            key.is_active
        )
        .execute(&state.db)
        .await
        .unwrap();

        key
    }

    /// Perform a test login
    pub async fn login(app: Router, email: &str, password: &str) -> AuthResponse {
        let login = LoginRequest {
            email: email.to_string(),
            password: password.to_string(),
        };

        let response = TestRequest::post("/api/v1/auth/login")
            .json(&login)
            .send(app)
            .await;

        response.assert_ok();
        response.json()
    }
}

/// Test data fixtures
pub mod fixtures {
    use neural_doc_flow_api::models::{Document, ProcessingJob, ProcessingStatus};
    use uuid::Uuid;

    pub fn test_document() -> Document {
        Document {
            id: Uuid::new_v4().to_string(),
            filename: "test.pdf".to_string(),
            content_type: "application/pdf".to_string(),
            size: 1024,
            hash: "test-hash".to_string(),
            storage_path: "/test/path".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            metadata: Some(serde_json::json!({"test": true})),
        }
    }

    pub fn test_processing_job() -> ProcessingJob {
        ProcessingJob {
            id: Uuid::new_v4().to_string(),
            document_id: Uuid::new_v4().to_string(),
            status: ProcessingStatus::Pending,
            created_at: chrono::Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
            result: None,
            processor_version: "1.0.0".to_string(),
        }
    }
}

/// Database test helpers
pub mod db {
    use sqlx::SqlitePool;

    /// Clear all data from the database
    pub async fn clear_database(pool: &SqlitePool) {
        sqlx::query!("DELETE FROM processing_results").execute(pool).await.unwrap();
        sqlx::query!("DELETE FROM processing_jobs").execute(pool).await.unwrap();
        sqlx::query!("DELETE FROM documents").execute(pool).await.unwrap();
        sqlx::query!("DELETE FROM api_keys").execute(pool).await.unwrap();
    }

    /// Insert test data
    pub async fn seed_test_data(pool: &SqlitePool) {
        // Add any common test data seeding here
    }
}