//! Neural Document Flow REST API
//! 
//! A production-ready REST API server for neural document processing with comprehensive
//! security, authentication, rate limiting, and monitoring capabilities.

pub mod auth;
pub mod config;
pub mod error;
pub mod handlers;
#[cfg(any(feature = "auth", feature = "metrics", feature = "rate-limiting"))]
pub mod middleware;
pub mod models;
pub mod routes;
#[cfg(feature = "security")]
pub mod security;
pub mod state;
pub mod utils;
#[cfg(feature = "background-jobs")]
pub mod jobs;
#[cfg(feature = "metrics")]
pub mod monitoring;

use axum::{
    extract::DefaultBodyLimit,
    middleware as axum_middleware,
    Router,
};
use std::sync::Arc;
use tower::ServiceBuilder;
use tower_http::{
    cors::CorsLayer,
    trace::TraceLayer,
    timeout::TimeoutLayer,
    compression::CompressionLayer,
};
#[cfg(feature = "docs")]
use utoipa::OpenApi;
#[cfg(feature = "docs")]
use utoipa_swagger_ui::SwaggerUi;

pub use config::ServerConfig;
pub use error::{ApiError, ApiResult};
pub use state::AppState;

/// OpenAPI documentation
#[cfg(feature = "docs")]
#[derive(OpenApi)]
#[openapi(
    paths(
        handlers::process::process_document,
        handlers::process::process_batch,
        handlers::status::get_status,
        handlers::result::get_result,
        handlers::health::health_check,
        handlers::auth::login,
        handlers::auth::register,
    ),
    components(
        schemas(
            models::ProcessRequest,
            models::ProcessResponse,
            models::BatchRequest,
            models::BatchResponse,
            models::StatusResponse,
            models::ResultResponse,
            models::HealthResponse,
            models::LoginRequest,
            models::LoginResponse,
            models::RegisterRequest,
            models::ErrorResponse,
        )
    ),
    tags(
        (name = "Processing", description = "Document processing endpoints"),
        (name = "Status", description = "Job status and result retrieval"),
        (name = "Health", description = "Health check and system status"),
        (name = "Authentication", description = "User authentication and authorization"),
    ),
    info(
        title = "Neural Document Flow API",
        version = "1.0.0",
        description = "REST API for neural-enhanced document processing",
        contact(
            name = "API Support",
            email = "support@neuraldocflow.com"
        ),
        license(
            name = "MIT",
            url = "https://opensource.org/licenses/MIT"
        )
    ),
    servers(
        (url = "http://localhost:8080", description = "Local development server"),
        (url = "https://api.neuraldocflow.com", description = "Production server")
    )
)]
pub struct ApiDoc;

/// Create the main application router
pub fn create_app(state: Arc<AppState>) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(tower_http::cors::Any)
        .allow_methods([
            http::Method::GET,
            http::Method::POST,
            http::Method::PUT,
            http::Method::DELETE,
            http::Method::OPTIONS,
        ])
        .allow_headers(tower_http::cors::Any);

    let mut middleware_stack = ServiceBuilder::new()
        .layer(TraceLayer::new_for_http())
        .layer(TimeoutLayer::new(std::time::Duration::from_secs(300))) // 5 minute timeout
        .layer(cors);

    #[cfg(feature = "compression")]
    let middleware_stack = middleware_stack.layer(CompressionLayer::new());

    #[cfg(feature = "rate-limiting")]
    let middleware_stack = middleware_stack.layer(axum_middleware::from_fn_with_state(
        state.clone(),
        middleware::rate_limit::rate_limit_middleware,
    ));

    #[cfg(feature = "auth")]
    let middleware_stack = middleware_stack.layer(axum_middleware::from_fn_with_state(
        state.clone(),
        middleware::auth::auth_middleware,
    ));

    #[cfg(any(feature = "auth", feature = "metrics", feature = "rate-limiting"))]
    let middleware_stack = middleware_stack
        .layer(axum_middleware::from_fn(middleware::logging::logging_middleware));

    #[cfg(feature = "metrics")]
    let middleware_stack = middleware_stack
        .layer(axum_middleware::from_fn(middleware::metrics::metrics_middleware));

    let mut app = Router::new()
        // API routes
        .nest("/api/v1", routes::api_routes())
        
        // Health check (no auth required)
        .route("/health", axum::routing::get(handlers::health::health_check))
        .route("/ready", axum::routing::get(handlers::health::readiness_check));

    #[cfg(feature = "metrics")]
    {
        app = app.route("/metrics", axum::routing::get(handlers::metrics::metrics_handler));
    }

    #[cfg(feature = "docs")]
    {
        app = app
            // OpenAPI documentation
            .merge(SwaggerUi::new("/docs")
                .url("/api-docs/openapi.json", ApiDoc::openapi()))
            .route("/openapi.json", axum::routing::get(handlers::openapi::openapi_spec));
    }

    app
        
        // Set body size limit to 100MB for large documents
        .layer(DefaultBodyLimit::max(100 * 1024 * 1024))
        .layer(middleware_stack)
        .with_state(state)
}

/// Start the API server
pub async fn start_server(config: ServerConfig) -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .json()
        .init();

    // Create application state
    let state = Arc::new(AppState::new(config.clone()).await?);

    // Start background job processor
    #[cfg(feature = "background-jobs")]
    jobs::start_job_processor(state.clone()).await;

    // Start metrics collector
    #[cfg(feature = "metrics")]
    monitoring::start_metrics_collector(state.clone()).await;

    // Create the application
    let app = create_app(state);

    // Start the server
    let listener = tokio::net::TcpListener::bind(&format!("{}:{}", config.host, config.port)).await?;
    
    tracing::info!(
        "🚀 Neural Document Flow API server starting on {}:{}",
        config.host, config.port
    );
    
    tracing::info!("📖 API Documentation available at: http://{}:{}/docs", config.host, config.port);
    tracing::info!("🔍 OpenAPI spec available at: http://{}:{}/openapi.json", config.host, config.port);
    tracing::info!("💚 Health check available at: http://{}:{}/health", config.host, config.port);

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

/// Graceful shutdown signal handler
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            tracing::info!("Received Ctrl+C, starting graceful shutdown");
        },
        _ = terminate => {
            tracing::info!("Received SIGTERM, starting graceful shutdown");
        },
    }
}