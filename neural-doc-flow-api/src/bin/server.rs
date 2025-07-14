//! Neural Document Flow API Server
//! 
//! Production-ready REST API server for neural document processing

use clap::Parser;
use neural_doc_flow_api::{ServerConfig, start_server};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load configuration from command line and environment
    let config = ServerConfig::parse();

    // Initialize tracing early
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| format!("neural_doc_flow_api={}", config.monitoring.log_level).into())
        )
        .with_target(false)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .init();

    tracing::info!("🧠 Neural Document Flow API Server");
    tracing::info!("Version: {}", env!("CARGO_PKG_VERSION"));
    tracing::info!("Configuration loaded successfully");

    // Start the server
    if let Err(e) = start_server(config).await {
        tracing::error!("Server failed to start: {}", e);
        std::process::exit(1);
    }

    Ok(())
}