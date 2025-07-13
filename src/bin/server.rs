use anyhow::Result;
use tracing::{info, Level};
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    info!("Starting doc-ingest server...");
    
    // TODO: Implement server functionality
    println!("Doc-ingest server is not yet implemented.");
    println!("This binary will host the REST API for document processing.");
    
    Ok(())
}