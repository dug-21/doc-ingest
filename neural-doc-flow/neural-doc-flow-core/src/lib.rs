//! # Neural Doc Flow Core
//! 
//! Core traits and types for neural-enhanced document processing with DAA coordination.
//! 
//! This crate provides the fundamental building blocks for distributed document processing:
//! - DAA-based coordination for parallel processing
//! - Neural enhancement through ruv-FANN integration
//! - Modular source plugin architecture
//! - Async processing pipeline
//! 
//! ## Quick Start
//! 
//! ```rust
//! use neural_doc_flow_core::{DocFlow, coordination::DocumentCoordinator};
//! 
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Initialize DAA coordination
//!     let coordinator = DocumentCoordinator::new().await?;
//!     
//!     // Create processing pipeline
//!     let docflow = DocFlow::with_coordinator(coordinator)?;
//!     
//!     // Process documents with DAA coordination
//!     let result = docflow.process_batch(documents).await?;
//!     
//!     Ok(())
//! }
//! ```

pub mod coordination;
pub mod types;
pub mod traits;
pub mod pipeline;
pub mod error;

// Re-export core components
pub use coordination::{DocumentCoordinator, DocumentTask, CoordinationConfig};
pub use types::{Document, ExtractedContent, ProcessingResult, Confidence};
pub use traits::{DocumentSource, DocumentProcessor, NeuralEnhancer};
pub use pipeline::{DocFlow, ProcessingPipeline};
pub use error::{CoreError, Result};

use once_cell::sync::Lazy;
use tracing::{info, debug};

/// Global initialization for the neural doc flow system
static INIT: Lazy<()> = Lazy::new(|| {
    // Initialize tracing for coordination logging
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "neural_doc_flow_core=info");
    }
    
    let subscriber = tracing_subscriber::fmt()
        .with_target(false)
        .compact()
        .finish();
    
    let _ = tracing::subscriber::set_global_default(subscriber);
    
    info!("Neural Doc Flow Core initialized with DAA coordination");
    debug!("DAA features: coordination, neural integration, memory persistence");
});

/// Initialize the neural doc flow system
/// 
/// This function should be called once at the start of your application.
/// It sets up logging and other global state required for DAA coordination.
pub fn init() {
    Lazy::force(&INIT);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() {
        init();
        // Should not panic
    }
}