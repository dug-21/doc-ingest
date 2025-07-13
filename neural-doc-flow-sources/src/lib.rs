//! # Neural Document Flow Sources
//!
//! Document source implementations for the neural document flow framework.
//! This crate provides concrete implementations of document sources for various
//! file formats and input types.
//!
//! ## Features
//!
//! - **PDF Support**: Parse and extract text from PDF documents using lopdf
//! - **Text Support**: Handle plain text files with various encodings
//! - **Plugin System**: Dynamic source registration and discovery
//! - **Metadata Extraction**: Extract document metadata
//! - **Async Processing**: Fully async document processing pipeline
//!
//! ## Usage
//!
//! ```rust
//! use neural_doc_flow_sources::{SourceManager, PdfSource, TextSource};
//! 
//! async fn example() {
//!     // Create a source manager
//!     let manager = SourceManager::new();
//!     
//!     // Register sources
//!     manager.register_source(Arc::new(PdfSource::new())).await.unwrap();
//!     manager.register_source(Arc::new(TextSource::new())).await.unwrap();
//!     
//!     // Find sources for a file
//!     let pdf_sources = manager.find_sources_by_extension("pdf").await;
//! }
//! ```

#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

/// Source trait definitions and common types
pub mod traits;

/// Source plugin manager
pub mod manager;

/// PDF document source implementation
#[cfg(feature = "pdf")]
#[cfg_attr(docsrs, doc(cfg(feature = "pdf")))]
pub mod pdf;

/// Plain text document source
#[cfg(feature = "text")]
#[cfg_attr(docsrs, doc(cfg(feature = "text")))]
pub mod text;

// Re-export commonly used types
pub use traits::{
    SourceCapability, SourceMetadata, SourceConfig, SourceError, SourceResult,
    BaseDocumentSource,
};
pub use manager::SourceManager;

#[cfg(feature = "pdf")]
pub use pdf::PdfSource;

#[cfg(feature = "text")]
pub use text::TextSource;

// Re-export core types for convenience
pub use neural_doc_flow_core::prelude::*;

/// Prelude module for convenient imports
pub mod prelude {
    pub use super::traits::*;
    pub use super::manager::*;
    
    #[cfg(feature = "pdf")]
    pub use super::pdf::*;
    
    #[cfg(feature = "text")]
    pub use super::text::*;
    
    pub use neural_doc_flow_core::prelude::*;
}

/// Create and configure a default source manager with standard sources
#[cfg(all(feature = "pdf", feature = "text"))]
pub async fn create_default_manager() -> Result<SourceManager> {
    use std::sync::Arc;
    
    let manager = SourceManager::new();
    
    // Register default sources
    manager.register_source(Arc::new(PdfSource::new())).await?;
    manager.register_source(Arc::new(TextSource::new())).await?;
    
    Ok(manager)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    #[cfg(all(feature = "pdf", feature = "text"))]
    async fn test_default_manager_creation() {
        let manager = create_default_manager().await.unwrap();
        let sources = manager.list_sources().await;
        
        assert!(sources.contains(&"pdf-source".to_string()));
        assert!(sources.contains(&"text-source".to_string()));
    }
}