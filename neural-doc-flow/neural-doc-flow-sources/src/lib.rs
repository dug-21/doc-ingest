//! # Neural Document Flow Sources
//!
//! This crate provides document source plugins for the Neural Document Flow system.
//! It includes built-in support for common document formats and a plugin manager
//! for discovering and managing source plugins.
//!
//! ## Features
//!
//! - **PDF Support**: Extract content from PDF documents using `lopdf` and `pdf-extract`
//! - **Plugin Manager**: Discover and manage document source plugins
//! - **Extensible**: Easy to add new document format support
//! - **Async**: All operations are async-first for better performance
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use neural_doc_flow_sources::{SourcePluginManager, ManagerConfig};
//! use neural_doc_flow_core::{SourceInput, DocumentSource};
//! use std::path::PathBuf;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create plugin manager
//!     let config = ManagerConfig {
//!         plugin_directories: vec![PathBuf::from("./plugins")],
//!         config_path: PathBuf::from("./config.yaml"),
//!     };
//!     let manager = SourcePluginManager::new(config)?;
//!
//!     // Find compatible source for a PDF file
//!     let input = SourceInput::File {
//!         path: PathBuf::from("document.pdf"),
//!         metadata: None,
//!     };
//!     
//!     let sources = manager.find_compatible_sources(&input).await?;
//!     if let Some(source) = sources.first() {
//!         // Extract content
//!         let document = source.extract(input).await?;
//!         println!("Extracted {} blocks from document", document.content.len());
//!     }
//!
//!     Ok(())
//! }
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

// Re-export core types for convenience
pub use neural_doc_flow_core::*;

// Built-in source implementations
#[cfg(feature = "pdf")]
pub mod pdf;

// Plugin management
pub mod manager;

// Re-export main types
pub use manager::{SourcePluginManager, ManagerConfig};

#[cfg(feature = "pdf")]
pub use pdf::PdfSource;