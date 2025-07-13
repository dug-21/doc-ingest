//! # Neural Document Flow Processors
//!
//! This crate provides processing pipeline implementations for the Neural Document Flow system.
//! It includes built-in processing stages and pipeline managers for document enhancement.
//!
//! ## Features
//!
//! - **Basic Pipeline**: Simple processing pipeline for Phase 1
//! - **Processing Stages**: Modular stages for different enhancement types
//! - **Pipeline Manager**: Orchestrate complex processing workflows
//! - **Extensible**: Easy to add new processing stages
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use neural_doc_flow_processors::{BasicPipeline, PipelineConfig};
//! use neural_doc_flow_core::{ExtractedDocument, DocumentMetadata, DocumentStructure, 
//!                           ExtractionMetrics, ContentBlock};
//! use std::collections::HashMap;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create a basic processing pipeline
//!     let mut pipeline = BasicPipeline::new();
//!     
//!     // Initialize with configuration
//!     let config = PipelineConfig {
//!         name: "basic_enhancement".to_string(),
//!         stages: vec![],
//!         settings: HashMap::new(),
//!         limits: Default::default(),
//!     };
//!     
//!     pipeline.initialize(config).await?;
//!
//!     // Create a document to process
//!     let document = ExtractedDocument {
//!         id: "test".to_string(),
//!         source_id: "pdf".to_string(),
//!         metadata: DocumentMetadata {
//!             title: Some("Test Document".to_string()),
//!             author: None,
//!             created_date: None,
//!             modified_date: None,
//!             page_count: 1,
//!             language: Some("en".to_string()),
//!             keywords: vec![],
//!             custom_metadata: HashMap::new(),
//!         },
//!         content: vec![],
//!         structure: DocumentStructure {
//!             sections: vec![],
//!             hierarchy: vec![],
//!             table_of_contents: vec![],
//!         },
//!         confidence: 0.8,
//!         metrics: ExtractionMetrics {
//!             extraction_time: std::time::Duration::from_millis(100),
//!             pages_processed: 1,
//!             blocks_extracted: 0,
//!             memory_used: 1024,
//!         },
//!     };
//!
//!     // Process the document
//!     let result = pipeline.process(document).await?;
//!     println!("Enhanced document with confidence: {}", result.document.confidence);
//!
//!     Ok(())
//! }
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

// Re-export core types for convenience
pub use neural_doc_flow_core::*;

// Pipeline implementations
pub mod pipeline;

// Processing stages
pub mod stages;

// Re-export main types
pub use pipeline::BasicPipeline;