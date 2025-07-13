//! # Neural Document Flow Core
//!
//! Core traits, types, and functionality for the neural document processing framework.
//! This crate provides the foundational abstractions for document sources, processors,
//! neural components, and output formatters.

pub mod traits;
pub mod types;
pub mod error;
pub mod neural;
pub mod document;
pub mod result;
pub mod config;
pub mod engine;

// Re-export specific items to avoid conflicts
pub use traits::{DocumentSource, DocumentSourceFactory, ProcessorPipeline, Processor, 
                OutputFormatter, TemplateFormatter, NeuralProcessor, NeuralProcessorFactory};
pub use types::*;
pub use error::*;
pub use document::*;
pub use result::*;

// Re-export config items with specific names to avoid conflicts
pub use config::{NeuralDocFlowConfig, SystemConfig, SourcesConfig, OutputConfig, 
                MonitoringConfig, CacheConfig, ResourceLimits};

#[cfg(feature = "neural")]
pub use neural::*;

/// Re-export commonly used types
pub mod prelude {
    pub use crate::traits::{DocumentSource, DocumentSourceFactory, ProcessorPipeline, 
                           Processor, OutputFormatter, TemplateFormatter, 
                           NeuralProcessor, NeuralProcessorFactory};
    pub use crate::types::*;
    pub use crate::error::*;
    pub use crate::document::*;
    pub use crate::result::*;
    pub use crate::config::{NeuralDocFlowConfig, SystemConfig, SourcesConfig, 
                           OutputConfig, MonitoringConfig};
    
    #[cfg(feature = "neural")]
    pub use crate::neural::*;
}