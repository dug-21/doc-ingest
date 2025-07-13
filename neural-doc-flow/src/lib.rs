//! Neural Document Flow - Main Library Crate
//! 
//! A neural-enhanced document processing and flow management system.
//! This crate provides a high-level interface to the neural document flow system,
//! orchestrating all components for document ingestion, processing, and output generation.

pub use neural_doc_flow_core as core;
pub use neural_doc_flow_sources as sources;
pub use neural_doc_flow_processors as processors;
pub use neural_doc_flow_outputs as outputs;
pub use neural_doc_flow_coordination as coordination;

use anyhow::Result;
use std::path::Path;

/// Main entry point for the Neural Document Flow system
#[derive(Debug, Clone)]
pub struct NeuralDocumentFlow {
    config: FlowConfig,
}

/// Configuration for the Neural Document Flow system
#[derive(Debug, Clone)]
pub struct FlowConfig {
    pub coordination_enabled: bool,
    pub neural_processing: bool,
    pub max_concurrent_documents: usize,
    pub output_formats: Vec<String>,
}

impl Default for FlowConfig {
    fn default() -> Self {
        Self {
            coordination_enabled: true,
            neural_processing: true,
            max_concurrent_documents: 10,
            output_formats: vec!["json".to_string(), "markdown".to_string()],
        }
    }
}

impl NeuralDocumentFlow {
    /// Create a new Neural Document Flow instance with default configuration
    pub fn new() -> Self {
        Self {
            config: FlowConfig::default(),
        }
    }
    
    /// Create a new Neural Document Flow instance with custom configuration
    pub fn with_config(config: FlowConfig) -> Self {
        Self { config }
    }
    
    /// Process a single document
    pub async fn process_document(&self, input_path: &Path) -> Result<String> {
        // TODO: Implement document processing pipeline
        // This would orchestrate sources -> processors -> outputs
        Ok(format!("Processed: {}", input_path.display()))
    }
    
    /// Process multiple documents in parallel
    pub async fn process_documents(&self, input_paths: &[&Path]) -> Result<Vec<String>> {
        // TODO: Implement parallel document processing
        let mut results = Vec::new();
        for path in input_paths {
            results.push(self.process_document(path).await?);
        }
        Ok(results)
    }
}

impl Default for NeuralDocumentFlow {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_flow_creation() {
        let flow = NeuralDocumentFlow::new();
        assert!(flow.config.coordination_enabled);
        assert!(flow.config.neural_processing);
    }
    
    #[tokio::test]
    async fn test_document_processing() {
        let flow = NeuralDocumentFlow::new();
        let path = Path::new("test.pdf");
        let result = flow.process_document(&path).await;
        assert!(result.is_ok());
    }
}