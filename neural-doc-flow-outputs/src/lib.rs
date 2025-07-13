//! Neural Document Flow Outputs
//! 
//! Output generation and formatting for neural document flow system.
//! This crate provides traits and implementations for various output formats.

use neural_doc_flow_core::{Document, Result};
use async_trait::async_trait;
use std::path::Path;

pub mod json;
pub mod markdown;
pub mod html;
pub mod pdf;
pub mod xml;

pub use json::JsonOutput;
pub use markdown::MarkdownOutput;
pub use html::HtmlOutput;
pub use pdf::PdfOutput;
pub use xml::XmlOutput;

/// Trait for document output generators
#[async_trait]
pub trait DocumentOutput: Send + Sync {
    /// Name of the output format
    fn name(&self) -> &str;
    
    /// File extensions supported by this output format
    fn supported_extensions(&self) -> &[&str];
    
    /// MIME type for this output format
    fn mime_type(&self) -> &str;
    
    /// Generate output from a document
    async fn generate(&self, document: &Document, output_path: &Path) -> Result<()>;
    
    /// Generate output as bytes (for in-memory processing)
    async fn generate_bytes(&self, document: &Document) -> Result<Vec<u8>>;
    
    /// Validate output configuration
    fn validate_config(&self) -> Result<()> {
        Ok(())
    }
}

/// Output manager that can handle multiple output formats
#[derive(Debug, Default)]
pub struct OutputManager {
    outputs: Vec<Box<dyn DocumentOutput>>,
}

impl OutputManager {
    /// Create a new output manager with default outputs
    pub fn new() -> Self {
        let mut manager = Self {
            outputs: Vec::new(),
        };
        
        // Register default outputs
        manager.register_output(Box::new(JsonOutput::new()));
        manager.register_output(Box::new(MarkdownOutput::new()));
        manager.register_output(Box::new(HtmlOutput::new()));
        manager.register_output(Box::new(PdfOutput::new()));
        manager.register_output(Box::new(XmlOutput::new()));
        
        manager
    }
    
    /// Register a new document output
    pub fn register_output(&mut self, output: Box<dyn DocumentOutput>) {
        self.outputs.push(output);
    }
    
    /// Find output by format name
    pub fn find_output(&self, format: &str) -> Option<&dyn DocumentOutput> {
        for output in &self.outputs {
            if output.name().eq_ignore_ascii_case(format) {
                return Some(output.as_ref());
            }
        }
        None
    }
    
    /// Find output by file extension
    pub fn find_output_by_extension(&self, extension: &str) -> Option<&dyn DocumentOutput> {
        for output in &self.outputs {
            if output.supported_extensions().contains(&extension) {
                return Some(output.as_ref());
            }
        }
        None
    }
    
    /// Generate output using the appropriate formatter
    pub async fn generate_output(&self, document: &Document, format: &str, output_path: &Path) -> Result<()> {
        if let Some(output) = self.find_output(format) {
            output.generate(document, output_path).await
        } else {
            Err(anyhow::anyhow!("No output formatter found for format: {}", format))
        }
    }
    
    /// Generate output as bytes
    pub async fn generate_bytes(&self, document: &Document, format: &str) -> Result<Vec<u8>> {
        if let Some(output) = self.find_output(format) {
            output.generate_bytes(document).await
        } else {
            Err(anyhow::anyhow!("No output formatter found for format: {}", format))
        }
    }
    
    /// List all available output formats
    pub fn available_formats(&self) -> Vec<&str> {
        self.outputs.iter().map(|o| o.name()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_output_manager_creation() {
        let manager = OutputManager::new();
        assert_eq!(manager.outputs.len(), 5);
    }
    
    #[test]
    fn test_find_output() {
        let manager = OutputManager::new();
        
        assert!(manager.find_output("json").is_some());
        assert!(manager.find_output("markdown").is_some());
        assert!(manager.find_output("html").is_some());
        assert!(manager.find_output("pdf").is_some());
        assert!(manager.find_output("xml").is_some());
        assert!(manager.find_output("unknown").is_none());
    }
    
    #[test]
    fn test_available_formats() {
        let manager = OutputManager::new();
        let formats = manager.available_formats();
        
        assert!(formats.contains(&"json"));
        assert!(formats.contains(&"markdown"));
        assert!(formats.contains(&"html"));
        assert!(formats.contains(&"pdf"));
        assert!(formats.contains(&"xml"));
    }
}