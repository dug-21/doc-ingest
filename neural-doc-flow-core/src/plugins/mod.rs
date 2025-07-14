//! Core plugins for the Neural Document Flow system
//!
//! This module provides the 5 core plugins that implement the DocumentSource trait:
//! - PDF Source Plugin
//! - DOCX Source Plugin  
//! - HTML Source Plugin
//! - Image Source Plugin
//! - CSV Source Plugin

pub mod pdf_source;
pub mod docx_source;
pub mod html_source;
pub mod image_source;
pub mod csv_source;

// Re-export all plugins
pub use pdf_source::PDFSource;
pub use docx_source::DOCXSource;
pub use html_source::HTMLSource;
pub use image_source::ImageSource;
pub use csv_source::CSVSource;

use crate::traits::DocumentSource;
use std::sync::Arc;

/// Plugin registry for all core plugins
pub struct CorePluginRegistry {
    plugins: Vec<Arc<dyn DocumentSource>>,
}

impl CorePluginRegistry {
    /// Create a new plugin registry with all core plugins
    pub fn new() -> Self {
        let mut registry = Self {
            plugins: Vec::new(),
        };
        
        // Register all core plugins
        registry.plugins.push(Arc::new(PDFSource::new()));
        registry.plugins.push(Arc::new(DOCXSource::new()));
        registry.plugins.push(Arc::new(HTMLSource::new()));
        registry.plugins.push(Arc::new(ImageSource::new()));
        registry.plugins.push(Arc::new(CSVSource::new()));
        
        registry
    }
    
    /// Get all registered plugins
    pub fn get_plugins(&self) -> &[Arc<dyn DocumentSource>] {
        &self.plugins
    }
    
    /// Find plugin by source type
    pub fn find_plugin(&self, source_type: &str) -> Option<Arc<dyn DocumentSource>> {
        self.plugins.iter()
            .find(|p| p.source_type() == source_type)
            .cloned()
    }
}

impl Default for CorePluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}