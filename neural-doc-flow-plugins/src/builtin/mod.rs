//! Built-in plugins for Neural Document Flow
//!
//! This module contains core plugins that are shipped with the system:
//! - DOCX Parser: Extract text, images, tables from Word documents
//! - Table Detection: Neural-based table boundary detection and extraction
//! - Image Processing: Extract and process images from documents

pub mod docx_parser;
pub mod table_detection;
pub mod image_processing;

#[cfg(test)]
mod tests;

use neural_doc_flow_core::ProcessingError;
use crate::{PluginManager, PluginConfig};

/// Register all built-in plugins with the plugin manager
pub async fn register_builtin_plugins(manager: &mut PluginManager) -> Result<(), ProcessingError> {
    // For built-in plugins, we can register them directly without loading from files
    // This provides better performance and security since they're compiled into the binary
    
    tracing::info!("Registering built-in plugins");
    
    // Note: In a full implementation, we would register the plugins here
    // For now, we'll focus on the plugin implementations themselves
    
    Ok(())
}

/// List all available built-in plugins
pub fn list_builtin_plugins() -> Vec<&'static str> {
    vec![
        "docx_parser",
        "table_detection", 
        "image_processing",
    ]
}