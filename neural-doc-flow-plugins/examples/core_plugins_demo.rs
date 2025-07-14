//! Core Plugins Demo
//! 
//! This example demonstrates how to use the three core plugins:
//! - DOCX Parser: Extract content from Word documents
//! - Table Detection: Find and extract tables from images/PDFs  
//! - Image Processing: Process images with neural enhancement

use neural_doc_flow_plugins::{
    create_plugin_manager_with_builtins,
    builtin::{
        docx_parser::{DocxParserPlugin, DocxConfig},
        table_detection::{TableDetectionPlugin, TableDetectionConfig},
        image_processing::{ImageProcessingPlugin, ImageProcessingConfig, ImageFormat},
    },
    Plugin, PluginManager, PluginConfig,
};
use neural_doc_flow_core::{DocumentSource, Document};
use std::path::{Path, PathBuf};
use tokio;
use tracing::{info, warn, error};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    info!("Starting Core Plugins Demo");
    
    // Demo 1: DOCX Parser Plugin
    demo_docx_parser().await?;
    
    // Demo 2: Table Detection Plugin  
    demo_table_detection().await?;
    
    // Demo 3: Image Processing Plugin
    demo_image_processing().await?;
    
    // Demo 4: Plugin Manager with All Plugins
    demo_plugin_manager().await?;
    
    info!("Core Plugins Demo completed successfully");
    Ok(())
}

/// Demonstrate DOCX Parser Plugin functionality
async fn demo_docx_parser() -> Result<(), Box<dyn std::error::Error>> {
    info!("=== DOCX Parser Plugin Demo ===");
    
    // Create DOCX parser with custom configuration
    let config = DocxConfig {
        extract_images: true,
        extract_tables: true,
        preserve_formatting: true,
        max_image_size_mb: 20,
        table_detection_threshold: 0.8,
    };
    
    let mut plugin = DocxParserPlugin::with_config(config);
    
    // Initialize the plugin
    plugin.initialize()?;
    info!("DOCX Parser initialized: {}", plugin.metadata().description);
    
    // Get the document source
    let source = plugin.document_source();
    
    // Test format detection
    let test_files = vec![
        "document.docx",
        "presentation.pptx", // Should not be supported
        "spreadsheet.xlsx",  // Should not be supported
        "macro_doc.docm",
        "text_file.txt",     // Should not be supported
    ];
    
    for file in test_files {
        let path = Path::new(file);
        let can_process = source.can_process(path);
        info!("Can process {}: {}", file, can_process);
    }
    
    // Simulate processing a DOCX file (would fail with real file, but shows the interface)
    let sample_docx = Path::new("sample_document.docx");
    if source.can_process(sample_docx) {
        info!("Would process DOCX file: {:?}", sample_docx);
        // In a real scenario: let document = source.extract_document(sample_docx)?;
    }
    
    // Shutdown the plugin
    plugin.shutdown()?;
    info!("DOCX Parser demo completed\n");
    
    Ok(())
}

/// Demonstrate Table Detection Plugin functionality
async fn demo_table_detection() -> Result<(), Box<dyn std::error::Error>> {
    info!("=== Table Detection Plugin Demo ===");
    
    // Create table detection plugin with custom configuration
    let config = TableDetectionConfig {
        confidence_threshold: 0.85,
        min_table_size: (150, 75),
        max_table_size: (1800, 1400),
        cell_detection_enabled: true,
        text_extraction_enabled: true,
        border_detection_threshold: 0.7,
        grid_analysis_enabled: true,
        neural_enhancement: true,
    };
    
    let mut plugin = TableDetectionPlugin::with_config(config);
    
    // Initialize the plugin
    plugin.initialize().await?;
    info!("Table Detection initialized: {}", plugin.metadata().description);
    
    // Get the document source
    let source = plugin.document_source();
    
    // Test format detection
    let test_files = vec![
        "scanned_document.pdf",
        "table_image.png",
        "photo.jpg",
        "chart.tiff",
        "word_doc.docx",    // Should not be supported
        "plain_text.txt",   // Should not be supported
    ];
    
    for file in test_files {
        let path = Path::new(file);
        let can_process = source.can_process(path);
        info!("Can process {}: {}", file, can_process);
    }
    
    // Demonstrate table detection capabilities
    info!("Table Detection Features:");
    info!("- Confidence threshold: {}", plugin.config.confidence_threshold);
    info!("- Min table size: {:?}", plugin.config.min_table_size);
    info!("- Neural enhancement: {}", plugin.config.neural_enhancement);
    info!("- Cell detection: {}", plugin.config.cell_detection_enabled);
    info!("- Text extraction: {}", plugin.config.text_extraction_enabled);
    
    // Shutdown the plugin
    plugin.shutdown().await?;
    info!("Table Detection demo completed\n");
    
    Ok(())
}

/// Demonstrate Image Processing Plugin functionality
async fn demo_image_processing() -> Result<(), Box<dyn std::error::Error>> {
    info!("=== Image Processing Plugin Demo ===");
    
    // Create image processing plugin with custom configuration
    let config = ImageProcessingConfig {
        enhance_quality: true,
        extract_text: true,
        detect_objects: false, // Disabled for performance
        analyze_layout: true,
        max_image_size_mb: 25,
        output_format: ImageFormat::PNG,
        enhancement_strength: 0.6,
        ocr_confidence_threshold: 0.75,
        object_detection_threshold: 0.8,
    };
    
    let mut plugin = ImageProcessingPlugin::with_config(config);
    
    // Initialize the plugin
    plugin.initialize().await?;
    info!("Image Processing initialized: {}", plugin.metadata().description);
    
    // Get the document source
    let source = plugin.document_source();
    
    // Test format detection
    let test_files = vec![
        "photo.jpg",
        "graphic.png",
        "scan.tiff",
        "bitmap.bmp", 
        "animation.gif",
        "modern.webp",
        "document.pdf",     // Should be supported for image extraction
        "spreadsheet.xlsx", // Should not be supported
        "video.mp4",        // Should not be supported
    ];
    
    for file in test_files {
        let path = Path::new(file);
        let can_process = source.can_process(path);
        info!("Can process {}: {}", file, can_process);
    }
    
    // Demonstrate image processing capabilities
    info!("Image Processing Features:");
    info!("- Quality enhancement: {}", plugin.config.enhance_quality);
    info!("- Text extraction (OCR): {}", plugin.config.extract_text);
    info!("- Object detection: {}", plugin.config.detect_objects);
    info!("- Layout analysis: {}", plugin.config.analyze_layout);
    info!("- Enhancement strength: {}", plugin.config.enhancement_strength);
    info!("- OCR confidence threshold: {}", plugin.config.ocr_confidence_threshold);
    
    // Test image format detection
    test_image_format_detection(&plugin);
    
    // Shutdown the plugin
    plugin.shutdown().await?;
    info!("Image Processing demo completed\n");
    
    Ok(())
}

/// Demonstrate Plugin Manager with all core plugins
async fn demo_plugin_manager() -> Result<(), Box<dyn std::error::Error>> {
    info!("=== Plugin Manager Demo ===");
    
    // Create plugin manager with built-in plugins
    let mut manager = create_plugin_manager_with_builtins().await?;
    
    // Initialize the manager (discover and load plugins)
    manager.initialize().await?;
    info!("Plugin Manager initialized");
    
    // List all loaded plugins
    let plugins = manager.list_plugins().await;
    info!("Loaded {} plugins:", plugins.len());
    for plugin_meta in &plugins {
        info!("  - {} v{}: {}", 
              plugin_meta.name, 
              plugin_meta.version, 
              plugin_meta.description);
        info!("    Formats: {:?}", plugin_meta.supported_formats);
        info!("    Memory limit: {}MB", plugin_meta.capabilities.max_memory_mb);
    }
    
    // Test file routing to appropriate plugins
    test_file_routing(&manager).await;
    
    // Get performance metrics if available
    let metrics = manager.get_reload_metrics().await;
    info!("Hot-reload metrics:");
    info!("  - Total reloads: {}", metrics.total_reloads);
    info!("  - Successful: {}", metrics.successful_reloads);
    info!("  - Failed: {}", metrics.failed_reloads);
    if metrics.total_reloads > 0 {
        info!("  - Success rate: {:.1}%", 
              (metrics.successful_reloads as f64 / metrics.total_reloads as f64) * 100.0);
        info!("  - Average reload time: {:.1}ms", metrics.average_reload_time_ms);
    }
    
    // Shutdown the manager
    manager.shutdown().await?;
    info!("Plugin Manager demo completed\n");
    
    Ok(())
}

/// Test which plugin should handle different file types
async fn test_file_routing(manager: &PluginManager) {
    info!("Testing file routing to plugins:");
    
    let test_files = vec![
        ("document.docx", "DOCX Parser"),
        ("report.pdf", "Table Detection or Image Processing"),
        ("scan.png", "Table Detection or Image Processing"), 
        ("photo.jpg", "Image Processing"),
        ("data.csv", "None (not supported)"),
        ("presentation.pptx", "None (not supported)"),
        ("macro_doc.docm", "DOCX Parser"),
        ("chart.tiff", "Table Detection or Image Processing"),
    ];
    
    for (filename, expected) in test_files {
        let plugins = manager.list_plugins().await;
        let mut handlers = Vec::new();
        
        for plugin_meta in &plugins {
            // Create a dummy source to test (in real implementation, we'd get the actual plugin)
            let path = Path::new(filename);
            let extension = path.extension()
                .and_then(|ext| ext.to_str())
                .unwrap_or("")
                .to_lowercase();
            
            if plugin_meta.supported_formats.contains(&extension) {
                handlers.push(plugin_meta.name.clone());
            }
        }
        
        if handlers.is_empty() {
            info!("  {} -> No handlers (expected: {})", filename, expected);
        } else {
            info!("  {} -> {} (expected: {})", filename, handlers.join(", "), expected);
        }
    }
}

/// Test image format detection
fn test_image_format_detection(plugin: &ImageProcessingPlugin) {
    info!("Testing image format detection:");
    
    // Test various image format signatures
    let test_formats = vec![
        (vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A], "PNG"),
        (vec![0xFF, 0xD8, 0xFF, 0xE0], "JPEG"),
        (vec![0x47, 0x49, 0x46, 0x38], "GIF"),
        (vec![0x42, 0x4D], "BMP"),
        (vec![0x49, 0x49, 0x2A, 0x00], "TIFF (Little Endian)"),
        (vec![0x4D, 0x4D, 0x00, 0x2A], "TIFF (Big Endian)"),
        (vec![0x00, 0x00, 0x00, 0x00], "Unknown"),
    ];
    
    for (signature, format_name) in test_formats {
        let mut data = signature;
        data.resize(8, 0); // Ensure minimum 8 bytes
        let detected = plugin.detect_format(&data);
        info!("  {} signature -> detected as: {}", format_name, detected);
    }
}

/// Helper function to demonstrate plugin error handling
async fn demonstrate_error_handling() {
    info!("=== Error Handling Demo ===");
    
    // Test plugin initialization with invalid configuration
    let invalid_config = PluginConfig {
        plugin_dir: PathBuf::from("/nonexistent/directory"),
        enable_hot_reload: true,
        enable_sandboxing: true,
        max_plugins: 0, // Invalid: no plugins allowed
    };
    
    match PluginManager::new(invalid_config) {
        Ok(_) => warn!("Expected error but plugin manager created successfully"),
        Err(e) => info!("Expected error caught: {}", e),
    }
    
    // Test processing non-existent file
    let source = neural_doc_flow_plugins::builtin::docx_parser::DocxSource::new(
        neural_doc_flow_plugins::builtin::docx_parser::DocxConfig::default()
    );
    
    match source.extract_document(Path::new("/nonexistent/file.docx")) {
        Ok(_) => warn!("Expected error but document extraction succeeded"),
        Err(e) => info!("Expected error caught: {}", e),
    }
    
    info!("Error handling demo completed\n");
}

/// Performance comparison between plugins
async fn benchmark_plugins() -> Result<(), Box<dyn std::error::Error>> {
    info!("=== Plugin Performance Benchmark ===");
    
    let start = std::time::Instant::now();
    
    // Benchmark plugin initialization
    let init_start = std::time::Instant::now();
    let mut docx_plugin = DocxParserPlugin::new();
    docx_plugin.initialize()?;
    let docx_init_time = init_start.elapsed();
    
    let init_start = std::time::Instant::now();
    let mut table_plugin = TableDetectionPlugin::new();
    table_plugin.initialize().await?;
    let table_init_time = init_start.elapsed();
    
    let init_start = std::time::Instant::now();
    let mut image_plugin = ImageProcessingPlugin::new();
    image_plugin.initialize().await?;
    let image_init_time = init_start.elapsed();
    
    info!("Plugin initialization times:");
    info!("  - DOCX Parser: {:?}", docx_init_time);
    info!("  - Table Detection: {:?}", table_init_time);
    info!("  - Image Processing: {:?}", image_init_time);
    
    // Benchmark document source creation
    let source_start = std::time::Instant::now();
    let _docx_source = docx_plugin.document_source();
    let _table_source = table_plugin.document_source();
    let _image_source = image_plugin.document_source();
    let source_time = source_start.elapsed();
    
    info!("Document source creation time: {:?}", source_time);
    
    // Benchmark shutdown
    let shutdown_start = std::time::Instant::now();
    docx_plugin.shutdown()?;
    table_plugin.shutdown().await?;
    image_plugin.shutdown().await?;
    let shutdown_time = shutdown_start.elapsed();
    
    info!("Plugin shutdown time: {:?}", shutdown_time);
    
    let total_time = start.elapsed();
    info!("Total benchmark time: {:?}", total_time);
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_all_plugins_can_initialize() {
        let mut docx_plugin = DocxParserPlugin::new();
        assert!(docx_plugin.initialize().is_ok());
        assert!(docx_plugin.shutdown().is_ok());
        
        let mut table_plugin = TableDetectionPlugin::new();
        assert!(table_plugin.initialize().await.is_ok());
        assert!(table_plugin.shutdown().await.is_ok());
        
        let mut image_plugin = ImageProcessingPlugin::new();
        assert!(image_plugin.initialize().await.is_ok());
        assert!(image_plugin.shutdown().await.is_ok());
    }
    
    #[tokio::test]
    async fn test_plugin_manager_creation() {
        let result = create_plugin_manager_with_builtins().await;
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_plugin_metadata_validity() {
        let docx_plugin = DocxParserPlugin::new();
        let table_plugin = TableDetectionPlugin::new();
        let image_plugin = ImageProcessingPlugin::new();
        
        // All plugins should have non-empty metadata
        assert!(!docx_plugin.metadata().name.is_empty());
        assert!(!table_plugin.metadata().name.is_empty());
        assert!(!image_plugin.metadata().name.is_empty());
        
        assert!(!docx_plugin.metadata().description.is_empty());
        assert!(!table_plugin.metadata().description.is_empty());
        assert!(!image_plugin.metadata().description.is_empty());
        
        assert!(!docx_plugin.metadata().supported_formats.is_empty());
        assert!(!table_plugin.metadata().supported_formats.is_empty());
        assert!(!image_plugin.metadata().supported_formats.is_empty());
    }
}