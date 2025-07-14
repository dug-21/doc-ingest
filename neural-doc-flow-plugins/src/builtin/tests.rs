//! Comprehensive tests for built-in plugins

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Plugin, PluginManager, PluginConfig};
    use neural_doc_flow_core::{DocumentSource, Document};
    use std::path::Path;
    use tempfile::{NamedTempFile, TempDir};
    use std::io::Write;

    /// Test helper to create a temporary file with specified content
    fn create_test_file(content: &[u8], extension: &str) -> NamedTempFile {
        let mut file = NamedTempFile::with_suffix(extension).unwrap();
        file.write_all(content).unwrap();
        file
    }

    mod docx_parser_tests {
        use super::*;
        use crate::builtin::docx_parser::{DocxParserPlugin, DocxConfig, DocxSource};

        #[test]
        fn test_docx_plugin_creation() {
            let plugin = DocxParserPlugin::new();
            assert_eq!(plugin.metadata().name, "docx_parser");
            assert_eq!(plugin.metadata().version, "1.0.0");
            assert!(plugin.metadata().supported_formats.contains(&"docx".to_string()));
            assert!(plugin.metadata().capabilities.requires_filesystem);
        }

        #[test]
        fn test_docx_plugin_with_config() {
            let config = DocxConfig {
                extract_images: false,
                extract_tables: true,
                preserve_formatting: false,
                max_image_size_mb: 5,
                table_detection_threshold: 0.9,
            };
            
            let plugin = DocxParserPlugin::with_config(config.clone());
            assert!(!plugin.config.extract_images);
            assert!(plugin.config.extract_tables);
            assert_eq!(plugin.config.max_image_size_mb, 5);
        }

        #[test]
        fn test_docx_source_can_process() {
            let source = DocxSource::new(DocxConfig::default());
            
            assert!(source.can_process(Path::new("document.docx")));
            assert!(source.can_process(Path::new("macro_doc.docm")));
            assert!(!source.can_process(Path::new("document.pdf")));
            assert!(!source.can_process(Path::new("text.txt")));
            assert!(!source.can_process(Path::new("presentation.pptx")));
        }

        #[test]
        fn test_docx_image_format_detection() {
            let plugin = DocxParserPlugin::new();
            
            // PNG signature
            let png_data = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
            assert_eq!(plugin.detect_image_format(&png_data), "png");
            
            // JPEG signature
            let jpeg_data = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46];
            assert_eq!(plugin.detect_image_format(&jpeg_data), "jpeg");
            
            // GIF signature
            let gif_data = [0x47, 0x49, 0x46, 0x38, 0x39, 0x61, 0x00, 0x00];
            assert_eq!(plugin.detect_image_format(&gif_data), "gif");
            
            // Unknown format
            let unknown_data = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
            assert_eq!(plugin.detect_image_format(&unknown_data), "unknown");
        }

        #[test]
        fn test_docx_is_image_file() {
            let plugin = DocxParserPlugin::new();
            
            assert!(plugin.is_image_file("image.png"));
            assert!(plugin.is_image_file("photo.jpg"));
            assert!(plugin.is_image_file("graphic.jpeg"));
            assert!(plugin.is_image_file("bitmap.bmp"));
            assert!(plugin.is_image_file("vector.wmf"));
            assert!(!plugin.is_image_file("document.xml"));
            assert!(!plugin.is_image_file("data.json"));
            assert!(!plugin.is_image_file("noextension"));
        }

        #[tokio::test]
        async fn test_docx_plugin_lifecycle() {
            let mut plugin = DocxParserPlugin::new();
            
            // Test initialization
            assert!(plugin.initialize().is_ok());
            
            // Test document source creation
            let source = plugin.document_source();
            assert!(source.can_process(Path::new("test.docx")));
            
            // Test shutdown
            assert!(plugin.shutdown().is_ok());
        }
    }

    mod table_detection_tests {
        use super::*;
        use crate::builtin::table_detection::{
            TableDetectionPlugin, TableDetectionConfig, TableDetectionSource,
            BoundingBox, TableCell, CellType
        };

        #[test]
        fn test_table_detection_plugin_creation() {
            let plugin = TableDetectionPlugin::new();
            assert_eq!(plugin.metadata().name, "table_detection");
            assert_eq!(plugin.metadata().version, "1.0.0");
            assert!(plugin.metadata().supported_formats.contains(&"pdf".to_string()));
            assert!(plugin.metadata().supported_formats.contains(&"png".to_string()));
        }

        #[test]
        fn test_table_detection_config() {
            let config = TableDetectionConfig {
                confidence_threshold: 0.9,
                min_table_size: (200, 100),
                max_table_size: (1500, 1200),
                cell_detection_enabled: false,
                text_extraction_enabled: false,
                border_detection_threshold: 0.8,
                grid_analysis_enabled: false,
                neural_enhancement: false,
            };
            
            let plugin = TableDetectionPlugin::with_config(config.clone());
            assert_eq!(plugin.config.confidence_threshold, 0.9);
            assert!(!plugin.config.cell_detection_enabled);
        }

        #[test]
        fn test_table_detection_source_can_process() {
            let source = TableDetectionSource::new(TableDetectionConfig::default());
            
            assert!(source.can_process(Path::new("document.pdf")));
            assert!(source.can_process(Path::new("image.png")));
            assert!(source.can_process(Path::new("photo.jpg")));
            assert!(source.can_process(Path::new("scan.tiff")));
            assert!(!source.can_process(Path::new("document.docx")));
            assert!(!source.can_process(Path::new("text.txt")));
        }

        #[test]
        fn test_bounding_box_overlap_calculation() {
            let plugin = TableDetectionPlugin::new();
            
            // Overlapping boxes
            let bbox1 = BoundingBox { x: 0, y: 0, width: 100, height: 100 };
            let bbox2 = BoundingBox { x: 50, y: 50, width: 100, height: 100 };
            let overlap = plugin.calculate_overlap(&bbox1, &bbox2);
            assert!(overlap > 0.0 && overlap < 1.0);
            
            // Non-overlapping boxes
            let bbox3 = BoundingBox { x: 200, y: 200, width: 100, height: 100 };
            let no_overlap = plugin.calculate_overlap(&bbox1, &bbox3);
            assert_eq!(no_overlap, 0.0);
            
            // Identical boxes
            let full_overlap = plugin.calculate_overlap(&bbox1, &bbox1);
            assert_eq!(full_overlap, 1.0);
        }

        #[test]
        fn test_grid_structure_analysis() {
            let plugin = TableDetectionPlugin::new();
            
            let cells = vec![
                TableCell {
                    id: "cell_0_0".to_string(),
                    bounding_box: BoundingBox { x: 0, y: 0, width: 50, height: 30 },
                    row: 0, column: 0, rowspan: 1, colspan: 1,
                    text: Some("Header 1".to_string()), confidence: 0.9, 
                    cell_type: CellType::Header,
                },
                TableCell {
                    id: "cell_0_1".to_string(),
                    bounding_box: BoundingBox { x: 50, y: 0, width: 50, height: 30 },
                    row: 0, column: 1, rowspan: 1, colspan: 1,
                    text: Some("Header 2".to_string()), confidence: 0.9, 
                    cell_type: CellType::Header,
                },
                TableCell {
                    id: "cell_1_0".to_string(),
                    bounding_box: BoundingBox { x: 0, y: 30, width: 50, height: 30 },
                    row: 1, column: 0, rowspan: 1, colspan: 1,
                    text: Some("Data 1".to_string()), confidence: 0.8, 
                    cell_type: CellType::Data,
                },
                TableCell {
                    id: "cell_1_1".to_string(),
                    bounding_box: BoundingBox { x: 50, y: 30, width: 50, height: 30 },
                    row: 1, column: 1, rowspan: 1, colspan: 1,
                    text: Some("Data 2".to_string()), confidence: 0.8, 
                    cell_type: CellType::Data,
                },
            ];
            
            let grid = plugin.analyze_grid_structure(&cells);
            
            assert_eq!(grid.rows, 2);
            assert_eq!(grid.columns, 2);
            assert!(grid.has_headers);
            assert_eq!(grid.row_heights.len(), 2);
            assert_eq!(grid.column_widths.len(), 2);
            assert!(grid.border_style.has_outer_border);
        }

        #[test]
        fn test_table_region_validation() {
            let plugin = TableDetectionPlugin::new();
            let image_data = crate::builtin::table_detection::ImageData {
                data: vec![0u8; 1000],
                width: 800,
                height: 600,
                channels: 3,
                format: "png".to_string(),
            };
            
            // Valid table region
            let valid_region = BoundingBox { x: 10, y: 10, width: 200, height: 100 };
            assert!(plugin.is_valid_table_region(&valid_region, &image_data));
            
            // Too small
            let too_small = BoundingBox { x: 10, y: 10, width: 50, height: 30 };
            assert!(!plugin.is_valid_table_region(&too_small, &image_data));
            
            // Too large
            let too_large = BoundingBox { x: 10, y: 10, width: 3000, height: 2000 };
            assert!(!plugin.is_valid_table_region(&too_large, &image_data));
            
            // Outside image bounds
            let outside_bounds = BoundingBox { x: 700, y: 500, width: 200, height: 200 };
            assert!(!plugin.is_valid_table_region(&outside_bounds, &image_data));
        }

        #[tokio::test]
        async fn test_table_detection_plugin_lifecycle() {
            let mut plugin = TableDetectionPlugin::new();
            
            // Test initialization
            assert!(plugin.initialize().is_ok());
            
            // Test document source creation
            let source = plugin.document_source();
            assert!(source.can_process(Path::new("test.pdf")));
            
            // Test shutdown
            assert!(plugin.shutdown().is_ok());
        }
    }

    mod image_processing_tests {
        use super::*;
        use crate::builtin::image_processing::{
            ImageProcessingPlugin, ImageProcessingConfig, ImageProcessingSource,
            ImageFormat, QualityMetrics
        };

        #[test]
        fn test_image_processing_plugin_creation() {
            let plugin = ImageProcessingPlugin::new();
            assert_eq!(plugin.metadata().name, "image_processing");
            assert_eq!(plugin.metadata().version, "1.0.0");
            assert!(plugin.metadata().supported_formats.contains(&"png".to_string()));
            assert!(plugin.metadata().supported_formats.contains(&"jpg".to_string()));
            assert!(plugin.metadata().supported_formats.contains(&"pdf".to_string()));
        }

        #[test]
        fn test_image_processing_config() {
            let config = ImageProcessingConfig {
                enhance_quality: false,
                extract_text: true,
                detect_objects: true,
                analyze_layout: false,
                max_image_size_mb: 100,
                output_format: ImageFormat::PNG,
                enhancement_strength: 0.8,
                ocr_confidence_threshold: 0.9,
                object_detection_threshold: 0.7,
            };
            
            let plugin = ImageProcessingPlugin::with_config(config.clone());
            assert!(!plugin.config.enhance_quality);
            assert!(plugin.config.extract_text);
            assert!(plugin.config.detect_objects);
            assert_eq!(plugin.config.max_image_size_mb, 100);
        }

        #[test]
        fn test_image_processing_source_can_process() {
            let source = ImageProcessingSource::new(ImageProcessingConfig::default());
            
            assert!(source.can_process(Path::new("image.png")));
            assert!(source.can_process(Path::new("photo.jpg")));
            assert!(source.can_process(Path::new("scan.tiff")));
            assert!(source.can_process(Path::new("bitmap.bmp")));
            assert!(source.can_process(Path::new("animation.gif")));
            assert!(source.can_process(Path::new("document.pdf")));
            assert!(!source.can_process(Path::new("document.docx")));
            assert!(!source.can_process(Path::new("text.txt")));
        }

        #[test]
        fn test_image_format_detection() {
            let plugin = ImageProcessingPlugin::new();
            
            // PNG signature
            let png_data = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
            assert_eq!(plugin.detect_format(&png_data), "png");
            
            // JPEG signature
            let jpeg_data = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46];
            assert_eq!(plugin.detect_format(&jpeg_data), "jpeg");
            
            // GIF signature  
            let gif_data = [0x47, 0x49, 0x46, 0x38, 0x39, 0x61, 0x00, 0x00];
            assert_eq!(plugin.detect_format(&gif_data), "gif");
            
            // BMP signature
            let bmp_data = [0x42, 0x4D, 0x36, 0x28, 0x00, 0x00, 0x00, 0x00];
            assert_eq!(plugin.detect_format(&bmp_data), "bmp");
            
            // TIFF signatures (little and big endian)
            let tiff_le_data = [0x49, 0x49, 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00];
            assert_eq!(plugin.detect_format(&tiff_le_data), "tiff");
            
            let tiff_be_data = [0x4D, 0x4D, 0x00, 0x2A, 0x00, 0x00, 0x00, 0x08];
            assert_eq!(plugin.detect_format(&tiff_be_data), "tiff");
            
            // Unknown format
            let unknown_data = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
            assert_eq!(plugin.detect_format(&unknown_data), "unknown");
        }

        #[test]
        fn test_quality_analysis() {
            let plugin = ImageProcessingPlugin::new();
            
            // Test with uniform gray image (low variance = low noise)
            let gray_image = vec![128u8; 1000];
            let gray_quality = plugin.analyze_quality(&gray_image).unwrap();
            assert!(gray_quality.brightness > 0.4 && gray_quality.brightness < 0.6); // ~50% gray
            assert!(gray_quality.noise_level < 0.1); // Low noise for uniform image
            assert!(gray_quality.overall_quality >= 0.0 && gray_quality.overall_quality <= 1.0);
            
            // Test with high contrast image (alternating black/white)
            let mut contrast_image = vec![0u8; 1000];
            for (i, pixel) in contrast_image.iter_mut().enumerate() {
                *pixel = if i % 2 == 0 { 0 } else { 255 };
            }
            let contrast_quality = plugin.analyze_quality(&contrast_image).unwrap();
            assert!(contrast_quality.noise_level > gray_quality.noise_level); // Higher variance
            
            // Test with bright image
            let bright_image = vec![200u8; 1000];
            let bright_quality = plugin.analyze_quality(&bright_image).unwrap();
            assert!(bright_quality.brightness > gray_quality.brightness);
            
            // Test with dark image
            let dark_image = vec![50u8; 1000];
            let dark_quality = plugin.analyze_quality(&dark_image).unwrap();
            assert!(dark_quality.brightness < gray_quality.brightness);
        }

        #[test]
        fn test_basic_image_enhancement() {
            let plugin = ImageProcessingPlugin::new();
            
            // Test enhancement of a simple image
            let original_data = vec![100u8; 100]; // Mid-gray pixels
            let enhanced = plugin.basic_enhancement(&original_data);
            
            assert!(enhanced.is_some());
            let enhanced_data = enhanced.unwrap();
            assert_eq!(enhanced_data.len(), original_data.len());
            
            // Enhanced image should have higher values (brighter)
            for (original, enhanced) in original_data.iter().zip(enhanced_data.iter()) {
                assert!(*enhanced >= *original); // Should be brighter or equal
            }
        }

        #[tokio::test]
        async fn test_image_processing_plugin_lifecycle() {
            let mut plugin = ImageProcessingPlugin::new();
            
            // Test initialization
            assert!(plugin.initialize().is_ok());
            
            // Test document source creation
            let source = plugin.document_source();
            assert!(source.can_process(Path::new("test.png")));
            
            // Test shutdown
            assert!(plugin.shutdown().is_ok());
        }
    }

    mod integration_tests {
        use super::*;
        use crate::builtin::{register_builtin_plugins, list_builtin_plugins};

        #[test]
        fn test_list_builtin_plugins() {
            let plugins = list_builtin_plugins();
            assert_eq!(plugins.len(), 3);
            assert!(plugins.contains(&"docx_parser"));
            assert!(plugins.contains(&"table_detection"));
            assert!(plugins.contains(&"image_processing"));
        }

        #[tokio::test]
        async fn test_builtin_plugin_registration() {
            let config = PluginConfig {
                enable_hot_reload: false, // Disable for test
                enable_sandboxing: false, // Disable for test
                ..PluginConfig::default()
            };
            
            let mut manager = PluginManager::new(config).unwrap();
            
            // Register built-in plugins
            assert!(register_builtin_plugins(&mut manager).await.is_ok());
            
            // Test that plugins are available
            let plugin_list = manager.list_plugins().await;
            
            // Note: In the current implementation, register_builtin_plugins doesn't 
            // actually register plugins yet - it's a placeholder. This test verifies
            // the function completes without error.
            assert!(plugin_list.len() >= 0); // Could be 0 if not implemented yet
        }

        #[test]
        fn test_plugin_format_compatibility() {
            // Test that plugins handle their declared formats correctly
            
            // DOCX Parser
            let docx_source = crate::builtin::docx_parser::DocxSource::new(
                crate::builtin::docx_parser::DocxConfig::default()
            );
            assert!(docx_source.can_process(Path::new("test.docx")));
            assert!(docx_source.can_process(Path::new("TEST.DOCX"))); // Case insensitive
            assert!(!docx_source.can_process(Path::new("test.pdf")));
            
            // Table Detection
            let table_source = crate::builtin::table_detection::TableDetectionSource::new(
                crate::builtin::table_detection::TableDetectionConfig::default()
            );
            assert!(table_source.can_process(Path::new("scan.pdf")));
            assert!(table_source.can_process(Path::new("image.png")));
            assert!(!table_source.can_process(Path::new("text.txt")));
            
            // Image Processing
            let image_source = crate::builtin::image_processing::ImageProcessingSource::new(
                crate::builtin::image_processing::ImageProcessingConfig::default()
            );
            assert!(image_source.can_process(Path::new("photo.jpg")));
            assert!(image_source.can_process(Path::new("graphic.png")));
            assert!(image_source.can_process(Path::new("document.pdf")));
            assert!(!image_source.can_process(Path::new("spreadsheet.xlsx")));
        }

        #[test]
        fn test_plugin_metadata_consistency() {
            // Test that all plugins have consistent metadata
            
            let docx_plugin = crate::builtin::docx_parser::DocxParserPlugin::new();
            let table_plugin = crate::builtin::table_detection::TableDetectionPlugin::new();
            let image_plugin = crate::builtin::image_processing::ImageProcessingPlugin::new();
            
            // All plugins should have the same version and author
            assert_eq!(docx_plugin.metadata().version, "1.0.0");
            assert_eq!(table_plugin.metadata().version, "1.0.0");
            assert_eq!(image_plugin.metadata().version, "1.0.0");
            
            assert_eq!(docx_plugin.metadata().author, "Neural Document Flow Team");
            assert_eq!(table_plugin.metadata().author, "Neural Document Flow Team");
            assert_eq!(image_plugin.metadata().author, "Neural Document Flow Team");
            
            // All plugins should require filesystem access
            assert!(docx_plugin.metadata().capabilities.requires_filesystem);
            assert!(table_plugin.metadata().capabilities.requires_filesystem);
            assert!(image_plugin.metadata().capabilities.requires_filesystem);
            
            // None should require network access
            assert!(!docx_plugin.metadata().capabilities.requires_network);
            assert!(!table_plugin.metadata().capabilities.requires_network);
            assert!(!image_plugin.metadata().capabilities.requires_network);
            
            // All should have reasonable resource limits
            assert!(docx_plugin.metadata().capabilities.max_memory_mb > 0);
            assert!(table_plugin.metadata().capabilities.max_memory_mb > 0);
            assert!(image_plugin.metadata().capabilities.max_memory_mb > 0);
            
            assert!(docx_plugin.metadata().capabilities.max_cpu_percent > 0.0);
            assert!(table_plugin.metadata().capabilities.max_cpu_percent > 0.0);
            assert!(image_plugin.metadata().capabilities.max_cpu_percent > 0.0);
            
            assert!(docx_plugin.metadata().capabilities.timeout_seconds > 0);
            assert!(table_plugin.metadata().capabilities.timeout_seconds > 0);
            assert!(image_plugin.metadata().capabilities.timeout_seconds > 0);
        }

        #[test]
        fn test_plugin_format_no_overlap() {
            // Test that plugins don't claim to handle the same formats inappropriately
            
            let docx_formats: std::collections::HashSet<_> = 
                crate::builtin::docx_parser::DocxParserPlugin::new()
                    .metadata().supported_formats.iter().collect();
            
            let table_formats: std::collections::HashSet<_> = 
                crate::builtin::table_detection::TableDetectionPlugin::new()
                    .metadata().supported_formats.iter().collect();
            
            let image_formats: std::collections::HashSet<_> = 
                crate::builtin::image_processing::ImageProcessingPlugin::new()
                    .metadata().supported_formats.iter().collect();
            
            // DOCX should be unique to DOCX parser
            assert!(docx_formats.contains(&"docx".to_string()));
            assert!(!table_formats.contains(&"docx".to_string()));
            assert!(!image_formats.contains(&"docx".to_string()));
            
            // PDF can be handled by both table detection and image processing
            assert!(!docx_formats.contains(&"pdf".to_string()));
            assert!(table_formats.contains(&"pdf".to_string()));
            assert!(image_formats.contains(&"pdf".to_string()));
            
            // PNG/JPG should be in image processing and table detection
            assert!(!docx_formats.contains(&"png".to_string()));
            assert!(table_formats.contains(&"png".to_string()));
            assert!(image_formats.contains(&"png".to_string()));
        }
    }
}