//! DOCX Parser Plugin
//!
//! Extracts text, images, tables, and metadata from Microsoft Word documents (.docx)
//! This plugin uses a combination of ZIP extraction and XML parsing to handle the
//! Open XML format used by modern Word documents.

use neural_doc_flow_core::{DocumentSource, ProcessingError, Document};
use crate::{Plugin, PluginMetadata, PluginCapabilities};
use std::path::Path;
use std::io::{Read, Cursor};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// DOCX Parser Plugin implementation
pub struct DocxParserPlugin {
    metadata: PluginMetadata,
    config: DocxConfig,
}

/// Configuration for DOCX parsing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocxConfig {
    pub extract_images: bool,
    pub extract_tables: bool,
    pub preserve_formatting: bool,
    pub max_image_size_mb: usize,
    pub table_detection_threshold: f32,
}

/// Extracted DOCX content structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocxContent {
    pub text: String,
    pub images: Vec<DocxImage>,
    pub tables: Vec<DocxTable>,
    pub headers: Vec<String>,
    pub footers: Vec<String>,
    pub comments: Vec<String>,
    pub styles: HashMap<String, String>,
}

/// Image extracted from DOCX
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocxImage {
    pub id: String,
    pub name: String,
    pub data: Vec<u8>,
    pub format: String,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub description: Option<String>,
}

/// Table extracted from DOCX
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocxTable {
    pub id: String,
    pub rows: Vec<DocxTableRow>,
    pub headers: Option<Vec<String>>,
    pub caption: Option<String>,
    pub style: Option<String>,
}

/// Table row
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocxTableRow {
    pub cells: Vec<DocxTableCell>,
}

/// Table cell
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocxTableCell {
    pub text: String,
    pub colspan: u32,
    pub rowspan: u32,
    pub style: Option<String>,
}

impl Default for DocxConfig {
    fn default() -> Self {
        Self {
            extract_images: true,
            extract_tables: true,
            preserve_formatting: true,
            max_image_size_mb: 10,
            table_detection_threshold: 0.8,
        }
    }
}

impl DocxParserPlugin {
    /// Create new DOCX parser plugin
    pub fn new() -> Self {
        Self {
            metadata: PluginMetadata {
                name: "docx_parser".to_string(),
                version: "1.0.0".to_string(),
                author: "Neural Document Flow Team".to_string(),
                description: "Extract text, images, tables from Microsoft Word documents".to_string(),
                supported_formats: vec!["docx".to_string(), "docm".to_string()],
                capabilities: PluginCapabilities {
                    requires_network: false,
                    requires_filesystem: true,
                    max_memory_mb: 200,
                    max_cpu_percent: 60.0,
                    timeout_seconds: 120,
                },
            },
            config: DocxConfig::default(),
        }
    }

    /// Create plugin with custom configuration
    pub fn with_config(config: DocxConfig) -> Self {
        let mut plugin = Self::new();
        plugin.config = config;
        plugin
    }

    /// Parse DOCX file and extract all content
    fn parse_docx(&self, path: &Path) -> Result<DocxContent, ProcessingError> {
        tracing::info!("Parsing DOCX file: {:?}", path);

        // Read the DOCX file as a ZIP archive
        let file = std::fs::File::open(path)
            .map_err(|e| ProcessingError::SourceNotFound(format!("Failed to open DOCX file: {}", e)))?;

        let mut archive = zip::ZipArchive::new(file)
            .map_err(|e| ProcessingError::ProcessorFailed {
                processor_name: "docx_parser".to_string(),
                reason: format!("Failed to read ZIP archive: {}", e),
            })?;

        let mut content = DocxContent {
            text: String::new(),
            images: Vec::new(),
            tables: Vec::new(),
            headers: Vec::new(),
            footers: Vec::new(),
            comments: Vec::new(),
            styles: HashMap::new(),
        };

        // Extract main document content
        if let Ok(mut main_doc) = archive.by_name("word/document.xml") {
            let mut doc_content = String::new();
            main_doc.read_to_string(&mut doc_content)
                .map_err(|e| ProcessingError::ProcessorFailed {
                    processor_name: "docx_parser".to_string(),
                    reason: format!("Failed to read document.xml: {}", e),
                })?;

            self.extract_text_and_tables(&doc_content, &mut content)?;
        }

        // Extract images if enabled
        if self.config.extract_images {
            self.extract_images(&mut archive, &mut content)?;
        }

        // Extract headers and footers
        self.extract_headers_footers(&mut archive, &mut content)?;

        // Extract comments
        self.extract_comments(&mut archive, &mut content)?;

        // Extract styles
        self.extract_styles(&mut archive, &mut content)?;

        tracing::info!("DOCX parsing complete. Text: {} chars, Images: {}, Tables: {}", 
                      content.text.len(), content.images.len(), content.tables.len());

        Ok(content)
    }

    /// Extract text content and tables from document XML
    fn extract_text_and_tables(&self, xml_content: &str, content: &mut DocxContent) -> Result<(), ProcessingError> {
        // Parse XML and extract text and table structures
        // This is a simplified implementation - a full version would use proper XML parsing
        
        let doc = roxmltree::Document::parse(xml_content)
            .map_err(|e| ProcessingError::ProcessorFailed {
                processor_name: "docx_parser".to_string(),
                reason: format!("Failed to parse document XML: {}", e),
            })?;

        let mut text_parts = Vec::new();
        let mut table_id = 0;

        // Walk through the document structure
        for node in doc.descendants() {
            match node.tag_name().name() {
                "t" => {
                    // Text content
                    if let Some(text) = node.text() {
                        text_parts.push(text.to_string());
                    }
                }
                "tbl" => {
                    // Table
                    if self.config.extract_tables {
                        if let Ok(table) = self.parse_table_node(&node, &format!("table_{}", table_id)) {
                            content.tables.push(table);
                            table_id += 1;
                        }
                    }
                }
                "p" => {
                    // Paragraph - add line break
                    if !text_parts.is_empty() && !text_parts.last().unwrap().ends_with('\n') {
                        text_parts.push("\n".to_string());
                    }
                }
                _ => {}
            }
        }

        content.text = text_parts.join("");
        Ok(())
    }

    /// Parse a table node from XML
    fn parse_table_node(&self, table_node: &roxmltree::Node, table_id: &str) -> Result<DocxTable, ProcessingError> {
        let mut rows = Vec::new();
        let mut headers = None;
        let mut caption = None;

        // Extract table rows
        for row_node in table_node.descendants() {
            if row_node.tag_name().name() == "tr" {
                let mut cells = Vec::new();
                
                for cell_node in row_node.descendants() {
                    if cell_node.tag_name().name() == "tc" {
                        let cell_text = self.extract_cell_text(&cell_node);
                        
                        // Get cell span attributes
                        let colspan = self.get_cell_span(&cell_node, "gridSpan").unwrap_or(1);
                        let rowspan = self.get_cell_span(&cell_node, "vMerge").unwrap_or(1);
                        
                        cells.push(DocxTableCell {
                            text: cell_text,
                            colspan,
                            rowspan,
                            style: None,
                        });
                    }
                }
                
                if !cells.is_empty() {
                    rows.push(DocxTableRow { cells });
                }
            }
        }

        // Determine if first row is header (heuristic)
        if !rows.is_empty() && self.is_likely_header_row(&rows[0]) {
            headers = Some(rows[0].cells.iter().map(|c| c.text.clone()).collect());
        }

        Ok(DocxTable {
            id: table_id.to_string(),
            rows,
            headers,
            caption,
            style: None,
        })
    }

    /// Extract text from a table cell
    fn extract_cell_text(&self, cell_node: &roxmltree::Node) -> String {
        let mut text_parts = Vec::new();
        
        for node in cell_node.descendants() {
            if node.tag_name().name() == "t" {
                if let Some(text) = node.text() {
                    text_parts.push(text.to_string());
                }
            }
        }
        
        text_parts.join(" ").trim().to_string()
    }

    /// Get cell span value from attributes
    fn get_cell_span(&self, cell_node: &roxmltree::Node, attr_name: &str) -> Option<u32> {
        for child in cell_node.children() {
            if child.tag_name().name() == "tcPr" {
                for prop in child.children() {
                    if prop.tag_name().name() == attr_name {
                        if let Some(val) = prop.attribute("val") {
                            return val.parse().ok();
                        }
                    }
                }
            }
        }
        None
    }

    /// Heuristic to determine if a row is likely a header
    fn is_likely_header_row(&self, row: &DocxTableRow) -> bool {
        // Simple heuristic: if cells are short and contain common header words
        let header_indicators = ["name", "date", "value", "type", "id", "description", "status"];
        
        row.cells.iter().any(|cell| {
            let text = cell.text.to_lowercase();
            text.len() < 50 && header_indicators.iter().any(|&indicator| text.contains(indicator))
        })
    }

    /// Extract images from the DOCX archive
    fn extract_images(&self, archive: &mut zip::ZipArchive<std::fs::File>, content: &mut DocxContent) -> Result<(), ProcessingError> {
        let mut image_id = 0;
        
        // Look for images in word/media/ directory
        for i in 0..archive.len() {
            let mut file = archive.by_index(i)
                .map_err(|e| ProcessingError::ProcessorFailed {
                    processor_name: "docx_parser".to_string(),
                    reason: format!("Failed to access archive entry: {}", e),
                })?;

            if file.name().starts_with("word/media/") {
                let file_name = file.name().to_string();
                
                // Check if it's an image file
                if self.is_image_file(&file_name) {
                    let mut image_data = Vec::new();
                    file.read_to_end(&mut image_data)
                        .map_err(|e| ProcessingError::ProcessorFailed {
                            processor_name: "docx_parser".to_string(),
                            reason: format!("Failed to read image data: {}", e),
                        })?;

                    // Check size limit
                    let size_mb = image_data.len() / (1024 * 1024);
                    if size_mb > self.config.max_image_size_mb {
                        tracing::warn!("Skipping large image: {} ({}MB > {}MB)", 
                                      file_name, size_mb, self.config.max_image_size_mb);
                        continue;
                    }

                    let format = self.detect_image_format(&image_data);
                    
                    content.images.push(DocxImage {
                        id: format!("image_{}", image_id),
                        name: file_name.clone(),
                        data: image_data,
                        format,
                        width: None,  // Would need image parsing library to get dimensions
                        height: None,
                        description: None,
                    });
                    
                    image_id += 1;
                }
            }
        }

        Ok(())
    }

    /// Check if file is an image based on extension
    fn is_image_file(&self, filename: &str) -> bool {
        let image_extensions = ["png", "jpg", "jpeg", "gif", "bmp", "tiff", "wmf", "emf"];
        
        if let Some(ext) = filename.split('.').last() {
            image_extensions.contains(&ext.to_lowercase().as_str())
        } else {
            false
        }
    }

    /// Detect image format from binary data
    fn detect_image_format(&self, data: &[u8]) -> String {
        if data.len() < 8 {
            return "unknown".to_string();
        }

        // Check magic bytes
        match &data[0..8] {
            [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A] => "png".to_string(),
            [0xFF, 0xD8, 0xFF, ..] => "jpeg".to_string(),
            [0x47, 0x49, 0x46, 0x38, ..] => "gif".to_string(),
            [0x42, 0x4D, ..] => "bmp".to_string(),
            _ => "unknown".to_string(),
        }
    }

    /// Extract headers and footers
    fn extract_headers_footers(&self, archive: &mut zip::ZipArchive<std::fs::File>, content: &mut DocxContent) -> Result<(), ProcessingError> {
        // Extract header files
        for i in 0..archive.len() {
            let mut file = archive.by_index(i)
                .map_err(|e| ProcessingError::ProcessorFailed {
                    processor_name: "docx_parser".to_string(),
                    reason: format!("Failed to access archive entry: {}", e),
                })?;

            let filename = file.name().to_string();
            
            if filename.starts_with("word/header") || filename.starts_with("word/footer") {
                let mut xml_content = String::new();
                file.read_to_string(&mut xml_content)
                    .map_err(|e| ProcessingError::ProcessorFailed {
                        processor_name: "docx_parser".to_string(),
                        reason: format!("Failed to read header/footer: {}", e),
                    })?;

                let text = self.extract_text_from_xml(&xml_content)?;
                
                if filename.contains("header") {
                    content.headers.push(text);
                } else {
                    content.footers.push(text);
                }
            }
        }

        Ok(())
    }

    /// Extract comments from the document
    fn extract_comments(&self, archive: &mut zip::ZipArchive<std::fs::File>, content: &mut DocxContent) -> Result<(), ProcessingError> {
        if let Ok(mut comments_file) = archive.by_name("word/comments.xml") {
            let mut xml_content = String::new();
            comments_file.read_to_string(&mut xml_content)
                .map_err(|e| ProcessingError::ProcessorFailed {
                    processor_name: "docx_parser".to_string(),
                    reason: format!("Failed to read comments: {}", e),
                })?;

            let doc = roxmltree::Document::parse(&xml_content)
                .map_err(|e| ProcessingError::ProcessorFailed {
                    processor_name: "docx_parser".to_string(),
                    reason: format!("Failed to parse comments XML: {}", e),
                })?;

            for node in doc.descendants() {
                if node.tag_name().name() == "comment" {
                    let comment_text = self.extract_text_from_node(&node);
                    if !comment_text.trim().is_empty() {
                        content.comments.push(comment_text);
                    }
                }
            }
        }

        Ok(())
    }

    /// Extract style information
    fn extract_styles(&self, archive: &mut zip::ZipArchive<std::fs::File>, content: &mut DocxContent) -> Result<(), ProcessingError> {
        if let Ok(mut styles_file) = archive.by_name("word/styles.xml") {
            let mut xml_content = String::new();
            styles_file.read_to_string(&mut xml_content)
                .map_err(|e| ProcessingError::ProcessorFailed {
                    processor_name: "docx_parser".to_string(),
                    reason: format!("Failed to read styles: {}", e),
                })?;

            let doc = roxmltree::Document::parse(&xml_content)
                .map_err(|e| ProcessingError::ProcessorFailed {
                    processor_name: "docx_parser".to_string(),
                    reason: format!("Failed to parse styles XML: {}", e),
                })?;

            for node in doc.descendants() {
                if node.tag_name().name() == "style" {
                    if let Some(style_id) = node.attribute("styleId") {
                        if let Some(name_node) = node.descendants().find(|n| n.tag_name().name() == "name") {
                            if let Some(style_name) = name_node.attribute("val") {
                                content.styles.insert(style_id.to_string(), style_name.to_string());
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Extract plain text from XML content
    fn extract_text_from_xml(&self, xml_content: &str) -> Result<String, ProcessingError> {
        let doc = roxmltree::Document::parse(xml_content)
            .map_err(|e| ProcessingError::ProcessorFailed {
                processor_name: "docx_parser".to_string(),
                reason: format!("Failed to parse XML: {}", e),
            })?;

        Ok(self.extract_text_from_node(&doc.root_element()))
    }

    /// Extract text from XML node
    fn extract_text_from_node(&self, node: &roxmltree::Node) -> String {
        let mut text_parts = Vec::new();
        
        for child in node.descendants() {
            if child.tag_name().name() == "t" {
                if let Some(text) = child.text() {
                    text_parts.push(text.to_string());
                }
            }
        }
        
        text_parts.join(" ")
    }
}

impl Plugin for DocxParserPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }
    
    fn initialize(&mut self) -> Result<(), ProcessingError> {
        tracing::info!("Initializing DOCX parser plugin");
        Ok(())
    }
    
    fn shutdown(&mut self) -> Result<(), ProcessingError> {
        tracing::info!("Shutting down DOCX parser plugin");
        Ok(())
    }
    
    fn document_source(&self) -> Box<dyn DocumentSource> {
        Box::new(DocxSource::new(self.config.clone()))
    }
}

/// Document source for DOCX files
pub struct DocxSource {
    config: DocxConfig,
}

impl DocxSource {
    pub fn new(config: DocxConfig) -> Self {
        Self { config }
    }
}

impl DocumentSource for DocxSource {
    fn can_process(&self, path: &Path) -> bool {
        if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
            matches!(ext.to_lowercase().as_str(), "docx" | "docm")
        } else {
            false
        }
    }
    
    fn extract_document(&self, path: &Path) -> Result<Document, ProcessingError> {
        let parser = DocxParserPlugin::with_config(self.config.clone());
        let docx_content = parser.parse_docx(path)?;
        
        // Convert to standard Document format
        let mut metadata = HashMap::new();
        
        // Add extracted metadata
        metadata.insert("format".to_string(), "docx".to_string());
        metadata.insert("images_count".to_string(), docx_content.images.len().to_string());
        metadata.insert("tables_count".to_string(), docx_content.tables.len().to_string());
        metadata.insert("headers_count".to_string(), docx_content.headers.len().to_string());
        metadata.insert("footers_count".to_string(), docx_content.footers.len().to_string());
        metadata.insert("comments_count".to_string(), docx_content.comments.len().to_string());
        
        // Serialize structured data as JSON for access
        if let Ok(images_json) = serde_json::to_string(&docx_content.images) {
            metadata.insert("images_data".to_string(), images_json);
        }
        if let Ok(tables_json) = serde_json::to_string(&docx_content.tables) {
            metadata.insert("tables_data".to_string(), tables_json);
        }

        Ok(Document {
            content: docx_content.text,
            metadata,
        })
    }
}

/// Entry point for plugin
#[no_mangle]
pub extern "C" fn create_docx_parser_plugin() -> *mut dyn Plugin {
    let plugin = DocxParserPlugin::new();
    Box::into_raw(Box::new(plugin))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;

    #[test]
    fn test_plugin_creation() {
        let plugin = DocxParserPlugin::new();
        assert_eq!(plugin.metadata().name, "docx_parser");
        assert!(plugin.metadata().supported_formats.contains(&"docx".to_string()));
    }

    #[test]
    fn test_source_can_process() {
        let source = DocxSource::new(DocxConfig::default());
        
        assert!(source.can_process(Path::new("test.docx")));
        assert!(source.can_process(Path::new("test.docm")));
        assert!(!source.can_process(Path::new("test.pdf")));
        assert!(!source.can_process(Path::new("test.txt")));
    }

    #[test]
    fn test_image_format_detection() {
        let plugin = DocxParserPlugin::new();
        
        // PNG magic bytes
        let png_data = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        assert_eq!(plugin.detect_image_format(&png_data), "png");
        
        // JPEG magic bytes
        let jpeg_data = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46];
        assert_eq!(plugin.detect_image_format(&jpeg_data), "jpeg");
    }

    #[test]
    fn test_is_image_file() {
        let plugin = DocxParserPlugin::new();
        
        assert!(plugin.is_image_file("image.png"));
        assert!(plugin.is_image_file("photo.jpg"));
        assert!(plugin.is_image_file("graphic.jpeg"));
        assert!(!plugin.is_image_file("document.xml"));
        assert!(!plugin.is_image_file("text.txt"));
    }
}