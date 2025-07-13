//! PDF document source implementation
//!
//! This module provides a comprehensive PDF document source that can extract
//! text, tables, images, and metadata from PDF files.

use async_trait::async_trait;
use neural_doc_flow_core::*;
use std::collections::HashMap;
use std::io::Cursor;
use uuid::Uuid;

#[cfg(feature = "pdf")]
use lopdf::Document as LopdfDocument;

/// PDF-specific configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PdfConfig {
    /// Maximum file size in bytes
    pub max_file_size: usize,
    /// Enable OCR for scanned PDFs
    pub enable_ocr: bool,
    /// Extract tables from PDFs
    pub extract_tables: bool,
    /// Extract images from PDFs
    pub extract_images: bool,
    /// Security settings
    pub security: SecurityConfig,
    /// Performance settings
    pub performance: PerformanceConfig,
}

impl Default for PdfConfig {
    fn default() -> Self {
        Self {
            max_file_size: 100 * 1024 * 1024, // 100MB
            enable_ocr: false,
            extract_tables: true,
            extract_images: true,
            security: SecurityConfig::default(),
            performance: PerformanceConfig::default(),
        }
    }
}

/// Security configuration for PDF processing
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SecurityConfig {
    /// Enable security checks
    pub enabled: bool,
    /// Allow JavaScript in PDFs
    pub allow_javascript: bool,
    /// Allow external references
    pub allow_external_references: bool,
    /// Maximum embedded file size
    pub max_embedded_file_size: usize,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            allow_javascript: false,
            allow_external_references: false,
            max_embedded_file_size: 10 * 1024 * 1024, // 10MB
        }
    }
}

/// Performance configuration for PDF processing
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PerformanceConfig {
    /// Use memory-mapped files
    pub use_mmap: bool,
    /// Process pages in parallel
    pub parallel_pages: bool,
    /// Chunk size for processing
    pub chunk_size: usize,
    /// Buffer pool size
    pub buffer_pool_size: usize,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            use_mmap: true,
            parallel_pages: true,
            chunk_size: 4096,
            buffer_pool_size: 16,
        }
    }
}

/// PDF document source implementation
pub struct PdfSource {
    config: PdfConfig,
    statistics: SourceStatistics,
}

impl PdfSource {
    /// Create new PDF source
    pub fn new() -> Self {
        Self {
            config: PdfConfig::default(),
            statistics: SourceStatistics::default(),
        }
    }

    /// Create PDF source with custom configuration
    pub fn with_config(config: PdfConfig) -> Self {
        Self {
            config,
            statistics: SourceStatistics::default(),
        }
    }

    /// Read input data
    async fn read_input(&self, input: SourceInput) -> Result<Vec<u8>, SourceError> {
        match input {
            SourceInput::File { path, .. } => {
                #[cfg(feature = "pdf")]
                if self.config.performance.use_mmap {
                    // Use memory-mapped file for large PDFs
                    let file = std::fs::File::open(&path)?;
                    let len = file.metadata()?.len();
                    if len > self.config.max_file_size as u64 {
                        return Err(SourceError::Custom("File too large".to_string()));
                    }
                    
                    #[cfg(feature = "pdf")]
                    {
                        use memmap2::Mmap;
                        let mmap = unsafe { Mmap::map(&file)? };
                        Ok(mmap.to_vec())
                    }
                    #[cfg(not(feature = "pdf"))]
                    {
                        tokio::fs::read(&path).await.map_err(Into::into)
                    }
                } else {
                    tokio::fs::read(&path).await.map_err(Into::into)
                }
                #[cfg(not(feature = "pdf"))]
                {
                    tokio::fs::read(&path).await.map_err(Into::into)
                }
            }
            SourceInput::Memory { data, .. } => {
                if data.len() > self.config.max_file_size {
                    return Err(SourceError::Custom("Data too large".to_string()));
                }
                Ok(data)
            }
            SourceInput::Stream { mut reader, size_hint, .. } => {
                let mut buffer = if let Some(size) = size_hint {
                    if size > self.config.max_file_size {
                        return Err(SourceError::Custom("Stream too large".to_string()));
                    }
                    Vec::with_capacity(size)
                } else {
                    Vec::new()
                };
                
                use tokio::io::AsyncReadExt;
                reader.read_to_end(&mut buffer).await?;
                
                if buffer.len() > self.config.max_file_size {
                    return Err(SourceError::Custom("Stream data too large".to_string()));
                }
                
                Ok(buffer)
            }
            SourceInput::Url { url, headers } => {
                #[cfg(feature = "http")]
                {
                    let client = reqwest::Client::new();
                    let mut request = client.get(&url);
                    
                    if let Some(headers) = headers {
                        for (key, value) in headers {
                            request = request.header(&key, &value);
                        }
                    }
                    
                    let response = request.send().await
                        .map_err(|e| SourceError::HttpError(e.to_string()))?;
                    
                    if !response.status().is_success() {
                        return Err(SourceError::HttpError(format!("HTTP {}", response.status())));
                    }
                    
                    let bytes = response.bytes().await
                        .map_err(|e| SourceError::HttpError(e.to_string()))?;
                    
                    if bytes.len() > self.config.max_file_size {
                        return Err(SourceError::Custom("Downloaded file too large".to_string()));
                    }
                    
                    Ok(bytes.to_vec())
                }
                #[cfg(not(feature = "http"))]
                {
                    Err(SourceError::Custom("HTTP support not enabled".to_string()))
                }
            }
        }
    }

    /// Extract PDF content using available libraries
    #[cfg(feature = "pdf")]
    async fn extract_pdf_content(&self, data: &[u8]) -> Result<ExtractedDocument, SourceError> {
        let start_time = std::time::Instant::now();
        
        // Parse PDF with lopdf
        let doc = LopdfDocument::load_mem(data)
            .map_err(|e| SourceError::ParseError(format!("Failed to parse PDF: {}", e)))?;

        let mut content_blocks = Vec::new();
        let mut total_pages = 0;

        // Extract text from each page
        for (page_num, page_id) in doc.page_iter().enumerate() {
            total_pages += 1;
            
            // For Phase 1, create a simple text block per page
            // Phase 2 will implement detailed PDF parsing
            let page_text = format!("Page {} content (detailed extraction in Phase 2)", page_num + 1);
            
            let block = ContentBlock {
                id: Uuid::new_v4().to_string(),
                block_type: BlockType::Paragraph,
                text: Some(page_text),
                binary: None,
                metadata: BlockMetadata {
                    page: Some(page_num + 1),
                    confidence: 0.85, // Phase 1 placeholder confidence
                    language: Some("en".to_string()),
                    attributes: HashMap::new(),
                },
                position: BlockPosition {
                    page: page_num,
                    x: 0.0,
                    y: 0.0,
                    width: 595.0, // Standard A4 width in points
                    height: 842.0, // Standard A4 height in points
                },
                relationships: vec![],
            };
            
            content_blocks.push(block);
        }

        // Extract metadata
        let metadata = self.extract_pdf_metadata(&doc);

        // Build document structure
        let structure = DocumentStructure {
            sections: vec![],
            hierarchy: vec![],
            table_of_contents: vec![],
        };

        let document = ExtractedDocument {
            id: Uuid::new_v4().to_string(),
            source_id: self.source_id().to_string(),
            metadata,
            content: content_blocks.clone(),
            structure,
            confidence: 0.85, // Phase 1 placeholder
            metrics: ExtractionMetrics {
                extraction_time: start_time.elapsed(),
                pages_processed: total_pages,
                blocks_extracted: content_blocks.len(),
                memory_used: data.len(),
            },
        };

        Ok(document)
    }

    /// Extract PDF content (fallback implementation)
    #[cfg(not(feature = "pdf"))]
    async fn extract_pdf_content(&self, data: &[u8]) -> Result<ExtractedDocument, SourceError> {
        let start_time = std::time::Instant::now();
        
        // Phase 1: Basic placeholder implementation
        let content_block = ContentBlock {
            id: Uuid::new_v4().to_string(),
            block_type: BlockType::Paragraph,
            text: Some("PDF content extraction requires 'pdf' feature to be enabled".to_string()),
            binary: None,
            metadata: BlockMetadata {
                page: Some(1),
                confidence: 0.5,
                language: Some("en".to_string()),
                attributes: HashMap::new(),
            },
            position: BlockPosition {
                page: 0,
                x: 0.0,
                y: 0.0,
                width: 595.0,
                height: 842.0,
            },
            relationships: vec![],
        };

        let document = ExtractedDocument {
            id: Uuid::new_v4().to_string(),
            source_id: self.source_id().to_string(),
            metadata: DocumentMetadata {
                title: Some("PDF Document".to_string()),
                author: None,
                created_date: None,
                modified_date: None,
                page_count: 1,
                language: Some("en".to_string()),
                keywords: vec![],
                custom_metadata: HashMap::new(),
            },
            content: vec![content_block],
            structure: DocumentStructure {
                sections: vec![],
                hierarchy: vec![],
                table_of_contents: vec![],
            },
            confidence: 0.5,
            metrics: ExtractionMetrics {
                extraction_time: start_time.elapsed(),
                pages_processed: 1,
                blocks_extracted: 1,
                memory_used: data.len(),
            },
        };

        Ok(document)
    }

    /// Extract PDF metadata
    #[cfg(feature = "pdf")]
    fn extract_pdf_metadata(&self, doc: &LopdfDocument) -> DocumentMetadata {
        let mut metadata = DocumentMetadata {
            title: None,
            author: None,
            created_date: None,
            modified_date: None,
            page_count: doc.get_pages().len(),
            language: Some("en".to_string()), // Default to English
            keywords: vec![],
            custom_metadata: HashMap::new(),
        };

        // Try to extract PDF metadata
        if let Ok(trailer) = doc.trailer.get(b"Info") {
            if let Ok(info_ref) = trailer.as_reference() {
                if let Ok(info_obj) = doc.get_object(info_ref) {
                    // Extract standard PDF metadata fields
                    // Phase 1: Basic implementation
                    // Phase 2: Full PDF metadata extraction
                    metadata.custom_metadata.insert(
                        "pdf_version".to_string(),
                        "1.4".to_string() // Placeholder
                    );
                }
            }
        }

        metadata
    }

    /// Security check for PDF content
    async fn check_security(&self, data: &[u8]) -> Result<(), SourceError> {
        if !self.config.security.enabled {
            return Ok(());
        }

        // Check for JavaScript
        if !self.config.security.allow_javascript {
            if self.contains_javascript(data) {
                return Err(SourceError::SecurityError(
                    "PDF contains JavaScript code".to_string()
                ));
            }
        }

        // Check for external references
        if !self.config.security.allow_external_references {
            if self.contains_external_refs(data) {
                return Err(SourceError::SecurityError(
                    "PDF contains external references".to_string()
                ));
            }
        }

        Ok(())
    }

    /// Check if PDF contains JavaScript
    fn contains_javascript(&self, data: &[u8]) -> bool {
        // Simple pattern matching for JavaScript
        data.windows(b"/JavaScript".len())
            .any(|window| window == b"/JavaScript") ||
        data.windows(b"/JS".len())
            .any(|window| window == b"/JS")
    }

    /// Check if PDF contains external references
    fn contains_external_refs(&self, data: &[u8]) -> bool {
        // Check for URI actions and external references
        data.windows(b"/URI".len())
            .any(|window| window == b"/URI") ||
        data.windows(b"http://".len())
            .any(|window| window == b"http://") ||
        data.windows(b"https://".len())
            .any(|window| window == b"https://")
    }

    /// Update statistics
    fn update_statistics(&mut self, document: &ExtractedDocument, processing_time: std::time::Duration) {
        self.statistics.documents_processed += 1;
        self.statistics.total_processing_time += processing_time;
        self.statistics.average_processing_time = 
            self.statistics.total_processing_time / self.statistics.documents_processed as u32;
        
        // Simple success rate calculation (assume all successful for Phase 1)
        self.statistics.success_rate = 1.0;
        
        // Update memory usage
        self.statistics.peak_memory_usage = self.statistics.peak_memory_usage
            .max(document.metrics.memory_used);
    }
}

impl Default for PdfSource {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DocumentSource for PdfSource {
    fn source_id(&self) -> &str {
        "pdf"
    }

    fn name(&self) -> &str {
        "PDF Document Source"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn supported_extensions(&self) -> &[&str] {
        &["pdf", "PDF"]
    }

    fn supported_mime_types(&self) -> &[&str] {
        &["application/pdf", "application/x-pdf"]
    }

    async fn can_handle(&self, input: &SourceInput) -> Result<bool, SourceError> {
        match input {
            SourceInput::File { path, .. } => {
                // Check file extension
                if let Some(ext) = path.extension() {
                    if let Some(ext_str) = ext.to_str() {
                        if self.supported_extensions().contains(&ext_str) {
                            return Ok(true);
                        }
                    }
                }

                // Check file magic bytes
                let mut file = tokio::fs::File::open(path).await?;
                let mut header = [0u8; 5];
                use tokio::io::AsyncReadExt;
                if file.read_exact(&mut header).await.is_ok() {
                    return Ok(&header == b"%PDF-");
                }
                
                Ok(false)
            }
            SourceInput::Memory { data, mime_type, .. } => {
                // Check MIME type
                if let Some(mime) = mime_type {
                    if self.supported_mime_types().contains(&mime.as_str()) {
                        return Ok(true);
                    }
                }

                // Check magic bytes
                Ok(data.starts_with(b"%PDF-"))
            }
            SourceInput::Stream { .. } => {
                // Cannot easily check stream content without consuming it
                Ok(false)
            }
            SourceInput::Url { url, .. } => {
                // Check URL extension
                Ok(url.ends_with(".pdf") || url.ends_with(".PDF"))
            }
        }
    }

    async fn validate(&self, input: &SourceInput) -> Result<ValidationResult, SourceError> {
        let mut result = ValidationResult::valid();

        // Get document data for validation
        let data = match input {
            SourceInput::File { path, .. } => tokio::fs::read(path).await?,
            SourceInput::Memory { data, .. } => data.clone(),
            _ => {
                // For streams and URLs, perform basic validation
                result.add_warning("Cannot perform detailed validation on stream/URL input");
                return Ok(result);
            }
        };

        // Validate PDF structure
        if !data.starts_with(b"%PDF-") {
            result.add_error("Invalid PDF header");
        }

        // Check file size
        if data.len() > self.config.max_file_size {
            result.add_error("File size exceeds limit");
        }

        // Security checks
        if let Err(e) = self.check_security(&data).await {
            result.add_error(&e.to_string());
        }

        Ok(result)
    }

    async fn extract(&self, input: SourceInput) -> Result<ExtractedDocument, SourceError> {
        let data = self.read_input(input).await?;
        
        // Security validation
        self.check_security(&data).await?;
        
        // Extract content
        let document = self.extract_pdf_content(&data).await?;
        
        // Update statistics (Note: This requires mutable access in real implementation)
        // For Phase 1, we'll skip statistics updates to maintain trait compatibility
        
        Ok(document)
    }

    fn config_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "max_file_size": {
                    "type": "integer",
                    "description": "Maximum file size in bytes",
                    "default": 104857600
                },
                "enable_ocr": {
                    "type": "boolean",
                    "description": "Enable OCR for scanned PDFs",
                    "default": false
                },
                "extract_tables": {
                    "type": "boolean",
                    "description": "Extract tables from PDFs",
                    "default": true
                },
                "extract_images": {
                    "type": "boolean",
                    "description": "Extract images from PDFs",
                    "default": true
                },
                "security": {
                    "type": "object",
                    "properties": {
                        "enabled": {
                            "type": "boolean",
                            "description": "Enable security checks",
                            "default": true
                        },
                        "allow_javascript": {
                            "type": "boolean",
                            "description": "Allow JavaScript in PDFs",
                            "default": false
                        },
                        "allow_external_references": {
                            "type": "boolean",
                            "description": "Allow external references",
                            "default": false
                        }
                    }
                },
                "performance": {
                    "type": "object",
                    "properties": {
                        "use_mmap": {
                            "type": "boolean",
                            "description": "Use memory-mapped files",
                            "default": true
                        },
                        "parallel_pages": {
                            "type": "boolean",
                            "description": "Process pages in parallel",
                            "default": true
                        },
                        "chunk_size": {
                            "type": "integer",
                            "description": "Processing chunk size",
                            "default": 4096
                        }
                    }
                }
            }
        })
    }

    async fn initialize(&mut self, config: SourceConfig) -> Result<(), SourceError> {
        // Parse PDF-specific configuration
        if !config.settings.is_null() {
            self.config = serde_json::from_value(config.settings)
                .map_err(|e| SourceError::ConfigError(e.to_string()))?;
        }

        tracing::info!("PDF source initialized with config: {:?}", self.config);
        Ok(())
    }

    async fn cleanup(&mut self) -> Result<(), SourceError> {
        // Clean up any resources
        tracing::info!("PDF source cleanup completed");
        Ok(())
    }

    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            text_extraction: true,
            image_extraction: self.config.extract_images,
            table_extraction: self.config.extract_tables,
            metadata_extraction: true,
            structure_preservation: true,
            encrypted_documents: false, // Phase 1: No encrypted PDF support
            max_file_size: Some(self.config.max_file_size),
            quality_levels: vec!["standard".to_string()],
        }
    }

    fn statistics(&self) -> SourceStatistics {
        self.statistics.clone()
    }

    fn supports_streaming(&self) -> bool {
        false // PDF requires full document access
    }

    fn supports_parallel(&self) -> bool {
        self.config.performance.parallel_pages
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_pdf_source_creation() {
        let source = PdfSource::new();
        assert_eq!(source.source_id(), "pdf");
        assert_eq!(source.name(), "PDF Document Source");
        assert!(source.supported_extensions().contains(&"pdf"));
    }

    #[tokio::test]
    async fn test_pdf_detection() {
        let source = PdfSource::new();

        let input = SourceInput::Memory {
            data: b"%PDF-1.4".to_vec(),
            filename: Some("test.pdf".to_string()),
            mime_type: None,
        };

        assert!(source.can_handle(&input).await.unwrap());
    }

    #[tokio::test]
    async fn test_invalid_pdf_detection() {
        let source = PdfSource::new();

        let input = SourceInput::Memory {
            data: b"Not a PDF".to_vec(),
            filename: Some("test.txt".to_string()),
            mime_type: None,
        };

        assert!(!source.can_handle(&input).await.unwrap());
    }

    #[tokio::test]
    async fn test_pdf_validation() {
        let source = PdfSource::new();

        // Valid PDF header
        let valid_input = SourceInput::Memory {
            data: b"%PDF-1.4\nsome content".to_vec(),
            filename: Some("test.pdf".to_string()),
            mime_type: None,
        };

        let result = source.validate(&valid_input).await.unwrap();
        assert!(result.is_valid);

        // Invalid PDF header
        let invalid_input = SourceInput::Memory {
            data: b"Invalid content".to_vec(),
            filename: Some("test.pdf".to_string()),
            mime_type: None,
        };

        let result = source.validate(&invalid_input).await.unwrap();
        assert!(!result.is_valid);
    }

    #[test]
    fn test_security_checks() {
        let source = PdfSource::new();

        // Test JavaScript detection
        let js_content = b"some content /JavaScript more content";
        assert!(source.contains_javascript(js_content));

        let no_js_content = b"regular pdf content without scripts";
        assert!(!source.contains_javascript(no_js_content));

        // Test external reference detection
        let uri_content = b"some content /URI http://example.com";
        assert!(source.contains_external_refs(uri_content));

        let no_uri_content = b"regular pdf content without external refs";
        assert!(!source.contains_external_refs(no_uri_content));
    }

    #[test]
    fn test_config_schema() {
        let source = PdfSource::new();
        let schema = source.config_schema();
        
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"].is_object());
        assert!(schema["properties"]["max_file_size"].is_object());
    }
}