// Example: Creating a Custom Document Source Plugin
// This example shows how to create a plugin for a proprietary document format

use async_trait::async_trait;
use neuraldocflow_sources::{
    DocumentSource, SourceInput, SourceConfig, SourceError,
    ExtractedDocument, ValidationResult, ContentBlock, BlockType,
    DocumentMetadata, DocumentStructure, ExtractionMetrics,
    BlockMetadata, BlockPosition,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Configuration for the custom source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomSourceConfig {
    /// API endpoint for document conversion service
    pub api_endpoint: String,
    
    /// API key for authentication
    pub api_key: String,
    
    /// Timeout for API calls
    pub timeout_seconds: u64,
    
    /// Maximum document size
    pub max_document_size: usize,
    
    /// Enable caching
    pub enable_cache: bool,
    
    /// Cache TTL in seconds
    pub cache_ttl: u64,
}

impl Default for CustomSourceConfig {
    fn default() -> Self {
        Self {
            api_endpoint: "https://api.example.com/convert".to_string(),
            api_key: String::new(),
            timeout_seconds: 30,
            max_document_size: 50 * 1024 * 1024, // 50MB
            enable_cache: true,
            cache_ttl: 3600, // 1 hour
        }
    }
}

/// Custom document format source
pub struct CustomFormatSource {
    config: CustomSourceConfig,
    http_client: reqwest::Client,
    cache: Option<DocumentCache>,
}

impl CustomFormatSource {
    /// Create new custom format source
    pub fn new() -> Self {
        Self {
            config: CustomSourceConfig::default(),
            http_client: reqwest::Client::new(),
            cache: None,
        }
    }
    
    /// Parse custom format
    async fn parse_custom_format(&self, data: &[u8]) -> Result<CustomDocument, SourceError> {
        // Custom format structure (example):
        // Magic bytes: "CDOC"
        // Version: 4 bytes
        // Metadata length: 4 bytes
        // Metadata: JSON
        // Content sections...
        
        if data.len() < 12 {
            return Err(SourceError::InvalidFormat("File too small".to_string()));
        }
        
        // Check magic bytes
        if &data[0..4] != b"CDOC" {
            return Err(SourceError::InvalidFormat("Invalid magic bytes".to_string()));
        }
        
        // Read version
        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        if version > 2 {
            return Err(SourceError::UnsupportedVersion(version));
        }
        
        // Read metadata length
        let metadata_len = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        
        if data.len() < 12 + metadata_len {
            return Err(SourceError::InvalidFormat("Truncated metadata".to_string()));
        }
        
        // Parse metadata
        let metadata_bytes = &data[12..12 + metadata_len];
        let metadata: CustomMetadata = serde_json::from_slice(metadata_bytes)?;
        
        // Parse content sections
        let mut sections = Vec::new();
        let mut offset = 12 + metadata_len;
        
        while offset < data.len() {
            let section = self.parse_section(&data[offset..])?;
            offset += section.total_size;
            sections.push(section);
        }
        
        Ok(CustomDocument {
            version,
            metadata,
            sections,
        })
    }
    
    /// Parse a content section
    fn parse_section(&self, data: &[u8]) -> Result<CustomSection, SourceError> {
        if data.len() < 12 {
            return Err(SourceError::InvalidFormat("Invalid section header".to_string()));
        }
        
        // Section header:
        // Type: 4 bytes
        // Length: 4 bytes
        // Flags: 4 bytes
        
        let section_type = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let length = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
        let flags = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
        
        if data.len() < 12 + length {
            return Err(SourceError::InvalidFormat("Truncated section".to_string()));
        }
        
        let content = data[12..12 + length].to_vec();
        
        Ok(CustomSection {
            section_type: section_type.into(),
            content,
            flags,
            total_size: 12 + length,
        })
    }
    
    /// Convert custom document to standard format
    fn convert_to_standard(&self, doc: CustomDocument) -> Result<ExtractedDocument, SourceError> {
        let mut content_blocks = Vec::new();
        
        // Convert each section to content blocks
        for (idx, section) in doc.sections.iter().enumerate() {
            match section.section_type {
                SectionType::Text => {
                    let text = String::from_utf8(section.content.clone())
                        .map_err(|_| SourceError::InvalidFormat("Invalid UTF-8".to_string()))?;
                    
                    content_blocks.push(ContentBlock {
                        id: Uuid::new_v4().to_string(),
                        block_type: BlockType::Paragraph,
                        text: Some(text),
                        binary: None,
                        metadata: BlockMetadata {
                            page: Some(idx / 10), // Approximate pages
                            confidence: 0.95,
                            language: doc.metadata.language.clone(),
                            attributes: HashMap::new(),
                        },
                        position: BlockPosition {
                            page: idx / 10,
                            x: 0.0,
                            y: (idx % 10) as f32 * 100.0,
                            width: 600.0,
                            height: 50.0,
                        },
                        relationships: vec![],
                    });
                }
                SectionType::Table => {
                    // Parse table data
                    let table_data: TableData = bincode::deserialize(&section.content)?;
                    
                    content_blocks.push(ContentBlock {
                        id: Uuid::new_v4().to_string(),
                        block_type: BlockType::Table,
                        text: Some(self.table_to_markdown(&table_data)),
                        binary: None,
                        metadata: BlockMetadata {
                            page: Some(idx / 10),
                            confidence: 0.9,
                            language: None,
                            attributes: {
                                let mut attrs = HashMap::new();
                                attrs.insert("rows".to_string(), table_data.rows.to_string());
                                attrs.insert("cols".to_string(), table_data.cols.to_string());
                                attrs
                            },
                        },
                        position: BlockPosition {
                            page: idx / 10,
                            x: 0.0,
                            y: (idx % 10) as f32 * 100.0,
                            width: 600.0,
                            height: 200.0,
                        },
                        relationships: vec![],
                    });
                }
                SectionType::Image => {
                    content_blocks.push(ContentBlock {
                        id: Uuid::new_v4().to_string(),
                        block_type: BlockType::Image,
                        text: None,
                        binary: Some(section.content.clone()),
                        metadata: BlockMetadata {
                            page: Some(idx / 10),
                            confidence: 1.0,
                            language: None,
                            attributes: {
                                let mut attrs = HashMap::new();
                                if section.flags & 0x01 != 0 {
                                    attrs.insert("format".to_string(), "png".to_string());
                                } else {
                                    attrs.insert("format".to_string(), "jpeg".to_string());
                                }
                                attrs
                            },
                        },
                        position: BlockPosition {
                            page: idx / 10,
                            x: 0.0,
                            y: (idx % 10) as f32 * 100.0,
                            width: 400.0,
                            height: 300.0,
                        },
                        relationships: vec![],
                    });
                }
                SectionType::Metadata => {
                    // Skip metadata sections
                }
            }
        }
        
        // Build document structure
        let structure = DocumentStructure {
            sections: vec![],
            hierarchy: vec![],
            table_of_contents: vec![],
        };
        
        // Create extracted document
        Ok(ExtractedDocument {
            id: Uuid::new_v4().to_string(),
            source_id: self.source_id().to_string(),
            metadata: DocumentMetadata {
                title: doc.metadata.title,
                author: doc.metadata.author,
                created_date: doc.metadata.created_date,
                modified_date: doc.metadata.modified_date,
                page_count: (content_blocks.len() / 10).max(1),
                language: doc.metadata.language,
                keywords: doc.metadata.keywords,
                custom_metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("version".to_string(), doc.version.to_string());
                    meta.insert("original_format".to_string(), "custom".to_string());
                    meta
                },
            },
            content: content_blocks,
            structure,
            confidence: 0.92,
            metrics: ExtractionMetrics {
                extraction_time: std::time::Duration::from_millis(100),
                pages_processed: 1,
                blocks_extracted: content_blocks.len(),
                memory_used: std::mem::size_of_val(&doc),
            },
        })
    }
    
    /// Convert table data to markdown
    fn table_to_markdown(&self, table: &TableData) -> String {
        let mut markdown = String::new();
        
        for row in &table.cells {
            markdown.push('|');
            for cell in row {
                markdown.push_str(&format!(" {} |", cell));
            }
            markdown.push('\n');
            
            // Add header separator after first row
            if markdown.lines().count() == 1 {
                markdown.push('|');
                for _ in row {
                    markdown.push_str(" --- |");
                }
                markdown.push('\n');
            }
        }
        
        markdown
    }
    
    /// Call external API for advanced processing
    async fn call_conversion_api(&self, data: &[u8]) -> Result<ExtractedDocument, SourceError> {
        let response = self.http_client
            .post(&self.config.api_endpoint)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .body(data.to_vec())
            .timeout(std::time::Duration::from_secs(self.config.timeout_seconds))
            .send()
            .await?;
        
        if !response.status().is_success() {
            return Err(SourceError::ApiError(response.status().to_string()));
        }
        
        let api_result: ApiConversionResult = response.json().await?;
        
        // Convert API result to ExtractedDocument
        self.convert_api_result(api_result)
    }
    
    fn convert_api_result(&self, result: ApiConversionResult) -> Result<ExtractedDocument, SourceError> {
        // Convert API-specific format to standard ExtractedDocument
        // Implementation depends on API response format
        todo!()
    }
}

#[async_trait]
impl DocumentSource for CustomFormatSource {
    fn source_id(&self) -> &str {
        "custom_format"
    }
    
    fn name(&self) -> &str {
        "Custom Document Format"
    }
    
    fn version(&self) -> &str {
        "1.0.0"
    }
    
    fn supported_extensions(&self) -> &[&str] {
        &["cdf", "custom"]
    }
    
    fn supported_mime_types(&self) -> &[&str] {
        &["application/x-custom-format"]
    }
    
    async fn can_handle(&self, input: &SourceInput) -> Result<bool, SourceError> {
        match input {
            SourceInput::File { path, .. } => {
                // Check extension
                if let Some(ext) = path.extension() {
                    if self.supported_extensions().contains(&ext.to_str().unwrap_or("")) {
                        return Ok(true);
                    }
                }
                
                // Check magic bytes
                let mut file = tokio::fs::File::open(path).await?;
                let mut header = [0u8; 4];
                use tokio::io::AsyncReadExt;
                file.read_exact(&mut header).await?;
                Ok(&header == b"CDOC")
            }
            SourceInput::Memory { data, .. } => {
                Ok(data.starts_with(b"CDOC"))
            }
            _ => Ok(false),
        }
    }
    
    async fn validate(&self, input: &SourceInput) -> Result<ValidationResult, SourceError> {
        let mut result = ValidationResult::default();
        
        let data = match input {
            SourceInput::File { path, .. } => tokio::fs::read(path).await?,
            SourceInput::Memory { data, .. } => data.clone(),
            _ => return Err(SourceError::UnsupportedInput),
        };
        
        // Check size
        if data.len() > self.config.max_document_size {
            result.add_error("Document exceeds size limit");
        }
        
        // Validate format
        if !data.starts_with(b"CDOC") {
            result.add_error("Invalid file format");
        }
        
        Ok(result)
    }
    
    async fn extract(&self, input: SourceInput) -> Result<ExtractedDocument, SourceError> {
        // Check cache first
        if let Some(cache) = &self.cache {
            if let Some(cached) = cache.get(&input).await {
                return Ok(cached);
            }
        }
        
        let data = match input {
            SourceInput::File { path, .. } => tokio::fs::read(path).await?,
            SourceInput::Memory { data, .. } => data,
            _ => return Err(SourceError::UnsupportedInput),
        };
        
        // Try local parsing first
        let result = match self.parse_custom_format(&data).await {
            Ok(custom_doc) => self.convert_to_standard(custom_doc)?,
            Err(_) if !self.config.api_endpoint.is_empty() => {
                // Fallback to API
                self.call_conversion_api(&data).await?
            }
            Err(e) => return Err(e),
        };
        
        // Cache result
        if let Some(cache) = &self.cache {
            cache.put(&input, &result).await;
        }
        
        Ok(result)
    }
    
    fn config_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "api_endpoint": {
                    "type": "string",
                    "description": "API endpoint for document conversion",
                    "default": ""
                },
                "api_key": {
                    "type": "string",
                    "description": "API key for authentication",
                    "default": ""
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "API timeout in seconds",
                    "default": 30
                },
                "max_document_size": {
                    "type": "integer",
                    "description": "Maximum document size in bytes",
                    "default": 52428800
                },
                "enable_cache": {
                    "type": "boolean",
                    "description": "Enable document caching",
                    "default": true
                },
                "cache_ttl": {
                    "type": "integer",
                    "description": "Cache TTL in seconds",
                    "default": 3600
                }
            }
        })
    }
    
    async fn initialize(&mut self, config: SourceConfig) -> Result<(), SourceError> {
        self.config = serde_json::from_value(config.settings)?;
        
        // Initialize HTTP client with custom settings
        self.http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(self.config.timeout_seconds))
            .build()?;
        
        // Initialize cache if enabled
        if self.config.enable_cache {
            self.cache = Some(DocumentCache::new(
                1000, // max entries
                self.config.cache_ttl,
            ));
        }
        
        Ok(())
    }
    
    async fn cleanup(&mut self) -> Result<(), SourceError> {
        if let Some(cache) = &mut self.cache {
            cache.clear().await;
        }
        Ok(())
    }
}

// Supporting types for the custom format

#[derive(Debug, Clone)]
struct CustomDocument {
    version: u32,
    metadata: CustomMetadata,
    sections: Vec<CustomSection>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CustomMetadata {
    title: Option<String>,
    author: Option<String>,
    created_date: Option<String>,
    modified_date: Option<String>,
    language: Option<String>,
    keywords: Vec<String>,
}

#[derive(Debug, Clone)]
struct CustomSection {
    section_type: SectionType,
    content: Vec<u8>,
    flags: u32,
    total_size: usize,
}

#[derive(Debug, Clone, Copy)]
enum SectionType {
    Text,
    Table,
    Image,
    Metadata,
}

impl From<u32> for SectionType {
    fn from(value: u32) -> Self {
        match value {
            0 => SectionType::Text,
            1 => SectionType::Table,
            2 => SectionType::Image,
            3 => SectionType::Metadata,
            _ => SectionType::Text,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TableData {
    rows: usize,
    cols: usize,
    cells: Vec<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ApiConversionResult {
    // API-specific response format
}

// Simple document cache implementation
struct DocumentCache {
    // Cache implementation
}

impl DocumentCache {
    fn new(max_entries: usize, ttl_seconds: u64) -> Self {
        Self {
            // Initialize cache
        }
    }
    
    async fn get(&self, input: &SourceInput) -> Option<ExtractedDocument> {
        // Retrieve from cache
        None
    }
    
    async fn put(&self, input: &SourceInput, doc: &ExtractedDocument) {
        // Store in cache
    }
    
    async fn clear(&mut self) {
        // Clear cache
    }
}

// Plugin exports
#[no_mangle]
pub extern "C" fn create_source() -> *mut dyn DocumentSource {
    Box::into_raw(Box::new(CustomFormatSource::new()))
}

#[no_mangle]
pub extern "C" fn plugin_metadata() -> PluginMetadata {
    PluginMetadata {
        id: "custom_format".to_string(),
        name: "Custom Document Format".to_string(),
        version: "1.0.0".to_string(),
        author: "Your Company".to_string(),
        description: "Support for proprietary document format with API fallback".to_string(),
        api_version: "1.0".to_string(),
        capabilities: vec![
            "text_extraction".to_string(),
            "table_extraction".to_string(),
            "image_extraction".to_string(),
            "api_conversion".to_string(),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_custom_format_detection() {
        let source = CustomFormatSource::new();
        
        let input = SourceInput::Memory {
            data: b"CDOC\x01\x00\x00\x00".to_vec(),
            filename: Some("test.cdf".to_string()),
            mime_type: None,
        };
        
        assert!(source.can_handle(&input).await.unwrap());
    }
    
    #[tokio::test]
    async fn test_custom_format_parsing() {
        let source = CustomFormatSource::new();
        
        // Create test document
        let metadata = CustomMetadata {
            title: Some("Test Document".to_string()),
            author: Some("Test Author".to_string()),
            created_date: Some("2024-01-01".to_string()),
            modified_date: None,
            language: Some("en".to_string()),
            keywords: vec!["test".to_string()],
        };
        
        let metadata_json = serde_json::to_vec(&metadata).unwrap();
        let metadata_len = metadata_json.len() as u32;
        
        let mut data = Vec::new();
        data.extend_from_slice(b"CDOC");                    // Magic bytes
        data.extend_from_slice(&1u32.to_le_bytes());       // Version
        data.extend_from_slice(&metadata_len.to_le_bytes()); // Metadata length
        data.extend_from_slice(&metadata_json);             // Metadata
        
        // Add text section
        data.extend_from_slice(&0u32.to_le_bytes());       // Section type (Text)
        data.extend_from_slice(&11u32.to_le_bytes());      // Length
        data.extend_from_slice(&0u32.to_le_bytes());       // Flags
        data.extend_from_slice(b"Hello World");            // Content
        
        let doc = source.parse_custom_format(&data).await.unwrap();
        assert_eq!(doc.version, 1);
        assert_eq!(doc.metadata.title, Some("Test Document".to_string()));
        assert_eq!(doc.sections.len(), 1);
    }
}