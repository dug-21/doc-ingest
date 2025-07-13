//! PDF document source implementation using lopdf
//!
//! Provides PDF parsing and text extraction capabilities.

use crate::traits::{BaseDocumentSource, SourceCapability, SourceMetadata, SourceConfig, SourceError, SourceResult};
use neural_doc_flow_core::prelude::*;
use async_trait::async_trait;
use lopdf::{Document as LopdfDocument, Object};
use std::collections::HashMap;
use std::path::Path;
use tokio::fs;
use tracing::{debug, info, warn, error};

/// PDF document source implementation
pub struct PdfSource {
    config: SourceConfig,
    metadata: SourceMetadata,
}

impl PdfSource {
    /// Create a new PDF source
    pub fn new() -> Self {
        Self {
            config: SourceConfig::default(),
            metadata: SourceMetadata {
                id: "pdf-source".to_string(),
                name: "PDF Document Source".to_string(),
                version: "1.0.0".to_string(),
                capabilities: vec![
                    SourceCapability::Pdf,
                    SourceCapability::MetadataExtraction,
                ],
                file_extensions: vec!["pdf".to_string()],
                mime_types: vec!["application/pdf".to_string()],
            },
        }
    }
    
    /// Extract text from a PDF document
    async fn extract_text(&self, pdf_bytes: &[u8]) -> SourceResult<String> {
        // Parse PDF document
        let doc = LopdfDocument::load_mem(pdf_bytes)
            .map_err(|e| SourceError::ParseError(format!("Failed to load PDF: {}", e)))?;
        
        let mut text = String::new();
        let pages = doc.get_pages();
        
        for (page_num, page_id) in pages {
            debug!("Processing page {}", page_num);
            
            if let Ok(content) = doc.extract_text(&[page_num]) {
                text.push_str(&content);
                text.push('\n');
            } else {
                warn!("Failed to extract text from page {}", page_num);
            }
        }
        
        Ok(text)
    }
    
    /// Extract metadata from a PDF document
    async fn extract_metadata(&self, pdf_bytes: &[u8]) -> SourceResult<HashMap<String, String>> {
        let doc = LopdfDocument::load_mem(pdf_bytes)
            .map_err(|e| SourceError::ParseError(format!("Failed to load PDF: {}", e)))?;
        
        let mut metadata = HashMap::new();
        
        // Extract document info
        if let Ok(info) = doc.trailer.get(b"Info") {
            if let Ok(Object::Reference(ref_id)) = info {
                if let Ok(Object::Dictionary(dict)) = doc.get_object(*ref_id) {
                    // Extract common metadata fields
                    let fields = vec![
                        ("Title", "title"),
                        ("Author", "author"),
                        ("Subject", "subject"),
                        ("Keywords", "keywords"),
                        ("Creator", "creator"),
                        ("Producer", "producer"),
                        ("CreationDate", "creation_date"),
                        ("ModDate", "modification_date"),
                    ];
                    
                    for (pdf_key, meta_key) in fields {
                        if let Ok(value) = dict.get(pdf_key.as_bytes()) {
                            if let Ok(Object::String(s, _)) = value {
                                metadata.insert(
                                    meta_key.to_string(),
                                    String::from_utf8_lossy(s).to_string()
                                );
                            }
                        }
                    }
                }
            }
        }
        
        // Add page count
        metadata.insert("page_count".to_string(), doc.get_pages().len().to_string());
        
        // Add PDF version
        metadata.insert("pdf_version".to_string(), format!("{}.{}", doc.version.0, doc.version.1));
        
        Ok(metadata)
    }
}

impl Default for PdfSource {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DocumentSource for PdfSource {
    async fn process(&self, mut input: Document) -> Result<Document> {
        info!("Processing PDF document: {}", input.id);
        
        // Get content based on source type
        let pdf_bytes = match &input.source {
            DocumentSourceType::File(path) => {
                // Check file size
                let file_path = Path::new(path);
                let metadata = fs::metadata(file_path).await
                    .map_err(|e| anyhow::anyhow!("Failed to read file metadata: {}", e))?;
                
                if let Some(max_size) = self.config.max_file_size {
                    if metadata.len() as usize > max_size {
                        return Err(anyhow::anyhow!(SourceError::FileTooLarge {
                            size: metadata.len() as usize,
                            max: max_size,
                        }));
                    }
                }
                
                fs::read(file_path).await
                    .map_err(|e| anyhow::anyhow!("Failed to read PDF file: {}", e))?
            }
            DocumentSourceType::Memory(data) => {
                // Check data size
                if let Some(max_size) = self.config.max_file_size {
                    if data.len() > max_size {
                        return Err(anyhow::anyhow!(SourceError::FileTooLarge {
                            size: data.len(),
                            max: max_size,
                        }));
                    }
                }
                data.clone()
            }
            _ => return Err(anyhow::anyhow!("Unsupported source type for PDF")),
        };
        
        // Extract text content
        let text_content = self.extract_text(&pdf_bytes).await
            .map_err(|e| anyhow::anyhow!("Failed to extract text: {}", e))?;
        
        // Update document content
        input.content = text_content;
        
        // Extract metadata if enabled
        if self.config.extract_metadata {
            match self.extract_metadata(&pdf_bytes).await {
                Ok(pdf_metadata) => {
                    // Merge PDF metadata with existing metadata
                    for (key, value) in pdf_metadata {
                        input.metadata.insert(key, value);
                    }
                }
                Err(e) => {
                    warn!("Failed to extract PDF metadata: {}", e);
                }
            }
        }
        
        // Update document type
        input.doc_type = DocumentType::Pdf;
        
        // Add processing metadata
        input.metadata.insert("processor".to_string(), "pdf-source".to_string());
        input.metadata.insert("processor_version".to_string(), self.metadata.version.clone());
        
        info!("Successfully processed PDF document: {}", input.id);
        Ok(input)
    }
}

#[async_trait]
impl BaseDocumentSource for PdfSource {
    fn metadata(&self) -> SourceMetadata {
        self.metadata.clone()
    }
    
    async fn can_handle(&self, input: &str) -> bool {
        // Check if input is a file path with .pdf extension
        if input.ends_with(".pdf") || input.ends_with(".PDF") {
            return true;
        }
        
        // Check if it's a path that exists
        let path = Path::new(input);
        if path.exists() && path.is_file() {
            if let Some(ext) = path.extension() {
                return ext.eq_ignore_ascii_case("pdf");
            }
        }
        
        false
    }
    
    fn config(&self) -> &SourceConfig {
        &self.config
    }
    
    fn set_config(&mut self, config: SourceConfig) {
        self.config = config;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    
    #[tokio::test]
    async fn test_pdf_source_creation() {
        let source = PdfSource::new();
        assert_eq!(source.metadata().id, "pdf-source");
        assert!(source.metadata().capabilities.contains(&SourceCapability::Pdf));
    }
    
    #[tokio::test]
    async fn test_can_handle() {
        let source = PdfSource::new();
        
        assert!(source.can_handle("document.pdf").await);
        assert!(source.can_handle("DOCUMENT.PDF").await);
        assert!(!source.can_handle("document.txt").await);
        assert!(!source.can_handle("document").await);
    }
    
    #[tokio::test]
    async fn test_config() {
        let mut source = PdfSource::new();
        let mut config = SourceConfig::default();
        config.max_file_size = Some(50 * 1024 * 1024);
        
        source.set_config(config.clone());
        assert_eq!(source.config().max_file_size, Some(50 * 1024 * 1024));
    }
}