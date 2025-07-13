//! Document utilities and extensions
//!
//! This module provides additional utilities for working with documents,
//! extending the core Document type defined in types.rs

use crate::types::{Document, DocumentMetadata, DocumentContent, ProcessingEvent,
                  DocumentType, DocumentSourceType, DocumentStructure};
use crate::error::{Result, NeuralDocFlowError};
use std::path::Path;
use uuid::Uuid;
use chrono::Utc;

/// Document builder for convenient document creation
#[derive(Debug, Default)]
pub struct DocumentBuilder {
    metadata: DocumentMetadata,
    content: DocumentContent,
    processing_history: Vec<ProcessingEvent>,
}

impl DocumentBuilder {
    /// Create a new document builder
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set document title
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.metadata.title = Some(title.into());
        self
    }
    
    /// Set document source
    pub fn source(mut self, source: impl Into<String>) -> Self {
        self.metadata.source = source.into();
        self
    }
    
    /// Set MIME type
    pub fn mime_type(mut self, mime_type: impl Into<String>) -> Self {
        self.metadata.mime_type = mime_type.into();
        self
    }
    
    /// Add author
    pub fn author(mut self, author: impl Into<String>) -> Self {
        self.metadata.authors.push(author.into());
        self
    }
    
    /// Set text content
    pub fn text_content(mut self, text: impl Into<String>) -> Self {
        self.content.text = Some(text.into());
        self
    }
    
    /// Set file size
    pub fn size(mut self, size: u64) -> Self {
        self.metadata.size = Some(size);
        self
    }
    
    /// Set language
    pub fn language(mut self, language: impl Into<String>) -> Self {
        self.metadata.language = Some(language.into());
        self
    }
    
    /// Add custom metadata
    pub fn custom_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.custom.insert(key.into(), value);
        self
    }
    
    /// Build the document
    pub fn build(self) -> Document {
        let now = Utc::now();
        let doc_type = match self.metadata.mime_type.as_str() {
            "application/pdf" => DocumentType::Pdf,
            "text/plain" => DocumentType::Text,
            "text/html" => DocumentType::Html,
            "text/markdown" => DocumentType::Markdown,
            "application/json" => DocumentType::Json,
            "application/xml" | "text/xml" => DocumentType::Xml,
            t if t.starts_with("image/") => DocumentType::Image,
            _ => DocumentType::Unknown,
        };
        
        Document {
            id: Uuid::new_v4(),
            doc_type,
            source_type: DocumentSourceType::File,
            raw_content: Vec::new(),
            metadata: self.metadata,
            content: self.content,
            structure: DocumentStructure::default(),
            attachments: Vec::new(),
            processing_history: self.processing_history,
            created_at: now,
            updated_at: now,
        }
    }
}

/// Document utilities
pub struct DocumentUtils;

impl DocumentUtils {
    /// Create a document from file path
    pub async fn from_file(path: impl AsRef<Path>) -> Result<Document> {
        let path = path.as_ref();
        let source = path.to_string_lossy().to_string();
        
        // Determine MIME type from extension
        let mime_type = Self::mime_from_path(path);
        
        // Get file metadata
        let metadata = tokio::fs::metadata(path).await
            .map_err(|e| NeuralDocFlowError::IoError(e))?;
        
        let size = metadata.len();
        
        // Read file content
        let content = tokio::fs::read(path).await
            .map_err(|e| NeuralDocFlowError::IoError(e))?;
        
        // Create document
        let mut doc = DocumentBuilder::new()
            .source(source)
            .mime_type(mime_type)
            .size(size)
            .build();
        
        // Store raw content
        doc.content.raw = Some(content);
        
        Ok(doc)
    }
    
    /// Determine MIME type from file path
    pub fn mime_from_path(path: &Path) -> String {
        match path.extension().and_then(|e| e.to_str()) {
            Some("pdf") => "application/pdf",
            Some("html") | Some("htm") => "text/html",
            Some("xml") => "application/xml",
            Some("json") => "application/json",
            Some("txt") => "text/plain",
            Some("md") => "text/markdown",
            Some("doc") => "application/msword",
            Some("docx") => "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            Some("xls") => "application/vnd.ms-excel",
            Some("xlsx") => "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            Some("ppt") => "application/vnd.ms-powerpoint",
            Some("pptx") => "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            Some("jpg") | Some("jpeg") => "image/jpeg",
            Some("png") => "image/png",
            Some("gif") => "image/gif",
            Some("svg") => "image/svg+xml",
            _ => "application/octet-stream",
        }.to_string()
    }
    
    /// Extract text content from document
    pub fn extract_text(doc: &Document) -> Option<String> {
        doc.content.text.clone()
    }
    
    /// Get document size in bytes
    pub fn get_size(doc: &Document) -> u64 {
        doc.metadata.size.unwrap_or_else(|| {
            // Calculate size from content if not in metadata
            let mut size = 0u64;
            if let Some(text) = &doc.content.text {
                size += text.len() as u64;
            }
            if let Some(raw) = &doc.content.raw {
                size += raw.len() as u64;
            }
            for image in &doc.content.images {
                size += match &image.data {
                    crate::types::ImageContent::Base64(s) => s.len() as u64,
                    crate::types::ImageContent::Binary(b) => b.len() as u64,
                    crate::types::ImageContent::File(_) => 0,
                };
            }
            size
        })
    }
    
    /// Check if document has been processed by a specific processor
    pub fn has_been_processed_by(doc: &Document, processor_name: &str) -> bool {
        doc.processing_history.iter()
            .any(|event| event.processor_name == processor_name)
    }
    
    /// Get processing confidence for a specific processor
    pub fn get_processor_confidence(doc: &Document, processor_name: &str) -> Option<f64> {
        doc.processing_history.iter()
            .filter(|event| event.processor_name == processor_name)
            .filter_map(|event| event.confidence)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }
    
    /// Merge two documents
    pub fn merge(doc1: Document, doc2: Document) -> Document {
        let mut merged = doc1;
        
        // Merge content
        if doc2.content.text.is_some() && merged.content.text.is_none() {
            merged.content.text = doc2.content.text;
        }
        
        merged.content.images.extend(doc2.content.images);
        merged.content.tables.extend(doc2.content.tables);
        
        for (key, value) in doc2.content.structured {
            merged.content.structured.entry(key).or_insert(value);
        }
        
        // Merge processing history
        merged.processing_history.extend(doc2.processing_history);
        
        // Update timestamp
        merged.updated_at = Utc::now();
        
        merged
    }
}

/// Document validation utilities
pub struct DocumentValidator;

impl DocumentValidator {
    /// Validate document structure
    pub fn validate(doc: &Document) -> Result<()> {
        // Check required fields
        if doc.metadata.source.is_empty() {
            return Err(NeuralDocFlowError::ConfigError {
                message: "Document source cannot be empty".to_string()
            });
        }
        
        if doc.metadata.mime_type.is_empty() {
            return Err(NeuralDocFlowError::ConfigError {
                message: "Document MIME type cannot be empty".to_string()
            });
        }
        
        // Check content
        let has_content = doc.content.text.is_some() 
            || !doc.content.images.is_empty()
            || !doc.content.tables.is_empty()
            || !doc.content.structured.is_empty()
            || doc.content.raw.is_some();
        
        if !has_content {
            return Err(NeuralDocFlowError::ConfigError {
                message: "Document has no content".to_string()
            });
        }
        
        Ok(())
    }
    
    /// Check if document is complete (has been fully processed)
    pub fn is_complete(doc: &Document, required_processors: &[String]) -> bool {
        required_processors.iter()
            .all(|processor| DocumentUtils::has_been_processed_by(doc, processor))
    }
    
    /// Calculate overall document quality score
    pub fn calculate_quality_score(doc: &Document) -> f64 {
        let confidence_scores: Vec<f64> = doc.processing_history.iter()
            .filter_map(|event| event.confidence)
            .collect();
        
        if confidence_scores.is_empty() {
            return 0.0;
        }
        
        // Calculate weighted average based on recency
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        
        for (i, score) in confidence_scores.iter().enumerate() {
            let weight = 1.0 + (i as f64 / confidence_scores.len() as f64);
            weighted_sum += score * weight;
            weight_sum += weight;
        }
        
        weighted_sum / weight_sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_document_builder() {
        let doc = DocumentBuilder::new()
            .title("Test Document")
            .source("test.pdf")
            .mime_type("application/pdf")
            .author("Test Author")
            .text_content("Test content")
            .size(1024)
            .language("en")
            .build();
        
        assert_eq!(doc.metadata.title, Some("Test Document".to_string()));
        assert_eq!(doc.metadata.source, "test.pdf");
        assert_eq!(doc.metadata.mime_type, "application/pdf");
        assert_eq!(doc.metadata.authors, vec!["Test Author"]);
        assert_eq!(doc.content.text, Some("Test content".to_string()));
        assert_eq!(doc.metadata.size, Some(1024));
        assert_eq!(doc.metadata.language, Some("en".to_string()));
    }
    
    #[test]
    fn test_mime_type_detection() {
        assert_eq!(DocumentUtils::mime_from_path(Path::new("test.pdf")), "application/pdf");
        assert_eq!(DocumentUtils::mime_from_path(Path::new("test.html")), "text/html");
        assert_eq!(DocumentUtils::mime_from_path(Path::new("test.json")), "application/json");
        assert_eq!(DocumentUtils::mime_from_path(Path::new("test.unknown")), "application/octet-stream");
    }
}