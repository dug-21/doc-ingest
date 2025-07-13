//! PDF output implementation

use crate::DocumentOutput;
use neural_doc_flow_core::{Document, Result};
use async_trait::async_trait;
use std::path::Path;
use tokio::fs;

/// PDF output formatter
#[derive(Debug, Default)]
pub struct PdfOutput;

impl PdfOutput {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DocumentOutput for PdfOutput {
    fn name(&self) -> &str {
        "pdf"
    }
    
    fn supported_extensions(&self) -> &[&str] {
        &["pdf"]
    }
    
    fn mime_type(&self) -> &str {
        "application/pdf"
    }
    
    async fn generate(&self, document: &Document, output_path: &Path) -> Result<()> {
        let bytes = self.generate_bytes(document).await?;
        fs::write(output_path, bytes).await?;
        Ok(())
    }
    
    async fn generate_bytes(&self, document: &Document) -> Result<Vec<u8>> {
        // TODO: Implement actual PDF generation using printpdf
        // For now, return a simple text representation
        
        let mut content = String::new();
        
        // Document title
        content.push_str(&format!("{}\n", document.metadata.title));
        content.push_str(&"=".repeat(document.metadata.title.len()));
        content.push_str("\n\n");
        
        // Metadata section
        content.push_str("METADATA\n");
        content.push_str("--------\n");
        content.push_str(&format!("ID: {}\n", document.metadata.id));
        if let Some(author) = &document.metadata.author {
            content.push_str(&format!("Author: {}\n", author));
        }
        content.push_str(&format!("Format: {}\n", document.metadata.format));
        content.push_str(&format!("Size: {} bytes\n", document.metadata.size));
        content.push_str(&format!("Created: {}\n", document.metadata.created_at.format("%Y-%m-%d %H:%M:%S UTC")));
        content.push_str(&format!("Modified: {}\n", document.metadata.modified_at.format("%Y-%m-%d %H:%M:%S UTC")));
        
        if let Some(language) = &document.metadata.language {
            content.push_str(&format!("Language: {}\n", language));
        }
        
        if !document.metadata.tags.is_empty() {
            content.push_str(&format!("Tags: {}\n", document.metadata.tags.join(", ")));
        }
        
        if let Some(source_path) = &document.metadata.source_path {
            content.push_str(&format!("Source: {}\n", source_path.display()));
        }
        
        content.push_str("\n");
        
        // Content section
        content.push_str("CONTENT\n");
        content.push_str("-------\n");
        content.push_str(&document.content);
        content.push_str("\n\n");
        
        // Extracted text section (if different from content)
        if document.extracted_text != document.content && !document.extracted_text.is_empty() {
            content.push_str("EXTRACTED TEXT\n");
            content.push_str("--------------\n");
            content.push_str(&document.extracted_text);
            content.push_str("\n\n");
        }
        
        // Structure section (if available)
        if let Some(structure) = &document.structure {
            content.push_str("STRUCTURE\n");
            content.push_str("---------\n");
            content.push_str(&serde_json::to_string_pretty(structure)?);
            content.push_str("\n\n");
        }
        
        // Attachments section (if any)
        if !document.attachments.is_empty() {
            content.push_str("ATTACHMENTS\n");
            content.push_str("-----------\n");
            for (i, attachment) in document.attachments.iter().enumerate() {
                content.push_str(&format!("{}. {} ({} bytes)\n", i + 1, attachment.name, attachment.size));
            }
            content.push_str("\n");
        }
        
        // For now, return as plain text bytes
        // TODO: Replace with actual PDF generation
        Ok(content.into_bytes())
    }
    
    fn validate_config(&self) -> Result<()> {
        // TODO: Validate PDF generation dependencies
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neural_doc_flow_core::{DocumentMetadata};
    use uuid::Uuid;
    use chrono::Utc;
    
    #[tokio::test]
    async fn test_pdf_output() {
        let output = PdfOutput::new();
        
        let metadata = DocumentMetadata {
            id: Uuid::new_v4(),
            title: "Test Document".to_string(),
            author: Some("Test Author".to_string()),
            created_at: Utc::now(),
            modified_at: Utc::now(),
            source_path: None,
            format: "test".to_string(),
            size: 100,
            language: Some("en".to_string()),
            tags: vec!["test".to_string(), "example".to_string()],
        };
        
        let document = Document {
            metadata,
            content: "Test content\n\nThis is a test document.".to_string(),
            raw_content: b"Test content".to_vec(),
            extracted_text: "Test content".to_string(),
            structure: None,
            attachments: Vec::new(),
        };
        
        let bytes = output.generate_bytes(&document).await.unwrap();
        let content_str = String::from_utf8(bytes).unwrap();
        
        assert!(content_str.contains("Test Document"));
        assert!(content_str.contains("METADATA"));
        assert!(content_str.contains("Author: Test Author"));
        assert!(content_str.contains("Tags: test, example"));
        assert!(content_str.contains("CONTENT"));
        assert!(content_str.contains("This is a test document."));
    }
}