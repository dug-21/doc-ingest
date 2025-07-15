//! Markdown output implementation

use crate::DocumentOutput;
use neural_doc_flow_core::{Document, Result};
use async_trait::async_trait;
use std::path::Path;
use tokio::fs;

/// Markdown output formatter
#[derive(Debug, Default)]
pub struct MarkdownOutput;

impl MarkdownOutput {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DocumentOutput for MarkdownOutput {
    fn name(&self) -> &str {
        "markdown"
    }
    
    fn supported_extensions(&self) -> &[&str] {
        &["md", "markdown"]
    }
    
    fn mime_type(&self) -> &str {
        "text/markdown"
    }
    
    async fn generate(&self, document: &Document, output_path: &Path) -> Result<()> {
        let bytes = self.generate_bytes(document).await?;
        fs::write(output_path, bytes).await?;
        Ok(())
    }
    
    async fn generate_bytes(&self, document: &Document) -> Result<Vec<u8>> {
        let mut markdown = String::new();
        
        // Document title
        let title = document.metadata.title.as_ref()
            .map(|s| s.as_str())
            .unwrap_or("Untitled Document");
        markdown.push_str(&format!("# {}\n\n", title));
        
        // Metadata section
        markdown.push_str("## Metadata\n\n");
        markdown.push_str(&format!("- **ID**: {}\n", document.id));
        if !document.metadata.authors.is_empty() {
            markdown.push_str(&format!("- **Authors**: {}\n", document.metadata.authors.join(", ")));
        }
        markdown.push_str(&format!("- **MIME Type**: {}\n", document.metadata.mime_type));
        if let Some(size) = document.metadata.size {
            markdown.push_str(&format!("- **Size**: {} bytes\n", size));
        }
        markdown.push_str(&format!("- **Created**: {}\n", document.created_at.format("%Y-%m-%d %H:%M:%S UTC")));
        markdown.push_str(&format!("- **Modified**: {}\n", document.updated_at.format("%Y-%m-%d %H:%M:%S UTC")));
        
        if let Some(language) = &document.metadata.language {
            markdown.push_str(&format!("- **Language**: {}\n", language));
        }
        
        markdown.push_str(&format!("- **Source**: {}\n", document.metadata.source));
        
        markdown.push('\n');
        
        // Content section
        markdown.push_str("## Content\n\n");
        if let Some(text) = &document.content.text {
            markdown.push_str(text);
        } else {
            markdown.push_str(&format!("{}", document.content));
        }
        markdown.push('\n');
        
        // Images section (if any)
        if !document.content.images.is_empty() {
            markdown.push_str("\n## Images\n\n");
            for (i, image) in document.content.images.iter().enumerate() {
                markdown.push_str(&format!("{}. {} ({}x{})\n", i + 1, image.id, image.width, image.height));
                if let Some(caption) = &image.caption {
                    markdown.push_str(&format!("   Caption: {}\n", caption));
                }
            }
        }
        
        // Tables section (if any)
        if !document.content.tables.is_empty() {
            markdown.push_str("\n## Tables\n\n");
            for (i, table) in document.content.tables.iter().enumerate() {
                markdown.push_str(&format!("### Table {} - {}\n\n", i + 1, table.caption.as_ref()
                    .map(|s| s.as_str())
                    .unwrap_or(&table.id)));
                
                // Table headers
                markdown.push_str(&format!("| {} |\n", table.headers.join(" | ")));
                markdown.push_str(&format!("| {} |\n", vec!["---"; table.headers.len()].join(" | ")));
                
                // Table rows
                for row in &table.rows {
                    markdown.push_str(&format!("| {} |\n", row.join(" | ")));
                }
                markdown.push('\n');
            }
        }
        
        // Structure section
        if document.structure.page_count.is_some() || !document.structure.sections.is_empty() {
            markdown.push_str("\n## Document Structure\n\n");
            if let Some(page_count) = document.structure.page_count {
                markdown.push_str(&format!("- **Pages**: {}\n", page_count));
            }
            if !document.structure.sections.is_empty() {
                markdown.push_str(&format!("- **Sections**: {}\n", document.structure.sections.len()));
            }
        }
        
        // Attachments section (if any)
        if !document.attachments.is_empty() {
            markdown.push_str("\n## Attachments\n\n");
            for (i, attachment) in document.attachments.iter().enumerate() {
                markdown.push_str(&format!("{}. {} ({} bytes)\n", i + 1, attachment.name, attachment.data.len()));
            }
            markdown.push('\n');
        }
        
        Ok(markdown.into_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neural_doc_flow_core::{DocumentMetadata, DocumentContent, DocumentStructure, DocumentType, DocumentSourceType};
    use uuid::Uuid;
    use chrono::Utc;
    use std::collections::HashMap;
    
    #[tokio::test]
    async fn test_markdown_output() {
        let output = MarkdownOutput::new();
        
        let metadata = DocumentMetadata {
            title: Some("Test Document".to_string()),
            authors: vec!["Test Author".to_string()],
            source: "test.txt".to_string(),
            mime_type: "text/plain".to_string(),
            size: Some(100),
            language: Some("en".to_string()),
            custom: HashMap::new(),
        };
        
        let content = DocumentContent {
            text: Some("Test content\n\nThis is a test document.".to_string()),
            images: Vec::new(),
            tables: Vec::new(),
            structured: HashMap::new(),
            raw: Some(b"Test content".to_vec()),
        };
        
        let now = Utc::now();
        let document = Document {
            id: Uuid::new_v4(),
            doc_type: DocumentType::Text,
            source_type: DocumentSourceType::File,
            raw_content: b"Test content".to_vec(),
            metadata,
            content,
            structure: DocumentStructure::default(),
            attachments: Vec::new(),
            processing_history: Vec::new(),
            created_at: now,
            updated_at: now,
        };
        
        let bytes = output.generate_bytes(&document).await.unwrap();
        let markdown_str = String::from_utf8(bytes).unwrap();
        
        assert!(markdown_str.contains("# Test Document"));
        assert!(markdown_str.contains("## Metadata"));
        assert!(markdown_str.contains("**Authors**: Test Author"));
        assert!(markdown_str.contains("## Content"));
        assert!(markdown_str.contains("This is a test document."));
    }
}