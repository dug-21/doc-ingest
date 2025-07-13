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
        markdown.push_str(&format!("# {}\n\n", document.metadata.title));
        
        // Metadata section
        markdown.push_str("## Metadata\n\n");
        markdown.push_str(&format!("- **ID**: {}\n", document.metadata.id));
        if let Some(author) = &document.metadata.author {
            markdown.push_str(&format!("- **Author**: {}\n", author));
        }
        markdown.push_str(&format!("- **Format**: {}\n", document.metadata.format));
        markdown.push_str(&format!("- **Size**: {} bytes\n", document.metadata.size));
        markdown.push_str(&format!("- **Created**: {}\n", document.metadata.created_at.format("%Y-%m-%d %H:%M:%S UTC")));
        markdown.push_str(&format!("- **Modified**: {}\n", document.metadata.modified_at.format("%Y-%m-%d %H:%M:%S UTC")));
        
        if let Some(language) = &document.metadata.language {
            markdown.push_str(&format!("- **Language**: {}\n", language));
        }
        
        if !document.metadata.tags.is_empty() {
            markdown.push_str(&format!("- **Tags**: {}\n", document.metadata.tags.join(", ")));
        }
        
        if let Some(source_path) = &document.metadata.source_path {
            markdown.push_str(&format!("- **Source**: {}\n", source_path.display()));
        }
        
        markdown.push('\n');
        
        // Content section
        markdown.push_str("## Content\n\n");
        markdown.push_str(&document.content);
        markdown.push('\n');
        
        // Extracted text section (if different from content)
        if document.extracted_text != document.content && !document.extracted_text.is_empty() {
            markdown.push_str("\n## Extracted Text\n\n");
            markdown.push_str(&document.extracted_text);
            markdown.push('\n');
        }
        
        // Structure section (if available)
        if let Some(structure) = &document.structure {
            markdown.push_str("\n## Structure\n\n");
            markdown.push_str(&format!("```json\n{}\n```\n", serde_json::to_string_pretty(structure)?));
        }
        
        // Attachments section (if any)
        if !document.attachments.is_empty() {
            markdown.push_str("\n## Attachments\n\n");
            for (i, attachment) in document.attachments.iter().enumerate() {
                markdown.push_str(&format!("{}. {} ({} bytes)\n", i + 1, attachment.name, attachment.size));
            }
            markdown.push('\n');
        }
        
        Ok(markdown.into_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neural_doc_flow_core::{DocumentMetadata};
    use uuid::Uuid;
    use chrono::Utc;
    
    #[tokio::test]
    async fn test_markdown_output() {
        let output = MarkdownOutput::new();
        
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
        let markdown_str = String::from_utf8(bytes).unwrap();
        
        assert!(markdown_str.contains("# Test Document"));
        assert!(markdown_str.contains("## Metadata"));
        assert!(markdown_str.contains("**Author**: Test Author"));
        assert!(markdown_str.contains("**Tags**: test, example"));
        assert!(markdown_str.contains("## Content"));
        assert!(markdown_str.contains("This is a test document."));
    }
}