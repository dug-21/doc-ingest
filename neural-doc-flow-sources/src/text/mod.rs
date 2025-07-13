//! Plain text document source implementation
//!
//! Handles plain text files with various encodings.

use crate::traits::{BaseDocumentSource, SourceCapability, SourceMetadata, SourceConfig, SourceError, SourceResult};
use neural_doc_flow_core::prelude::*;
use async_trait::async_trait;
use std::collections::HashMap;
use std::path::Path;
use tokio::fs;
use tracing::{debug, info, warn};

/// Plain text document source implementation
pub struct TextSource {
    config: SourceConfig,
    metadata: SourceMetadata,
}

impl TextSource {
    /// Create a new text source
    pub fn new() -> Self {
        Self {
            config: SourceConfig::default(),
            metadata: SourceMetadata {
                id: "text-source".to_string(),
                name: "Plain Text Document Source".to_string(),
                version: "1.0.0".to_string(),
                capabilities: vec![
                    SourceCapability::PlainText,
                    SourceCapability::MetadataExtraction,
                ],
                file_extensions: vec![
                    "txt".to_string(),
                    "text".to_string(),
                    "log".to_string(),
                    "md".to_string(),
                    "markdown".to_string(),
                    "rst".to_string(),
                    "csv".to_string(),
                    "tsv".to_string(),
                ],
                mime_types: vec![
                    "text/plain".to_string(),
                    "text/markdown".to_string(),
                    "text/x-markdown".to_string(),
                    "text/csv".to_string(),
                    "text/tab-separated-values".to_string(),
                ],
            },
        }
    }
    
    /// Detect encoding of text content
    async fn detect_encoding(&self, data: &[u8]) -> String {
        // For now, use UTF-8 as default
        // In a real implementation, you might use encoding_rs or chardet
        "UTF-8".to_string()
    }
    
    /// Extract basic metadata from text content
    async fn extract_text_metadata(&self, content: &str) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        
        // Line count
        let line_count = content.lines().count();
        metadata.insert("line_count".to_string(), line_count.to_string());
        
        // Word count (approximate)
        let word_count = content.split_whitespace().count();
        metadata.insert("word_count".to_string(), word_count.to_string());
        
        // Character count
        metadata.insert("char_count".to_string(), content.len().to_string());
        
        // Check if it looks like markdown
        if content.contains("# ") || content.contains("## ") || content.contains("```") {
            metadata.insert("format_hint".to_string(), "markdown".to_string());
        }
        
        // Check if it looks like CSV
        if content.lines().take(5).all(|line| line.contains(',')) {
            metadata.insert("format_hint".to_string(), "csv".to_string());
        }
        
        // Check if it looks like TSV
        if content.lines().take(5).all(|line| line.contains('\t')) {
            metadata.insert("format_hint".to_string(), "tsv".to_string());
        }
        
        metadata
    }
}

impl Default for TextSource {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DocumentSource for TextSource {
    async fn process(&self, mut input: Document) -> Result<Document> {
        info!("Processing text document: {}", input.id);
        
        // Get content based on source type
        let raw_bytes = match &input.source {
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
                    .map_err(|e| anyhow::anyhow!("Failed to read text file: {}", e))?
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
            _ => return Err(anyhow::anyhow!("Unsupported source type for text")),
        };
        
        // Detect encoding
        let encoding = if let Some(enc) = &self.config.encoding {
            enc.clone()
        } else {
            self.detect_encoding(&raw_bytes).await
        };
        
        // Convert to string
        let text_content = match encoding.as_str() {
            "UTF-8" | "utf-8" => {
                String::from_utf8(raw_bytes)
                    .map_err(|e| anyhow::anyhow!("Failed to decode UTF-8: {}", e))?
            }
            _ => {
                // For other encodings, we'd use encoding_rs or similar
                // For now, fall back to lossy UTF-8
                warn!("Unsupported encoding: {}, falling back to lossy UTF-8", encoding);
                String::from_utf8_lossy(&raw_bytes).to_string()
            }
        };
        
        // Update document content
        input.content = text_content.clone();
        
        // Extract metadata if enabled
        if self.config.extract_metadata {
            let text_metadata = self.extract_text_metadata(&text_content).await;
            for (key, value) in text_metadata {
                input.metadata.insert(key, value);
            }
        }
        
        // Update document type based on extension or content
        input.doc_type = match input.metadata.get("format_hint").map(|s| s.as_str()) {
            Some("markdown") => DocumentType::Markdown,
            Some("csv") => DocumentType::Text, // Could add CSV type
            _ => DocumentType::Text,
        };
        
        // Add processing metadata
        input.metadata.insert("processor".to_string(), "text-source".to_string());
        input.metadata.insert("processor_version".to_string(), self.metadata.version.clone());
        input.metadata.insert("encoding".to_string(), encoding);
        
        info!("Successfully processed text document: {}", input.id);
        Ok(input)
    }
}

#[async_trait]
impl BaseDocumentSource for TextSource {
    fn metadata(&self) -> SourceMetadata {
        self.metadata.clone()
    }
    
    async fn can_handle(&self, input: &str) -> bool {
        // Check if input matches any of our extensions
        for ext in &self.metadata.file_extensions {
            if input.ends_with(&format!(".{}", ext)) {
                return true;
            }
        }
        
        // Check if it's a path that exists
        let path = Path::new(input);
        if path.exists() && path.is_file() {
            if let Some(ext) = path.extension() {
                let ext_str = ext.to_string_lossy().to_lowercase();
                return self.metadata.file_extensions.contains(&ext_str);
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
    async fn test_text_source_creation() {
        let source = TextSource::new();
        assert_eq!(source.metadata().id, "text-source");
        assert!(source.metadata().capabilities.contains(&SourceCapability::PlainText));
    }
    
    #[tokio::test]
    async fn test_can_handle() {
        let source = TextSource::new();
        
        assert!(source.can_handle("document.txt").await);
        assert!(source.can_handle("readme.md").await);
        assert!(source.can_handle("data.csv").await);
        assert!(source.can_handle("log.log").await);
        assert!(!source.can_handle("document.pdf").await);
        assert!(!source.can_handle("image.png").await);
    }
    
    #[tokio::test]
    async fn test_metadata_extraction() {
        let source = TextSource::new();
        let content = "Line 1\nLine 2\nLine 3\nThis is a test document.";
        
        let metadata = source.extract_text_metadata(content).await;
        
        assert_eq!(metadata.get("line_count"), Some(&"4".to_string()));
        assert_eq!(metadata.get("word_count"), Some(&"9".to_string()));
        assert!(metadata.contains_key("char_count"));
    }
    
    #[tokio::test]
    async fn test_process_text_document() {
        let source = TextSource::new();
        let content = "Hello, World!\nThis is a test.";
        
        let doc = Document {
            id: "test-doc".to_string(),
            source: DocumentSourceType::Memory(content.as_bytes().to_vec()),
            doc_type: DocumentType::Unknown,
            content: String::new(),
            metadata: HashMap::new(),
            embeddings: None,
            chunks: vec![],
            processing_state: ProcessingState::Pending,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        
        let result = source.process(doc).await.unwrap();
        
        assert_eq!(result.content, content);
        assert_eq!(result.doc_type, DocumentType::Text);
        assert_eq!(result.metadata.get("processor"), Some(&"text-source".to_string()));
        assert!(result.metadata.contains_key("line_count"));
        assert!(result.metadata.contains_key("word_count"));
    }
}