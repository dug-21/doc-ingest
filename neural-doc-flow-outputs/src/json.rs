//! JSON output implementation

use crate::DocumentOutput;
use neural_doc_flow_core::{Document, Result};
use async_trait::async_trait;
use std::path::Path;
use tokio::fs;

/// JSON output formatter
#[derive(Debug, Default)]
pub struct JsonOutput {
    pretty: bool,
}

impl JsonOutput {
    pub fn new() -> Self {
        Self { pretty: true }
    }
    
    pub fn with_pretty(mut self, pretty: bool) -> Self {
        self.pretty = pretty;
        self
    }
}

#[async_trait]
impl DocumentOutput for JsonOutput {
    fn name(&self) -> &str {
        "json"
    }
    
    fn supported_extensions(&self) -> &[&str] {
        &["json"]
    }
    
    fn mime_type(&self) -> &str {
        "application/json"
    }
    
    async fn generate(&self, document: &Document, output_path: &Path) -> Result<()> {
        let bytes = self.generate_bytes(document).await?;
        fs::write(output_path, bytes).await?;
        Ok(())
    }
    
    async fn generate_bytes(&self, document: &Document) -> Result<Vec<u8>> {
        let json = if self.pretty {
            serde_json::to_string_pretty(document)?
        } else {
            serde_json::to_string(document)?
        };
        
        Ok(json.into_bytes())
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
    async fn test_json_output() {
        let output = JsonOutput::new();
        
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
            text: Some("Test content".to_string()),
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
        let json_str = String::from_utf8(bytes).unwrap();
        
        assert!(json_str.contains("Test Document"));
        assert!(json_str.contains("Test Author"));
        assert!(json_str.contains("Test content"));
    }
}