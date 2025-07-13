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
    use neural_doc_flow_core::{DocumentMetadata};
    use uuid::Uuid;
    use chrono::Utc;
    
    #[tokio::test]
    async fn test_json_output() {
        let output = JsonOutput::new();
        
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
            tags: vec!["test".to_string()],
        };
        
        let document = Document {
            metadata,
            content: "Test content".to_string(),
            raw_content: b"Test content".to_vec(),
            extracted_text: "Test content".to_string(),
            structure: None,
            attachments: Vec::new(),
        };
        
        let bytes = output.generate_bytes(&document).await.unwrap();
        let json_str = String::from_utf8(bytes).unwrap();
        
        assert!(json_str.contains("Test Document"));
        assert!(json_str.contains("Test Author"));
        assert!(json_str.contains("Test content"));
    }
}