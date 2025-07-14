//! DOCX Source Plugin for Neural Document Flow
//!
//! This plugin handles Microsoft Word DOCX document extraction and processing.

use crate::{
    traits::{DocumentSource, source::{DocumentMetadata, ValidationResult, ValidationIssue, ValidationSeverity}},
    document::{Document, DocumentBuilder},
    error::ProcessingError,
    SourceResult,
};
use async_trait::async_trait;
use std::collections::HashMap;

/// DOCX document source implementation
#[derive(Debug, Clone)]
pub struct DOCXSource {
    name: String,
    version: String,
}

impl DOCXSource {
    /// Create a new DOCX source instance
    pub fn new() -> Self {
        Self {
            name: "DOCX Source Plugin".to_string(),
            version: "1.0.0".to_string(),
        }
    }
    
    /// Extract text from DOCX using basic XML parsing
    async fn extract_docx_text(&self, docx_data: &[u8]) -> Result<String, ProcessingError> {
        // In a real implementation, this would use a DOCX library like docx-rs
        // For now, return placeholder text
        if docx_data.len() < 1000 {
            return Ok("Sample DOCX text content extracted".to_string());
        }
        
        // Simulate DOCX text extraction
        let text = format!("Microsoft Word Document Content\n\nExtracted from {} bytes of DOCX data\n\nThis is a placeholder for actual DOCX text extraction using libraries like docx-rs or zip + XML parsing.", docx_data.len());
        
        Ok(text)
    }
    
    /// Extract metadata from DOCX
    async fn extract_docx_metadata(&self, docx_data: &[u8]) -> Result<HashMap<String, serde_json::Value>, ProcessingError> {
        let mut metadata = HashMap::new();
        
        // In a real implementation, this would extract actual DOCX metadata from core.xml
        metadata.insert("application".to_string(), serde_json::Value::String("Microsoft Word".to_string()));
        metadata.insert("document_type".to_string(), serde_json::Value::String("Word Document".to_string()));
        metadata.insert("docx_version".to_string(), serde_json::Value::String("2016".to_string()));
        metadata.insert("word_count".to_string(), serde_json::Value::Number(serde_json::Number::from(500)));
        metadata.insert("file_size".to_string(), serde_json::Value::Number(serde_json::Number::from(docx_data.len())));
        
        Ok(metadata)
    }
    
    /// Validate DOCX file structure
    async fn validate_docx(&self, docx_data: &[u8]) -> Result<ValidationResult, ProcessingError> {
        let mut validation_result = ValidationResult::success();
        
        // Check ZIP header (DOCX is a ZIP file)
        if docx_data.len() < 4 || !docx_data.starts_with(b"PK") {
            validation_result.add_issue(ValidationIssue {
                severity: ValidationSeverity::Error,
                message: "Invalid DOCX header - file does not start with ZIP signature".to_string(),
                suggestion: Some("Ensure the file is a valid DOCX document".to_string()),
            });
        }
        
        // Check minimum file size
        if docx_data.len() < 1000 {
            validation_result.add_issue(ValidationIssue {
                severity: ValidationSeverity::Warning,
                message: "DOCX file appears to be very small".to_string(),
                suggestion: Some("Verify the DOCX file is not corrupted".to_string()),
            });
        }
        
        // In a real implementation, would check for required DOCX internal structure
        // - [Content_Types].xml
        // - _rels/.rels
        // - word/document.xml
        // - word/_rels/document.xml.rels
        
        // Estimate processing time based on file size
        let estimated_time = (docx_data.len() as f64 / 1024.0 / 1024.0) * 1.5; // 1.5 seconds per MB
        validation_result.estimated_processing_time = Some(estimated_time);
        
        Ok(validation_result)
    }
}

impl Default for DOCXSource {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DocumentSource for DOCXSource {
    fn source_type(&self) -> &'static str {
        "docx"
    }
    
    async fn can_handle(&self, input: &str) -> bool {
        // Check file extension
        if input.to_lowercase().ends_with(".docx") {
            return true;
        }
        
        // Check if it's a file path that exists
        if let Ok(metadata) = std::fs::metadata(input) {
            if metadata.is_file() {
                // Try to read first few bytes to check ZIP header
                if let Ok(mut file) = std::fs::File::open(input) {
                    use std::io::Read;
                    let mut buffer = [0; 2];
                    if file.read_exact(&mut buffer).is_ok() {
                        return buffer.starts_with(b"PK");
                    }
                }
            }
        }
        
        false
    }
    
    async fn load_document(&self, input: &str) -> SourceResult<Document> {
        // Read DOCX file
        let docx_data = std::fs::read(input)
            .map_err(|e| ProcessingError::IoError(e.to_string()))?;
        
        // Validate DOCX
        let validation_result = self.validate_docx(&docx_data).await?;
        if validation_result.has_critical_issues() {
            return Err(ProcessingError::ValidationError(
                format!("DOCX validation failed: {:?}", validation_result.issues)
            ));
        }
        
        // Extract text content
        let text_content = self.extract_docx_text(&docx_data).await?;
        
        // Extract metadata
        let custom_metadata = self.extract_docx_metadata(&docx_data).await?;
        
        // Build document
        let document = DocumentBuilder::new()
            .source("docx_source")
            .mime_type("application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            .size(docx_data.len() as u64)
            .with_text_content(text_content)
            .with_raw_content(docx_data)
            .with_custom_metadata(custom_metadata)
            .build();
        
        Ok(document)
    }
    
    async fn get_metadata(&self, input: &str) -> SourceResult<DocumentMetadata> {
        let metadata = std::fs::metadata(input)
            .map_err(|e| ProcessingError::IoError(e.to_string()))?;
        
        let mut attributes = HashMap::new();
        attributes.insert("file_type".to_string(), serde_json::Value::String("DOCX".to_string()));
        attributes.insert("readonly".to_string(), serde_json::Value::Bool(metadata.permissions().readonly()));
        
        Ok(DocumentMetadata {
            name: std::path::Path::new(input)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            size: Some(metadata.len()),
            mime_type: "application/vnd.openxmlformats-officedocument.wordprocessingml.document".to_string(),
            modified: metadata.modified().ok().and_then(|t| {
                use std::time::SystemTime;
                let duration = t.duration_since(SystemTime::UNIX_EPOCH).ok()?;
                Some(chrono::DateTime::from_timestamp(duration.as_secs() as i64, 0)?)
            }),
            attributes,
        })
    }
    
    async fn validate(&self, input: &str) -> SourceResult<ValidationResult> {
        // Check if file exists
        if !std::path::Path::new(input).exists() {
            return Ok(ValidationResult::failure(vec![
                ValidationIssue {
                    severity: ValidationSeverity::Critical,
                    message: format!("File does not exist: {}", input),
                    suggestion: Some("Verify the file path is correct".to_string()),
                }
            ]));
        }
        
        // Read and validate DOCX
        let docx_data = std::fs::read(input)
            .map_err(|e| ProcessingError::IoError(e.to_string()))?;
        
        self.validate_docx(&docx_data).await
    }
    
    fn supported_extensions(&self) -> Vec<&'static str> {
        vec!["docx"]
    }
    
    fn supported_mime_types(&self) -> Vec<&'static str> {
        vec!["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_docx_source_creation() {
        let source = DOCXSource::new();
        assert_eq!(source.source_type(), "docx");
        assert_eq!(source.supported_extensions(), vec!["docx"]);
    }
    
    #[tokio::test]
    async fn test_can_handle_docx() {
        let source = DOCXSource::new();
        assert!(source.can_handle("document.docx").await);
        assert!(source.can_handle("DOCUMENT.DOCX").await);
        assert!(!source.can_handle("document.doc").await);
    }
    
    #[tokio::test]
    async fn test_docx_validation() {
        let source = DOCXSource::new();
        
        // Valid ZIP header (DOCX is a ZIP file)
        let valid_docx = b"PK\x03\x04...docx content...";
        let result = source.validate_docx(valid_docx).await.unwrap();
        assert!(result.is_valid);
        
        // Invalid ZIP header
        let invalid_docx = b"Not a DOCX file";
        let result = source.validate_docx(invalid_docx).await.unwrap();
        assert!(!result.is_valid);
        assert!(result.issues.iter().any(|i| matches!(i.severity, ValidationSeverity::Error)));
    }
}