//! PDF Source Plugin for Neural Document Flow
//!
//! This plugin handles PDF document extraction and processing using the DocumentSource trait.

use crate::{
    traits::{DocumentSource, source::{DocumentMetadata, ValidationResult, ValidationIssue, ValidationSeverity}},
    document::{Document, DocumentBuilder},
    error::ProcessingError,
    SourceResult,
};
use async_trait::async_trait;
use std::collections::HashMap;

/// PDF document source implementation
#[derive(Debug, Clone)]
pub struct PDFSource {
    name: String,
    version: String,
}

impl PDFSource {
    /// Create a new PDF source instance
    pub fn new() -> Self {
        Self {
            name: "PDF Source Plugin".to_string(),
            version: "1.0.0".to_string(),
        }
    }
    
    /// Extract text from PDF using basic extraction
    async fn extract_pdf_text(&self, pdf_data: &[u8]) -> Result<String, ProcessingError> {
        // In a real implementation, this would use a PDF library like pdf-extract or poppler
        // For now, return placeholder text
        if pdf_data.len() < 1000 {
            return Ok("Sample PDF text content extracted".to_string());
        }
        
        // Simulate PDF text extraction
        let text = format!("PDF Document Content\n\nExtracted from {} bytes of PDF data\n\nThis is a placeholder for actual PDF text extraction using libraries like pdf-extract or poppler-rs.", pdf_data.len());
        
        Ok(text)
    }
    
    /// Extract metadata from PDF
    async fn extract_pdf_metadata(&self, pdf_data: &[u8]) -> Result<HashMap<String, serde_json::Value>, ProcessingError> {
        let mut metadata = HashMap::new();
        
        // In a real implementation, this would extract actual PDF metadata
        metadata.insert("creator".to_string(), serde_json::Value::String("PDF Creator".to_string()));
        metadata.insert("producer".to_string(), serde_json::Value::String("PDF Producer".to_string()));
        metadata.insert("pdf_version".to_string(), serde_json::Value::String("1.4".to_string()));
        metadata.insert("page_count".to_string(), serde_json::Value::Number(serde_json::Number::from(1)));
        metadata.insert("file_size".to_string(), serde_json::Value::Number(serde_json::Number::from(pdf_data.len())));
        
        Ok(metadata)
    }
    
    /// Validate PDF file structure
    async fn validate_pdf(&self, pdf_data: &[u8]) -> Result<ValidationResult, ProcessingError> {
        let mut validation_result = ValidationResult::success();
        
        // Check PDF header
        if pdf_data.len() < 5 || !pdf_data.starts_with(b"%PDF-") {
            validation_result.add_issue(ValidationIssue {
                severity: ValidationSeverity::Error,
                message: "Invalid PDF header - file does not start with %PDF-".to_string(),
                suggestion: Some("Ensure the file is a valid PDF document".to_string()),
            });
        }
        
        // Check minimum file size
        if pdf_data.len() < 100 {
            validation_result.add_issue(ValidationIssue {
                severity: ValidationSeverity::Warning,
                message: "PDF file appears to be very small".to_string(),
                suggestion: Some("Verify the PDF file is not corrupted".to_string()),
            });
        }
        
        // Check for EOF marker
        if pdf_data.len() > 10 && !pdf_data.ends_with(b"%%EOF") {
            validation_result.add_issue(ValidationIssue {
                severity: ValidationSeverity::Warning,
                message: "PDF file does not end with %%EOF marker".to_string(),
                suggestion: Some("File may be truncated or corrupted".to_string()),
            });
        }
        
        // Estimate processing time based on file size
        let estimated_time = (pdf_data.len() as f64 / 1024.0 / 1024.0) * 2.0; // 2 seconds per MB
        validation_result.estimated_processing_time = Some(estimated_time);
        
        Ok(validation_result)
    }
}

impl Default for PDFSource {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DocumentSource for PDFSource {
    fn source_type(&self) -> &'static str {
        "pdf"
    }
    
    async fn can_handle(&self, input: &str) -> bool {
        // Check file extension
        if input.to_lowercase().ends_with(".pdf") {
            return true;
        }
        
        // Check if it's a file path that exists
        if let Ok(metadata) = std::fs::metadata(input) {
            if metadata.is_file() {
                // Try to read first few bytes to check PDF header
                if let Ok(mut file) = std::fs::File::open(input) {
                    use std::io::Read;
                    let mut buffer = [0; 5];
                    if file.read_exact(&mut buffer).is_ok() {
                        return buffer.starts_with(b"%PDF-");
                    }
                }
            }
        }
        
        false
    }
    
    async fn load_document(&self, input: &str) -> SourceResult<Document> {
        // Read PDF file
        let pdf_data = std::fs::read(input)
            .map_err(|e| ProcessingError::IoError(e.to_string()))?;
        
        // Validate PDF
        let validation_result = self.validate_pdf(&pdf_data).await?;
        if validation_result.has_critical_issues() {
            return Err(ProcessingError::ValidationError(
                format!("PDF validation failed: {:?}", validation_result.issues)
            ));
        }
        
        // Extract text content
        let text_content = self.extract_pdf_text(&pdf_data).await?;
        
        // Extract metadata
        let custom_metadata = self.extract_pdf_metadata(&pdf_data).await?;
        
        // Build document
        let document = DocumentBuilder::new()
            .source("pdf_source")
            .mime_type("application/pdf")
            .size(pdf_data.len() as u64)
            .with_text_content(text_content)
            .with_raw_content(pdf_data)
            .with_custom_metadata(custom_metadata)
            .build();
        
        Ok(document)
    }
    
    async fn get_metadata(&self, input: &str) -> SourceResult<DocumentMetadata> {
        let metadata = std::fs::metadata(input)
            .map_err(|e| ProcessingError::IoError(e.to_string()))?;
        
        let mut attributes = HashMap::new();
        attributes.insert("file_type".to_string(), serde_json::Value::String("PDF".to_string()));
        attributes.insert("readonly".to_string(), serde_json::Value::Bool(metadata.permissions().readonly()));
        
        Ok(DocumentMetadata {
            name: std::path::Path::new(input)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            size: Some(metadata.len()),
            mime_type: "application/pdf".to_string(),
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
        
        // Read and validate PDF
        let pdf_data = std::fs::read(input)
            .map_err(|e| ProcessingError::IoError(e.to_string()))?;
        
        self.validate_pdf(&pdf_data).await
    }
    
    fn supported_extensions(&self) -> Vec<&'static str> {
        vec!["pdf"]
    }
    
    fn supported_mime_types(&self) -> Vec<&'static str> {
        vec!["application/pdf"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_pdf_source_creation() {
        let source = PDFSource::new();
        assert_eq!(source.source_type(), "pdf");
        assert_eq!(source.supported_extensions(), vec!["pdf"]);
        assert_eq!(source.supported_mime_types(), vec!["application/pdf"]);
    }
    
    #[tokio::test]
    async fn test_can_handle_pdf() {
        let source = PDFSource::new();
        assert!(source.can_handle("document.pdf").await);
        assert!(source.can_handle("DOCUMENT.PDF").await);
        assert!(!source.can_handle("document.txt").await);
    }
    
    #[tokio::test]
    async fn test_pdf_validation() {
        let source = PDFSource::new();
        
        // Valid PDF header
        let valid_pdf = b"%PDF-1.4\n...content...%%EOF";
        let result = source.validate_pdf(valid_pdf).await.unwrap();
        assert!(result.is_valid);
        
        // Invalid PDF header
        let invalid_pdf = b"Not a PDF file";
        let result = source.validate_pdf(invalid_pdf).await.unwrap();
        assert!(!result.is_valid);
        assert!(result.issues.iter().any(|i| matches!(i.severity, ValidationSeverity::Error)));
    }
}