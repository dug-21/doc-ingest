//! Image Source Plugin for Neural Document Flow
//!
//! This plugin handles image document extraction using OCR and processing.

use crate::{
    traits::{DocumentSource, source::{DocumentMetadata, ValidationResult, ValidationIssue, ValidationSeverity}},
    document::{Document, DocumentBuilder},
    error::ProcessingError,
    SourceResult,
};
use async_trait::async_trait;
use std::collections::HashMap;

/// Image document source implementation
#[derive(Debug, Clone)]
pub struct ImageSource {
    name: String,
    version: String,
}

impl ImageSource {
    /// Create a new Image source instance
    pub fn new() -> Self {
        Self {
            name: "Image Source Plugin".to_string(),
            version: "1.0.0".to_string(),
        }
    }
    
    /// Extract text from image using OCR
    async fn extract_image_text(&self, image_data: &[u8]) -> Result<String, ProcessingError> {
        // In a real implementation, this would use OCR libraries like tesseract-rs
        // For now, return placeholder OCR text
        let text = format!("OCR Text Extracted from Image\n\nImage size: {} bytes\n\nThis is a placeholder for actual OCR text extraction using libraries like tesseract-rs or cloud OCR services.\n\nDetected text content would appear here in a real implementation.", image_data.len());
        
        Ok(text)
    }
    
    /// Extract metadata from image
    async fn extract_image_metadata(&self, image_data: &[u8]) -> Result<HashMap<String, serde_json::Value>, ProcessingError> {
        let mut metadata = HashMap::new();
        
        // Detect image format
        let format = self.detect_image_format(image_data);
        metadata.insert("image_format".to_string(), serde_json::Value::String(format.clone()));
        
        // Basic image info
        metadata.insert("content_type".to_string(), serde_json::Value::String("Image".to_string()));
        metadata.insert("file_size".to_string(), serde_json::Value::Number(serde_json::Number::from(image_data.len())));
        
        // In a real implementation, would extract:
        // - Image dimensions
        // - Color depth
        // - EXIF data
        // - Creation date
        // - Camera info
        
        // Placeholder metadata
        metadata.insert("estimated_width".to_string(), serde_json::Value::Number(serde_json::Number::from(1920)));
        metadata.insert("estimated_height".to_string(), serde_json::Value::Number(serde_json::Number::from(1080)));
        metadata.insert("has_text".to_string(), serde_json::Value::Bool(true));
        metadata.insert("ocr_confidence".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(0.85).unwrap()));
        
        Ok(metadata)
    }
    
    /// Detect image format from header bytes
    fn detect_image_format(&self, image_data: &[u8]) -> String {
        if image_data.len() < 8 {
            return "unknown".to_string();
        }
        
        // Check common image format signatures
        if image_data.starts_with(&[0xFF, 0xD8, 0xFF]) {
            "JPEG".to_string()
        } else if image_data.starts_with(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) {
            "PNG".to_string()
        } else if image_data.starts_with(b"GIF87a") || image_data.starts_with(b"GIF89a") {
            "GIF".to_string()
        } else if image_data.starts_with(b"RIFF") && image_data.len() > 11 && &image_data[8..12] == b"WEBP" {
            "WEBP".to_string()
        } else if image_data.starts_with(b"BM") {
            "BMP".to_string()
        } else if image_data.starts_with(&[0x49, 0x49, 0x2A, 0x00]) || image_data.starts_with(&[0x4D, 0x4D, 0x00, 0x2A]) {
            "TIFF".to_string()
        } else {
            "unknown".to_string()
        }
    }
    
    /// Validate image file structure
    async fn validate_image(&self, image_data: &[u8]) -> Result<ValidationResult, ProcessingError> {
        let mut validation_result = ValidationResult::success();
        
        // Check minimum file size
        if image_data.len() < 100 {
            validation_result.add_issue(ValidationIssue {
                severity: ValidationSeverity::Error,
                message: "Image file appears to be too small".to_string(),
                suggestion: Some("Verify the image file is not corrupted".to_string()),
            });
        }
        
        // Check if we can detect the format
        let format = self.detect_image_format(image_data);
        if format == "unknown" {
            validation_result.add_issue(ValidationIssue {
                severity: ValidationSeverity::Warning,
                message: "Unable to detect image format".to_string(),
                suggestion: Some("Ensure the file is a valid image format".to_string()),
            });
        }
        
        // Check maximum file size (100MB limit)
        if image_data.len() > 100 * 1024 * 1024 {
            validation_result.add_issue(ValidationIssue {
                severity: ValidationSeverity::Warning,
                message: "Image file is very large".to_string(),
                suggestion: Some("Consider compressing the image for better performance".to_string()),
            });
        }
        
        // Estimate OCR processing time based on file size
        let estimated_time = (image_data.len() as f64 / 1024.0 / 1024.0) * 5.0; // 5 seconds per MB for OCR
        validation_result.estimated_processing_time = Some(estimated_time);
        
        Ok(validation_result)
    }
    
    /// Get MIME type for image format
    fn get_mime_type(&self, format: &str) -> &'static str {
        match format.to_lowercase().as_str() {
            "jpeg" => "image/jpeg",
            "png" => "image/png",
            "gif" => "image/gif",
            "webp" => "image/webp",
            "bmp" => "image/bmp",
            "tiff" => "image/tiff",
            _ => "application/octet-stream",
        }
    }
}

impl Default for ImageSource {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DocumentSource for ImageSource {
    fn source_type(&self) -> &'static str {
        "image"
    }
    
    async fn can_handle(&self, input: &str) -> bool {
        // Check file extension
        let lower_input = input.to_lowercase();
        let image_extensions = ["jpg", "jpeg", "png", "gif", "webp", "bmp", "tiff", "tif"];
        
        for ext in &image_extensions {
            if lower_input.ends_with(&format!(".{}", ext)) {
                return true;
            }
        }
        
        // Check if it's a file path that exists and has image format
        if let Ok(metadata) = std::fs::metadata(input) {
            if metadata.is_file() {
                // Try to read first few bytes to check image headers
                if let Ok(mut file) = std::fs::File::open(input) {
                    use std::io::Read;
                    let mut buffer = [0; 12];
                    if file.read(&mut buffer).is_ok() {
                        let format = self.detect_image_format(&buffer);
                        return format != "unknown";
                    }
                }
            }
        }
        
        false
    }
    
    async fn load_document(&self, input: &str) -> SourceResult<Document> {
        // Read image file
        let image_data = std::fs::read(input)
            .map_err(|e| ProcessingError::IoError(e.to_string()))?;
        
        // Validate image
        let validation_result = self.validate_image(&image_data).await?;
        if validation_result.has_critical_issues() {
            return Err(ProcessingError::ValidationError(
                format!("Image validation failed: {:?}", validation_result.issues)
            ));
        }
        
        // Extract text content using OCR
        let text_content = self.extract_image_text(&image_data).await?;
        
        // Extract metadata
        let custom_metadata = self.extract_image_metadata(&image_data).await?;
        
        // Detect format and get MIME type
        let format = self.detect_image_format(&image_data);
        let mime_type = self.get_mime_type(&format);
        
        // Build document
        let document = DocumentBuilder::new()
            .source("image_source")
            .mime_type(mime_type)
            .size(image_data.len() as u64)
            .with_text_content(text_content)
            .with_raw_content(image_data)
            .with_custom_metadata(custom_metadata)
            .build();
        
        Ok(document)
    }
    
    async fn get_metadata(&self, input: &str) -> SourceResult<DocumentMetadata> {
        let metadata = std::fs::metadata(input)
            .map_err(|e| ProcessingError::IoError(e.to_string()))?;
        
        // Read a small portion to detect format
        let mut file = std::fs::File::open(input)
            .map_err(|e| ProcessingError::IoError(e.to_string()))?;
        
        use std::io::Read;
        let mut buffer = [0; 12];
        file.read(&mut buffer).map_err(|e| ProcessingError::IoError(e.to_string()))?;
        
        let format = self.detect_image_format(&buffer);
        let mime_type = self.get_mime_type(&format);
        
        let mut attributes = HashMap::new();
        attributes.insert("file_type".to_string(), serde_json::Value::String("Image".to_string()));
        attributes.insert("image_format".to_string(), serde_json::Value::String(format));
        attributes.insert("readonly".to_string(), serde_json::Value::Bool(metadata.permissions().readonly()));
        
        Ok(DocumentMetadata {
            name: std::path::Path::new(input)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            size: Some(metadata.len()),
            mime_type: mime_type.to_string(),
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
        
        // Read and validate image
        let image_data = std::fs::read(input)
            .map_err(|e| ProcessingError::IoError(e.to_string()))?;
        
        self.validate_image(&image_data).await
    }
    
    fn supported_extensions(&self) -> Vec<&'static str> {
        vec!["jpg", "jpeg", "png", "gif", "webp", "bmp", "tiff", "tif"]
    }
    
    fn supported_mime_types(&self) -> Vec<&'static str> {
        vec![
            "image/jpeg",
            "image/png",
            "image/gif",
            "image/webp",
            "image/bmp",
            "image/tiff"
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_image_source_creation() {
        let source = ImageSource::new();
        assert_eq!(source.source_type(), "image");
        assert!(source.supported_extensions().contains(&"jpg"));
        assert!(source.supported_extensions().contains(&"png"));
    }
    
    #[tokio::test]
    async fn test_can_handle_image() {
        let source = ImageSource::new();
        assert!(source.can_handle("photo.jpg").await);
        assert!(source.can_handle("image.png").await);
        assert!(source.can_handle("animation.gif").await);
        assert!(!source.can_handle("document.txt").await);
    }
    
    #[tokio::test]
    async fn test_detect_image_format() {
        let source = ImageSource::new();
        
        // JPEG signature
        let jpeg_data = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46];
        assert_eq!(source.detect_image_format(&jpeg_data), "JPEG");
        
        // PNG signature
        let png_data = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        assert_eq!(source.detect_image_format(&png_data), "PNG");
        
        // Unknown format
        let unknown_data = [0x00, 0x00, 0x00, 0x00];
        assert_eq!(source.detect_image_format(&unknown_data), "unknown");
    }
    
    #[tokio::test]
    async fn test_get_mime_type() {
        let source = ImageSource::new();
        assert_eq!(source.get_mime_type("JPEG"), "image/jpeg");
        assert_eq!(source.get_mime_type("PNG"), "image/png");
        assert_eq!(source.get_mime_type("unknown"), "application/octet-stream");
    }
}