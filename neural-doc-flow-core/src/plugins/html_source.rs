//! HTML Source Plugin for Neural Document Flow
//!
//! This plugin handles HTML document extraction and processing.

use crate::{
    traits::{DocumentSource, source::{DocumentMetadata, ValidationResult, ValidationIssue, ValidationSeverity}},
    document::{Document, DocumentBuilder},
    error::{ProcessingError, SourceError},
    SourceResult,
};
use async_trait::async_trait;
use std::collections::HashMap;

/// HTML document source implementation
#[derive(Debug, Clone)]
pub struct HTMLSource {
    name: String,
    version: String,
}

impl HTMLSource {
    /// Create a new HTML source instance
    pub fn new() -> Self {
        Self {
            name: "HTML Source Plugin".to_string(),
            version: "1.0.0".to_string(),
        }
    }
    
    /// Extract text from HTML by removing tags
    async fn extract_html_text(&self, html_data: &[u8]) -> Result<String, ProcessingError> {
        let html_string = String::from_utf8_lossy(html_data);
        
        // In a real implementation, this would use an HTML parser like scraper or html5ever
        // For now, do basic tag removal
        let text = self.strip_html_tags(&html_string);
        
        Ok(text)
    }
    
    /// Basic HTML tag removal (in practice, use a proper HTML parser)
    fn strip_html_tags(&self, html: &str) -> String {
        let mut result = String::new();
        let mut in_tag = false;
        let mut chars = html.chars().peekable();
        
        while let Some(ch) = chars.next() {
            match ch {
                '<' => {
                    in_tag = true;
                    // Add space before certain closing tags for better text flow
                    if let Some(&'/') = chars.peek() {
                        if !result.is_empty() && !result.ends_with(' ') {
                            result.push(' ');
                        }
                    }
                }
                '>' => {
                    in_tag = false;
                    // Add space after certain tags
                    if !result.is_empty() && !result.ends_with(' ') {
                        result.push(' ');
                    }
                }
                _ if !in_tag => {
                    result.push(ch);
                }
                _ => {} // Skip characters inside tags
            }
        }
        
        // Clean up extra whitespace
        result.split_whitespace().collect::<Vec<_>>().join(" ")
    }
    
    /// Extract metadata from HTML
    async fn extract_html_metadata(&self, html_data: &[u8]) -> Result<HashMap<String, serde_json::Value>, ProcessingError> {
        let html_string = String::from_utf8_lossy(html_data);
        let mut metadata = HashMap::new();
        
        // Extract title
        if let Some(title) = self.extract_html_title(&html_string) {
            metadata.insert("title".to_string(), serde_json::Value::String(title));
        }
        
        // Extract meta description
        if let Some(description) = self.extract_meta_description(&html_string) {
            metadata.insert("description".to_string(), serde_json::Value::String(description));
        }
        
        // Extract meta keywords
        if let Some(keywords) = self.extract_meta_keywords(&html_string) {
            metadata.insert("keywords".to_string(), serde_json::Value::String(keywords));
        }
        
        // Basic document info
        metadata.insert("content_type".to_string(), serde_json::Value::String("HTML".to_string()));
        metadata.insert("file_size".to_string(), serde_json::Value::Number(serde_json::Number::from(html_data.len())));
        
        // Count basic HTML elements
        let tag_counts = self.count_html_tags(&html_string);
        metadata.insert("tag_counts".to_string(), serde_json::Value::Object(
            tag_counts.into_iter()
                .map(|(k, v)| (k, serde_json::Value::Number(serde_json::Number::from(v))))
                .collect()
        ));
        
        Ok(metadata)
    }
    
    /// Extract title from HTML
    fn extract_html_title(&self, html: &str) -> Option<String> {
        // Basic regex-like extraction (in practice, use proper HTML parser)
        let lower_html = html.to_lowercase();
        if let Some(start) = lower_html.find("<title>") {
            if let Some(end) = lower_html[start..].find("</title>") {
                let title_start = start + 7; // "<title>".len()
                let title_end = start + end;
                if title_start < title_end {
                    return Some(html[title_start..title_end].trim().to_string());
                }
            }
        }
        None
    }
    
    /// Extract meta description
    fn extract_meta_description(&self, html: &str) -> Option<String> {
        let lower_html = html.to_lowercase();
        if let Some(meta_start) = lower_html.find("<meta name=\"description\"") {
            if let Some(content_start) = lower_html[meta_start..].find("content=\"") {
                let content_pos = meta_start + content_start + 9; // "content=\"".len()
                if let Some(quote_end) = html[content_pos..].find('"') {
                    return Some(html[content_pos..content_pos + quote_end].to_string());
                }
            }
        }
        None
    }
    
    /// Extract meta keywords
    fn extract_meta_keywords(&self, html: &str) -> Option<String> {
        let lower_html = html.to_lowercase();
        if let Some(meta_start) = lower_html.find("<meta name=\"keywords\"") {
            if let Some(content_start) = lower_html[meta_start..].find("content=\"") {
                let content_pos = meta_start + content_start + 9; // "content=\"".len()
                if let Some(quote_end) = html[content_pos..].find('"') {
                    return Some(html[content_pos..content_pos + quote_end].to_string());
                }
            }
        }
        None
    }
    
    /// Count HTML tags
    fn count_html_tags(&self, html: &str) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        let tags = ["p", "div", "span", "h1", "h2", "h3", "h4", "h5", "h6", "a", "img", "table", "tr", "td"];
        
        for tag in &tags {
            let count = html.matches(&format!("<{}", tag)).count();
            if count > 0 {
                counts.insert(tag.to_string(), count);
            }
        }
        
        counts
    }
    
    /// Validate HTML file structure
    async fn validate_html(&self, html_data: &[u8]) -> Result<ValidationResult, SourceError> {
        let html_string = String::from_utf8_lossy(html_data);
        let mut validation_result = ValidationResult::success();
        
        // Check for basic HTML structure
        let lower_html = html_string.to_lowercase();
        
        if !lower_html.contains("<html") {
            validation_result.add_issue(ValidationIssue {
                severity: ValidationSeverity::Warning,
                message: "No <html> tag found".to_string(),
                suggestion: Some("Consider adding proper HTML structure".to_string()),
            });
        }
        
        if !lower_html.contains("<head") {
            validation_result.add_issue(ValidationIssue {
                severity: ValidationSeverity::Warning,
                message: "No <head> tag found".to_string(),
                suggestion: Some("Consider adding a <head> section".to_string()),
            });
        }
        
        if !lower_html.contains("<body") {
            validation_result.add_issue(ValidationIssue {
                severity: ValidationSeverity::Warning,
                message: "No <body> tag found".to_string(),
                suggestion: Some("Consider adding a <body> section".to_string()),
            });
        }
        
        // Check for unclosed tags (basic check)
        let open_tags = html_string.matches('<').count();
        let close_tags = html_string.matches('>').count();
        if open_tags != close_tags {
            validation_result.add_issue(ValidationIssue {
                severity: ValidationSeverity::Error,
                message: "Mismatched HTML tags detected".to_string(),
                suggestion: Some("Check for unclosed or malformed HTML tags".to_string()),
            });
        }
        
        // Estimate processing time
        let estimated_time = (html_data.len() as f64 / 1024.0 / 1024.0) * 0.5; // 0.5 seconds per MB
        validation_result.estimated_processing_time = Some(estimated_time);
        
        Ok(validation_result)
    }
}

impl Default for HTMLSource {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DocumentSource for HTMLSource {
    fn source_type(&self) -> &'static str {
        "html"
    }
    
    async fn can_handle(&self, input: &str) -> bool {
        // Check file extension
        let lower_input = input.to_lowercase();
        if lower_input.ends_with(".html") || lower_input.ends_with(".htm") {
            return true;
        }
        
        // Check if it's a URL
        if input.starts_with("http://") || input.starts_with("https://") {
            return true;
        }
        
        // Check if it's a file path that exists and contains HTML
        if let Ok(metadata) = std::fs::metadata(input) {
            if metadata.is_file() {
                // Try to read first few bytes to check for HTML content
                if let Ok(content) = std::fs::read_to_string(input) {
                    let lower_content = content.to_lowercase();
                    return lower_content.contains("<html") || lower_content.contains("<!doctype html");
                }
            }
        }
        
        false
    }
    
    async fn load_document(&self, input: &str) -> SourceResult<Document> {
        let html_data = if input.starts_with("http://") || input.starts_with("https://") {
            // For URLs, we'd use an HTTP client to fetch the content
            // For now, just return an error
            return Err(SourceError::UnsupportedFormat {
                format: "URL fetching not implemented".to_string()
            });
        } else {
            // Read HTML file
            std::fs::read(input)
                .map_err(|e| SourceError::DocumentNotFound { path: input.to_string() })?
        };
        
        // Validate HTML
        let validation_result = self.validate_html(&html_data).await?;
        if validation_result.has_critical_issues() {
            return Err(SourceError::ParseError {
                reason: format!("HTML validation failed: {:?}", validation_result.issues)
            });
        }
        
        // Extract text content
        let text_content = self.extract_html_text(&html_data).await?;
        
        // Extract metadata
        let custom_metadata = self.extract_html_metadata(&html_data).await?;
        
        // Build document
        let document = DocumentBuilder::new()
            .source("html_source")
            .mime_type("text/html")
            .size(html_data.len() as u64)
            .build();
        
        Ok(document)
    }
    
    async fn get_metadata(&self, input: &str) -> SourceResult<DocumentMetadata> {
        if input.starts_with("http://") || input.starts_with("https://") {
            // For URLs, we'd need to fetch headers
            return Err(SourceError::UnsupportedFormat {
                format: "URL metadata fetching not implemented".to_string()
            });
        }
        
        let metadata = std::fs::metadata(input)
            .map_err(|e| ProcessingError::IoError(e.to_string()))?;
        
        let mut attributes = HashMap::new();
        attributes.insert("file_type".to_string(), serde_json::Value::String("HTML".to_string()));
        attributes.insert("readonly".to_string(), serde_json::Value::Bool(metadata.permissions().readonly()));
        
        Ok(DocumentMetadata {
            name: std::path::Path::new(input)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            size: Some(metadata.len()),
            mime_type: "text/html".to_string(),
            modified: metadata.modified().ok().and_then(|t| {
                use std::time::SystemTime;
                let duration = t.duration_since(SystemTime::UNIX_EPOCH).ok()?;
                Some(chrono::DateTime::from_timestamp(duration.as_secs() as i64, 0)?)
            }),
            attributes,
        })
    }
    
    async fn validate(&self, input: &str) -> SourceResult<ValidationResult> {
        if input.starts_with("http://") || input.starts_with("https://") {
            // For URLs, we'd need to fetch and validate
            return Err(SourceError::UnsupportedFormat {
                format: "URL validation not implemented".to_string()
            });
        }
        
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
        
        // Read and validate HTML
        let html_data = std::fs::read(input)
            .map_err(|e| ProcessingError::IoError(e.to_string()))?;
        
        self.validate_html(&html_data).await
    }
    
    fn supported_extensions(&self) -> Vec<&'static str> {
        vec!["html", "htm"]
    }
    
    fn supported_mime_types(&self) -> Vec<&'static str> {
        vec!["text/html", "application/xhtml+xml"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_html_source_creation() {
        let source = HTMLSource::new();
        assert_eq!(source.source_type(), "html");
        assert_eq!(source.supported_extensions(), vec!["html", "htm"]);
    }
    
    #[tokio::test]
    async fn test_can_handle_html() {
        let source = HTMLSource::new();
        assert!(source.can_handle("document.html").await);
        assert!(source.can_handle("document.htm").await);
        assert!(source.can_handle("https://example.com").await);
        assert!(!source.can_handle("document.txt").await);
    }
    
    #[tokio::test]
    async fn test_strip_html_tags() {
        let source = HTMLSource::new();
        let html = "<p>Hello <b>World</b></p>";
        let text = source.strip_html_tags(html);
        assert_eq!(text, "Hello World");
    }
    
    #[tokio::test]
    async fn test_extract_html_title() {
        let source = HTMLSource::new();
        let html = "<html><head><title>Test Title</title></head><body></body></html>";
        let title = source.extract_html_title(html);
        assert_eq!(title, Some("Test Title".to_string()));
    }
}