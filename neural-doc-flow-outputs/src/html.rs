//! HTML output implementation

use crate::DocumentOutput;
use neural_doc_flow_core::{Document, Result};
use async_trait::async_trait;
use std::path::Path;
use tokio::fs;

/// HTML output formatter
#[derive(Debug, Default)]
pub struct HtmlOutput;

impl HtmlOutput {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DocumentOutput for HtmlOutput {
    fn name(&self) -> &str {
        "html"
    }
    
    fn supported_extensions(&self) -> &[&str] {
        &["html", "htm"]
    }
    
    fn mime_type(&self) -> &str {
        "text/html"
    }
    
    async fn generate(&self, document: &Document, output_path: &Path) -> Result<()> {
        let bytes = self.generate_bytes(document).await?;
        fs::write(output_path, bytes).await?;
        Ok(())
    }
    
    async fn generate_bytes(&self, document: &Document) -> Result<Vec<u8>> {
        let mut html = String::new();
        
        // HTML document structure
        html.push_str("<!DOCTYPE html>\n");
        html.push_str("<html lang=\"en\">\n");
        html.push_str("<head>\n");
        html.push_str("    <meta charset=\"UTF-8\">\n");
        html.push_str("    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n");
        html.push_str(&format!("    <title>{}</title>\n", html_escape(&document.metadata.title)));
        
        if let Some(author) = &document.metadata.author {
            html.push_str(&format!("    <meta name=\"author\" content=\"{}\">\n", html_escape(author)));
        }
        
        if !document.metadata.tags.is_empty() {
            html.push_str(&format!("    <meta name=\"keywords\" content=\"{}\">\n", 
                html_escape(&document.metadata.tags.join(", "))));
        }
        
        // Basic CSS styling
        html.push_str("    <style>\n");
        html.push_str("        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }\n");
        html.push_str("        .metadata { background: #f5f5f5; padding: 20px; border-radius: 5px; margin-bottom: 20px; }\n");
        html.push_str("        .metadata h2 { margin-top: 0; }\n");
        html.push_str("        .metadata-item { margin: 10px 0; }\n");
        html.push_str("        .content { margin: 20px 0; }\n");
        html.push_str("        .structure { background: #f9f9f9; padding: 15px; border-left: 4px solid #ccc; }\n");
        html.push_str("        pre { background: #f0f0f0; padding: 10px; overflow-x: auto; }\n");
        html.push_str("    </style>\n");
        html.push_str("</head>\n");
        html.push_str("<body>\n");
        
        // Document title
        html.push_str(&format!("    <h1>{}</h1>\n", html_escape(&document.metadata.title)));
        
        // Metadata section
        html.push_str("    <div class=\"metadata\">\n");
        html.push_str("        <h2>Metadata</h2>\n");
        html.push_str(&format!("        <div class=\"metadata-item\"><strong>ID:</strong> {}</div>\n", document.metadata.id));
        
        if let Some(author) = &document.metadata.author {
            html.push_str(&format!("        <div class=\"metadata-item\"><strong>Author:</strong> {}</div>\n", html_escape(author)));
        }
        
        html.push_str(&format!("        <div class=\"metadata-item\"><strong>Format:</strong> {}</div>\n", document.metadata.format));
        html.push_str(&format!("        <div class=\"metadata-item\"><strong>Size:</strong> {} bytes</div>\n", document.metadata.size));
        html.push_str(&format!("        <div class=\"metadata-item\"><strong>Created:</strong> {}</div>\n", 
            document.metadata.created_at.format("%Y-%m-%d %H:%M:%S UTC")));
        html.push_str(&format!("        <div class=\"metadata-item\"><strong>Modified:</strong> {}</div>\n", 
            document.metadata.modified_at.format("%Y-%m-%d %H:%M:%S UTC")));
        
        if let Some(language) = &document.metadata.language {
            html.push_str(&format!("        <div class=\"metadata-item\"><strong>Language:</strong> {}</div>\n", language));
        }
        
        if !document.metadata.tags.is_empty() {
            html.push_str(&format!("        <div class=\"metadata-item\"><strong>Tags:</strong> {}</div>\n", 
                html_escape(&document.metadata.tags.join(", "))));
        }
        
        if let Some(source_path) = &document.metadata.source_path {
            html.push_str(&format!("        <div class=\"metadata-item\"><strong>Source:</strong> {}</div>\n", 
                html_escape(&source_path.display().to_string())));
        }
        
        html.push_str("    </div>\n");
        
        // Content section
        html.push_str("    <div class=\"content\">\n");
        html.push_str("        <h2>Content</h2>\n");
        html.push_str(&format!("        <div>{}</div>\n", html_escape(&document.content).replace('\n', "<br>\n")));
        html.push_str("    </div>\n");
        
        // Extracted text section (if different from content)
        if document.extracted_text != document.content && !document.extracted_text.is_empty() {
            html.push_str("    <div class=\"content\">\n");
            html.push_str("        <h2>Extracted Text</h2>\n");
            html.push_str(&format!("        <div>{}</div>\n", html_escape(&document.extracted_text).replace('\n', "<br>\n")));
            html.push_str("    </div>\n");
        }
        
        // Structure section (if available)
        if let Some(structure) = &document.structure {
            html.push_str("    <div class=\"structure\">\n");
            html.push_str("        <h2>Structure</h2>\n");
            html.push_str("        <pre><code>");
            html.push_str(&html_escape(&serde_json::to_string_pretty(structure)?));
            html.push_str("</code></pre>\n");
            html.push_str("    </div>\n");
        }
        
        // Attachments section (if any)
        if !document.attachments.is_empty() {
            html.push_str("    <div class=\"content\">\n");
            html.push_str("        <h2>Attachments</h2>\n");
            html.push_str("        <ul>\n");
            for attachment in &document.attachments {
                html.push_str(&format!("            <li>{} ({} bytes)</li>\n", 
                    html_escape(&attachment.name), attachment.size));
            }
            html.push_str("        </ul>\n");
            html.push_str("    </div>\n");
        }
        
        html.push_str("</body>\n");
        html.push_str("</html>\n");
        
        Ok(html.into_bytes())
    }
}

fn html_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use neural_doc_flow_core::{DocumentMetadata};
    use uuid::Uuid;
    use chrono::Utc;
    
    #[test]
    fn test_html_escape() {
        assert_eq!(html_escape("Hello & <World>"), "Hello &amp; &lt;World&gt;");
        assert_eq!(html_escape("\"quoted\" 'text'"), "&quot;quoted&quot; &#x27;text&#x27;");
    }
    
    #[tokio::test]
    async fn test_html_output() {
        let output = HtmlOutput::new();
        
        let metadata = DocumentMetadata {
            id: Uuid::new_v4(),
            title: "Test Document & Example".to_string(),
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
            content: "Test content\n\nThis is a <test> document.".to_string(),
            raw_content: b"Test content".to_vec(),
            extracted_text: "Test content".to_string(),
            structure: None,
            attachments: Vec::new(),
        };
        
        let bytes = output.generate_bytes(&document).await.unwrap();
        let html_str = String::from_utf8(bytes).unwrap();
        
        assert!(html_str.contains("<!DOCTYPE html>"));
        assert!(html_str.contains("<title>Test Document &amp; Example</title>"));
        assert!(html_str.contains("<h1>Test Document &amp; Example</h1>"));
        assert!(html_str.contains("<strong>Author:</strong> Test Author"));
        assert!(html_str.contains("<strong>Tags:</strong> test, example"));
        assert!(html_str.contains("This is a &lt;test&gt; document."));
        assert!(html_str.contains("</html>"));
    }
}