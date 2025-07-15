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
        let title = document.metadata.title.as_deref().unwrap_or("Untitled Document");
        html.push_str(&format!("    <title>{}</title>\n", html_escape(title)));
        
        if !document.metadata.authors.is_empty() {
            html.push_str(&format!("    <meta name=\"author\" content=\"{}\">\n", html_escape(&document.metadata.authors.join(", "))));
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
        let title = document.metadata.title.as_deref().unwrap_or("Untitled Document");
        html.push_str(&format!("    <h1>{}</h1>\n", html_escape(title)));
        
        // Metadata section
        html.push_str("    <div class=\"metadata\">\n");
        html.push_str("        <h2>Metadata</h2>\n");
        html.push_str(&format!("        <div class=\"metadata-item\"><strong>ID:</strong> {}</div>\n", document.id));
        
        if !document.metadata.authors.is_empty() {
            html.push_str(&format!("        <div class=\"metadata-item\"><strong>Authors:</strong> {}</div>\n", html_escape(&document.metadata.authors.join(", "))));
        }
        
        html.push_str(&format!("        <div class=\"metadata-item\"><strong>MIME Type:</strong> {}</div>\n", document.metadata.mime_type));
        if let Some(size) = document.metadata.size {
            html.push_str(&format!("        <div class=\"metadata-item\"><strong>Size:</strong> {} bytes</div>\n", size));
        }
        html.push_str(&format!("        <div class=\"metadata-item\"><strong>Created:</strong> {}</div>\n", 
            document.created_at.format("%Y-%m-%d %H:%M:%S UTC")));
        html.push_str(&format!("        <div class=\"metadata-item\"><strong>Modified:</strong> {}</div>\n", 
            document.updated_at.format("%Y-%m-%d %H:%M:%S UTC")));
        
        if let Some(language) = &document.metadata.language {
            html.push_str(&format!("        <div class=\"metadata-item\"><strong>Language:</strong> {}</div>\n", language));
        }
        
        html.push_str(&format!("        <div class=\"metadata-item\"><strong>Source:</strong> {}</div>\n", 
            html_escape(&document.metadata.source)));
        
        html.push_str("    </div>\n");
        
        // Content section
        html.push_str("    <div class=\"content\">\n");
        html.push_str("        <h2>Content</h2>\n");
        if let Some(text) = &document.content.text {
            html.push_str(&format!("        <div>{}</div>\n", html_escape(text).replace('\n', "<br>\n")));
        } else {
            let content_str = document.content.to_string();
            html.push_str(&format!("        <div>{}</div>\n", html_escape(&content_str).replace('\n', "<br>\n")));
        }
        html.push_str("    </div>\n");
        
        // Images section (if any)
        if !document.content.images.is_empty() {
            html.push_str("    <div class=\"content\">\n");
            html.push_str("        <h2>Images</h2>\n");
            html.push_str("        <ul>\n");
            for image in &document.content.images {
                html.push_str(&format!("            <li>{} ({}x{})</li>\n", 
                    html_escape(&image.id), image.width, image.height));
            }
            html.push_str("        </ul>\n");
            html.push_str("    </div>\n");
        }
        
        // Tables section (if any)
        if !document.content.tables.is_empty() {
            html.push_str("    <div class=\"content\">\n");
            html.push_str("        <h2>Tables</h2>\n");
            for table in &document.content.tables {
                html.push_str(&format!("        <h3>{}</h3>\n", html_escape(&table.id)));
                html.push_str("        <table border=\"1\">\n");
                
                // Headers
                html.push_str("            <tr>\n");
                for header in &table.headers {
                    html.push_str(&format!("                <th>{}</th>\n", html_escape(header)));
                }
                html.push_str("            </tr>\n");
                
                // Rows
                for row in &table.rows {
                    html.push_str("            <tr>\n");
                    for cell in row {
                        html.push_str(&format!("                <td>{}</td>\n", html_escape(cell)));
                    }
                    html.push_str("            </tr>\n");
                }
                html.push_str("        </table>\n");
            }
            html.push_str("    </div>\n");
        }
        
        // Structure section
        if document.structure.page_count.is_some() || !document.structure.sections.is_empty() {
            html.push_str("    <div class=\"structure\">\n");
            html.push_str("        <h2>Document Structure</h2>\n");
            html.push_str("        <pre><code>");
            html.push_str(&html_escape(&serde_json::to_string_pretty(&document.structure)?));
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
                    html_escape(&attachment.name), attachment.data.len()));
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
    use neural_doc_flow_core::{DocumentMetadata, DocumentContent, DocumentStructure, DocumentType, DocumentSourceType};
    use uuid::Uuid;
    use chrono::Utc;
    use std::collections::HashMap;
    
    #[test]
    fn test_html_escape() {
        assert_eq!(html_escape("Hello & <World>"), "Hello &amp; &lt;World&gt;");
        assert_eq!(html_escape("\"quoted\" 'text'"), "&quot;quoted&quot; &#x27;text&#x27;");
    }
    
    #[tokio::test]
    async fn test_html_output() {
        let output = HtmlOutput::new();
        
        let metadata = DocumentMetadata {
            title: Some("Test Document & Example".to_string()),
            authors: vec!["Test Author".to_string()],
            source: "test.txt".to_string(),
            mime_type: "text/plain".to_string(),
            size: Some(100),
            language: Some("en".to_string()),
            custom: HashMap::new(),
        };
        
        let content = DocumentContent {
            text: Some("Test content\n\nThis is a <test> document.".to_string()),
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
        let html_str = String::from_utf8(bytes).unwrap();
        
        assert!(html_str.contains("<!DOCTYPE html>"));
        assert!(html_str.contains("<title>Test Document &amp; Example</title>"));
        assert!(html_str.contains("<h1>Test Document &amp; Example</h1>"));
        assert!(html_str.contains("<strong>Authors:</strong> Test Author"));
        assert!(html_str.contains("This is a &lt;test&gt; document."));
        assert!(html_str.contains("</html>"));
    }
}