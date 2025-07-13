//! XML output implementation

use crate::DocumentOutput;
use neural_doc_flow_core::{Document, Result};
use async_trait::async_trait;
use std::path::Path;
use tokio::fs;

/// XML output formatter
#[derive(Debug, Default)]
pub struct XmlOutput;

impl XmlOutput {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DocumentOutput for XmlOutput {
    fn name(&self) -> &str {
        "xml"
    }
    
    fn supported_extensions(&self) -> &[&str] {
        &["xml"]
    }
    
    fn mime_type(&self) -> &str {
        "application/xml"
    }
    
    async fn generate(&self, document: &Document, output_path: &Path) -> Result<()> {
        let bytes = self.generate_bytes(document).await?;
        fs::write(output_path, bytes).await?;
        Ok(())
    }
    
    async fn generate_bytes(&self, document: &Document) -> Result<Vec<u8>> {
        let mut xml = String::new();
        
        // XML declaration
        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        
        // Root document element
        xml.push_str("<document>\n");
        
        // Metadata section
        xml.push_str("  <metadata>\n");
        xml.push_str(&format!("    <id>{}</id>\n", document.metadata.id));
        xml.push_str(&format!("    <title>{}</title>\n", xml_escape(&document.metadata.title)));
        
        if let Some(author) = &document.metadata.author {
            xml.push_str(&format!("    <author>{}</author>\n", xml_escape(author)));
        }
        
        xml.push_str(&format!("    <format>{}</format>\n", xml_escape(&document.metadata.format)));
        xml.push_str(&format!("    <size>{}</size>\n", document.metadata.size));
        xml.push_str(&format!("    <created>{}</created>\n", document.metadata.created_at.to_rfc3339()));
        xml.push_str(&format!("    <modified>{}</modified>\n", document.metadata.modified_at.to_rfc3339()));
        
        if let Some(language) = &document.metadata.language {
            xml.push_str(&format!("    <language>{}</language>\n", xml_escape(language)));
        }
        
        if !document.metadata.tags.is_empty() {
            xml.push_str("    <tags>\n");
            for tag in &document.metadata.tags {
                xml.push_str(&format!("      <tag>{}</tag>\n", xml_escape(tag)));
            }
            xml.push_str("    </tags>\n");
        }
        
        if let Some(source_path) = &document.metadata.source_path {
            xml.push_str(&format!("    <source>{}</source>\n", xml_escape(&source_path.display().to_string())));
        }
        
        xml.push_str("  </metadata>\n");
        
        // Content section
        xml.push_str("  <content>\n");
        xml.push_str(&format!("    <![CDATA[{}]]>\n", document.content));
        xml.push_str("  </content>\n");
        
        // Extracted text section (if different from content)
        if document.extracted_text != document.content && !document.extracted_text.is_empty() {
            xml.push_str("  <extracted_text>\n");
            xml.push_str(&format!("    <![CDATA[{}]]>\n", document.extracted_text));
            xml.push_str("  </extracted_text>\n");
        }
        
        // Raw content (base64 encoded)
        if !document.raw_content.is_empty() {
            xml.push_str("  <raw_content encoding=\"base64\">\n");
            xml.push_str(&format!("    {}\n", base64_encode(&document.raw_content)));
            xml.push_str("  </raw_content>\n");
        }
        
        // Structure section (if available)
        if let Some(structure) = &document.structure {
            xml.push_str("  <structure>\n");
            xml.push_str(&format!("    <![CDATA[{}]]>\n", serde_json::to_string_pretty(structure)?));
            xml.push_str("  </structure>\n");
        }
        
        // Attachments section (if any)
        if !document.attachments.is_empty() {
            xml.push_str("  <attachments>\n");
            for attachment in &document.attachments {
                xml.push_str("    <attachment>\n");
                xml.push_str(&format!("      <name>{}</name>\n", xml_escape(&attachment.name)));
                xml.push_str(&format!("      <size>{}</size>\n", attachment.size));
                xml.push_str(&format!("      <mime_type>{}</mime_type>\n", xml_escape(&attachment.mime_type)));
                if !attachment.data.is_empty() {
                    xml.push_str("      <data encoding=\"base64\">\n");
                    xml.push_str(&format!("        {}\n", base64_encode(&attachment.data)));
                    xml.push_str("      </data>\n");
                }
                xml.push_str("    </attachment>\n");
            }
            xml.push_str("  </attachments>\n");
        }
        
        xml.push_str("</document>\n");
        
        Ok(xml.into_bytes())
    }
}

fn xml_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn base64_encode(data: &[u8]) -> String {
    // Simple base64 encoding - in real implementation, use base64 crate
    use std::collections::HashMap;
    
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    
    let mut result = String::new();
    let mut i = 0;
    
    while i < data.len() {
        let b1 = data[i];
        let b2 = if i + 1 < data.len() { data[i + 1] } else { 0 };
        let b3 = if i + 2 < data.len() { data[i + 2] } else { 0 };
        
        let n = ((b1 as u32) << 16) | ((b2 as u32) << 8) | (b3 as u32);
        
        result.push(CHARS[((n >> 18) & 63) as usize] as char);
        result.push(CHARS[((n >> 12) & 63) as usize] as char);
        
        if i + 1 < data.len() {
            result.push(CHARS[((n >> 6) & 63) as usize] as char);
        } else {
            result.push('=');
        }
        
        if i + 2 < data.len() {
            result.push(CHARS[(n & 63) as usize] as char);
        } else {
            result.push('=');
        }
        
        i += 3;
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use neural_doc_flow_core::{DocumentMetadata};
    use uuid::Uuid;
    use chrono::Utc;
    
    #[test]
    fn test_xml_escape() {
        assert_eq!(xml_escape("Hello & <World>"), "Hello &amp; &lt;World&gt;");
        assert_eq!(xml_escape("\"quoted\" 'text'"), "&quot;quoted&quot; &apos;text&apos;");
    }
    
    #[test]
    fn test_base64_encode() {
        assert_eq!(base64_encode(b"hello"), "aGVsbG8=");
        assert_eq!(base64_encode(b""), "");
    }
    
    #[tokio::test]
    async fn test_xml_output() {
        let output = XmlOutput::new();
        
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
        let xml_str = String::from_utf8(bytes).unwrap();
        
        assert!(xml_str.contains("<?xml version=\"1.0\" encoding=\"UTF-8\"?>"));
        assert!(xml_str.contains("<document>"));
        assert!(xml_str.contains("<title>Test Document &amp; Example</title>"));
        assert!(xml_str.contains("<author>Test Author</author>"));
        assert!(xml_str.contains("<tag>test</tag>"));
        assert!(xml_str.contains("<tag>example</tag>"));
        assert!(xml_str.contains("This is a <test> document."));
        assert!(xml_str.contains("</document>"));
    }
}