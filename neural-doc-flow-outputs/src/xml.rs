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
        xml.push_str(&format!("    <id>{}</id>\n", document.id));
        let title = document.metadata.title.as_deref().unwrap_or("Untitled Document");
        xml.push_str(&format!("    <title>{}</title>\n", xml_escape(title)));
        
        if !document.metadata.authors.is_empty() {
            xml.push_str("    <authors>\n");
            for author in &document.metadata.authors {
                xml.push_str(&format!("      <author>{}</author>\n", xml_escape(author)));
            }
            xml.push_str("    </authors>\n");
        }
        
        xml.push_str(&format!("    <mime_type>{}</mime_type>\n", xml_escape(&document.metadata.mime_type)));
        if let Some(size) = document.metadata.size {
            xml.push_str(&format!("    <size>{}</size>\n", size));
        }
        xml.push_str(&format!("    <created>{}</created>\n", document.created_at.to_rfc3339()));
        xml.push_str(&format!("    <modified>{}</modified>\n", document.updated_at.to_rfc3339()));
        
        if let Some(language) = &document.metadata.language {
            xml.push_str(&format!("    <language>{}</language>\n", xml_escape(language)));
        }
        
        xml.push_str(&format!("    <source>{}</source>\n", xml_escape(&document.metadata.source)));
        
        xml.push_str("  </metadata>\n");
        
        // Content section
        xml.push_str("  <content>\n");
        if let Some(text) = &document.content.text {
            xml.push_str(&format!("    <![CDATA[{}]]>\n", text));
        } else {
            xml.push_str(&format!("    <![CDATA[{}]]>\n", document.content.to_string()));
        }
        xml.push_str("  </content>\n");
        
        // Images section (if any)
        if !document.content.images.is_empty() {
            xml.push_str("  <images>\n");
            for image in &document.content.images {
                xml.push_str("    <image>\n");
                xml.push_str(&format!("      <id>{}</id>\n", xml_escape(&image.id)));
                xml.push_str(&format!("      <format>{}</format>\n", xml_escape(&image.format)));
                xml.push_str(&format!("      <width>{}</width>\n", image.width));
                xml.push_str(&format!("      <height>{}</height>\n", image.height));
                if let Some(caption) = &image.caption {
                    xml.push_str(&format!("      <caption>{}</caption>\n", xml_escape(caption)));
                }
                xml.push_str("    </image>\n");
            }
            xml.push_str("  </images>\n");
        }
        
        // Tables section (if any)
        if !document.content.tables.is_empty() {
            xml.push_str("  <tables>\n");
            for table in &document.content.tables {
                xml.push_str("    <table>\n");
                xml.push_str(&format!("      <id>{}</id>\n", xml_escape(&table.id)));
                if let Some(caption) = &table.caption {
                    xml.push_str(&format!("      <caption>{}</caption>\n", xml_escape(caption)));
                }
                xml.push_str("      <headers>\n");
                for header in &table.headers {
                    xml.push_str(&format!("        <header>{}</header>\n", xml_escape(header)));
                }
                xml.push_str("      </headers>\n");
                xml.push_str("      <rows>\n");
                for row in &table.rows {
                    xml.push_str("        <row>\n");
                    for cell in row {
                        xml.push_str(&format!("          <cell>{}</cell>\n", xml_escape(cell)));
                    }
                    xml.push_str("        </row>\n");
                }
                xml.push_str("      </rows>\n");
                xml.push_str("    </table>\n");
            }
            xml.push_str("  </tables>\n");
        }
        
        // Raw content (base64 encoded)
        if !document.raw_content.is_empty() {
            xml.push_str("  <raw_content encoding=\"base64\">\n");
            xml.push_str(&format!("    {}\n", base64_encode(&document.raw_content)));
            xml.push_str("  </raw_content>\n");
        }
        
        // Structure section
        if document.structure.page_count.is_some() || !document.structure.sections.is_empty() {
            xml.push_str("  <structure>\n");
            xml.push_str(&format!("    <![CDATA[{}]]>\n", serde_json::to_string_pretty(&document.structure)?));
            xml.push_str("  </structure>\n");
        }
        
        // Attachments section (if any)
        if !document.attachments.is_empty() {
            xml.push_str("  <attachments>\n");
            for attachment in &document.attachments {
                xml.push_str("    <attachment>\n");
                xml.push_str(&format!("      <id>{}</id>\n", xml_escape(&attachment.id)));
                xml.push_str(&format!("      <name>{}</name>\n", xml_escape(&attachment.name)));
                xml.push_str(&format!("      <size>{}</size>\n", attachment.data.len()));
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
    use neural_doc_flow_core::{DocumentMetadata, DocumentContent, DocumentStructure, DocumentType, DocumentSourceType};
    use uuid::Uuid;
    use chrono::Utc;
    use std::collections::HashMap;
    
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
        let xml_str = String::from_utf8(bytes).unwrap();
        
        assert!(xml_str.contains("<?xml version=\"1.0\" encoding=\"UTF-8\"?>"));
        assert!(xml_str.contains("<document>"));
        assert!(xml_str.contains("<title>Test Document &amp; Example</title>"));
        assert!(xml_str.contains("<author>Test Author</author>"));
        assert!(xml_str.contains("This is a <test> document."));
        assert!(xml_str.contains("</document>"));
    }
}