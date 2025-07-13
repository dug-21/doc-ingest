use neural_doc_flow_outputs::*;
use neural_doc_flow_core::{Document, ProcessingStatus};
use uuid::Uuid;
use chrono::Utc;
use std::path::Path;
use tempfile::TempDir;
use tokio::fs;

/// Create a test document
fn create_test_document() -> Document {
    Document {
        id: Uuid::new_v4(),
        content: "# Test Document\n\nThis is a test document with **bold** and *italic* text.\n\n## Section 1\n\nSome content here.\n\n## Section 2\n\nMore content with a [link](https://example.com).".to_string(),
        metadata: serde_json::json!({
            "title": "Test Document",
            "author": "Test Suite",
            "created": "2024-01-01",
            "tags": ["test", "example"],
            "custom_field": "custom_value"
        }),
        source: "test_source".to_string(),
        document_type: "text/markdown".to_string(),
        size: 150,
        created_at: Utc::now(),
        updated_at: Utc::now(),
        hash: "test_hash_12345".to_string(),
        tags: vec!["test".to_string(), "example".to_string()],
        relationships: vec![],
        processing_status: ProcessingStatus::Completed,
    }
}

#[test]
fn test_output_manager_creation() {
    let manager = OutputManager::new();
    assert_eq!(manager.available_formats().len(), 5);
    
    let formats = manager.available_formats();
    assert!(formats.contains(&"json"));
    assert!(formats.contains(&"markdown"));
    assert!(formats.contains(&"html"));
    assert!(formats.contains(&"pdf"));
    assert!(formats.contains(&"xml"));
}

#[test]
fn test_find_output_by_name() {
    let manager = OutputManager::new();
    
    // Test case-insensitive lookup
    assert!(manager.find_output("json").is_some());
    assert!(manager.find_output("JSON").is_some());
    assert!(manager.find_output("Json").is_some());
    
    assert!(manager.find_output("markdown").is_some());
    assert!(manager.find_output("MARKDOWN").is_some());
    
    assert!(manager.find_output("nonexistent").is_none());
}

#[test]
fn test_find_output_by_extension() {
    let manager = OutputManager::new();
    
    assert!(manager.find_output_by_extension("json").is_some());
    assert!(manager.find_output_by_extension("md").is_some());
    assert!(manager.find_output_by_extension("html").is_some());
    assert!(manager.find_output_by_extension("pdf").is_some());
    assert!(manager.find_output_by_extension("xml").is_some());
    
    assert!(manager.find_output_by_extension("xyz").is_none());
}

#[tokio::test]
async fn test_json_output_generation() {
    let json_output = JsonOutput::new();
    let document = create_test_document();
    
    // Test metadata
    assert_eq!(json_output.name(), "json");
    assert_eq!(json_output.mime_type(), "application/json");
    assert!(json_output.supported_extensions().contains(&"json"));
    
    // Test byte generation
    let bytes = json_output.generate_bytes(&document).await.unwrap();
    assert!(!bytes.is_empty());
    
    // Parse and verify JSON
    let json_str = String::from_utf8(bytes).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
    
    assert_eq!(parsed["id"], document.id.to_string());
    assert_eq!(parsed["content"], document.content);
    assert_eq!(parsed["metadata"]["title"], "Test Document");
    assert_eq!(parsed["tags"], serde_json::json!(["test", "example"]));
}

#[tokio::test]
async fn test_json_output_file_generation() {
    let json_output = JsonOutput::new();
    let document = create_test_document();
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("output.json");
    
    // Generate file
    json_output.generate(&document, &output_path).await.unwrap();
    
    // Verify file exists and content
    assert!(output_path.exists());
    let content = fs::read_to_string(&output_path).await.unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();
    
    assert_eq!(parsed["document_type"], "text/markdown");
    assert_eq!(parsed["source"], "test_source");
}

#[tokio::test]
async fn test_markdown_output_generation() {
    let md_output = MarkdownOutput::new();
    let document = create_test_document();
    
    // Test metadata
    assert_eq!(md_output.name(), "markdown");
    assert_eq!(md_output.mime_type(), "text/markdown");
    assert!(md_output.supported_extensions().contains(&"md"));
    assert!(md_output.supported_extensions().contains(&"markdown"));
    
    // Test byte generation
    let bytes = md_output.generate_bytes(&document).await.unwrap();
    let md_content = String::from_utf8(bytes).unwrap();
    
    // Verify markdown content
    assert!(md_content.contains("# Test Document"));
    assert!(md_content.contains("## Metadata"));
    assert!(md_content.contains("- **Author**: Test Suite"));
    assert!(md_content.contains("## Content"));
    assert!(md_content.contains("## Section 1"));
    assert!(md_content.contains("## Section 2"));
}

#[tokio::test]
async fn test_html_output_generation() {
    let html_output = HtmlOutput::new();
    let document = create_test_document();
    
    // Test metadata
    assert_eq!(html_output.name(), "html");
    assert_eq!(html_output.mime_type(), "text/html");
    assert!(html_output.supported_extensions().contains(&"html"));
    assert!(html_output.supported_extensions().contains(&"htm"));
    
    // Test byte generation
    let bytes = html_output.generate_bytes(&document).await.unwrap();
    let html_content = String::from_utf8(bytes).unwrap();
    
    // Verify HTML structure
    assert!(html_content.contains("<!DOCTYPE html>"));
    assert!(html_content.contains("<html lang=\"en\">"));
    assert!(html_content.contains("<title>Test Document</title>"));
    assert!(html_content.contains("<h1>Test Document</h1>"));
    assert!(html_content.contains("<strong>bold</strong>"));
    assert!(html_content.contains("<em>italic</em>"));
    assert!(html_content.contains("<h2>Section 1</h2>"));
    assert!(html_content.contains("<h2>Section 2</h2>"));
    assert!(html_content.contains("<a href=\"https://example.com\">"));
}

#[tokio::test]
async fn test_xml_output_generation() {
    let xml_output = XmlOutput::new();
    let document = create_test_document();
    
    // Test metadata
    assert_eq!(xml_output.name(), "xml");
    assert_eq!(xml_output.mime_type(), "application/xml");
    assert!(xml_output.supported_extensions().contains(&"xml"));
    
    // Test byte generation
    let bytes = xml_output.generate_bytes(&document).await.unwrap();
    let xml_content = String::from_utf8(bytes).unwrap();
    
    // Verify XML structure
    assert!(xml_content.contains("<?xml version=\"1.0\" encoding=\"UTF-8\"?>"));
    assert!(xml_content.contains("<document>"));
    assert!(xml_content.contains("</document>"));
    assert!(xml_content.contains("<id>"));
    assert!(xml_content.contains("<content>"));
    assert!(xml_content.contains("<metadata>"));
    assert!(xml_content.contains("<title>Test Document</title>"));
    assert!(xml_content.contains("<author>Test Suite</author>"));
    assert!(xml_content.contains("<tags>"));
    assert!(xml_content.contains("<tag>test</tag>"));
    assert!(xml_content.contains("<tag>example</tag>"));
}

#[tokio::test]
async fn test_pdf_output_generation() {
    let pdf_output = PdfOutput::new();
    let document = create_test_document();
    
    // Test metadata
    assert_eq!(pdf_output.name(), "pdf");
    assert_eq!(pdf_output.mime_type(), "application/pdf");
    assert!(pdf_output.supported_extensions().contains(&"pdf"));
    
    // Test byte generation
    let bytes = pdf_output.generate_bytes(&document).await.unwrap();
    assert!(!bytes.is_empty());
    
    // Verify PDF header
    let pdf_str = String::from_utf8_lossy(&bytes[0..10]);
    assert!(pdf_str.starts_with("%PDF-"));
}

#[tokio::test]
async fn test_output_manager_generate() {
    let manager = OutputManager::new();
    let document = create_test_document();
    let temp_dir = TempDir::new().unwrap();
    
    // Test JSON output
    let json_path = temp_dir.path().join("output.json");
    manager.generate_output(&document, "json", &json_path).await.unwrap();
    assert!(json_path.exists());
    
    // Test Markdown output
    let md_path = temp_dir.path().join("output.md");
    manager.generate_output(&document, "markdown", &md_path).await.unwrap();
    assert!(md_path.exists());
    
    // Test HTML output
    let html_path = temp_dir.path().join("output.html");
    manager.generate_output(&document, "html", &html_path).await.unwrap();
    assert!(html_path.exists());
    
    // Test invalid format
    let invalid_path = temp_dir.path().join("output.xyz");
    let result = manager.generate_output(&document, "xyz", &invalid_path).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_output_manager_generate_bytes() {
    let manager = OutputManager::new();
    let document = create_test_document();
    
    // Test all formats
    let json_bytes = manager.generate_bytes(&document, "json").await.unwrap();
    assert!(!json_bytes.is_empty());
    
    let md_bytes = manager.generate_bytes(&document, "markdown").await.unwrap();
    assert!(!md_bytes.is_empty());
    
    let html_bytes = manager.generate_bytes(&document, "html").await.unwrap();
    assert!(!html_bytes.is_empty());
    
    let pdf_bytes = manager.generate_bytes(&document, "pdf").await.unwrap();
    assert!(!pdf_bytes.is_empty());
    
    let xml_bytes = manager.generate_bytes(&document, "xml").await.unwrap();
    assert!(!xml_bytes.is_empty());
    
    // Test invalid format
    let result = manager.generate_bytes(&document, "invalid").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_empty_document_handling() {
    let manager = OutputManager::new();
    let empty_doc = Document {
        id: Uuid::new_v4(),
        content: String::new(),
        metadata: serde_json::json!({}),
        source: String::new(),
        document_type: String::new(),
        size: 0,
        created_at: Utc::now(),
        updated_at: Utc::now(),
        hash: String::new(),
        tags: vec![],
        relationships: vec![],
        processing_status: ProcessingStatus::Completed,
    };
    
    // All formats should handle empty documents
    let json_bytes = manager.generate_bytes(&empty_doc, "json").await.unwrap();
    assert!(!json_bytes.is_empty());
    
    let md_bytes = manager.generate_bytes(&empty_doc, "markdown").await.unwrap();
    assert!(!md_bytes.is_empty());
    
    let html_bytes = manager.generate_bytes(&empty_doc, "html").await.unwrap();
    assert!(!html_bytes.is_empty());
}

#[tokio::test]
async fn test_special_characters_handling() {
    let special_doc = Document {
        id: Uuid::new_v4(),
        content: "Special chars: < > & \" ' \n\t😀".to_string(),
        metadata: serde_json::json!({
            "special": "< > & \" '"
        }),
        source: "test".to_string(),
        document_type: "text/plain".to_string(),
        size: 50,
        created_at: Utc::now(),
        updated_at: Utc::now(),
        hash: "hash".to_string(),
        tags: vec![],
        relationships: vec![],
        processing_status: ProcessingStatus::Completed,
    };
    
    let manager = OutputManager::new();
    
    // HTML should escape special characters
    let html_bytes = manager.generate_bytes(&special_doc, "html").await.unwrap();
    let html_content = String::from_utf8(html_bytes).unwrap();
    assert!(html_content.contains("&lt;"));
    assert!(html_content.contains("&gt;"));
    assert!(html_content.contains("&amp;"));
    assert!(html_content.contains("&quot;"));
    
    // XML should escape special characters
    let xml_bytes = manager.generate_bytes(&special_doc, "xml").await.unwrap();
    let xml_content = String::from_utf8(xml_bytes).unwrap();
    assert!(xml_content.contains("&lt;"));
    assert!(xml_content.contains("&gt;"));
    assert!(xml_content.contains("&amp;"));
    
    // JSON should handle properly
    let json_bytes = manager.generate_bytes(&special_doc, "json").await.unwrap();
    let json_str = String::from_utf8(json_bytes).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
    assert!(parsed["content"].as_str().unwrap().contains("😀"));
}

#[tokio::test]
async fn test_large_document_handling() {
    let large_content = "Lorem ipsum dolor sit amet. ".repeat(10000);
    let large_doc = Document {
        id: Uuid::new_v4(),
        content: large_content.clone(),
        metadata: serde_json::json!({
            "size": "large"
        }),
        source: "test".to_string(),
        document_type: "text/plain".to_string(),
        size: large_content.len(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        hash: "large_hash".to_string(),
        tags: vec![],
        relationships: vec![],
        processing_status: ProcessingStatus::Completed,
    };
    
    let manager = OutputManager::new();
    
    // All formats should handle large documents
    let json_bytes = manager.generate_bytes(&large_doc, "json").await.unwrap();
    assert!(json_bytes.len() > large_content.len()); // JSON adds structure
    
    let md_bytes = manager.generate_bytes(&large_doc, "markdown").await.unwrap();
    assert!(md_bytes.len() > large_content.len()); // Markdown adds headers
    
    let html_bytes = manager.generate_bytes(&large_doc, "html").await.unwrap();
    assert!(html_bytes.len() > large_content.len()); // HTML adds tags
}

#[tokio::test]
async fn test_metadata_preservation() {
    let doc_with_metadata = Document {
        id: Uuid::new_v4(),
        content: "Content".to_string(),
        metadata: serde_json::json!({
            "nested": {
                "field1": "value1",
                "field2": 42,
                "field3": true,
                "field4": [1, 2, 3],
                "field5": null
            },
            "array": ["a", "b", "c"],
            "number": 3.14,
            "boolean": false
        }),
        source: "test".to_string(),
        document_type: "text/plain".to_string(),
        size: 7,
        created_at: Utc::now(),
        updated_at: Utc::now(),
        hash: "hash".to_string(),
        tags: vec![],
        relationships: vec![],
        processing_status: ProcessingStatus::Completed,
    };
    
    let json_output = JsonOutput::new();
    let bytes = json_output.generate_bytes(&doc_with_metadata).await.unwrap();
    let json_str = String::from_utf8(bytes).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
    
    // Verify nested metadata is preserved
    assert_eq!(parsed["metadata"]["nested"]["field1"], "value1");
    assert_eq!(parsed["metadata"]["nested"]["field2"], 42);
    assert_eq!(parsed["metadata"]["nested"]["field3"], true);
    assert_eq!(parsed["metadata"]["array"], serde_json::json!(["a", "b", "c"]));
    assert_eq!(parsed["metadata"]["number"], 3.14);
    assert_eq!(parsed["metadata"]["boolean"], false);
}

#[test]
fn test_custom_output_registration() {
    use async_trait::async_trait;
    
    // Define a custom output format
    struct CustomOutput;
    
    #[async_trait]
    impl DocumentOutput for CustomOutput {
        fn name(&self) -> &str {
            "custom"
        }
        
        fn supported_extensions(&self) -> &[&str] {
            &["custom", "cst"]
        }
        
        fn mime_type(&self) -> &str {
            "application/x-custom"
        }
        
        async fn generate(&self, _document: &Document, _output_path: &Path) -> neural_doc_flow_core::Result<()> {
            Ok(())
        }
        
        async fn generate_bytes(&self, _document: &Document) -> neural_doc_flow_core::Result<Vec<u8>> {
            Ok(b"CUSTOM OUTPUT".to_vec())
        }
    }
    
    let mut manager = OutputManager::new();
    let initial_count = manager.available_formats().len();
    
    // Register custom output
    manager.register_output(Box::new(CustomOutput));
    
    // Verify registration
    assert_eq!(manager.available_formats().len(), initial_count + 1);
    assert!(manager.find_output("custom").is_some());
    assert!(manager.find_output_by_extension("custom").is_some());
    assert!(manager.find_output_by_extension("cst").is_some());
}

#[tokio::test]
async fn test_concurrent_output_generation() {
    use tokio::task;
    
    let manager = OutputManager::new();
    let document = create_test_document();
    
    // Generate multiple outputs concurrently
    let handles: Vec<_> = vec!["json", "markdown", "html", "xml"]
        .into_iter()
        .map(|format| {
            let doc = document.clone();
            let fmt = format.to_string();
            task::spawn(async move {
                let output_manager = OutputManager::new();
                output_manager.generate_bytes(&doc, &fmt).await
            })
        })
        .collect();
    
    // Wait for all tasks and verify results
    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok());
        assert!(!result.unwrap().is_empty());
    }
}
