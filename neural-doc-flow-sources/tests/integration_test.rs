//! Integration tests for neural-doc-flow-sources

use neural_doc_flow_sources::prelude::*;
use std::sync::Arc;
use tokio;

#[tokio::test]
async fn test_source_manager_lifecycle() {
    // Create manager
    let manager = SourceManager::new();
    
    // Register PDF source
    let pdf_source = Arc::new(PdfSource::new());
    manager.register_source(pdf_source).await.unwrap();
    
    // Register text source
    let text_source = Arc::new(TextSource::new());
    manager.register_source(text_source).await.unwrap();
    
    // Verify sources are registered
    let sources = manager.list_sources().await;
    assert_eq!(sources.len(), 2);
    assert!(sources.contains(&"pdf-source".to_string()));
    assert!(sources.contains(&"text-source".to_string()));
    
    // Test finding by extension
    let pdf_sources = manager.find_sources_by_extension("pdf").await;
    assert_eq!(pdf_sources.len(), 1);
    assert_eq!(pdf_sources[0], "pdf-source");
    
    let txt_sources = manager.find_sources_by_extension("txt").await;
    assert_eq!(txt_sources.len(), 1);
    assert_eq!(txt_sources[0], "text-source");
    
    // Test finding by MIME type
    let pdf_mime_sources = manager.find_sources_by_mime("application/pdf").await;
    assert_eq!(pdf_mime_sources.len(), 1);
    
    let text_mime_sources = manager.find_sources_by_mime("text/plain").await;
    assert_eq!(text_mime_sources.len(), 1);
    
    // Test finding by capability
    let pdf_capable = manager.find_sources_by_capability(SourceCapability::Pdf).await;
    assert_eq!(pdf_capable.len(), 1);
    
    let text_capable = manager.find_sources_by_capability(SourceCapability::PlainText).await;
    assert_eq!(text_capable.len(), 1);
    
    // Test metadata listing
    let all_metadata = manager.list_metadata().await;
    assert_eq!(all_metadata.len(), 2);
    
    // Test capability report
    let report = manager.capability_report().await;
    assert!(report.contains_key(&SourceCapability::Pdf));
    assert!(report.contains_key(&SourceCapability::PlainText));
    assert!(report.contains_key(&SourceCapability::MetadataExtraction));
    
    // Test source validation
    assert!(manager.validate_source("pdf-source").await.unwrap());
    assert!(manager.validate_source("text-source").await.unwrap());
    
    // Test unregistering
    manager.unregister_source("pdf-source").await.unwrap();
    let remaining = manager.list_sources().await;
    assert_eq!(remaining.len(), 1);
    assert!(!remaining.contains(&"pdf-source".to_string()));
}

#[tokio::test]
async fn test_text_source_processing() {
    let source = TextSource::new();
    let content = "Hello, World!\nThis is a test document.\nWith multiple lines.";
    
    let doc = Document {
        id: "test-text-doc".to_string(),
        source: DocumentSourceType::Memory(content.as_bytes().to_vec()),
        doc_type: DocumentType::Unknown,
        content: String::new(),
        metadata: std::collections::HashMap::new(),
        embeddings: None,
        chunks: vec![],
        processing_state: ProcessingState::Pending,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    };
    
    let result = source.process(doc).await.unwrap();
    
    assert_eq!(result.content, content);
    assert_eq!(result.doc_type, DocumentType::Text);
    assert_eq!(result.metadata.get("processor"), Some(&"text-source".to_string()));
    assert_eq!(result.metadata.get("line_count"), Some(&"3".to_string()));
    assert_eq!(result.metadata.get("word_count"), Some(&"9".to_string()));
    assert!(result.metadata.contains_key("encoding"));
}

#[tokio::test]
async fn test_source_config() {
    let mut pdf_source = PdfSource::new();
    let mut text_source = TextSource::new();
    
    // Test default config
    assert_eq!(pdf_source.config().max_file_size, Some(100 * 1024 * 1024));
    assert!(pdf_source.config().extract_metadata);
    
    // Test config update
    let mut new_config = SourceConfig::default();
    new_config.max_file_size = Some(50 * 1024 * 1024);
    new_config.extract_metadata = false;
    
    pdf_source.set_config(new_config.clone());
    text_source.set_config(new_config);
    
    assert_eq!(pdf_source.config().max_file_size, Some(50 * 1024 * 1024));
    assert!(!pdf_source.config().extract_metadata);
}

#[tokio::test]
#[cfg(all(feature = "pdf", feature = "text"))]
async fn test_default_manager() {
    let manager = create_default_manager().await.unwrap();
    
    let sources = manager.list_sources().await;
    assert_eq!(sources.len(), 2);
    
    // Both default sources should be registered
    assert!(sources.contains(&"pdf-source".to_string()));
    assert!(sources.contains(&"text-source".to_string()));
}