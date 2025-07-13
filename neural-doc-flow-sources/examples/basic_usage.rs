//! Basic usage example for neural-doc-flow-sources

use neural_doc_flow_sources::prelude::*;
use std::sync::Arc;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    println!("Neural Document Flow Sources - Basic Usage Example\n");
    
    // Create a source manager
    let manager = SourceManager::new();
    println!("✓ Created source manager");
    
    // Register PDF source
    let pdf_source = Arc::new(PdfSource::new());
    manager.register_source(pdf_source).await?;
    println!("✓ Registered PDF source");
    
    // Register text source
    let text_source = Arc::new(TextSource::new());
    manager.register_source(text_source).await?;
    println!("✓ Registered text source");
    
    // List all registered sources
    println!("\nRegistered sources:");
    for source_id in manager.list_sources().await {
        println!("  - {}", source_id);
    }
    
    // Show metadata for each source
    println!("\nSource details:");
    for metadata in manager.list_metadata().await {
        println!("\n  {}:", metadata.name);
        println!("    ID: {}", metadata.id);
        println!("    Version: {}", metadata.version);
        println!("    Extensions: {:?}", metadata.file_extensions);
        println!("    MIME types: {:?}", metadata.mime_types);
        println!("    Capabilities: {:?}", metadata.capabilities);
    }
    
    // Demonstrate finding sources by file extension
    println!("\nFinding sources by extension:");
    let pdf_sources = manager.find_sources_by_extension("pdf").await;
    println!("  PDF handlers: {:?}", pdf_sources);
    
    let txt_sources = manager.find_sources_by_extension("txt").await;
    println!("  TXT handlers: {:?}", txt_sources);
    
    // Show capability report
    println!("\nCapability report:");
    let report = manager.capability_report().await;
    for (capability, sources) in report {
        println!("  {:?}: {:?}", capability, sources);
    }
    
    // Example: Process a text document
    println!("\nProcessing a sample text document:");
    if let Some(source) = manager.get_source("text-source").await {
        let content = "# Sample Document\n\nThis is a test document.\nIt has multiple lines.\n\n## Section 1\n\nSome content here.";
        
        let doc = Document {
            id: "sample-001".to_string(),
            source: DocumentSourceType::Memory(content.as_bytes().to_vec()),
            doc_type: DocumentType::Unknown,
            content: String::new(),
            metadata: HashMap::new(),
            embeddings: None,
            chunks: vec![],
            processing_state: ProcessingState::Pending,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        
        match source.process(doc).await {
            Ok(processed) => {
                println!("  ✓ Successfully processed document");
                println!("    Type: {:?}", processed.doc_type);
                println!("    Content length: {} chars", processed.content.len());
                println!("    Metadata:");
                for (key, value) in &processed.metadata {
                    println!("      {}: {}", key, value);
                }
            }
            Err(e) => {
                println!("  ✗ Error processing document: {}", e);
            }
        }
    }
    
    // Configure a source
    println!("\nConfiguring sources:");
    if let Some(source) = manager.get_source("pdf-source").await {
        // This would work if we could downcast, but for now we'll just show the concept
        println!("  PDF source is configured with default settings");
        println!("  (In real usage, you would configure before registration)");
    }
    
    Ok(())
}