//! Basic PDF extraction example
//!
//! This example demonstrates the simplest way to extract content from a PDF
//! document using NeuralDocFlow's default configuration.

use neuraldocflow::{DocFlow, SourceInput};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the document extraction engine with default settings
    println!("Initializing NeuralDocFlow...");
    let docflow = DocFlow::new()?;
    
    // Create input from a PDF file
    // Replace "example.pdf" with your actual PDF file path
    let input = SourceInput::File {
        path: PathBuf::from("example.pdf"),
        metadata: None,
    };
    
    println!("Extracting content from PDF...");
    
    // Extract document content
    let start_time = std::time::Instant::now();
    let document = docflow.extract(input).await?;
    let extraction_time = start_time.elapsed();
    
    // Display extraction results
    println!("\n=== Extraction Results ===");
    println!("Document ID: {}", document.id);
    println!("Source: {}", document.source_id);
    println!("Extraction time: {:.2?}", extraction_time);
    println!("Overall confidence: {:.1}%", document.confidence * 100.0);
    println!("Content blocks extracted: {}", document.content.len());
    
    // Show document metadata
    if let Some(title) = &document.metadata.title {
        println!("Title: {}", title);
    }
    if let Some(author) = &document.metadata.author {
        println!("Author: {}", author);
    }
    println!("Pages: {}", document.metadata.page_count);
    
    // Display content statistics
    let stats = document.get_content_stats();
    println!("\n=== Content Statistics ===");
    println!("Paragraphs: {}", stats.paragraph_count);
    println!("Headings: {}", stats.heading_count);
    println!("Tables: {}", stats.table_count);
    println!("Images: {}", stats.image_count);
    println!("Lists: {}", stats.list_count);
    println!("Total words: {}", stats.total_words);
    println!("Total characters: {}", stats.total_characters);
    
    // Show first few content blocks
    println!("\n=== Content Preview ===");
    for (i, block) in document.content.iter().take(5).enumerate() {
        println!("\nBlock {} ({:?}):", i + 1, block.block_type);
        println!("  Page: {}", block.position.page);
        println!("  Position: ({:.1}, {:.1})", block.position.x, block.position.y);
        println!("  Size: {:.1} x {:.1}", block.position.width, block.position.height);
        println!("  Confidence: {:.1}%", block.metadata.confidence * 100.0);
        
        if let Some(text) = &block.text {
            let preview = if text.len() > 100 {
                format!("{}...", &text[..97])
            } else {
                text.clone()
            };
            println!("  Text: {}", preview);
        }
        
        if let Some(binary) = &block.binary {
            println!("  Binary data: {} bytes", binary.len());
        }
    }
    
    if document.content.len() > 5 {
        println!("\n... and {} more blocks", document.content.len() - 5);
    }
    
    // Extract and display full text
    println!("\n=== Full Text Content ===");
    let full_text = document.get_text();
    if full_text.len() > 500 {
        println!("{}...\n\n[Text truncated - {} total characters]", 
            &full_text[..497], full_text.len());
    } else {
        println!("{}", full_text);
    }
    
    // Show table information if any tables were found
    let tables = document.get_tables();
    if !tables.is_empty() {
        println!("\n=== Tables Found ===");
        for (i, table) in tables.iter().enumerate() {
            println!("Table {}: {} confidence", i + 1, table.metadata.confidence);
            if let Some(text) = &table.text {
                let lines: Vec<&str> = text.lines().take(3).collect();
                for line in lines {
                    println!("  {}", line);
                }
                if text.lines().count() > 3 {
                    println!("  ... ({} total lines)", text.lines().count());
                }
            }
        }
    }
    
    // Show image information if any images were found
    let images = document.get_images();
    if !images.is_empty() {
        println!("\n=== Images Found ===");
        for (i, image) in images.iter().enumerate() {
            println!("Image {}: {:.1} x {:.1} at page {}", 
                i + 1, 
                image.position.width, 
                image.position.height,
                image.position.page
            );
            if let Some(binary) = &image.binary {
                println!("  Size: {} bytes", binary.len());
            }
        }
    }
    
    // Show processing metrics
    println!("\n=== Processing Metrics ===");
    println!("Pages processed: {}", document.metrics.pages_processed);
    println!("Blocks extracted: {}", document.metrics.blocks_extracted);
    println!("Memory used: {} MB", document.metrics.memory_used / (1024 * 1024));
    
    println!("\n✅ Extraction completed successfully!");
    
    Ok(())
}

/// Example with error handling for missing files
#[allow(dead_code)]
async fn extract_with_error_handling() -> Result<(), Box<dyn std::error::Error>> {
    let docflow = DocFlow::new()?;
    
    let input = SourceInput::File {
        path: PathBuf::from("nonexistent.pdf"),
        metadata: None,
    };
    
    match docflow.extract(input).await {
        Ok(document) => {
            println!("Successfully extracted {} blocks", document.content.len());
        }
        Err(e) => {
            eprintln!("Extraction failed: {}", e);
            eprintln!("Make sure the PDF file exists and is readable");
        }
    }
    
    Ok(())
}

/// Example with memory-based input instead of file
#[allow(dead_code)]
async fn extract_from_memory() -> Result<(), Box<dyn std::error::Error>> {
    let docflow = DocFlow::new()?;
    
    // Read PDF data into memory
    let pdf_data = std::fs::read("example.pdf")?;
    
    let input = SourceInput::Memory {
        data: pdf_data,
        filename: Some("example.pdf".to_string()),
        mime_type: Some("application/pdf".to_string()),
    };
    
    let document = docflow.extract(input).await?;
    
    println!("Extracted from memory: {} blocks", document.content.len());
    
    Ok(())
}

/// Example with custom metadata
#[allow(dead_code)]
async fn extract_with_metadata() -> Result<(), Box<dyn std::error::Error>> {
    use std::collections::HashMap;
    
    let docflow = DocFlow::new()?;
    
    let mut metadata = HashMap::new();
    metadata.insert("department".to_string(), "finance".to_string());
    metadata.insert("priority".to_string(), "high".to_string());
    metadata.insert("classification".to_string(), "confidential".to_string());
    
    let input = SourceInput::File {
        path: PathBuf::from("financial_report.pdf"),
        metadata: Some(metadata),
    };
    
    let document = docflow.extract(input).await?;
    
    println!("Extracted document with custom metadata");
    println!("Classification: {}", 
        document.metadata.custom_metadata
            .get("classification")
            .unwrap_or(&"unknown".to_string())
    );
    
    Ok(())
}