//! Batch document processing example
//!
//! This example demonstrates how to process multiple documents efficiently
//! using NeuralDocFlow's batch processing capabilities.

use neuraldocflow::{DocFlow, SourceInput, Config, ExtractionConfig, DaaConfig};
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure for high-throughput batch processing
    let config = Config {
        extraction: ExtractionConfig {
            enable_parallel_processing: true,
            worker_threads: num_cpus::get(),
            batch_size: 10,
            processing_timeout: Duration::from_secs(300), // 5 minutes per document
            ..Default::default()
        },
        daa: DaaConfig {
            max_agents: num_cpus::get() * 2,
            enable_consensus: false, // Disable for speed in batch processing
            coordination_timeout: Duration::from_secs(60),
            ..Default::default()
        },
        ..Default::default()
    };
    
    println!("Initializing NeuralDocFlow for batch processing...");
    println!("CPU cores: {}, Max agents: {}", num_cpus::get(), config.daa.max_agents);
    
    let docflow = DocFlow::with_config(config)?;
    
    // Prepare batch inputs
    let inputs = prepare_batch_inputs()?;
    println!("Prepared {} documents for batch processing", inputs.len());
    
    // Process batch
    let start_time = Instant::now();
    let results = docflow.extract_batch(inputs).await?;
    let total_time = start_time.elapsed();
    
    // Analyze results
    analyze_batch_results(&results, total_time);
    
    // Save results
    save_batch_results(&results).await?;
    
    println!("\n✅ Batch processing completed successfully!");
    
    Ok(())
}

/// Prepare a batch of input documents
fn prepare_batch_inputs() -> Result<Vec<SourceInput>, std::io::Error> {
    let mut inputs = Vec::new();
    
    // Example: Process all PDFs in a directory
    let pdf_dir = PathBuf::from("./sample_documents");
    
    if pdf_dir.exists() {
        for entry in std::fs::read_dir(&pdf_dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.extension().and_then(|s| s.to_str()) == Some("pdf") {
                inputs.push(SourceInput::File {
                    path: path.clone(),
                    metadata: None,
                });
            }
        }
    } else {
        // Create sample inputs for demonstration
        println!("Sample documents directory not found, creating demo inputs...");
        
        for i in 1..=5 {
            inputs.push(SourceInput::File {
                path: PathBuf::from(format!("document_{}.pdf", i)),
                metadata: None,
            });
        }
    }
    
    Ok(inputs)
}

/// Analyze batch processing results
fn analyze_batch_results(results: &[neuraldocflow::ExtractedDocument], total_time: Duration) {
    println!("\n=== Batch Processing Analysis ===");
    println!("Total documents: {}", results.len());
    println!("Total processing time: {:.2?}", total_time);
    
    if !results.is_empty() {
        let avg_time = total_time / results.len() as u32;
        println!("Average time per document: {:.2?}", avg_time);
        
        let throughput = results.len() as f64 / total_time.as_secs_f64() * 60.0;
        println!("Throughput: {:.1} documents/minute", throughput);
    }
    
    // Confidence statistics
    let confidences: Vec<f32> = results.iter().map(|d| d.confidence).collect();
    let avg_confidence = confidences.iter().sum::<f32>() / confidences.len() as f32;
    let min_confidence = confidences.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_confidence = confidences.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    
    println!("\n=== Quality Metrics ===");
    println!("Average confidence: {:.1}%", avg_confidence * 100.0);
    println!("Minimum confidence: {:.1}%", min_confidence * 100.0);
    println!("Maximum confidence: {:.1}%", max_confidence * 100.0);
    
    // Content statistics
    let total_blocks: usize = results.iter().map(|d| d.content.len()).sum();
    let total_words: usize = results.iter()
        .map(|d| d.get_content_stats().total_words)
        .sum();
    let total_pages: usize = results.iter()
        .map(|d| d.metadata.page_count)
        .sum();
    
    println!("\n=== Content Statistics ===");
    println!("Total content blocks: {}", total_blocks);
    println!("Total words extracted: {}", total_words);
    println!("Total pages processed: {}", total_pages);
    
    if !results.is_empty() {
        println!("Average blocks per document: {:.1}", total_blocks as f64 / results.len() as f64);
        println!("Average words per document: {:.1}", total_words as f64 / results.len() as f64);
        println!("Average pages per document: {:.1}", total_pages as f64 / results.len() as f64);
    }
    
    // Document type analysis
    analyze_document_types(results);
    
    // Quality distribution
    analyze_quality_distribution(&confidences);
}

/// Analyze document types and content
fn analyze_document_types(results: &[neuraldocflow::ExtractedDocument]) {
    println!("\n=== Document Analysis ===");
    
    let mut table_docs = 0;
    let mut image_docs = 0;
    let mut text_heavy_docs = 0;
    
    for doc in results {
        let stats = doc.get_content_stats();
        
        if stats.table_count > 0 {
            table_docs += 1;
        }
        
        if stats.image_count > 0 {
            image_docs += 1;
        }
        
        if stats.total_words > 1000 {
            text_heavy_docs += 1;
        }
    }
    
    println!("Documents with tables: {} ({:.1}%)", 
        table_docs, 
        table_docs as f64 / results.len() as f64 * 100.0
    );
    println!("Documents with images: {} ({:.1}%)", 
        image_docs, 
        image_docs as f64 / results.len() as f64 * 100.0
    );
    println!("Text-heavy documents (>1000 words): {} ({:.1}%)", 
        text_heavy_docs, 
        text_heavy_docs as f64 / results.len() as f64 * 100.0
    );
}

/// Analyze confidence distribution
fn analyze_quality_distribution(confidences: &[f32]) {
    println!("\n=== Quality Distribution ===");
    
    let high_quality = confidences.iter().filter(|&&c| c >= 0.9).count();
    let good_quality = confidences.iter().filter(|&&c| c >= 0.8 && c < 0.9).count();
    let medium_quality = confidences.iter().filter(|&&c| c >= 0.7 && c < 0.8).count();
    let low_quality = confidences.iter().filter(|&&c| c < 0.7).count();
    
    let total = confidences.len();
    
    println!("High quality (≥90%): {} ({:.1}%)", 
        high_quality, 
        high_quality as f64 / total as f64 * 100.0
    );
    println!("Good quality (80-89%): {} ({:.1}%)", 
        good_quality, 
        good_quality as f64 / total as f64 * 100.0
    );
    println!("Medium quality (70-79%): {} ({:.1}%)", 
        medium_quality, 
        medium_quality as f64 / total as f64 * 100.0
    );
    println!("Low quality (<70%): {} ({:.1}%)", 
        low_quality, 
        low_quality as f64 / total as f64 * 100.0
    );
}

/// Save batch results to files
async fn save_batch_results(results: &[neuraldocflow::ExtractedDocument]) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs;
    use serde_json;
    
    // Create output directory
    let output_dir = PathBuf::from("./batch_results");
    fs::create_dir_all(&output_dir)?;
    
    println!("\n=== Saving Results ===");
    println!("Output directory: {:?}", output_dir);
    
    // Save individual documents
    for (i, doc) in results.iter().enumerate() {
        let filename = format!("document_{:03}.json", i + 1);
        let filepath = output_dir.join(&filename);
        
        let json_content = serde_json::to_string_pretty(doc)?;
        fs::write(&filepath, json_content)?;
        
        // Also save just the text content
        let text_filename = format!("document_{:03}.txt", i + 1);
        let text_filepath = output_dir.join(&text_filename);
        let text_content = doc.get_text();
        fs::write(&text_filepath, text_content)?;
    }
    
    // Save batch summary
    let summary = create_batch_summary(results);
    let summary_json = serde_json::to_string_pretty(&summary)?;
    fs::write(output_dir.join("batch_summary.json"), summary_json)?;
    
    println!("Saved {} document files", results.len());
    println!("Saved batch summary to batch_summary.json");
    
    Ok(())
}

/// Create a summary of the batch processing results
fn create_batch_summary(results: &[neuraldocflow::ExtractedDocument]) -> serde_json::Value {
    let total_blocks: usize = results.iter().map(|d| d.content.len()).sum();
    let total_words: usize = results.iter()
        .map(|d| d.get_content_stats().total_words)
        .sum();
    let total_pages: usize = results.iter()
        .map(|d| d.metadata.page_count)
        .sum();
    
    let confidences: Vec<f32> = results.iter().map(|d| d.confidence).collect();
    let avg_confidence = confidences.iter().sum::<f32>() / confidences.len() as f32;
    
    serde_json::json!({
        "batch_summary": {
            "total_documents": results.len(),
            "total_content_blocks": total_blocks,
            "total_words": total_words,
            "total_pages": total_pages,
            "average_confidence": avg_confidence,
            "processing_timestamp": chrono::Utc::now().to_rfc3339()
        },
        "documents": results.iter().enumerate().map(|(i, doc)| {
            serde_json::json!({
                "index": i,
                "id": doc.id,
                "source_id": doc.source_id,
                "confidence": doc.confidence,
                "content_blocks": doc.content.len(),
                "pages": doc.metadata.page_count,
                "title": doc.metadata.title,
                "content_stats": doc.get_content_stats()
            })
        }).collect::<Vec<_>>()
    })
}

/// Example: Process documents with different configurations
#[allow(dead_code)]
async fn batch_with_different_configs() -> Result<(), Box<dyn std::error::Error>> {
    let inputs = prepare_batch_inputs()?;
    
    // Configuration 1: High accuracy (slower)
    let high_accuracy_config = Config {
        neural: neuraldocflow::NeuralConfig {
            enabled: true,
            ..Default::default()
        },
        daa: DaaConfig {
            enable_consensus: true,
            consensus_threshold: 0.9,
            max_agents: 4,
            ..Default::default()
        },
        ..Default::default()
    };
    
    // Configuration 2: High speed (lower accuracy)
    let high_speed_config = Config {
        neural: neuraldocflow::NeuralConfig {
            enabled: false,
            ..Default::default()
        },
        daa: DaaConfig {
            enable_consensus: false,
            max_agents: num_cpus::get() * 3,
            ..Default::default()
        },
        extraction: ExtractionConfig {
            enable_parallel_processing: true,
            worker_threads: num_cpus::get(),
            ..Default::default()
        },
        ..Default::default()
    };
    
    println!("Testing high accuracy configuration...");
    let docflow_accurate = DocFlow::with_config(high_accuracy_config)?;
    let start = Instant::now();
    let accurate_results = docflow_accurate.extract_batch(inputs.clone()).await?;
    let accurate_time = start.elapsed();
    
    println!("Testing high speed configuration...");
    let docflow_fast = DocFlow::with_config(high_speed_config)?;
    let start = Instant::now();
    let fast_results = docflow_fast.extract_batch(inputs).await?;
    let fast_time = start.elapsed();
    
    // Compare results
    println!("\n=== Configuration Comparison ===");
    println!("High Accuracy - Time: {:.2?}, Avg Confidence: {:.1}%", 
        accurate_time,
        accurate_results.iter().map(|d| d.confidence).sum::<f32>() / accurate_results.len() as f32 * 100.0
    );
    println!("High Speed - Time: {:.2?}, Avg Confidence: {:.1}%", 
        fast_time,
        fast_results.iter().map(|d| d.confidence).sum::<f32>() / fast_results.len() as f32 * 100.0
    );
    println!("Speed improvement: {:.1}x", accurate_time.as_secs_f64() / fast_time.as_secs_f64());
    
    Ok(())
}

/// Example: Progress tracking for long batch jobs
#[allow(dead_code)]
async fn batch_with_progress_tracking() -> Result<(), Box<dyn std::error::Error>> {
    let inputs = prepare_batch_inputs()?;
    let docflow = DocFlow::new()?;
    
    println!("Processing {} documents with progress tracking...", inputs.len());
    
    let mut completed = 0;
    let total = inputs.len();
    let start_time = Instant::now();
    
    // Process documents individually to track progress
    let mut results = Vec::new();
    
    for (i, input) in inputs.into_iter().enumerate() {
        let doc_start = Instant::now();
        
        match docflow.extract(input).await {
            Ok(document) => {
                completed += 1;
                let doc_time = doc_start.elapsed();
                
                results.push(document);
                
                // Calculate progress
                let progress = (completed as f64 / total as f64) * 100.0;
                let elapsed = start_time.elapsed();
                let estimated_total = elapsed.as_secs_f64() / progress * 100.0;
                let eta = Duration::from_secs_f64(estimated_total - elapsed.as_secs_f64());
                
                println!("✅ Document {} ({:.1}%) - {:.2?} - ETA: {:.0?}", 
                    i + 1, progress, doc_time, eta);
            }
            Err(e) => {
                println!("❌ Document {} failed: {}", i + 1, e);
            }
        }
    }
    
    let total_time = start_time.elapsed();
    println!("\n✅ Completed {} of {} documents in {:.2?}", 
        completed, total, total_time);
    
    Ok(())
}