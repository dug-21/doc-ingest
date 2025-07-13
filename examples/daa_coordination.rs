//! DAA coordination example
//!
//! This example demonstrates how to configure and monitor the Distributed
//! Autonomous Agents (DAA) coordination system in NeuralDocFlow.

use neuraldocflow::{
    DocFlow, SourceInput, Config, DaaConfig,
    daa::{Agent, AgentType, TaskPriority, AgentMessage, MessageType},
    error::Result,
};
use std::path::PathBuf;
use std::time::Duration;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== DAA Coordination Example ===\n");
    
    // Configure DAA system
    let daa_config = DaaConfig {
        max_agents: 6,
        enable_consensus: true,
        consensus_threshold: 0.8,
        coordination_timeout: Duration::from_secs(60),
        message_queue_size: 1000,
        health_check_interval: Duration::from_secs(5),
        agent_spawn_strategy: neuraldocflow::config::AgentSpawnStrategy::Adaptive,
        load_balancing: neuraldocflow::config::LoadBalancingStrategy::WorkStealing,
        enable_work_stealing: true,
        fault_tolerance: neuraldocflow::config::FaultToleranceConfig {
            enable_recovery: true,
            max_retries: 3,
            recovery_timeout: Duration::from_secs(30),
            health_threshold: 0.7,
        },
    };
    
    let config = Config {
        daa: daa_config,
        ..Default::default()
    };
    
    println!("🤖 Initializing DAA-coordinated DocFlow...");
    println!("Max agents: {}", config.daa.max_agents);
    println!("Consensus enabled: {}", config.daa.enable_consensus);
    println!("Consensus threshold: {:.1}%", config.daa.consensus_threshold * 100.0);
    
    let docflow = DocFlow::with_config(config)?;
    
    // Create sample documents for coordination testing
    create_sample_documents().await?;
    
    // Test single document coordination
    println!("\n📄 Testing single document coordination...");
    test_single_document_coordination(&docflow).await?;
    
    // Test batch coordination
    println!("\n📊 Testing batch coordination...");
    test_batch_coordination(&docflow).await?;
    
    // Test load balancing
    println!("\n⚖️  Testing load balancing...");
    test_load_balancing(&docflow).await?;
    
    // Test fault tolerance
    println!("\n🛡️  Testing fault tolerance...");
    test_fault_tolerance(&docflow).await?;
    
    // Monitor agent performance
    println!("\n📈 Monitoring agent performance...");
    monitor_agent_performance(&docflow).await?;
    
    // Test consensus validation
    println!("\n🤝 Testing consensus validation...");
    test_consensus_validation(&docflow).await?;
    
    // Cleanup
    cleanup_sample_documents().await?;
    
    println!("\n✅ DAA coordination demonstration completed!");
    
    Ok(())
}

/// Test coordination for a single document
async fn test_single_document_coordination(docflow: &DocFlow) -> Result<()> {
    let input = SourceInput::File {
        path: PathBuf::from("simple_document.txt"),
        metadata: Some({
            let mut meta = HashMap::new();
            meta.insert("priority".to_string(), "high".to_string());
            meta.insert("task_type".to_string(), "single_extraction".to_string());
            meta
        }),
    };
    
    let start_time = std::time::Instant::now();
    let document = docflow.extract(input).await?;
    let extraction_time = start_time.elapsed();
    
    println!("Single document coordination results:");
    println!("  - Extraction time: {:.2?}", extraction_time);
    println!("  - Confidence: {:.1}%", document.confidence * 100.0);
    println!("  - Content blocks: {}", document.content.len());
    
    // Show agent statistics
    let stats = docflow.get_stats();
    println!("  - Active agents: {}", stats.active_agents);
    println!("  - Total processing time: {:.2?}", stats.total_processing_time);
    
    Ok(())
}

/// Test coordination for batch processing
async fn test_batch_coordination(docflow: &DocFlow) -> Result<()> {
    let inputs = vec![
        SourceInput::File {
            path: PathBuf::from("document1.txt"),
            metadata: Some({
                let mut meta = HashMap::new();
                meta.insert("batch_id".to_string(), "batch_001".to_string());
                meta.insert("priority".to_string(), "medium".to_string());
                meta
            }),
        },
        SourceInput::File {
            path: PathBuf::from("document2.txt"),
            metadata: Some({
                let mut meta = HashMap::new();
                meta.insert("batch_id".to_string(), "batch_001".to_string());
                meta.insert("priority".to_string(), "medium".to_string());
                meta
            }),
        },
        SourceInput::File {
            path: PathBuf::from("document3.txt"),
            metadata: Some({
                let mut meta = HashMap::new();
                meta.insert("batch_id".to_string(), "batch_001".to_string());
                meta.insert("priority".to_string(), "low".to_string());
                meta
            }),
        },
    ];
    
    let start_time = std::time::Instant::now();
    let documents = docflow.extract_batch(inputs).await?;
    let batch_time = start_time.elapsed();
    
    println!("Batch coordination results:");
    println!("  - Total documents: {}", documents.len());
    println!("  - Batch processing time: {:.2?}", batch_time);
    println!("  - Average time per document: {:.2?}", batch_time / documents.len() as u32);
    
    let avg_confidence = documents.iter()
        .map(|d| d.confidence)
        .sum::<f32>() / documents.len() as f32;
    println!("  - Average confidence: {:.1}%", avg_confidence * 100.0);
    
    // Show agent coordination efficiency
    let stats = docflow.get_stats();
    println!("  - Final active agents: {}", stats.active_agents);
    
    // Calculate throughput
    let throughput = documents.len() as f64 / batch_time.as_secs_f64() * 60.0;
    println!("  - Throughput: {:.1} documents/minute", throughput);
    
    Ok(())
}

/// Test load balancing across agents
async fn test_load_balancing(docflow: &DocFlow) -> Result<()> {
    println!("Creating workload simulation...");
    
    // Create varied workload
    let mut inputs = Vec::new();
    
    // Light documents
    for i in 0..3 {
        inputs.push(SourceInput::Memory {
            data: format!("Light document {} with minimal content.", i).into_bytes(),
            filename: Some(format!("light_{}.txt", i)),
            mime_type: Some("text/plain".to_string()),
        });
    }
    
    // Medium documents  
    for i in 0..2 {
        let content = format!("Medium document {} with more substantial content. {}",
            i, "This document has multiple paragraphs and sections. ".repeat(10));
        inputs.push(SourceInput::Memory {
            data: content.into_bytes(),
            filename: Some(format!("medium_{}.txt", i)),
            mime_type: Some("text/plain".to_string()),
        });
    }
    
    // Heavy document
    let heavy_content = format!("Heavy document with extensive content. {}",
        "This is a very long document with lots of text that will take more time to process. ".repeat(50));
    inputs.push(SourceInput::Memory {
        data: heavy_content.into_bytes(),
        filename: Some("heavy.txt".to_string()),
        mime_type: Some("text/plain".to_string()),
    });
    
    println!("Processing varied workload: {} documents", inputs.len());
    
    let start_time = std::time::Instant::now();
    let documents = docflow.extract_batch(inputs).await?;
    let total_time = start_time.elapsed();
    
    println!("Load balancing results:");
    println!("  - Documents processed: {}", documents.len());
    println!("  - Total time: {:.2?}", total_time);
    
    // Analyze load distribution
    for (i, doc) in documents.iter().enumerate() {
        let doc_type = if doc.content.len() < 5 { "Light" }
                      else if doc.content.len() < 15 { "Medium" }
                      else { "Heavy" };
        
        println!("  - Doc {}: {} ({} blocks, {:.1}% confidence)", 
            i + 1, doc_type, doc.content.len(), doc.confidence * 100.0);
    }
    
    Ok(())
}

/// Test fault tolerance and recovery
async fn test_fault_tolerance(docflow: &DocFlow) -> Result<()> {
    println!("Testing fault tolerance with problematic inputs...");
    
    let inputs = vec![
        // Valid document
        SourceInput::Memory {
            data: "Valid document content".into_bytes(),
            filename: Some("valid.txt".to_string()),
            mime_type: Some("text/plain".to_string()),
        },
        
        // Empty document (edge case)
        SourceInput::Memory {
            data: Vec::new(),
            filename: Some("empty.txt".to_string()),
            mime_type: Some("text/plain".to_string()),
        },
        
        // Very large document (stress test)
        SourceInput::Memory {
            data: "Large content. ".repeat(10000).into_bytes(),
            filename: Some("large.txt".to_string()),
            mime_type: Some("text/plain".to_string()),
        },
        
        // Binary data (should fail gracefully)
        SourceInput::Memory {
            data: vec![0; 1000], // Binary data
            filename: Some("binary.bin".to_string()),
            mime_type: Some("application/octet-stream".to_string()),
        },
    ];
    
    println!("Processing {} test cases including edge cases...", inputs.len());
    
    let mut successful = 0;
    let mut failed = 0;
    
    for (i, input) in inputs.into_iter().enumerate() {
        match docflow.extract(input).await {
            Ok(document) => {
                successful += 1;
                println!("  ✅ Test case {}: Success (confidence: {:.1}%)", 
                    i + 1, document.confidence * 100.0);
            }
            Err(e) => {
                failed += 1;
                println!("  ❌ Test case {}: Failed gracefully ({})", i + 1, e);
            }
        }
    }
    
    println!("Fault tolerance results:");
    println!("  - Successful: {}", successful);
    println!("  - Failed gracefully: {}", failed);
    println!("  - System stability: {}%", 
        if successful + failed > 0 { 
            successful * 100 / (successful + failed) 
        } else { 
            0 
        });
    
    Ok(())
}

/// Monitor agent performance and health
async fn monitor_agent_performance(docflow: &DocFlow) -> Result<()> {
    println!("Monitoring agent performance...");
    
    // Get initial stats
    let initial_stats = docflow.get_stats();
    
    // Process some documents while monitoring
    let monitoring_inputs = vec![
        SourceInput::Memory {
            data: "Monitoring document 1".into_bytes(),
            filename: Some("monitor1.txt".to_string()),
            mime_type: Some("text/plain".to_string()),
        },
        SourceInput::Memory {
            data: "Monitoring document 2 with more content for testing".into_bytes(),
            filename: Some("monitor2.txt".to_string()),
            mime_type: Some("text/plain".to_string()),
        },
    ];
    
    // Monitor during processing
    let monitor_start = std::time::Instant::now();
    
    for (i, input) in monitoring_inputs.into_iter().enumerate() {
        let doc_start = std::time::Instant::now();
        
        match docflow.extract(input).await {
            Ok(document) => {
                let doc_time = doc_start.elapsed();
                let current_stats = docflow.get_stats();
                
                println!("  Document {} processed in {:.2?}", i + 1, doc_time);
                println!("    - Active agents: {}", current_stats.active_agents);
                println!("    - Documents processed: {}", current_stats.documents_processed);
                println!("    - Average confidence: {:.1}%", 
                    current_stats.average_confidence * 100.0);
                
                // Simulate some delay for monitoring
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
            Err(e) => {
                println!("  Document {} failed: {}", i + 1, e);
            }
        }
    }
    
    let monitoring_time = monitor_start.elapsed();
    let final_stats = docflow.get_stats();
    
    println!("Performance monitoring summary:");
    println!("  - Monitoring duration: {:.2?}", monitoring_time);
    println!("  - Total documents processed: {}", final_stats.documents_processed);
    println!("  - Total processing time: {:.2?}", final_stats.total_processing_time);
    println!("  - Final agent count: {}", final_stats.active_agents);
    
    // Calculate efficiency metrics
    if final_stats.documents_processed > initial_stats.documents_processed {
        let docs_processed = final_stats.documents_processed - initial_stats.documents_processed;
        let avg_time_per_doc = final_stats.total_processing_time / docs_processed as u32;
        println!("  - Average time per document: {:.2?}", avg_time_per_doc);
        
        let efficiency = docs_processed as f64 / monitoring_time.as_secs_f64();
        println!("  - Processing efficiency: {:.1} docs/second", efficiency);
    }
    
    Ok(())
}

/// Test consensus validation mechanism
async fn test_consensus_validation(docflow: &DocFlow) -> Result<()> {
    println!("Testing consensus validation...");
    
    // Create document that might have ambiguous content for consensus testing
    let ambiguous_content = r#"
This document contains some ambiguous elements that might
be interpreted differently by different agents.

Table or List?
Item 1    Value A
Item 2    Value B  
Item 3    Value C

Is this a table with columns, or a list with descriptions?
The consensus system should help resolve this ambiguity.

Additional text that might be formatted differently
depending on the agent's interpretation.
"#;
    
    let input = SourceInput::Memory {
        data: ambiguous_content.into_bytes(),
        filename: Some("ambiguous.txt".to_string()),
        mime_type: Some("text/plain".to_string()),
    };
    
    println!("Processing ambiguous document for consensus validation...");
    
    let start_time = std::time::Instant::now();
    let document = docflow.extract(input).await?;
    let consensus_time = start_time.elapsed();
    
    println!("Consensus validation results:");
    println!("  - Processing time: {:.2?}", consensus_time);
    println!("  - Final confidence: {:.1}%", document.confidence * 100.0);
    println!("  - Content blocks: {}", document.content.len());
    
    // Analyze block types to see how consensus resolved ambiguity
    let mut block_type_counts = HashMap::new();
    for block in &document.content {
        let type_name = format!("{:?}", block.block_type);
        *block_type_counts.entry(type_name).or_insert(0) += 1;
    }
    
    println!("  - Block type distribution:");
    for (block_type, count) in block_type_counts {
        println!("    - {}: {}", block_type, count);
    }
    
    // Check if tables were detected (consensus on ambiguous structure)
    let tables = document.get_tables();
    if !tables.is_empty() {
        println!("  ✅ Consensus reached: {} table(s) detected", tables.len());
        for (i, table) in tables.iter().enumerate() {
            println!("    - Table {}: {:.1}% confidence", 
                i + 1, table.metadata.confidence * 100.0);
        }
    } else {
        println!("  ℹ️  Consensus: No tables detected in ambiguous content");
    }
    
    // Show confidence distribution across blocks
    let confidences: Vec<f32> = document.content.iter()
        .map(|block| block.metadata.confidence)
        .collect();
    
    if !confidences.is_empty() {
        let min_conf = confidences.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_conf = confidences.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let avg_conf = confidences.iter().sum::<f32>() / confidences.len() as f32;
        
        println!("  - Block confidence range: {:.1}% - {:.1}% (avg: {:.1}%)", 
            min_conf * 100.0, max_conf * 100.0, avg_conf * 100.0);
    }
    
    Ok(())
}

/// Create sample documents for testing
async fn create_sample_documents() -> Result<()> {
    let documents = [
        ("simple_document.txt", "This is a simple document with basic content for testing DAA coordination."),
        ("document1.txt", "First document in batch processing test. Contains multiple sentences and paragraphs."),
        ("document2.txt", "Second document with different content structure. Has lists and formatted text."),
        ("document3.txt", "Third document focusing on technical content and specialized terminology."),
    ];
    
    for (filename, content) in &documents {
        std::fs::write(filename, content)?;
    }
    
    println!("📝 Created {} sample documents", documents.len());
    Ok(())
}

/// Cleanup sample documents
async fn cleanup_sample_documents() -> Result<()> {
    let files = [
        "simple_document.txt",
        "document1.txt", 
        "document2.txt",
        "document3.txt",
    ];
    
    for file in &files {
        if std::path::Path::new(file).exists() {
            std::fs::remove_file(file)?;
        }
    }
    
    println!("🧹 Cleaned up sample documents");
    Ok(())
}

/// Example of advanced DAA configuration
#[allow(dead_code)]
async fn example_advanced_daa_config() -> Result<()> {
    let advanced_config = Config {
        daa: DaaConfig {
            max_agents: 12,
            enable_consensus: true,
            consensus_threshold: 0.85,
            coordination_timeout: Duration::from_secs(120),
            message_queue_size: 2000,
            health_check_interval: Duration::from_secs(3),
            agent_spawn_strategy: neuraldocflow::config::AgentSpawnStrategy::Predictive,
            load_balancing: neuraldocflow::config::LoadBalancingStrategy::WeightedRoundRobin,
            enable_work_stealing: true,
            fault_tolerance: neuraldocflow::config::FaultToleranceConfig {
                enable_recovery: true,
                max_retries: 5,
                recovery_timeout: Duration::from_secs(45),
                health_threshold: 0.8,
            },
            performance_tuning: neuraldocflow::config::PerformanceTuningConfig {
                enable_agent_pooling: true,
                agent_pool_size: 8,
                enable_task_batching: true,
                batch_timeout: Duration::from_millis(100),
                enable_adaptive_scheduling: true,
                cpu_usage_threshold: 0.8,
                memory_usage_threshold: 0.75,
            },
        },
        ..Default::default()
    };
    
    let docflow = DocFlow::with_config(advanced_config)?;
    
    println!("✅ Configured advanced DAA system");
    
    // The system will now use advanced coordination strategies
    // including predictive agent spawning, weighted load balancing,
    // and adaptive performance tuning
    
    Ok(())
}

/// Example of custom agent implementation
#[allow(dead_code)]
async fn example_custom_agent() -> Result<()> {
    // This would be implemented if extending the DAA system
    // with custom agent types and behaviors
    
    println!("Custom agents can be implemented by extending the Agent trait");
    println!("and registering them with the DAA coordinator");
    
    Ok(())
}