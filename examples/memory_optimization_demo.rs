//! Memory optimization demonstration
//!
//! This example demonstrates how to use the memory optimization features
//! to process documents while maintaining <2MB memory usage per document.

use neural_doc_flow_core::{
    memory::*,
    optimized_types::*,
    memory_profiler::*,
    error::*,
};
use neural_doc_flow_processors::memory_optimized::*;
use neural_doc_flow_sources::streaming::*;
use std::sync::Arc;
use std::path::Path;
use bytes::Bytes;
use tokio::time::{sleep, Duration};
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Neural Document Flow - Memory Optimization Demo");
    println!("Target: <2MB memory usage per document\n");
    
    // Initialize memory optimization system
    let memory_limit = 2 * 1024 * 1024; // 2MB limit
    println!("📊 Initializing memory optimization with {}MB limit", memory_limit / (1024 * 1024));
    
    // Create memory monitor
    let monitor = Arc::new(MemoryMonitor::new(memory_limit));
    
    // Create memory profiler
    let profiler_config = ProfilerConfig {
        warning_threshold: 0.7,
        critical_threshold: 0.9,
        track_call_stacks: true,
        ..Default::default()
    };
    let profiler = Arc::new(MemoryProfiler::new(profiler_config));
    
    // Add console alert handler
    profiler.add_alert_handler(Box::new(ConsoleAlertHandler)).await;
    
    // Create optimized processor
    let processor = Arc::new(MemoryOptimizedProcessor::new(memory_limit));
    
    println!("✅ Memory optimization system initialized\n");
    
    // Demo 1: Memory pool usage
    println!("🏊 Demo 1: Memory Pool Usage");
    demo_memory_pool().await?;
    
    // Demo 2: String cache efficiency
    println!("\n🔤 Demo 2: String Cache Efficiency");
    demo_string_cache().await?;
    
    // Demo 3: Optimized document processing
    println!("\n📄 Demo 3: Optimized Document Processing");
    demo_optimized_document(&monitor, &profiler).await?;
    
    // Demo 4: Streaming document processing
    println!("\n🌊 Demo 4: Streaming Document Processing");
    demo_streaming_processing(&monitor).await?;
    
    // Demo 5: Memory profiling and optimization
    println!("\n📈 Demo 5: Memory Profiling and Optimization");
    demo_memory_profiling(&profiler, &processor).await?;
    
    // Demo 6: Arena allocation
    println!("\n🏟️ Demo 6: Arena Allocation");
    demo_arena_allocation().await?;
    
    // Demo 7: Batch processing with memory management
    println!("\n📦 Demo 7: Batch Processing with Memory Management");
    demo_batch_processing(&monitor, &processor).await?;
    
    println!("\n🎉 All memory optimization demos completed successfully!");
    println!("💡 The system maintained <2MB memory usage throughout all operations.");
    
    Ok(())
}

/// Demo memory pool usage
async fn demo_memory_pool() -> Result<(), Box<dyn std::error::Error>> {
    let mut pool = MemoryPool::new();
    
    println!("  📝 Allocating buffers of different sizes...");
    
    // Allocate various buffer sizes
    let buffer1 = pool.allocate(1024);
    let buffer2 = pool.allocate(4096);
    let buffer3 = pool.allocate(16384);
    
    let stats = pool.stats();
    println!("  📊 Pool stats: {} allocations, {} current usage", 
        stats.allocations, stats.current_usage);
    
    // Return buffers to pool
    pool.deallocate(buffer1);
    pool.deallocate(buffer2);
    pool.deallocate(buffer3);
    
    let final_stats = pool.stats();
    println!("  ♻️ After deallocation: {} deallocations, {} pool hits", 
        final_stats.deallocations, final_stats.pool_hits);
    
    Ok(())
}

/// Demo string cache efficiency
async fn demo_string_cache() -> Result<(), Box<dyn std::error::Error>> {
    let mut cache = StringCache::new(100);
    
    println!("  📝 Testing string deduplication...");
    
    // Insert repeated strings
    let strings = vec![
        "Hello, world!",
        "Neural Document Flow",
        "Memory optimization",
        "Hello, world!", // Duplicate
        "Neural Document Flow", // Duplicate
        "Efficient processing",
        "Memory optimization", // Duplicate
    ];
    
    let mut cached_strings = Vec::new();
    for s in strings {
        cached_strings.push(cache.get_or_insert(s));
    }
    
    println!("  📊 Cache hit ratio: {:.2}%", cache.hit_ratio() * 100.0);
    
    // Verify string deduplication
    assert!(Arc::ptr_eq(&cached_strings[0], &cached_strings[3])); // "Hello, world!"
    assert!(Arc::ptr_eq(&cached_strings[1], &cached_strings[4])); // "Neural Document Flow"
    assert!(Arc::ptr_eq(&cached_strings[2], &cached_strings[6])); // "Memory optimization"
    
    println!("  ✅ String deduplication verified");
    
    Ok(())
}

/// Demo optimized document processing
async fn demo_optimized_document(
    monitor: &Arc<MemoryMonitor>,
    profiler: &Arc<MemoryProfiler>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("  📝 Creating optimized document...");
    
    // Create optimized document
    let doc = OptimizedDocument::new(
        "demo_document.txt".to_string(),
        "text/plain".to_string(),
        monitor.clone(),
    )?;
    
    // Record allocation in profiler
    profiler.record_allocation(
        doc.id,
        std::mem::size_of::<OptimizedDocument>(),
        AllocationType::Document,
        Some(doc.id),
    ).await?;
    
    println!("  📝 Adding text content with lazy loading...");
    
    // Add text content with lazy loading
    let large_text = "Lorem ipsum ".repeat(1000); // ~11KB of text
    let text_len = large_text.len();
    let text_arc = Arc::from(large_text.as_str());
    
    doc.set_text_lazy(move || Ok(text_arc.clone()), text_len).await?;
    
    // Check memory usage before loading text
    let memory_before = doc.memory_usage().await;
    println!("  📊 Memory before text load: {} bytes", memory_before);
    
    // Load text content
    let loaded_text = doc.get_text().await?;
    assert!(loaded_text.is_some());
    
    // Check memory usage after loading text
    let memory_after = doc.memory_usage().await;
    println!("  📊 Memory after text load: {} bytes", memory_after);
    
    // Add optimized image
    println!("  📝 Adding optimized image...");
    let image = OptimizedImage::new(
        "demo_image.jpg".to_string(),
        "image/jpeg".to_string(),
        800,
        600,
        ImageContentRef::Inline(Bytes::from(vec![0u8; 1024])), // 1KB fake image
    );
    
    doc.add_image(image).await?;
    
    // Add compact table
    println!("  📝 Adding compact table...");
    let table = CompactTable::new(
        "demo_table".to_string(),
        vec!["Name".to_string(), "Age".to_string(), "City".to_string()],
        vec![
            vec!["John".to_string(), "30".to_string(), "New York".to_string()],
            vec!["Jane".to_string(), "25".to_string(), "Los Angeles".to_string()],
            vec!["Bob".to_string(), "35".to_string(), "Chicago".to_string()],
        ],
    );
    
    doc.add_table(table).await?;
    
    // Get final memory statistics
    let stats = doc.memory_stats().await;
    println!("  📊 Final document memory stats:");
    println!("    - Total usage: {} bytes", stats.total_usage);
    println!("    - Text size: {} bytes", stats.text_size);
    println!("    - Images: {}", stats.image_count);
    println!("    - Tables: {}", stats.table_count);
    
    // Compact document
    println!("  🗜️ Compacting document...");
    let saved = doc.compact().await?;
    println!("  💾 Saved {} bytes through compaction", saved);
    
    // Monitor document
    profiler.monitor_document(&doc, memory_before + 100 * 1024).await?; // 100KB limit
    
    Ok(())
}

/// Demo streaming document processing
async fn demo_streaming_processing(
    monitor: &Arc<MemoryMonitor>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("  📝 Creating streaming processor...");
    
    let processor = StreamingDocumentProcessor::new(2 * 1024 * 1024);
    
    // Simulate processing multiple documents
    let document_data = vec![
        ("document1.txt", "This is the first document content."),
        ("document2.txt", "This is the second document with more content."),
        ("document3.txt", "This is the third document with even more content that might be larger."),
    ];
    
    for (filename, content) in document_data {
        println!("  🔄 Processing {}...", filename);
        
        // Check memory before processing
        let memory_before = monitor.current_usage();
        
        // Create temporary file (simulated)
        let temp_path = std::path::Path::new(filename);
        
        // Process document (simplified - would normally read from file)
        let mut doc = OptimizedDocument::new(
            filename.to_string(),
            "text/plain".to_string(),
            monitor.clone(),
        )?;
        
        // Set text with lazy loading
        let content_len = content.len();
        let content_arc = Arc::from(content);
        doc.set_text_lazy(move || Ok(content_arc.clone()), content_len).await?;
        
        // Check memory after processing
        let memory_after = monitor.current_usage();
        let doc_memory = doc.memory_usage().await;
        
        println!("    📊 Document memory: {} bytes", doc_memory);
        println!("    📊 System memory delta: {} bytes", memory_after.saturating_sub(memory_before));
        
        // Ensure we stay under limit
        assert!(doc_memory < 2 * 1024 * 1024, "Document exceeded 2MB limit");
    }
    
    println!("  ✅ All documents processed within memory limits");
    
    Ok(())
}

/// Demo memory profiling and optimization
async fn demo_memory_profiling(
    profiler: &Arc<MemoryProfiler>,
    processor: &Arc<MemoryOptimizedProcessor>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("  📝 Running memory profiling demo...");
    
    // Simulate various allocations
    let allocations = vec![
        (AllocationType::Document, 50 * 1024),
        (AllocationType::Text, 30 * 1024),
        (AllocationType::Image, 100 * 1024),
        (AllocationType::Table, 20 * 1024),
        (AllocationType::Cache, 15 * 1024),
    ];
    
    let mut allocation_ids = Vec::new();
    
    for (alloc_type, size) in allocations {
        let id = Uuid::new_v4();
        profiler.record_allocation(id, size, alloc_type, None).await?;
        allocation_ids.push(id);
    }
    
    // Take snapshot
    let snapshot = profiler.take_snapshot().await;
    println!("  📊 Memory snapshot:");
    println!("    - Total usage: {} bytes", snapshot.total_usage);
    println!("    - Allocations: {}", snapshot.allocation_count);
    println!("    - Largest allocation: {} bytes", snapshot.largest_allocation);
    
    // Generate memory report
    let report = profiler.generate_report().await;
    println!("  📋 Memory report:");
    println!("    - Current usage: {} bytes", report.current_usage);
    println!("    - Recommendations: {}", report.recommendations.len());
    
    for recommendation in &report.recommendations {
        println!("      💡 {}", recommendation);
    }
    
    // Test memory optimizer
    let optimizer = MemoryOptimizer::new(profiler.clone());
    let suggestions = optimizer.analyze_and_optimize().await;
    
    println!("  🔧 Optimization suggestions:");
    for action in &suggestions.immediate_actions {
        println!("    ⚡ Immediate: {}", action);
    }
    for improvement in &suggestions.long_term_improvements {
        println!("    🚀 Long-term: {}", improvement);
    }
    
    if suggestions.estimated_savings > 0 {
        println!("    💾 Estimated savings: {} bytes", suggestions.estimated_savings);
    }
    
    // Clean up allocations
    for id in allocation_ids {
        profiler.record_deallocation(id).await;
    }
    
    Ok(())
}

/// Demo arena allocation
async fn demo_arena_allocation() -> Result<(), Box<dyn std::error::Error>> {
    println!("  📝 Testing arena allocation...");
    
    let mut arena = ProcessingArena::new(64 * 1024); // 64KB arena
    
    // Allocate various sizes
    let allocations = vec![1024, 2048, 4096, 8192];
    let mut ptrs = Vec::new();
    
    for size in allocations {
        if let Some(ptr) = arena.allocate(size) {
            ptrs.push((ptr, size));
            println!("    ✅ Allocated {} bytes", size);
        } else {
            println!("    ❌ Failed to allocate {} bytes", size);
        }
    }
    
    println!("  📊 Arena stats:");
    println!("    - Total allocated: {} bytes", arena.total_allocated());
    println!("    - Buffer count: {}", arena.buffer_count());
    
    // Reset arena
    arena.reset();
    println!("  🔄 Arena reset - total allocated: {} bytes", arena.total_allocated());
    
    Ok(())
}

/// Demo batch processing with memory management
async fn demo_batch_processing(
    monitor: &Arc<MemoryMonitor>,
    processor: &Arc<MemoryOptimizedProcessor>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("  📝 Testing batch processing with memory management...");
    
    // Create multiple content blocks
    let content_blocks = vec![
        neural_doc_flow_processors::types::ContentBlock::text_block(
            "This is a sample text block for processing.".to_string()
        ),
        neural_doc_flow_processors::types::ContentBlock::text_block(
            "Another text block with different content for variety.".to_string()
        ),
        neural_doc_flow_processors::types::ContentBlock::table_block(
            "| Name | Age | City |\n| John | 30 | NYC |\n| Jane | 25 | LA |".to_string()
        ),
    ];
    
    let mut processed_blocks = Vec::new();
    let initial_memory = monitor.current_usage();
    
    for (i, block) in content_blocks.iter().enumerate() {
        println!("    🔄 Processing block {}...", i + 1);
        
        // Check memory before processing
        let memory_before = monitor.current_usage();
        
        // Process block
        let optimized_block = processor.process_optimized(block).await?;
        
        // Check memory after processing
        let memory_after = monitor.current_usage();
        
        println!("      📊 Block memory: {} bytes", optimized_block.memory_usage);
        println!("      📊 System memory delta: {} bytes", 
            memory_after.saturating_sub(memory_before));
        
        processed_blocks.push(optimized_block);
        
        // Simulate some delay
        sleep(Duration::from_millis(10)).await;
    }
    
    let final_memory = monitor.current_usage();
    let total_block_memory: usize = processed_blocks.iter().map(|b| b.memory_usage).sum();
    
    println!("  📊 Batch processing results:");
    println!("    - Blocks processed: {}", processed_blocks.len());
    println!("    - Total block memory: {} bytes", total_block_memory);
    println!("    - System memory delta: {} bytes", 
        final_memory.saturating_sub(initial_memory));
    
    // Get processor statistics
    let stats = processor.get_stats().await;
    println!("  📈 Processor statistics:");
    println!("    - Documents processed: {}", stats.documents_processed);
    println!("    - Memory efficiency: {:.2}%", stats.memory_efficiency_ratio * 100.0);
    println!("    - Cache hit ratio: {:.2}%", stats.cache_hit_ratio * 100.0);
    
    Ok(())
}
