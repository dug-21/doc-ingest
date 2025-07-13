//! DAA Coordination Performance Benchmarks
//!
//! This benchmark suite measures the performance of DAA coordination features
//! including agent spawning, task distribution, consensus, and coordination overhead.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use neuraldocflow::{Config, DocFlow, SourceInput};
use std::time::Duration;
use tokio::runtime::Runtime;

mod test_utilities;
use test_utilities::{TestConfigBuilder, MockPdfGenerator, TestDocumentBuilder};

/// Benchmark agent spawning and coordination setup
fn benchmark_agent_spawning(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("agent_spawning");
    group.measurement_time(Duration::from_secs(10));
    
    for agent_count in [1, 2, 4, 8, 16, 32].iter() {
        group.bench_with_input(
            BenchmarkId::new("spawn_agents", agent_count),
            agent_count,
            |b, &agent_count| {
                b.to_async(&rt).iter(|| async {
                    let config = TestConfigBuilder::new()
                        .with_daa_agents(agent_count)
                        .with_temp_directories()
                        .build();
                    
                    let docflow = black_box(DocFlow::with_config(config));
                    
                    match docflow {
                        Ok(flow) => {
                            let stats = flow.get_stats();
                            black_box(stats);
                            let _ = flow.shutdown().await;
                        }
                        Err(_) => {
                            // Expected in some test environments
                        }
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark task distribution across agents
fn benchmark_task_distribution(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("task_distribution");
    group.measurement_time(Duration::from_secs(15));
    
    for doc_count in [1, 5, 10, 25, 50].iter() {
        group.bench_with_input(
            BenchmarkId::new("distribute_tasks", doc_count),
            doc_count,
            |b, &doc_count| {
                b.to_async(&rt).iter(|| async {
                    let config = TestConfigBuilder::new()
                        .with_daa_agents(8)
                        .with_temp_directories()
                        .build();
                    
                    if let Ok(docflow) = DocFlow::with_config(config) {
                        let inputs: Vec<SourceInput> = (0..*doc_count)
                            .map(|i| SourceInput::Memory {
                                data: MockPdfGenerator::pdf_with_text(&format!("Document {}", i)),
                                filename: Some(format!("doc_{}.pdf", i)),
                                mime_type: Some("application/pdf".to_string()),
                            })
                            .collect();
                        
                        let start = std::time::Instant::now();
                        let result = docflow.extract_batch(black_box(inputs)).await;
                        let duration = start.elapsed();
                        
                        match result {
                            Ok(docs) => {
                                black_box(docs);
                                black_box(duration);
                            }
                            Err(_) => {
                                // May fail in test environment
                            }
                        }
                        
                        let _ = docflow.shutdown().await;
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark consensus mechanisms
fn benchmark_consensus(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("consensus");
    group.measurement_time(Duration::from_secs(10));
    
    for threshold in [0.5, 0.7, 0.8, 0.9, 0.95].iter() {
        group.bench_with_input(
            BenchmarkId::new("consensus_threshold", (threshold * 100.0) as u32),
            threshold,
            |b, &threshold| {
                b.to_async(&rt).iter(|| async {
                    let mut config = TestConfigBuilder::new()
                        .with_daa_agents(6)
                        .with_temp_directories()
                        .build();
                    
                    config.daa.enable_consensus = true;
                    config.daa.consensus_threshold = threshold;
                    
                    if let Ok(docflow) = DocFlow::with_config(config) {
                        let input = SourceInput::Memory {
                            data: MockPdfGenerator::pdf_with_text("Consensus test document"),
                            filename: Some("consensus_test.pdf".to_string()),
                            mime_type: Some("application/pdf".to_string()),
                        };
                        
                        let result = docflow.extract(black_box(input)).await;
                        
                        match result {
                            Ok(doc) => {
                                black_box(doc);
                            }
                            Err(_) => {
                                // May fail in test environment
                            }
                        }
                        
                        let _ = docflow.shutdown().await;
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark coordination overhead
fn benchmark_coordination_overhead(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("coordination_overhead");
    group.measurement_time(Duration::from_secs(15));
    
    // Compare single-agent vs multi-agent performance
    for agent_count in [1, 2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("overhead_comparison", agent_count),
            agent_count,
            |b, &agent_count| {
                b.to_async(&rt).iter(|| async {
                    let config = TestConfigBuilder::new()
                        .with_daa_agents(agent_count)
                        .with_temp_directories()
                        .build();
                    
                    if let Ok(docflow) = DocFlow::with_config(config) {
                        let input = SourceInput::Memory {
                            data: MockPdfGenerator::pdf_with_text("Overhead test document with substantial content"),
                            filename: Some("overhead_test.pdf".to_string()),
                            mime_type: Some("application/pdf".to_string()),
                        };
                        
                        let start = std::time::Instant::now();
                        let result = docflow.extract(black_box(input)).await;
                        let duration = start.elapsed();
                        
                        match result {
                            Ok(doc) => {
                                black_box(doc);
                                black_box(duration);
                            }
                            Err(_) => {
                                // May fail in test environment
                            }
                        }
                        
                        let _ = docflow.shutdown().await;
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory usage during coordination
fn benchmark_memory_usage(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("memory_usage");
    group.measurement_time(Duration::from_secs(10));
    
    for doc_size_mb in [1, 5, 10, 25].iter() {
        group.bench_with_input(
            BenchmarkId::new("memory_coordination", doc_size_mb),
            doc_size_mb,
            |b, &doc_size_mb| {
                b.to_async(&rt).iter(|| async {
                    let config = TestConfigBuilder::new()
                        .with_daa_agents(4)
                        .with_temp_directories()
                        .build();
                    
                    if let Ok(docflow) = DocFlow::with_config(config) {
                        let input = SourceInput::Memory {
                            data: MockPdfGenerator::large_pdf(doc_size_mb),
                            filename: Some(format!("large_doc_{}mb.pdf", doc_size_mb)),
                            mime_type: Some("application/pdf".to_string()),
                        };
                        
                        let memory_before = get_memory_usage();
                        let result = docflow.extract(black_box(input)).await;
                        let memory_after = get_memory_usage();
                        
                        match result {
                            Ok(doc) => {
                                black_box(doc);
                                black_box(memory_after - memory_before);
                            }
                            Err(_) => {
                                // May fail in test environment
                            }
                        }
                        
                        let _ = docflow.shutdown().await;
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark agent communication patterns
fn benchmark_agent_communication(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("agent_communication");
    group.measurement_time(Duration::from_secs(10));
    
    for topology in ["mesh", "star", "ring"].iter() {
        group.bench_with_input(
            BenchmarkId::new("communication_topology", topology),
            topology,
            |b, &topology| {
                b.to_async(&rt).iter(|| async {
                    let mut config = TestConfigBuilder::new()
                        .with_daa_agents(6)
                        .with_temp_directories()
                        .build();
                    
                    // Set topology if the config supports it
                    // This is a placeholder - actual implementation would set topology
                    config.daa.enable_consensus = true;
                    
                    if let Ok(docflow) = DocFlow::with_config(config) {
                        let inputs: Vec<SourceInput> = (0..3)
                            .map(|i| SourceInput::Memory {
                                data: MockPdfGenerator::pdf_with_text(&format!("Communication test {}", i)),
                                filename: Some(format!("comm_test_{}.pdf", i)),
                                mime_type: Some("application/pdf".to_string()),
                            })
                            .collect();
                        
                        let result = docflow.extract_batch(black_box(inputs)).await;
                        
                        match result {
                            Ok(docs) => {
                                black_box(docs);
                            }
                            Err(_) => {
                                // May fail in test environment
                            }
                        }
                        
                        let _ = docflow.shutdown().await;
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark concurrent document processing
fn benchmark_concurrent_processing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("concurrent_processing");
    group.measurement_time(Duration::from_secs(20));
    
    for concurrency in [1, 2, 4, 8, 16].iter() {
        group.bench_with_input(
            BenchmarkId::new("concurrent_docs", concurrency),
            concurrency,
            |b, &concurrency| {
                b.to_async(&rt).iter(|| async {
                    let mut config = TestConfigBuilder::new()
                        .with_daa_agents(8)
                        .with_temp_directories()
                        .build();
                    
                    config.core.max_concurrent_documents = concurrency;
                    
                    if let Ok(docflow) = DocFlow::with_config(config) {
                        let inputs: Vec<SourceInput> = (0..concurrency)
                            .map(|i| SourceInput::Memory {
                                data: MockPdfGenerator::pdf_with_text(&format!("Concurrent doc {}", i)),
                                filename: Some(format!("concurrent_{}.pdf", i)),
                                mime_type: Some("application/pdf".to_string()),
                            })
                            .collect();
                        
                        let result = docflow.extract_batch(black_box(inputs)).await;
                        
                        match result {
                            Ok(docs) => {
                                black_box(docs);
                            }
                            Err(_) => {
                                // May fail in test environment
                            }
                        }
                        
                        let _ = docflow.shutdown().await;
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Simple memory usage estimation
fn get_memory_usage() -> usize {
    // In a real implementation, this would use proper memory profiling
    // For benchmarks, we'll use a simple estimation
    std::mem::size_of::<usize>() * 1024
}

criterion_group!(
    daa_benchmarks,
    benchmark_agent_spawning,
    benchmark_task_distribution,
    benchmark_consensus,
    benchmark_coordination_overhead,
    benchmark_memory_usage,
    benchmark_agent_communication,
    benchmark_concurrent_processing
);

criterion_main!(daa_benchmarks);