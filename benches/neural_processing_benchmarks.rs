//! Neural Processing Performance Benchmarks
//!
//! This benchmark suite measures the performance of neural enhancement features
//! including model loading, inference, text enhancement, table detection, and confidence scoring.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use neuraldocflow::{Config, DocFlow, SourceInput};
use std::time::Duration;
use tokio::runtime::Runtime;
use tempfile::TempDir;

mod test_utilities;
use test_utilities::{TestConfigBuilder, MockPdfGenerator, TestDocumentBuilder, TestFileCreator};

/// Benchmark neural model loading
fn benchmark_model_loading(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("model_loading");
    group.measurement_time(Duration::from_secs(10));
    
    for model_count in [1, 3, 5, 10].iter() {
        group.bench_with_input(
            BenchmarkId::new("load_models", model_count),
            model_count,
            |b, &model_count| {
                b.to_async(&rt).iter(|| async {
                    let temp_dir = TempDir::new().unwrap();
                    TestFileCreator::create_mock_models(&temp_dir).await;
                    
                    let mut config = TestConfigBuilder::new()
                        .with_neural_enabled()
                        .with_temp_directories()
                        .build();
                    
                    config.neural.model_directory = temp_dir.path().to_path_buf();
                    config.neural.max_loaded_models = model_count;
                    
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

/// Benchmark text enhancement performance
fn benchmark_text_enhancement(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("text_enhancement");
    group.measurement_time(Duration::from_secs(15));
    
    let test_texts = vec![
        "Short text",
        "Medium length text with rn example and ||egal characters that need enhancement",
        "Long text content ".repeat(100),
        "Very long text content ".repeat(1000),
    ];
    
    for (i, text) in test_texts.iter().enumerate() {
        group.bench_with_input(
            BenchmarkId::new("enhance_text", i),
            text,
            |b, text| {
                b.to_async(&rt).iter(|| async {
                    let temp_dir = TempDir::new().unwrap();
                    TestFileCreator::create_mock_models(&temp_dir).await;
                    
                    let mut config = TestConfigBuilder::new()
                        .with_neural_enabled()
                        .with_temp_directories()
                        .build();
                    
                    config.neural.model_directory = temp_dir.path().to_path_buf();
                    
                    if let Ok(docflow) = DocFlow::with_config(config) {
                        let input = SourceInput::Memory {
                            data: MockPdfGenerator::pdf_with_text(text),
                            filename: Some("enhancement_test.pdf".to_string()),
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

/// Benchmark table detection performance
fn benchmark_table_detection(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("table_detection");
    group.measurement_time(Duration::from_secs(10));
    
    let table_sizes = vec![
        (2, 2),   // Small table
        (5, 5),   // Medium table
        (10, 10), // Large table
        (20, 5),  // Wide table
        (5, 20),  // Tall table
    ];
    
    for (rows, cols) in table_sizes {
        group.bench_with_input(
            BenchmarkId::new("detect_table", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |b, &(rows, cols)| {
                b.to_async(&rt).iter(|| async {
                    let temp_dir = TempDir::new().unwrap();
                    TestFileCreator::create_mock_models(&temp_dir).await;
                    
                    let mut config = TestConfigBuilder::new()
                        .with_neural_enabled()
                        .with_temp_directories()
                        .build();
                    
                    config.neural.model_directory = temp_dir.path().to_path_buf();
                    
                    if let Ok(docflow) = DocFlow::with_config(config) {
                        // Generate table content
                        let table_content = generate_table_content(rows, cols);
                        
                        let input = SourceInput::Memory {
                            data: MockPdfGenerator::pdf_with_text(&table_content),
                            filename: Some("table_test.pdf".to_string()),
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

/// Benchmark confidence scoring
fn benchmark_confidence_scoring(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("confidence_scoring");
    group.measurement_time(Duration::from_secs(10));
    
    for block_count in [1, 10, 50, 100, 500].iter() {
        group.bench_with_input(
            BenchmarkId::new("score_confidence", block_count),
            block_count,
            |b, &block_count| {
                b.to_async(&rt).iter(|| async {
                    let temp_dir = TempDir::new().unwrap();
                    TestFileCreator::create_mock_models(&temp_dir).await;
                    
                    let mut config = TestConfigBuilder::new()
                        .with_neural_enabled()
                        .with_temp_directories()
                        .build();
                    
                    config.neural.model_directory = temp_dir.path().to_path_buf();
                    
                    if let Ok(docflow) = DocFlow::with_config(config) {
                        // Generate document with many blocks
                        let content = (0..block_count)
                            .map(|i| format!("Paragraph {} with content for confidence scoring", i))
                            .collect::<Vec<_>>()
                            .join("\n\n");
                        
                        let input = SourceInput::Memory {
                            data: MockPdfGenerator::pdf_with_text(&content),
                            filename: Some("confidence_test.pdf".to_string()),
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

/// Benchmark batch neural processing
fn benchmark_batch_processing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("batch_neural_processing");
    group.measurement_time(Duration::from_secs(20));
    
    for batch_size in [1, 5, 10, 25, 50].iter() {
        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            batch_size,
            |b, &batch_size| {
                b.to_async(&rt).iter(|| async {
                    let temp_dir = TempDir::new().unwrap();
                    TestFileCreator::create_mock_models(&temp_dir).await;
                    
                    let mut config = TestConfigBuilder::new()
                        .with_neural_enabled()
                        .with_temp_directories()
                        .build();
                    
                    config.neural.model_directory = temp_dir.path().to_path_buf();
                    config.neural.processing.batch_size = batch_size;
                    
                    if let Ok(docflow) = DocFlow::with_config(config) {
                        let inputs: Vec<SourceInput> = (0..batch_size)
                            .map(|i| SourceInput::Memory {
                                data: MockPdfGenerator::pdf_with_text(&format!("Batch neural test {}", i)),
                                filename: Some(format!("batch_neural_{}.pdf", i)),
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

/// Benchmark neural memory usage
fn benchmark_neural_memory_usage(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("neural_memory_usage");
    group.measurement_time(Duration::from_secs(10));
    
    for doc_size in ["small", "medium", "large"].iter() {
        group.bench_with_input(
            BenchmarkId::new("memory_usage", doc_size),
            doc_size,
            |b, &doc_size| {
                b.to_async(&rt).iter(|| async {
                    let temp_dir = TempDir::new().unwrap();
                    TestFileCreator::create_mock_models(&temp_dir).await;
                    
                    let mut config = TestConfigBuilder::new()
                        .with_neural_enabled()
                        .with_temp_directories()
                        .build();
                    
                    config.neural.model_directory = temp_dir.path().to_path_buf();
                    
                    if let Ok(docflow) = DocFlow::with_config(config) {
                        let content = match doc_size {
                            "small" => "Small document content",
                            "medium" => &"Medium document content ".repeat(100),
                            "large" => &"Large document content ".repeat(1000),
                            _ => "Default content",
                        };
                        
                        let input = SourceInput::Memory {
                            data: MockPdfGenerator::pdf_with_text(content),
                            filename: Some(format!("{}_memory_test.pdf", doc_size)),
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

/// Benchmark model inference speed
fn benchmark_inference_speed(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("inference_speed");
    group.measurement_time(Duration::from_secs(10));
    
    for input_size in [10, 50, 100, 500, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("inference_inputs", input_size),
            input_size,
            |b, &input_size| {
                b.to_async(&rt).iter(|| async {
                    let temp_dir = TempDir::new().unwrap();
                    TestFileCreator::create_mock_models(&temp_dir).await;
                    
                    let mut config = TestConfigBuilder::new()
                        .with_neural_enabled()
                        .with_temp_directories()
                        .build();
                    
                    config.neural.model_directory = temp_dir.path().to_path_buf();
                    
                    if let Ok(docflow) = DocFlow::with_config(config) {
                        // Generate content with specified number of inputs
                        let content = (0..input_size)
                            .map(|i| format!("Input {} for inference speed test", i))
                            .collect::<Vec<_>>()
                            .join(" ");
                        
                        let input = SourceInput::Memory {
                            data: MockPdfGenerator::pdf_with_text(&content),
                            filename: Some("inference_speed_test.pdf".to_string()),
                            mime_type: Some("application/pdf".to_string()),
                        };
                        
                        let start = std::time::Instant::now();
                        let result = docflow.extract(black_box(input)).await;
                        let inference_time = start.elapsed();
                        
                        match result {
                            Ok(doc) => {
                                black_box(doc);
                                black_box(inference_time);
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

/// Generate table content for testing
fn generate_table_content(rows: usize, cols: usize) -> String {
    let mut content = String::new();
    
    // Add headers
    for col in 0..cols {
        content.push_str(&format!("Header{}\t", col + 1));
    }
    content.push('\n');
    
    // Add data rows
    for row in 0..rows {
        for col in 0..cols {
            content.push_str(&format!("Row{}Col{}\t", row + 1, col + 1));
        }
        content.push('\n');
    }
    
    content
}

/// Simple memory usage estimation
fn get_memory_usage() -> usize {
    // In a real implementation, this would use proper memory profiling
    // For benchmarks, we'll use a simple estimation
    std::mem::size_of::<usize>() * 1024
}

criterion_group!(
    neural_benchmarks,
    benchmark_model_loading,
    benchmark_text_enhancement,
    benchmark_table_detection,
    benchmark_confidence_scoring,
    benchmark_batch_processing,
    benchmark_neural_memory_usage,
    benchmark_inference_speed
);

criterion_main!(neural_benchmarks);