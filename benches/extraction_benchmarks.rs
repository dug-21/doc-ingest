//! Performance benchmarks for document extraction
//!
//! These benchmarks measure extraction performance across different
//! document types and sizes to ensure we meet >85% coverage targets.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use neuraldocflow::{Config, DocFlow, SourceInput};
use std::time::Duration;
use tempfile::NamedTempFile;

/// Benchmark single PDF extraction
fn bench_pdf_extraction(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("pdf_extraction");
    
    for size in [1, 10, 50, 100].iter() {
        let pdf_content = create_mock_pdf_content(*size);
        
        group.throughput(Throughput::Bytes(pdf_content.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("single_pdf", format!("{}kb", pdf_content.len() / 1024)),
            &pdf_content,
            |b, content| {
                b.to_async(&rt).iter(|| async {
                    let config = create_benchmark_config();
                    let docflow = DocFlow::with_config(config).unwrap();
                    
                    let input = SourceInput::Memory {
                        data: content.clone(),
                        filename: Some("benchmark.pdf".to_string()),
                        mime_type: Some("application/pdf".to_string()),
                    };
                    
                    black_box(docflow.extract(input).await)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark batch extraction
fn bench_batch_extraction(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("batch_extraction");
    
    for batch_size in [5, 10, 25, 50].iter() {
        let inputs = create_batch_inputs(*batch_size);
        
        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_pdf", batch_size),
            &inputs,
            |b, inputs| {
                b.to_async(&rt).iter(|| async {
                    let config = create_benchmark_config();
                    let docflow = DocFlow::with_config(config).unwrap();
                    
                    black_box(docflow.extract_batch(inputs.clone()).await)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark concurrent extraction
fn bench_concurrent_extraction(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("concurrent_extraction");
    group.sample_size(10); // Fewer samples for concurrent tests
    
    for concurrency in [2, 4, 8, 16].iter() {
        let inputs = create_batch_inputs(*concurrency);
        
        group.throughput(Throughput::Elements(*concurrency as u64));
        group.bench_with_input(
            BenchmarkId::new("concurrent", concurrency),
            &inputs,
            |b, inputs| {
                b.to_async(&rt).iter(|| async {
                    let config = create_benchmark_config();
                    let docflow = DocFlow::with_config(config).unwrap();
                    
                    let tasks: Vec<_> = inputs
                        .iter()
                        .map(|input| {
                            let docflow = &docflow;
                            let input = input.clone();
                            tokio::spawn(async move {
                                docflow.extract(input).await
                            })
                        })
                        .collect();
                    
                    let results: Vec<_> = futures::future::join_all(tasks).await
                        .into_iter()
                        .collect();
                    
                    black_box(results)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark neural enhancement
fn bench_neural_enhancement(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("neural_enhancement");
    
    for complexity in ["simple", "complex"].iter() {
        let content = match *complexity {
            "simple" => create_simple_content(),
            "complex" => create_complex_content(),
            _ => unreachable!(),
        };
        
        group.bench_with_input(
            BenchmarkId::new("neural", complexity),
            &content,
            |b, content| {
                b.to_async(&rt).iter(|| async {
                    let mut config = create_benchmark_config();
                    config.neural.enabled = true;
                    
                    let docflow = DocFlow::with_config(config).unwrap();
                    
                    let input = SourceInput::Memory {
                        data: content.clone(),
                        filename: Some("neural_test.pdf".to_string()),
                        mime_type: Some("application/pdf".to_string()),
                    };
                    
                    black_box(docflow.extract(input).await)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark DAA coordination
fn bench_daa_coordination(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("daa_coordination");
    
    for agent_count in [2, 4, 8, 16].iter() {
        let content = create_mock_pdf_content(50); // 50KB documents
        
        group.bench_with_input(
            BenchmarkId::new("daa_agents", agent_count),
            &content,
            |b, content| {
                b.to_async(&rt).iter(|| async {
                    let mut config = create_benchmark_config();
                    config.daa.max_agents = *agent_count;
                    config.daa.enable_consensus = true;
                    
                    let docflow = DocFlow::with_config(config).unwrap();
                    
                    let input = SourceInput::Memory {
                        data: content.clone(),
                        filename: Some("daa_test.pdf".to_string()),
                        mime_type: Some("application/pdf".to_string()),
                    };
                    
                    black_box(docflow.extract(input).await)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory usage patterns
fn bench_memory_usage(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("memory_usage");
    
    for memory_limit in [50, 100, 200, 500].iter() {
        let content = create_large_pdf_content(*memory_limit / 2); // Half the limit
        
        group.bench_with_input(
            BenchmarkId::new("memory_limit", format!("{}mb", memory_limit)),
            &content,
            |b, content| {
                b.to_async(&rt).iter(|| async {
                    let mut config = create_benchmark_config();
                    config.performance.max_memory_usage = memory_limit * 1024 * 1024;
                    
                    let docflow = DocFlow::with_config(config).unwrap();
                    
                    let input = SourceInput::Memory {
                        data: content.clone(),
                        filename: Some("memory_test.pdf".to_string()),
                        mime_type: Some("application/pdf".to_string()),
                    };
                    
                    black_box(docflow.extract(input).await)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark error recovery scenarios
fn bench_error_recovery(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("error_recovery");
    
    let test_cases = vec![
        ("corrupted_pdf", create_corrupted_pdf()),
        ("invalid_format", b"This is not a PDF file".to_vec()),
        ("empty_file", Vec::new()),
        ("malformed_header", b"%PDF-999.9\nmalformed".to_vec()),
    ];
    
    for (name, content) in test_cases {
        group.bench_with_input(
            BenchmarkId::new("error_recovery", name),
            &content,
            |b, content| {
                b.to_async(&rt).iter(|| async {
                    let config = create_benchmark_config();
                    let docflow = DocFlow::with_config(config).unwrap();
                    
                    let input = SourceInput::Memory {
                        data: content.clone(),
                        filename: Some(format!("{}.pdf", name)),
                        mime_type: Some("application/pdf".to_string()),
                    };
                    
                    // This should handle errors gracefully
                    let result = docflow.extract(input).await;
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark different source types
fn bench_multi_source_extraction(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("multi_source");
    
    let sources = vec![
        ("pdf", create_mock_pdf_content(25), "application/pdf"),
        ("docx", create_mock_docx_content(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        ("html", create_mock_html_content(), "text/html"),
    ];
    
    for (source_type, content, mime_type) in sources {
        group.bench_with_input(
            BenchmarkId::new("source_type", source_type),
            &(content, mime_type),
            |b, (content, mime_type)| {
                b.to_async(&rt).iter(|| async {
                    let config = create_benchmark_config();
                    let docflow = DocFlow::with_config(config).unwrap();
                    
                    let input = SourceInput::Memory {
                        data: content.clone(),
                        filename: Some(format!("test.{}", source_type)),
                        mime_type: Some(mime_type.to_string()),
                    };
                    
                    black_box(docflow.extract(input).await)
                });
            },
        );
    }
    
    group.finish();
}

// Helper functions for benchmark setup

fn create_benchmark_config() -> Config {
    let mut config = Config::default();
    
    // Optimize for benchmarking
    config.core.worker_threads = num_cpus::get();
    config.core.max_concurrent_documents = 100;
    config.daa.max_agents = 8;
    config.neural.enabled = false; // Disable by default for speed
    config.performance.max_memory_usage = 1024 * 1024 * 1024; // 1GB
    config.security.enabled = false; // Skip security checks for benchmarking
    
    // Use temp directories
    let temp_dir = std::env::temp_dir().join("neuraldocflow_bench");
    config.core.temp_directory = temp_dir.clone();
    config.neural.model_directory = temp_dir.join("models");
    
    // Ensure directories exist
    std::fs::create_dir_all(&config.core.temp_directory).unwrap_or_default();
    std::fs::create_dir_all(&config.neural.model_directory).unwrap_or_default();
    
    config
}

fn create_mock_pdf_content(size_kb: usize) -> Vec<u8> {
    let mut content = Vec::new();
    content.extend_from_slice(b"%PDF-1.4\n");
    content.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    content.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    content.extend_from_slice(b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n");
    
    // Pad to desired size
    let target_size = size_kb * 1024;
    while content.len() < target_size {
        content.extend_from_slice(b"% Padding content to reach target size\n");
        content.extend_from_slice(b"4 0 obj\n<< /Length 50 >>\nstream\nThis is some test content in the PDF stream.\nendstream\nendobj\n");
    }
    
    content.extend_from_slice(b"xref\ntrailer\n<< /Size 5 /Root 1 0 R >>\n%%EOF\n");
    content
}

fn create_large_pdf_content(size_mb: usize) -> Vec<u8> {
    create_mock_pdf_content(size_mb * 1024)
}

fn create_batch_inputs(count: usize) -> Vec<SourceInput> {
    (0..count)
        .map(|i| SourceInput::Memory {
            data: create_mock_pdf_content(10 + i), // Varying sizes
            filename: Some(format!("batch_{}.pdf", i)),
            mime_type: Some("application/pdf".to_string()),
        })
        .collect()
}

fn create_simple_content() -> Vec<u8> {
    let content = "%PDF-1.4\nSimple document with basic text content for neural processing.";
    content.as_bytes().to_vec()
}

fn create_complex_content() -> Vec<u8> {
    let content = r#"%PDF-1.4
Complex document with:
- Multiple paragraphs
- Tables with data
- Headers and footers
- Special characters: àáâãäåæçèéêë
- Numbers: 123,456.78
- Dates: 2024-01-15
- Mathematical formulas: E=mc²
- Foreign text: 中文 العربية русский
"#;
    content.as_bytes().to_vec()
}

fn create_corrupted_pdf() -> Vec<u8> {
    let mut content = create_mock_pdf_content(10);
    // Corrupt some bytes in the middle
    if content.len() > 100 {
        for i in 50..60 {
            content[i] = 0xFF;
        }
    }
    content
}

fn create_mock_docx_content() -> Vec<u8> {
    // Mock DOCX content (would be actual ZIP structure in real implementation)
    b"PK\x03\x04mock docx content with word processing features".to_vec()
}

fn create_mock_html_content() -> Vec<u8> {
    let html = r#"<!DOCTYPE html>
<html>
<head><title>Test Document</title></head>
<body>
<h1>Main Heading</h1>
<p>This is a paragraph with some text content.</p>
<table>
<tr><th>Name</th><th>Value</th></tr>
<tr><td>Test</td><td>123</td></tr>
</table>
</body>
</html>"#;
    html.as_bytes().to_vec()
}

// Criterion benchmark groups
criterion_group!(
    benches,
    bench_pdf_extraction,
    bench_batch_extraction,
    bench_concurrent_extraction,
    bench_neural_enhancement,
    bench_daa_coordination,
    bench_memory_usage,
    bench_error_recovery,
    bench_multi_source_extraction
);

criterion_main!(benches);