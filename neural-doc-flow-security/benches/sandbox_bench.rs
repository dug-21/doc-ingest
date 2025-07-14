//! Benchmarks for sandbox performance

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use neural_doc_flow_security::sandbox::SandboxManager;
use tokio::runtime::Runtime;

fn bench_sandbox_creation(c: &mut Criterion) {
    c.bench_function("sandbox_creation", |b| {
        b.iter(|| {
            let manager = SandboxManager::new();
            black_box(manager);
        });
    });
}

fn bench_sandbox_execution(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("sandbox_execution", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut manager = SandboxManager::new().unwrap();
                let result = manager.execute_sandboxed("bench_plugin", || {
                    Ok(black_box(42))
                }).await;
                black_box(result);
            });
        });
    });
}

fn bench_sandbox_with_work(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("sandbox_with_work", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut manager = SandboxManager::new().unwrap();
                let result = manager.execute_sandboxed("work_plugin", || {
                    // Simulate some computational work
                    let mut sum = 0u64;
                    for i in 0..1000 {
                        sum += i;
                    }
                    Ok(black_box(sum))
                }).await;
                black_box(result);
            });
        });
    });
}

fn bench_sandbox_isolation_overhead(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("sandbox_isolation_overhead", |b| {
        let mut manager = SandboxManager::new().unwrap();
        
        b.iter(|| {
            rt.block_on(async {
                // Measure overhead of namespace setup
                let result = manager.execute_sandboxed("overhead_plugin", || {
                    // Minimal work to measure pure overhead
                    Ok(())
                }).await;
                black_box(result);
            });
        });
    });
}

criterion_group!(
    benches,
    bench_sandbox_creation,
    bench_sandbox_execution,
    bench_sandbox_with_work,
    bench_sandbox_isolation_overhead
);
criterion_main!(benches);