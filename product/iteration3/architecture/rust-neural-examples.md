# Rust Neural Integration Examples

## 1. Quick Start Example

```rust
use neural_doc_processor::prelude::*;
use tokio;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize the neural processor
    let config = NeuralConfig::from_file("config/neural.yaml")?;
    let processor = NeuralDocumentProcessor::new(config).await?;
    
    // Load models
    processor.load_model("models/layoutlmv3.onnx", ModelType::LayoutLMv3).await?;
    processor.load_model("models/finbert.onnx", ModelType::FinBERT).await?;
    
    // Process a document
    let document = Document::from_pdf("financial_report.pdf")?;
    let results = processor.process_financial_document(document).await?;
    
    // Output results
    println!("Document Analysis Complete!");
    println!("Risk Score: {:.2}", results.overall_risk_score());
    println!("Sentiment: {:?}", results.overall_sentiment());
    
    for (section, analysis) in results.sections() {
        println!("\nSection: {}", section);
        println!("  Entities: {:?}", analysis.entities);
        println!("  Sentiment: {:?}", analysis.sentiment);
    }
    
    Ok(())
}
```

## 2. Custom Neural Network Integration

```rust
use ruv_fann::{Fann, ActivationFunc, TrainAlgorithm};
use neural_doc_processor::{CustomNetwork, NetworkConfig};

// Define a custom risk assessment network
pub struct RiskAssessmentNetwork {
    fann: Fann,
    input_normalizer: InputNormalizer,
    output_interpreter: OutputInterpreter,
}

impl RiskAssessmentNetwork {
    pub fn new() -> Result<Self> {
        // Create a 4-layer network for risk assessment
        let mut fann = Fann::new(&[
            128,  // Input features
            64,   // Hidden layer 1
            32,   // Hidden layer 2
            3,    // Output (low, medium, high risk)
        ])?;
        
        // Configure the network
        fann.set_activation_function(ActivationFunc::ReLU);
        fann.set_activation_function_output(ActivationFunc::Softmax);
        fann.set_training_algorithm(TrainAlgorithm::Adam);
        fann.set_learning_rate(0.001);
        
        Ok(Self {
            fann,
            input_normalizer: InputNormalizer::new(),
            output_interpreter: OutputInterpreter::new(),
        })
    }
    
    pub fn assess_risk(&self, features: &DocumentFeatures) -> Result<RiskAssessment> {
        // Normalize input features
        let input = self.input_normalizer.normalize(features)?;
        
        // Run inference
        let output = self.fann.run(&input)?;
        
        // Interpret results
        let risk_level = self.output_interpreter.get_risk_level(&output);
        let confidence = output.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        
        Ok(RiskAssessment {
            level: risk_level,
            confidence: *confidence,
            contributing_factors: self.analyze_factors(features, &output)?,
        })
    }
    
    pub async fn train_on_dataset(&mut self, dataset: &RiskDataset) -> Result<TrainingStats> {
        let mut stats = TrainingStats::new();
        
        // Prepare training data
        let (inputs, outputs) = dataset.prepare_training_data()?;
        
        // Train with early stopping
        for epoch in 0..1000 {
            let error = self.fann.train_epoch(&inputs, &outputs)?;
            stats.record_epoch(epoch, error);
            
            if error < 0.001 {
                break;
            }
            
            // Validate every 10 epochs
            if epoch % 10 == 0 {
                let val_error = self.validate(dataset.validation_set())?;
                stats.record_validation(epoch, val_error);
                
                if stats.should_stop_early() {
                    break;
                }
            }
        }
        
        Ok(stats)
    }
}
```

## 3. Hybrid Processing Pipeline

```rust
use neural_doc_processor::{HybridPipeline, PipelineBuilder};

pub async fn create_sec_filing_pipeline() -> Result<HybridPipeline> {
    let mut builder = PipelineBuilder::new("SEC Filing Analyzer");
    
    // Stage 1: Document Understanding with LayoutLMv3
    builder.add_stage("layout_analysis", |stage| {
        stage
            .with_model(ModelType::LayoutLMv3)
            .with_preprocessor(|doc| {
                // Extract visual features
                doc.extract_visual_features()
                   .normalize_layout()
                   .detect_tables()
            })
            .with_postprocessor(|features| {
                // Group by semantic sections
                features.group_by_section()
                        .identify_key_sections()
            })
    });
    
    // Stage 2: Financial Analysis with FinBERT
    builder.add_stage("financial_analysis", |stage| {
        stage
            .with_model(ModelType::FinBERT)
            .with_batch_size(16)
            .with_section_filter(|section| {
                // Only analyze relevant sections
                section.is_financial() || section.is_risk_disclosure()
            })
            .with_aggregator(|results| {
                // Aggregate sentiment across sections
                results.weighted_average_by_importance()
            })
    });
    
    // Stage 3: Custom Risk Assessment
    builder.add_stage("risk_assessment", |stage| {
        stage
            .with_custom_network(RiskAssessmentNetwork::new()?)
            .with_feature_extractor(|doc, layout, sentiment| {
                // Combine features from previous stages
                FeatureVector::new()
                    .add_layout_features(layout)
                    .add_sentiment_features(sentiment)
                    .add_statistical_features(doc.compute_statistics())
                    .add_temporal_features(doc.extract_dates())
            })
    });
    
    // Stage 4: Anomaly Detection with Ensemble
    builder.add_stage("anomaly_detection", |stage| {
        stage
            .with_ensemble(vec![
                Box::new(IsolationForestDetector::new()),
                Box::new(AutoencoderDetector::new()),
                Box::new(StatisticalDetector::new()),
            ])
            .with_voting_strategy(VotingStrategy::Weighted)
            .with_threshold(0.7)
    });
    
    // Build and optimize pipeline
    let pipeline = builder
        .with_memory_pool_size(4 * 1024 * 1024 * 1024) // 4GB
        .with_cache_strategy(CacheStrategy::Adaptive)
        .with_parallel_stages(true)
        .build()
        .await?;
    
    Ok(pipeline)
}

// Usage example
pub async fn analyze_10k_filing(file_path: &str) -> Result<FilingAnalysis> {
    let pipeline = create_sec_filing_pipeline().await?;
    let document = Document::from_file(file_path)?;
    
    // Process through pipeline
    let results = pipeline.process(document).await?;
    
    // Generate comprehensive analysis
    Ok(FilingAnalysis {
        layout_structure: results.get("layout_analysis")?,
        financial_sentiment: results.get("financial_analysis")?,
        risk_assessment: results.get("risk_assessment")?,
        anomalies: results.get("anomaly_detection")?,
        executive_summary: generate_summary(&results)?,
    })
}
```

## 4. Real-time Streaming Processing

```rust
use tokio::stream::{Stream, StreamExt};
use neural_doc_processor::streaming::*;

pub struct StreamingDocumentProcessor {
    pipeline: Arc<NeuralPipeline>,
    buffer_size: usize,
    batch_timeout: Duration,
}

impl StreamingDocumentProcessor {
    pub fn process_document_stream<S>(&self, stream: S) -> impl Stream<Item = Result<ProcessedDocument>>
    where
        S: Stream<Item = Document> + Send + 'static,
    {
        let pipeline = self.pipeline.clone();
        let buffer_size = self.buffer_size;
        let timeout = self.batch_timeout;
        
        stream
            // Buffer documents for batching
            .chunks_timeout(buffer_size, timeout)
            // Process batches in parallel
            .map(move |batch| {
                let pipeline = pipeline.clone();
                tokio::spawn(async move {
                    pipeline.process_batch(batch).await
                })
            })
            .buffer_unordered(4) // Process up to 4 batches concurrently
            .flat_map(|result| {
                match result {
                    Ok(Ok(docs)) => stream::iter(docs.into_iter().map(Ok)),
                    Ok(Err(e)) => stream::once(async { Err(e) }),
                    Err(e) => stream::once(async { Err(e.into()) }),
                }
            })
    }
    
    pub async fn process_with_feedback(&self, 
        document: Document,
        feedback_channel: mpsc::Sender<ProcessingFeedback>
    ) -> Result<ProcessedDocument> {
        let stages = vec![
            "Layout Analysis",
            "Text Extraction", 
            "Entity Recognition",
            "Sentiment Analysis",
            "Risk Assessment",
            "Final Aggregation"
        ];
        
        let mut current_stage = 0;
        let total_stages = stages.len();
        
        // Process with progress updates
        let result = self.pipeline
            .process_with_callbacks(document, |stage_name, progress| {
                let _ = feedback_channel.try_send(ProcessingFeedback {
                    stage: stage_name.to_string(),
                    progress: (current_stage as f32 + progress) / total_stages as f32,
                    message: format!("Processing: {}", stages[current_stage]),
                });
                current_stage += 1;
            })
            .await?;
        
        let _ = feedback_channel.send(ProcessingFeedback {
            stage: "Complete".to_string(),
            progress: 1.0,
            message: "Document processing complete".to_string(),
        }).await;
        
        Ok(result)
    }
}
```

## 5. Advanced Caching and Optimization

```rust
use dashmap::DashMap;
use moka::future::Cache;

pub struct OptimizedNeuralProcessor {
    model_cache: Arc<ModelCache>,
    embedding_cache: Arc<Cache<String, Vec<f32>>>,
    result_cache: Arc<DashMap<u64, ProcessingResult>>,
    memory_pool: Arc<MemoryPool>,
}

impl OptimizedNeuralProcessor {
    pub async fn new(config: OptimizationConfig) -> Result<Self> {
        let model_cache = Arc::new(ModelCache::new(config.max_models));
        
        let embedding_cache = Arc::new(
            Cache::builder()
                .max_capacity(config.embedding_cache_size)
                .time_to_live(Duration::from_secs(3600))
                .build()
        );
        
        let memory_pool = Arc::new(
            MemoryPool::new()
                .with_segment_size(config.memory_segment_size)
                .with_max_segments(config.max_memory_segments)
                .build()?
        );
        
        Ok(Self {
            model_cache,
            embedding_cache,
            result_cache: Arc::new(DashMap::new()),
            memory_pool,
        })
    }
    
    pub async fn process_with_cache(&self, document: Document) -> Result<ProcessingResult> {
        // Check if we've processed this document before
        let doc_hash = document.compute_hash();
        
        if let Some(cached) = self.result_cache.get(&doc_hash) {
            return Ok(cached.clone());
        }
        
        // Get embeddings from cache or compute
        let embeddings = self.get_or_compute_embeddings(&document).await?;
        
        // Process using zero-copy operations
        let result = self.process_with_zero_copy(document, embeddings).await?;
        
        // Cache the result
        self.result_cache.insert(doc_hash, result.clone());
        
        Ok(result)
    }
    
    async fn get_or_compute_embeddings(&self, document: &Document) -> Result<Vec<f32>> {
        let cache_key = format!("emb:{}", document.id);
        
        // Try to get from cache
        if let Some(cached) = self.embedding_cache.get(&cache_key).await {
            return Ok(cached);
        }
        
        // Compute embeddings using pooled memory
        let buffer = self.memory_pool.acquire(document.estimated_embedding_size())?;
        let embeddings = self.compute_embeddings_in_place(document, buffer).await?;
        
        // Store in cache
        self.embedding_cache.insert(cache_key, embeddings.clone()).await;
        
        Ok(embeddings)
    }
    
    async fn process_with_zero_copy(&self, 
        document: Document, 
        embeddings: Vec<f32>
    ) -> Result<ProcessingResult> {
        // Create shared memory views
        let doc_view = self.memory_pool.create_view(&document.raw_data)?;
        let emb_view = self.memory_pool.create_view(&embeddings)?;
        
        // Process without copying data
        let layout_features = self.model_cache
            .get_model("layoutlm")
            .await?
            .process_zero_copy(&doc_view, &emb_view)
            .await?;
        
        // Aggregate results in place
        let result = ProcessingResult::build_in_place(
            self.memory_pool.acquire(1024 * 1024)?, // 1MB for results
            layout_features,
            document.metadata,
        )?;
        
        Ok(result)
    }
}
```

## 6. Production Deployment Example

```rust
use actix_web::{web, App, HttpServer, HttpResponse};
use prometheus::{Encoder, TextEncoder, Counter, Histogram};

pub struct NeuralProcessingService {
    processor: Arc<NeuralDocumentProcessor>,
    metrics: Arc<ServiceMetrics>,
}

#[derive(Clone)]
pub struct ServiceMetrics {
    requests_total: Counter,
    request_duration: Histogram,
    processing_errors: Counter,
}

impl NeuralProcessingService {
    pub async fn process_document_endpoint(
        data: web::Data<Self>,
        document: web::Bytes,
    ) -> Result<HttpResponse, actix_web::Error> {
        let start = Instant::now();
        data.metrics.requests_total.inc();
        
        // Parse document
        let doc = Document::from_bytes(&document)
            .map_err(|e| actix_web::error::ErrorBadRequest(e))?;
        
        // Process with timeout
        let result = timeout(Duration::from_secs(30), 
            data.processor.process_financial_document(doc)
        ).await
        .map_err(|_| actix_web::error::ErrorRequestTimeout("Processing timeout"))?
        .map_err(|e| {
            data.metrics.processing_errors.inc();
            actix_web::error::ErrorInternalServerError(e)
        })?;
        
        // Record metrics
        data.metrics.request_duration.observe(start.elapsed().as_secs_f64());
        
        // Return JSON response
        Ok(HttpResponse::Ok().json(&result))
    }
    
    pub async fn health_check(
        data: web::Data<Self>,
    ) -> Result<HttpResponse, actix_web::Error> {
        let health = HealthStatus {
            status: "healthy",
            models_loaded: data.processor.loaded_models().await,
            memory_usage: data.processor.memory_stats().await,
            cache_stats: data.processor.cache_stats().await,
        };
        
        Ok(HttpResponse::Ok().json(&health))
    }
    
    pub async fn metrics_endpoint(
        data: web::Data<Self>,
    ) -> Result<HttpResponse, actix_web::Error> {
        let encoder = TextEncoder::new();
        let metric_families = prometheus::gather();
        let mut buffer = vec![];
        encoder.encode(&metric_families, &mut buffer).unwrap();
        
        Ok(HttpResponse::Ok()
            .content_type("text/plain; version=0.0.4")
            .body(buffer))
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Initialize processor
    let config = NeuralConfig::from_env()?;
    let processor = Arc::new(NeuralDocumentProcessor::new(config).await?);
    
    // Load models
    processor.load_all_models().await?;
    
    // Create service
    let service = web::Data::new(NeuralProcessingService {
        processor,
        metrics: Arc::new(ServiceMetrics {
            requests_total: Counter::new("requests_total", "Total requests")?,
            request_duration: Histogram::with_opts(
                HistogramOpts::new("request_duration_seconds", "Request duration")
            )?,
            processing_errors: Counter::new("processing_errors", "Processing errors")?,
        }),
    });
    
    // Start server
    HttpServer::new(move || {
        App::new()
            .app_data(service.clone())
            .route("/process", web::post().to(NeuralProcessingService::process_document_endpoint))
            .route("/health", web::get().to(NeuralProcessingService::health_check))
            .route("/metrics", web::get().to(NeuralProcessingService::metrics_endpoint))
    })
    .bind("0.0.0.0:8080")?
    .workers(num_cpus::get())
    .run()
    .await
}
```

## 7. Testing and Benchmarking

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use criterion::{black_box, criterion_group, criterion_main, Criterion};
    
    #[tokio::test]
    async fn test_end_to_end_processing() {
        let processor = create_test_processor().await;
        let document = load_test_document("test_data/sample_10k.pdf");
        
        let result = processor.process_financial_document(document).await.unwrap();
        
        assert!(!result.sections.is_empty());
        assert!(result.risk_scores.contains_key("overall"));
        assert!(result.sentiments.len() > 0);
    }
    
    #[tokio::test]
    async fn test_parallel_processing() {
        let processor = Arc::new(create_test_processor().await);
        let documents: Vec<_> = (0..10)
            .map(|i| load_test_document(&format!("test_data/doc_{}.pdf", i)))
            .collect();
        
        let start = Instant::now();
        
        let futures: Vec<_> = documents
            .into_iter()
            .map(|doc| {
                let proc = processor.clone();
                tokio::spawn(async move {
                    proc.process_financial_document(doc).await
                })
            })
            .collect();
        
        let results = futures::future::try_join_all(futures).await.unwrap();
        let elapsed = start.elapsed();
        
        println!("Processed {} documents in {:?}", results.len(), elapsed);
        assert!(elapsed.as_secs() < 10); // Should process 10 docs in under 10 seconds
    }
    
    fn benchmark_inference(c: &mut Criterion) {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let processor = runtime.block_on(create_test_processor());
        
        c.bench_function("layoutlm_inference", |b| {
            b.to_async(&runtime).iter(|| async {
                let input = create_test_input();
                processor.layout_model.infer(black_box(&input)).await
            });
        });
        
        c.bench_function("finbert_inference", |b| {
            b.to_async(&runtime).iter(|| async {
                let text = "The company reported strong financial results.";
                processor.finbert_model.analyze_sentiment(black_box(text)).await
            });
        });
    }
    
    criterion_group!(benches, benchmark_inference);
    criterion_main!(benches);
}
```

## 8. Integration with Claude Flow

```rust
use claude_flow::{SwarmCoordinator, AgentMessage};

pub struct ClaudeFlowNeuralBridge {
    processor: Arc<NeuralDocumentProcessor>,
    coordinator: Arc<SwarmCoordinator>,
}

impl ClaudeFlowNeuralBridge {
    pub async fn process_with_swarm(&self, 
        document: Document,
        swarm_id: &str
    ) -> Result<ProcessingResult> {
        // Notify swarm of processing start
        self.coordinator.send_message(swarm_id, AgentMessage {
            agent_type: "neural_processor",
            action: "processing_start",
            data: serde_json::json!({
                "document_id": document.id,
                "document_type": document.doc_type,
                "size": document.size,
            }),
        }).await?;
        
        // Process document
        let result = self.processor.process_financial_document(document).await?;
        
        // Share results with swarm
        self.coordinator.send_message(swarm_id, AgentMessage {
            agent_type: "neural_processor",
            action: "processing_complete",
            data: serde_json::to_value(&result)?,
        }).await?;
        
        // Store in swarm memory
        self.coordinator.store_memory(
            &format!("neural_results/{}", document.id),
            &result,
            Some(3600) // 1 hour TTL
        ).await?;
        
        Ok(result)
    }
}
```

These examples demonstrate the complete integration of neural models in pure Rust, from basic usage to production deployment, with emphasis on performance, safety, and seamless integration with the existing swarm architecture.