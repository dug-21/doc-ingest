# NeuralDocFlow Rust Implementation Examples

This document provides comprehensive examples of using NeuralDocFlow's modular architecture in Rust, demonstrating various use cases from basic PDF extraction to advanced neural enhancement and distributed processing.

## Table of Contents

1. [Basic PDF Extraction](#1-basic-pdf-extraction)
2. [Neural Enhancement with ruv-FANN](#2-neural-enhancement-with-ruv-fann)
3. [DAA Coordination for Parallel Processing](#3-daa-coordination-for-parallel-processing)
4. [User-Defined Extraction Schemas](#4-user-defined-extraction-schemas)
5. [Custom Output Format Templates](#5-custom-output-format-templates)
6. [Domain Configuration Examples](#6-domain-configuration-examples)
7. [Python Usage via PyO3 Bindings](#7-python-usage-via-pyo3-bindings)
8. [WASM Usage in Browser](#8-wasm-usage-in-browser)

## 1. Basic PDF Extraction

### Simple PDF Text Extraction

```rust
use neuraldocflow::{SourcePluginManager, SourceInput, ManagerConfig};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the plugin manager with default configuration
    let config = ManagerConfig {
        plugin_directories: vec![
            PathBuf::from("./plugins"),
            PathBuf::from("/usr/local/lib/neuraldocflow/sources"),
        ],
        config_path: PathBuf::from("./config/sources.yaml"),
        enable_hot_reload: false,
    };
    
    let manager = SourcePluginManager::new(config)?;
    
    // Create input for a PDF file
    let input = SourceInput::File {
        path: PathBuf::from("documents/report.pdf"),
        metadata: None,
    };
    
    // Find compatible sources (PDF source should be available)
    let sources = manager.find_compatible_sources(&input).await?;
    
    if let Some(pdf_source) = sources.first() {
        // Extract the document
        let extracted = pdf_source.extract(input).await?;
        
        println!("Document ID: {}", extracted.id);
        println!("Extracted {} content blocks", extracted.content.len());
        println!("Confidence: {:.2}%", extracted.confidence * 100.0);
        
        // Process extracted content
        for block in &extracted.content {
            match &block.block_type {
                BlockType::Paragraph => {
                    if let Some(text) = &block.text {
                        println!("Paragraph: {}", text);
                    }
                }
                BlockType::Heading(level) => {
                    if let Some(text) = &block.text {
                        println!("Heading (Level {}): {}", level, text);
                    }
                }
                BlockType::Table => {
                    if let Some(text) = &block.text {
                        println!("Table:\n{}", text);
                    }
                }
                _ => {}
            }
        }
        
        // Access document structure
        println!("\nDocument Structure:");
        for section in &extracted.structure.sections {
            println!("- Section: {:?} (Level {})", section.title, section.level);
        }
    }
    
    Ok(())
}
```

### Streaming Large PDF Files

```rust
use neuraldocflow::{SourceInput, PdfSource, DocumentSource};
use tokio::io::AsyncRead;
use futures::stream::StreamExt;

async fn process_large_pdf<R: AsyncRead + Send + Unpin + 'static>(
    reader: R,
    size_hint: Option<usize>
) -> Result<(), Box<dyn std::error::Error>> {
    // Create PDF source with optimized configuration
    let mut pdf_source = PdfSource::new();
    
    // Configure for streaming
    let config = SourceConfig {
        enabled: true,
        priority: 100,
        timeout: Duration::from_secs(300), // 5 minutes for large files
        memory_limit: Some(100 * 1024 * 1024), // 100MB limit
        thread_pool_size: Some(4),
        retry: RetryConfig::default(),
        settings: serde_json::json!({
            "performance": {
                "use_mmap": false, // Don't use mmap for streaming
                "parallel_pages": true,
                "chunk_size": 8192,
                "buffer_pool_size": 32
            }
        }),
    };
    
    pdf_source.initialize(config).await?;
    
    // Create streaming input
    let input = SourceInput::Stream {
        reader: Box::new(reader),
        size_hint,
        mime_type: Some("application/pdf".to_string()),
    };
    
    // Extract with progress tracking
    let extracted = pdf_source.extract(input).await?;
    
    // Process in chunks to manage memory
    let chunk_size = 100;
    for chunk in extracted.content.chunks(chunk_size) {
        process_content_chunk(chunk).await?;
        
        // Yield to prevent blocking
        tokio::task::yield_now().await;
    }
    
    Ok(())
}

async fn process_content_chunk(blocks: &[ContentBlock]) -> Result<(), Box<dyn std::error::Error>> {
    // Process each block
    for block in blocks {
        // Your processing logic here
        println!("Processing block: {}", block.id);
    }
    Ok(())
}
```

### Error Handling and Validation

```rust
use neuraldocflow::{SourceInput, SourceError, ValidationResult};

async fn safe_pdf_extraction(
    pdf_path: &str
) -> Result<ExtractedDocument, Box<dyn std::error::Error>> {
    let manager = SourcePluginManager::new(Default::default())?;
    
    let input = SourceInput::File {
        path: PathBuf::from(pdf_path),
        metadata: None,
    };
    
    // Find PDF source
    let sources = manager.find_compatible_sources(&input).await?;
    let pdf_source = sources.first()
        .ok_or_else(|| SourceError::NoCompatibleSource)?;
    
    // Validate before extraction
    let validation = pdf_source.validate(&input).await?;
    
    if !validation.is_valid() {
        eprintln!("Validation errors:");
        for error in &validation.errors {
            eprintln!("  - {}", error);
        }
        
        if !validation.security_issues.is_empty() {
            eprintln!("Security issues detected:");
            for issue in &validation.security_issues {
                eprintln!("  - {}", issue);
            }
            return Err("Security validation failed".into());
        }
        
        // Decide whether to continue despite errors
        if validation.errors.iter().any(|e| e.contains("size exceeds")) {
            return Err("File too large".into());
        }
    }
    
    // Perform extraction with timeout
    let extraction_future = pdf_source.extract(input);
    let timeout_duration = Duration::from_secs(60);
    
    match tokio::time::timeout(timeout_duration, extraction_future).await {
        Ok(Ok(document)) => Ok(document),
        Ok(Err(e)) => Err(format!("Extraction failed: {}", e).into()),
        Err(_) => Err("Extraction timed out".into()),
    }
}
```

## 2. Neural Enhancement with ruv-FANN

### Basic Neural Text Enhancement

```rust
use neuraldocflow::{ExtractedDocument, ContentBlock};
use ruv_fann::{Fann, ActivationFunc, TrainAlgorithm};

/// Neural network for enhancing extracted text quality
pub struct TextEnhancer {
    network: Fann,
    vocabulary: HashMap<String, usize>,
    confidence_threshold: f32,
}

impl TextEnhancer {
    /// Create a new text enhancer with pre-trained model
    pub fn new(model_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let network = Fann::from_file(model_path)?;
        let vocabulary = Self::load_vocabulary("models/vocabulary.json")?;
        
        Ok(Self {
            network,
            vocabulary,
            confidence_threshold: 0.8,
        })
    }
    
    /// Train a new enhancement model
    pub fn train_new(
        training_data: &[(Vec<f32>, Vec<f32>)]
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Create network: input -> hidden -> output
        let layers = &[100, 50, 25, 100]; // Word embedding size = 100
        let mut network = Fann::new(layers)?;
        
        // Configure network
        network.set_activation_function_hidden(ActivationFunc::SigmoidSymmetric);
        network.set_activation_function_output(ActivationFunc::Linear);
        network.set_train_algorithm(TrainAlgorithm::RpropPlus);
        
        // Train the network
        for epoch in 0..1000 {
            let mut total_error = 0.0;
            
            for (input, target) in training_data {
                network.train(input, target);
                let output = network.run(input)?;
                let error = Self::calculate_error(target, &output);
                total_error += error;
            }
            
            if epoch % 100 == 0 {
                println!("Epoch {}: Error = {:.6}", epoch, total_error / training_data.len() as f32);
            }
            
            // Early stopping
            if total_error / training_data.len() as f32 < 0.001 {
                break;
            }
        }
        
        let vocabulary = Self::build_vocabulary(training_data);
        
        Ok(Self {
            network,
            vocabulary,
            confidence_threshold: 0.8,
        })
    }
    
    /// Enhance extracted document text
    pub async fn enhance_document(
        &self,
        document: &mut ExtractedDocument
    ) -> Result<(), Box<dyn std::error::Error>> {
        for block in &mut document.content {
            if let Some(text) = &block.text {
                let enhanced = self.enhance_text(text).await?;
                
                if enhanced.confidence > self.confidence_threshold {
                    block.text = Some(enhanced.text);
                    block.metadata.confidence = enhanced.confidence;
                    block.metadata.attributes.insert(
                        "neural_enhanced".to_string(),
                        "true".to_string()
                    );
                }
            }
        }
        
        Ok(())
    }
    
    /// Enhance individual text block
    async fn enhance_text(&self, text: &str) -> Result<EnhancedText, Box<dyn std::error::Error>> {
        // Tokenize and encode text
        let tokens = self.tokenize(text);
        let encoded = self.encode_tokens(&tokens)?;
        
        let mut enhanced_tokens = Vec::new();
        let mut total_confidence = 0.0;
        
        // Process each token through the network
        for (i, token_vec) in encoded.iter().enumerate() {
            let output = self.network.run(token_vec)?;
            let (best_token, confidence) = self.decode_output(&output)?;
            
            enhanced_tokens.push(best_token);
            total_confidence += confidence;
        }
        
        let enhanced_text = enhanced_tokens.join(" ");
        let avg_confidence = total_confidence / enhanced_tokens.len() as f32;
        
        Ok(EnhancedText {
            text: enhanced_text,
            confidence: avg_confidence,
            changes: self.identify_changes(text, &enhanced_text),
        })
    }
    
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|s| s.to_lowercase())
            .collect()
    }
    
    fn encode_tokens(&self, tokens: &[String]) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        tokens.iter()
            .map(|token| {
                let idx = self.vocabulary.get(token).unwrap_or(&0);
                let mut vec = vec![0.0; self.vocabulary.len()];
                vec[*idx] = 1.0;
                Ok(vec)
            })
            .collect()
    }
    
    fn decode_output(&self, output: &[f32]) -> Result<(String, f32), Box<dyn std::error::Error>> {
        let (max_idx, max_val) = output.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        
        let token = self.vocabulary.iter()
            .find(|(_, &idx)| idx == max_idx)
            .map(|(token, _)| token.clone())
            .unwrap_or_else(|| "<unknown>".to_string());
        
        Ok((token, *max_val))
    }
}

#[derive(Debug)]
struct EnhancedText {
    text: String,
    confidence: f32,
    changes: Vec<TextChange>,
}

#[derive(Debug)]
struct TextChange {
    original: String,
    enhanced: String,
    position: usize,
}
```

### Advanced Neural Pattern Recognition

```rust
use ruv_fann::{Fann, ActivationFunc};
use neuraldocflow::{ContentBlock, BlockType};

/// Neural pattern recognizer for document structure
pub struct DocumentPatternRecognizer {
    heading_detector: Fann,
    table_detector: Fann,
    list_detector: Fann,
    confidence_threshold: f32,
}

impl DocumentPatternRecognizer {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            heading_detector: Self::create_heading_network()?,
            table_detector: Self::create_table_network()?,
            list_detector: Self::create_list_network()?,
            confidence_threshold: 0.85,
        })
    }
    
    fn create_heading_network() -> Result<Fann, Box<dyn std::error::Error>> {
        // Features: font_size, bold, position, caps_ratio, length
        let layers = &[5, 10, 5, 1];
        let mut network = Fann::new(layers)?;
        
        network.set_activation_function_hidden(ActivationFunc::Sigmoid);
        network.set_activation_function_output(ActivationFunc::Sigmoid);
        
        Ok(network)
    }
    
    fn create_table_network() -> Result<Fann, Box<dyn std::error::Error>> {
        // Features: delimiter_count, alignment_score, row_similarity, col_count
        let layers = &[4, 8, 4, 1];
        let mut network = Fann::new(layers)?;
        
        network.set_activation_function_hidden(ActivationFunc::SigmoidSymmetric);
        network.set_activation_function_output(ActivationFunc::Sigmoid);
        
        Ok(network)
    }
    
    fn create_list_network() -> Result<Fann, Box<dyn std::error::Error>> {
        // Features: bullet_present, indentation, item_similarity, sequence_score
        let layers = &[4, 6, 3, 1];
        let mut network = Fann::new(layers)?;
        
        network.set_activation_function_hidden(ActivationFunc::Sigmoid);
        network.set_activation_function_output(ActivationFunc::Sigmoid);
        
        Ok(network)
    }
    
    pub async fn recognize_patterns(
        &self,
        blocks: &mut Vec<ContentBlock>
    ) -> Result<(), Box<dyn std::error::Error>> {
        for block in blocks.iter_mut() {
            if let Some(text) = &block.text {
                // Extract features from block
                let features = self.extract_features(block, text);
                
                // Check if it's a heading
                let heading_score = self.heading_detector.run(&features.heading_features)?[0];
                if heading_score > self.confidence_threshold {
                    let level = self.determine_heading_level(&features);
                    block.block_type = BlockType::Heading(level);
                    block.metadata.attributes.insert(
                        "neural_confidence".to_string(),
                        heading_score.to_string()
                    );
                    continue;
                }
                
                // Check if it's a table
                let table_score = self.table_detector.run(&features.table_features)?[0];
                if table_score > self.confidence_threshold {
                    block.block_type = BlockType::Table;
                    block.metadata.attributes.insert(
                        "neural_confidence".to_string(),
                        table_score.to_string()
                    );
                    continue;
                }
                
                // Check if it's a list
                let list_score = self.list_detector.run(&features.list_features)?[0];
                if list_score > self.confidence_threshold {
                    let list_type = self.determine_list_type(&features);
                    block.block_type = BlockType::List(list_type);
                    block.metadata.attributes.insert(
                        "neural_confidence".to_string(),
                        list_score.to_string()
                    );
                }
            }
        }
        
        Ok(())
    }
    
    fn extract_features(&self, block: &ContentBlock, text: &str) -> BlockFeatures {
        BlockFeatures {
            heading_features: vec![
                self.extract_font_size(block),
                self.is_bold(block) as i32 as f32,
                self.extract_position_score(block),
                self.calculate_caps_ratio(text),
                text.len() as f32 / 100.0, // Normalized length
            ],
            table_features: vec![
                self.count_delimiters(text),
                self.calculate_alignment_score(text),
                self.calculate_row_similarity(text),
                self.estimate_column_count(text),
            ],
            list_features: vec![
                self.has_bullet(text) as i32 as f32,
                self.extract_indentation(block),
                self.calculate_item_similarity(text),
                self.calculate_sequence_score(text),
            ],
        }
    }
}

struct BlockFeatures {
    heading_features: Vec<f32>,
    table_features: Vec<f32>,
    list_features: Vec<f32>,
}
```

### Continuous Learning System

```rust
use ruv_fann::Fann;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Adaptive neural enhancement system that learns from user feedback
pub struct AdaptiveNeuralEnhancer {
    enhancer: Arc<Mutex<TextEnhancer>>,
    feedback_buffer: Arc<Mutex<Vec<FeedbackSample>>>,
    update_threshold: usize,
    learning_rate: f32,
}

#[derive(Clone)]
struct FeedbackSample {
    original: String,
    enhanced: String,
    user_correction: String,
    confidence: f32,
}

impl AdaptiveNeuralEnhancer {
    pub fn new(base_model: TextEnhancer) -> Self {
        Self {
            enhancer: Arc::new(Mutex::new(base_model)),
            feedback_buffer: Arc::new(Mutex::new(Vec::new())),
            update_threshold: 100,
            learning_rate: 0.01,
        }
    }
    
    /// Process document with learning capability
    pub async fn enhance_with_learning(
        &self,
        document: &mut ExtractedDocument,
        enable_feedback: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let enhancer = self.enhancer.lock().await;
        
        for block in &mut document.content {
            if let Some(text) = &block.text {
                let enhanced = enhancer.enhance_text(text).await?;
                
                if enable_feedback {
                    // Store for potential feedback
                    block.metadata.attributes.insert(
                        "enhancement_id".to_string(),
                        Uuid::new_v4().to_string()
                    );
                    block.metadata.attributes.insert(
                        "original_text".to_string(),
                        text.clone()
                    );
                }
                
                block.text = Some(enhanced.text);
                block.metadata.confidence = enhanced.confidence;
            }
        }
        
        drop(enhancer);
        
        // Check if we should update the model
        if self.should_update().await {
            self.update_model().await?;
        }
        
        Ok(())
    }
    
    /// Record user feedback for continuous learning
    pub async fn record_feedback(
        &self,
        enhancement_id: &str,
        user_correction: String
    ) -> Result<(), Box<dyn std::error::Error>> {
        // In a real implementation, retrieve original and enhanced from storage
        let sample = FeedbackSample {
            original: "original text".to_string(),
            enhanced: "enhanced text".to_string(),
            user_correction,
            confidence: 0.9,
        };
        
        let mut buffer = self.feedback_buffer.lock().await;
        buffer.push(sample);
        
        Ok(())
    }
    
    async fn should_update(&self) -> bool {
        let buffer = self.feedback_buffer.lock().await;
        buffer.len() >= self.update_threshold
    }
    
    async fn update_model(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut buffer = self.feedback_buffer.lock().await;
        let samples = buffer.drain(..).collect::<Vec<_>>();
        drop(buffer);
        
        // Prepare training data from feedback
        let training_data = self.prepare_training_data(&samples)?;
        
        // Update model in background
        let enhancer = Arc::clone(&self.enhancer);
        let learning_rate = self.learning_rate;
        
        tokio::spawn(async move {
            if let Ok(mut e) = enhancer.lock().await {
                // Perform incremental training
                for (input, target) in training_data {
                    e.network.train_with_rate(&input, &target, learning_rate);
                }
                
                println!("Model updated with {} feedback samples", samples.len());
            }
        });
        
        Ok(())
    }
    
    fn prepare_training_data(
        &self,
        samples: &[FeedbackSample]
    ) -> Result<Vec<(Vec<f32>, Vec<f32>)>, Box<dyn std::error::Error>> {
        // Convert feedback samples to training data
        // Implementation would encode texts to vectors
        Ok(vec![])
    }
}
```

## 3. DAA Coordination for Parallel Processing

### Basic DAA Setup for Multi-Document Processing

```rust
use neuraldocflow_daa::{DAACoordinator, Agent, Task, CoordinationStrategy};
use neuraldocflow::{SourcePluginManager, SourceInput};
use std::sync::Arc;
use tokio::sync::mpsc;

/// Distributed processing coordinator for document extraction
pub struct DistributedDocumentProcessor {
    coordinator: Arc<DAACoordinator>,
    source_manager: Arc<SourcePluginManager>,
    agent_pool_size: usize,
}

impl DistributedDocumentProcessor {
    pub async fn new(
        agent_count: usize
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let coordinator = Arc::new(DAACoordinator::new(
            CoordinationStrategy::WorkStealing
        ));
        
        let source_manager = Arc::new(
            SourcePluginManager::new(Default::default())?
        );
        
        let processor = Self {
            coordinator: Arc::clone(&coordinator),
            source_manager: Arc::clone(&source_manager),
            agent_pool_size: agent_count,
        };
        
        // Initialize agent pool
        processor.initialize_agents().await?;
        
        Ok(processor)
    }
    
    async fn initialize_agents(&self) -> Result<(), Box<dyn std::error::Error>> {
        for i in 0..self.agent_pool_size {
            let agent = DocumentProcessingAgent::new(
                format!("agent-{}", i),
                Arc::clone(&self.source_manager)
            );
            
            self.coordinator.register_agent(Box::new(agent)).await?;
        }
        
        Ok(())
    }
    
    /// Process multiple documents in parallel
    pub async fn process_documents(
        &self,
        document_paths: Vec<PathBuf>
    ) -> Result<Vec<ProcessingResult>, Box<dyn std::error::Error>> {
        let (result_tx, mut result_rx) = mpsc::channel(document_paths.len());
        
        // Create tasks for each document
        let tasks: Vec<Task> = document_paths
            .into_iter()
            .enumerate()
            .map(|(idx, path)| Task {
                id: format!("task-{}", idx),
                task_type: TaskType::ExtractDocument,
                payload: TaskPayload::DocumentPath(path),
                priority: 1,
                dependencies: vec![],
                result_channel: Some(result_tx.clone()),
            })
            .collect();
        
        // Submit tasks to coordinator
        for task in tasks {
            self.coordinator.submit_task(task).await?;
        }
        
        // Start processing
        let coordinator = Arc::clone(&self.coordinator);
        let processing_handle = tokio::spawn(async move {
            coordinator.start_processing().await
        });
        
        // Collect results
        let mut results = Vec::new();
        let total_tasks = document_paths.len();
        
        for _ in 0..total_tasks {
            if let Some(result) = result_rx.recv().await {
                results.push(result);
                
                // Log progress
                println!("Processed {}/{} documents", results.len(), total_tasks);
            }
        }
        
        // Stop processing
        self.coordinator.stop_processing().await?;
        processing_handle.await??;
        
        Ok(results)
    }
}

/// Agent implementation for document processing
struct DocumentProcessingAgent {
    id: String,
    source_manager: Arc<SourcePluginManager>,
    neural_enhancer: Option<TextEnhancer>,
}

impl DocumentProcessingAgent {
    fn new(id: String, source_manager: Arc<SourcePluginManager>) -> Self {
        Self {
            id,
            source_manager,
            neural_enhancer: None,
        }
    }
}

#[async_trait]
impl Agent for DocumentProcessingAgent {
    fn id(&self) -> &str {
        &self.id
    }
    
    async fn process_task(&mut self, task: Task) -> Result<TaskResult, Box<dyn std::error::Error>> {
        match task.task_type {
            TaskType::ExtractDocument => {
                if let TaskPayload::DocumentPath(path) = task.payload {
                    let input = SourceInput::File {
                        path: path.clone(),
                        metadata: None,
                    };
                    
                    // Find compatible source
                    let sources = self.source_manager
                        .find_compatible_sources(&input)
                        .await?;
                    
                    if let Some(source) = sources.first() {
                        let start = Instant::now();
                        
                        // Extract document
                        let mut document = source.extract(input).await?;
                        
                        // Apply neural enhancement if available
                        if let Some(enhancer) = &self.neural_enhancer {
                            enhancer.enhance_document(&mut document).await?;
                        }
                        
                        let duration = start.elapsed();
                        
                        return Ok(TaskResult {
                            task_id: task.id,
                            success: true,
                            data: Some(Box::new(ProcessingResult {
                                document_id: document.id,
                                path: path.to_string_lossy().to_string(),
                                blocks_extracted: document.content.len(),
                                processing_time: duration,
                                confidence: document.confidence,
                            })),
                            error: None,
                        });
                    }
                }
            }
            _ => return Err("Unsupported task type".into()),
        }
        
        Err("Failed to process task".into())
    }
    
    fn capabilities(&self) -> Vec<String> {
        vec![
            "pdf_extraction".to_string(),
            "docx_extraction".to_string(),
            "neural_enhancement".to_string(),
        ]
    }
}

#[derive(Debug, Clone)]
struct ProcessingResult {
    document_id: String,
    path: String,
    blocks_extracted: usize,
    processing_time: Duration,
    confidence: f32,
}
```

### Advanced DAA Pipeline with Neural Coordination

```rust
use neuraldocflow_daa::{Pipeline, Stage, StageResult};

/// Multi-stage processing pipeline with neural coordination
pub struct NeuralProcessingPipeline {
    stages: Vec<Box<dyn Stage>>,
    coordinator: Arc<DAACoordinator>,
    neural_router: NeuralRouter,
}

impl NeuralProcessingPipeline {
    pub async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let coordinator = Arc::new(DAACoordinator::new(
            CoordinationStrategy::NeuralRouting
        ));
        
        let stages = vec![
            Box::new(ExtractionStage::new()) as Box<dyn Stage>,
            Box::new(ValidationStage::new()) as Box<dyn Stage>,
            Box::new(EnhancementStage::new()) as Box<dyn Stage>,
            Box::new(StructuringStage::new()) as Box<dyn Stage>,
            Box::new(OutputStage::new()) as Box<dyn Stage>,
        ];
        
        let neural_router = NeuralRouter::new()?;
        
        Ok(Self {
            stages,
            coordinator,
            neural_router,
        })
    }
    
    pub async fn process(
        &self,
        inputs: Vec<SourceInput>
    ) -> Result<Vec<ProcessedDocument>, Box<dyn std::error::Error>> {
        let mut pipeline_data = Vec::new();
        
        // Initialize pipeline data
        for (idx, input) in inputs.into_iter().enumerate() {
            pipeline_data.push(PipelineData {
                id: format!("doc-{}", idx),
                input,
                intermediate_results: HashMap::new(),
                current_stage: 0,
            });
        }
        
        // Process through stages
        for (stage_idx, stage) in self.stages.iter().enumerate() {
            println!("Processing stage {}: {}", stage_idx, stage.name());
            
            // Route documents to appropriate agents based on neural analysis
            let routing_decisions = self.neural_router
                .route_documents(&pipeline_data, stage.name())
                .await?;
            
            // Create tasks for this stage
            let tasks = self.create_stage_tasks(
                &pipeline_data,
                stage_idx,
                routing_decisions
            );
            
            // Submit and process tasks
            let results = self.coordinator
                .process_tasks_parallel(tasks)
                .await?;
            
            // Update pipeline data with results
            self.update_pipeline_data(&mut pipeline_data, results)?;
        }
        
        // Convert to final output
        Ok(self.finalize_documents(pipeline_data))
    }
}

/// Neural router for intelligent task distribution
struct NeuralRouter {
    routing_network: Fann,
    agent_profiles: HashMap<String, AgentProfile>,
}

impl NeuralRouter {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        // Create routing network
        // Input: document features + agent capabilities
        // Output: agent suitability scores
        let layers = &[20, 15, 10, 5];
        let mut network = Fann::new(layers)?;
        
        network.set_activation_function_hidden(ActivationFunc::SigmoidSymmetric);
        network.set_activation_function_output(ActivationFunc::Softmax);
        
        Ok(Self {
            routing_network: network,
            agent_profiles: HashMap::new(),
        })
    }
    
    pub async fn route_documents(
        &self,
        documents: &[PipelineData],
        stage: &str
    ) -> Result<HashMap<String, String>, Box<dyn std::error::Error>> {
        let mut routing = HashMap::new();
        
        for doc in documents {
            // Extract document features
            let doc_features = self.extract_document_features(doc);
            
            // Get best agent for this document
            let best_agent = self.find_best_agent(&doc_features, stage)?;
            
            routing.insert(doc.id.clone(), best_agent);
        }
        
        Ok(routing)
    }
    
    fn find_best_agent(
        &self,
        features: &[f32],
        stage: &str
    ) -> Result<String, Box<dyn std::error::Error>> {
        // Get available agents for this stage
        let available_agents = self.get_stage_agents(stage);
        
        let mut best_score = 0.0;
        let mut best_agent = String::new();
        
        for agent in available_agents {
            let agent_features = self.get_agent_features(&agent);
            let combined_features = [features, &agent_features].concat();
            
            let scores = self.routing_network.run(&combined_features)?;
            let score = scores[0]; // Suitability score
            
            if score > best_score {
                best_score = score;
                best_agent = agent;
            }
        }
        
        Ok(best_agent)
    }
}
```

### Load Balancing and Fault Tolerance

```rust
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::RwLock;

/// Fault-tolerant DAA coordinator with load balancing
pub struct FaultTolerantCoordinator {
    agents: Arc<RwLock<Vec<AgentHandle>>>,
    task_queue: Arc<SegmentedQueue<Task>>,
    load_balancer: Arc<LoadBalancer>,
    health_monitor: Arc<HealthMonitor>,
    retry_policy: RetryPolicy,
}

struct AgentHandle {
    agent: Box<dyn Agent>,
    load: AtomicUsize,
    failures: AtomicUsize,
    last_heartbeat: Arc<RwLock<Instant>>,
}

impl FaultTolerantCoordinator {
    pub async fn new(retry_policy: RetryPolicy) -> Self {
        Self {
            agents: Arc::new(RwLock::new(Vec::new())),
            task_queue: Arc::new(SegmentedQueue::new(4)), // 4 priority levels
            load_balancer: Arc::new(LoadBalancer::new(BalancingStrategy::LeastLoaded)),
            health_monitor: Arc::new(HealthMonitor::new()),
            retry_policy,
        }
    }
    
    pub async fn process_with_resilience(
        &self,
        tasks: Vec<Task>
    ) -> Vec<Result<TaskResult, ProcessingError>> {
        let mut results = Vec::with_capacity(tasks.len());
        let (result_tx, mut result_rx) = mpsc::channel(tasks.len());
        
        // Submit all tasks
        for task in tasks {
            self.task_queue.push(task.priority as usize, task);
        }
        
        // Start health monitoring
        self.start_health_monitoring();
        
        // Process tasks with fault tolerance
        let agents = Arc::clone(&self.agents);
        let queue = Arc::clone(&self.task_queue);
        let balancer = Arc::clone(&self.load_balancer);
        let retry_policy = self.retry_policy.clone();
        
        let processor = tokio::spawn(async move {
            while let Some(task) = queue.pop().await {
                let result_tx = result_tx.clone();
                let agents = Arc::clone(&agents);
                let balancer = Arc::clone(&balancer);
                let retry_policy = retry_policy.clone();
                
                tokio::spawn(async move {
                    let result = Self::process_task_with_retry(
                        task,
                        &agents,
                        &balancer,
                        &retry_policy
                    ).await;
                    
                    let _ = result_tx.send(result).await;
                });
            }
        });
        
        // Collect results
        while let Some(result) = result_rx.recv().await {
            results.push(result);
            
            if results.len() == tasks.len() {
                break;
            }
        }
        
        results
    }
    
    async fn process_task_with_retry(
        task: Task,
        agents: &Arc<RwLock<Vec<AgentHandle>>>,
        balancer: &Arc<LoadBalancer>,
        retry_policy: &RetryPolicy,
    ) -> Result<TaskResult, ProcessingError> {
        let mut attempts = 0;
        let mut last_error = None;
        
        while attempts < retry_policy.max_attempts {
            // Select best agent
            let agent_idx = balancer.select_agent(agents).await?;
            
            // Try processing
            match Self::try_process_on_agent(task.clone(), agent_idx, agents).await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    last_error = Some(e);
                    attempts += 1;
                    
                    if attempts < retry_policy.max_attempts {
                        // Exponential backoff
                        let delay = retry_policy.base_delay * 2u32.pow(attempts as u32);
                        tokio::time::sleep(delay).await;
                    }
                }
            }
        }
        
        Err(ProcessingError::MaxRetriesExceeded {
            task_id: task.id,
            attempts,
            last_error: last_error.map(|e| e.to_string()),
        })
    }
    
    async fn try_process_on_agent(
        task: Task,
        agent_idx: usize,
        agents: &Arc<RwLock<Vec<AgentHandle>>>,
    ) -> Result<TaskResult, Box<dyn std::error::Error>> {
        let agents_read = agents.read().await;
        let agent_handle = &agents_read[agent_idx];
        
        // Increment load
        agent_handle.load.fetch_add(1, Ordering::Relaxed);
        
        // Process task
        let result = match tokio::time::timeout(
            Duration::from_secs(60),
            agent_handle.agent.process_task(task.clone())
        ).await {
            Ok(Ok(result)) => {
                // Success - update heartbeat
                *agent_handle.last_heartbeat.write().await = Instant::now();
                Ok(result)
            }
            Ok(Err(e)) => {
                // Processing error
                agent_handle.failures.fetch_add(1, Ordering::Relaxed);
                Err(e)
            }
            Err(_) => {
                // Timeout
                agent_handle.failures.fetch_add(1, Ordering::Relaxed);
                Err("Task processing timeout".into())
            }
        };
        
        // Decrement load
        agent_handle.load.fetch_sub(1, Ordering::Relaxed);
        
        result
    }
    
    fn start_health_monitoring(&self) {
        let agents = Arc::clone(&self.agents);
        let monitor = Arc::clone(&self.health_monitor);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(5));
            
            loop {
                interval.tick().await;
                
                let mut agents_write = agents.write().await;
                let now = Instant::now();
                
                // Check each agent's health
                agents_write.retain(|agent| {
                    let last_heartbeat = agent.last_heartbeat.blocking_read();
                    let failures = agent.failures.load(Ordering::Relaxed);
                    
                    // Remove unhealthy agents
                    let is_healthy = now.duration_since(*last_heartbeat) < Duration::from_secs(30)
                        && failures < 5;
                    
                    if !is_healthy {
                        monitor.record_agent_failure(&agent.agent.id());
                    }
                    
                    is_healthy
                });
            }
        });
    }
}

/// Load balancing strategies
struct LoadBalancer {
    strategy: BalancingStrategy,
}

enum BalancingStrategy {
    RoundRobin,
    LeastLoaded,
    WeightedRandom,
    NeuralPredictive,
}

impl LoadBalancer {
    async fn select_agent(
        &self,
        agents: &Arc<RwLock<Vec<AgentHandle>>>
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let agents_read = agents.read().await;
        
        if agents_read.is_empty() {
            return Err("No available agents".into());
        }
        
        let idx = match self.strategy {
            BalancingStrategy::LeastLoaded => {
                agents_read
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, agent)| agent.load.load(Ordering::Relaxed))
                    .map(|(idx, _)| idx)
                    .unwrap()
            }
            _ => 0, // Simplified for example
        };
        
        Ok(idx)
    }
}
```

## 4. User-Defined Extraction Schemas

### Basic Schema Definition and Validation

```rust
use serde::{Deserialize, Serialize};
use jsonschema::{JSONSchema, Draft};

/// User-defined extraction schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionSchema {
    pub name: String,
    pub version: String,
    pub description: Option<String>,
    pub fields: Vec<FieldDefinition>,
    pub rules: Vec<ExtractionRule>,
    pub output_format: OutputFormat,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDefinition {
    pub name: String,
    pub field_type: FieldType,
    pub required: bool,
    pub multiple: bool,
    pub validators: Vec<Validator>,
    pub extractors: Vec<ExtractorConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum FieldType {
    Text { max_length: Option<usize> },
    Number { min: Option<f64>, max: Option<f64> },
    Date { format: String },
    Email,
    Phone,
    Currency { currency: Option<String> },
    Custom { schema: serde_json::Value },
}

/// Schema-based document extractor
pub struct SchemaBasedExtractor {
    schema: ExtractionSchema,
    validator: JSONSchema,
    field_extractors: HashMap<String, Box<dyn FieldExtractor>>,
}

impl SchemaBasedExtractor {
    pub fn new(schema: ExtractionSchema) -> Result<Self, Box<dyn std::error::Error>> {
        // Compile JSON schema for validation
        let json_schema = Self::schema_to_json_schema(&schema);
        let validator = JSONSchema::options()
            .with_draft(Draft::Draft7)
            .compile(&json_schema)?;
        
        // Initialize field extractors
        let mut field_extractors = HashMap::new();
        
        for field in &schema.fields {
            let extractor = Self::create_field_extractor(field)?;
            field_extractors.insert(field.name.clone(), extractor);
        }
        
        Ok(Self {
            schema,
            validator,
            field_extractors,
        })
    }
    
    /// Extract data according to schema
    pub async fn extract(
        &self,
        document: &ExtractedDocument
    ) -> Result<ExtractedData, Box<dyn std::error::Error>> {
        let mut data = HashMap::new();
        let mut extraction_context = ExtractionContext::new(&document);
        
        // Extract each field
        for field in &self.schema.fields {
            let extractor = self.field_extractors
                .get(&field.name)
                .ok_or("Missing extractor")?;
            
            let values = extractor.extract(&document, &mut extraction_context).await?;
            
            // Validate extracted values
            for value in &values {
                self.validate_field_value(field, value)?;
            }
            
            // Store based on multiplicity
            if field.multiple {
                data.insert(field.name.clone(), json!(values));
            } else if let Some(value) = values.first() {
                data.insert(field.name.clone(), value.clone());
            } else if field.required {
                return Err(format!("Required field '{}' not found", field.name).into());
            }
        }
        
        // Apply extraction rules
        self.apply_rules(&mut data, &extraction_context)?;
        
        // Validate complete data
        self.validate_data(&data)?;
        
        Ok(ExtractedData {
            schema_name: self.schema.name.clone(),
            schema_version: self.schema.version.clone(),
            data,
            metadata: extraction_context.metadata,
        })
    }
    
    fn create_field_extractor(
        field: &FieldDefinition
    ) -> Result<Box<dyn FieldExtractor>, Box<dyn std::error::Error>> {
        match &field.field_type {
            FieldType::Text { .. } => Ok(Box::new(TextFieldExtractor::new(field.extractors.clone()))),
            FieldType::Number { .. } => Ok(Box::new(NumberFieldExtractor::new(field.extractors.clone()))),
            FieldType::Date { format } => Ok(Box::new(DateFieldExtractor::new(format.clone(), field.extractors.clone()))),
            FieldType::Email => Ok(Box::new(EmailFieldExtractor::new(field.extractors.clone()))),
            FieldType::Phone => Ok(Box::new(PhoneFieldExtractor::new(field.extractors.clone()))),
            FieldType::Currency { currency } => Ok(Box::new(CurrencyFieldExtractor::new(currency.clone(), field.extractors.clone()))),
            FieldType::Custom { schema } => Ok(Box::new(CustomFieldExtractor::new(schema.clone(), field.extractors.clone()))),
        }
    }
    
    fn validate_field_value(
        &self,
        field: &FieldDefinition,
        value: &serde_json::Value
    ) -> Result<(), Box<dyn std::error::Error>> {
        for validator in &field.validators {
            validator.validate(value)?;
        }
        Ok(())
    }
    
    fn apply_rules(
        &self,
        data: &mut HashMap<String, serde_json::Value>,
        context: &ExtractionContext
    ) -> Result<(), Box<dyn std::error::Error>> {
        for rule in &self.schema.rules {
            rule.apply(data, context)?;
        }
        Ok(())
    }
    
    fn validate_data(
        &self,
        data: &HashMap<String, serde_json::Value>
    ) -> Result<(), Box<dyn std::error::Error>> {
        let data_json = serde_json::to_value(data)?;
        
        if let Err(errors) = self.validator.validate(&data_json) {
            let error_messages: Vec<String> = errors
                .map(|e| format!("{}: {}", e.instance_path, e.instance))
                .collect();
            
            return Err(format!("Validation errors: {}", error_messages.join(", ")).into());
        }
        
        Ok(())
    }
}

/// Field extractor trait
#[async_trait]
trait FieldExtractor: Send + Sync {
    async fn extract(
        &self,
        document: &ExtractedDocument,
        context: &mut ExtractionContext
    ) -> Result<Vec<serde_json::Value>, Box<dyn std::error::Error>>;
}

/// Example: Invoice extraction schema
pub fn create_invoice_schema() -> ExtractionSchema {
    ExtractionSchema {
        name: "invoice".to_string(),
        version: "1.0.0".to_string(),
        description: Some("Standard invoice data extraction".to_string()),
        fields: vec![
            FieldDefinition {
                name: "invoice_number".to_string(),
                field_type: FieldType::Text { max_length: Some(50) },
                required: true,
                multiple: false,
                validators: vec![
                    Validator::Pattern(r"^INV-\d{6}$".to_string()),
                ],
                extractors: vec![
                    ExtractorConfig::Regex {
                        pattern: r"Invoice\s*#?\s*:?\s*(INV-\d{6})".to_string(),
                        group: 1,
                    },
                ],
            },
            FieldDefinition {
                name: "date".to_string(),
                field_type: FieldType::Date {
                    format: "%Y-%m-%d".to_string(),
                },
                required: true,
                multiple: false,
                validators: vec![],
                extractors: vec![
                    ExtractorConfig::Regex {
                        pattern: r"Date\s*:?\s*(\d{4}-\d{2}-\d{2})".to_string(),
                        group: 1,
                    },
                ],
            },
            FieldDefinition {
                name: "total_amount".to_string(),
                field_type: FieldType::Currency {
                    currency: Some("USD".to_string()),
                },
                required: true,
                multiple: false,
                validators: vec![
                    Validator::Range { min: 0.0, max: 1_000_000.0 },
                ],
                extractors: vec![
                    ExtractorConfig::Regex {
                        pattern: r"Total\s*:?\s*\$?([\d,]+\.?\d*)".to_string(),
                        group: 1,
                    },
                ],
            },
            FieldDefinition {
                name: "line_items".to_string(),
                field_type: FieldType::Custom {
                    schema: json!({
                        "type": "object",
                        "properties": {
                            "description": { "type": "string" },
                            "quantity": { "type": "number" },
                            "unit_price": { "type": "number" },
                            "total": { "type": "number" }
                        }
                    }),
                },
                required: false,
                multiple: true,
                validators: vec![],
                extractors: vec![
                    ExtractorConfig::Table {
                        headers: vec![
                            "Description".to_string(),
                            "Qty".to_string(),
                            "Unit Price".to_string(),
                            "Total".to_string(),
                        ],
                    },
                ],
            },
        ],
        rules: vec![
            ExtractionRule::CrossFieldValidation {
                fields: vec!["line_items".to_string(), "total_amount".to_string()],
                validator: "sum_line_items_equals_total".to_string(),
            },
        ],
        output_format: OutputFormat::Json,
    }
}
```

### Advanced Schema with Machine Learning

```rust
use tensorflow::{Graph, Session, Tensor};

/// ML-powered schema extractor
pub struct MLSchemaExtractor {
    schema: ExtractionSchema,
    models: HashMap<String, MLFieldModel>,
    feature_extractor: DocumentFeatureExtractor,
}

struct MLFieldModel {
    graph: Graph,
    session: Session,
    input_placeholder: String,
    output_tensor: String,
    preprocessing: Box<dyn Fn(&str) -> Tensor<f32>>,
}

impl MLSchemaExtractor {
    pub async fn new(
        schema: ExtractionSchema,
        model_dir: &Path
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut models = HashMap::new();
        
        // Load ML models for each field
        for field in &schema.fields {
            if let Some(model_config) = Self::get_ml_config(&field) {
                let model = Self::load_field_model(
                    &field.name,
                    model_dir,
                    model_config
                ).await?;
                
                models.insert(field.name.clone(), model);
            }
        }
        
        let feature_extractor = DocumentFeatureExtractor::new();
        
        Ok(Self {
            schema,
            models,
            feature_extractor,
        })
    }
    
    pub async fn extract_with_ml(
        &self,
        document: &ExtractedDocument
    ) -> Result<ExtractedData, Box<dyn std::error::Error>> {
        let mut data = HashMap::new();
        let mut confidence_scores = HashMap::new();
        
        // Extract document features
        let doc_features = self.feature_extractor.extract_features(document)?;
        
        for field in &self.schema.fields {
            if let Some(model) = self.models.get(&field.name) {
                // ML-based extraction
                let (value, confidence) = self.extract_field_ml(
                    field,
                    model,
                    document,
                    &doc_features
                ).await?;
                
                if confidence > 0.8 || field.required {
                    data.insert(field.name.clone(), value);
                    confidence_scores.insert(field.name.clone(), confidence);
                }
            } else {
                // Fallback to rule-based extraction
                let value = self.extract_field_rules(field, document)?;
                data.insert(field.name.clone(), value);
            }
        }
        
        Ok(ExtractedData {
            schema_name: self.schema.name.clone(),
            schema_version: self.schema.version.clone(),
            data,
            metadata: json!({
                "extraction_method": "ml_enhanced",
                "confidence_scores": confidence_scores,
            }),
        })
    }
    
    async fn extract_field_ml(
        &self,
        field: &FieldDefinition,
        model: &MLFieldModel,
        document: &ExtractedDocument,
        features: &DocumentFeatures,
    ) -> Result<(serde_json::Value, f32), Box<dyn std::error::Error>> {
        // Prepare input tensor
        let input = self.prepare_ml_input(field, document, features)?;
        
        // Run inference
        let mut run_args = tensorflow::SessionRunArgs::new();
        run_args.add_feed(&model.input_placeholder, 0, &input);
        
        let output_token = run_args.request_fetch(&model.output_tensor, 0);
        model.session.run(&mut run_args)?;
        
        let output: Tensor<f32> = run_args.fetch(output_token)?;
        
        // Post-process output
        let (value, confidence) = self.post_process_ml_output(field, output)?;
        
        Ok((value, confidence))
    }
}

/// Document feature extractor for ML models
struct DocumentFeatureExtractor {
    text_embedder: TextEmbedder,
    layout_analyzer: LayoutAnalyzer,
}

impl DocumentFeatureExtractor {
    fn extract_features(
        &self,
        document: &ExtractedDocument
    ) -> Result<DocumentFeatures, Box<dyn std::error::Error>> {
        let text_features = self.text_embedder.embed_document(document)?;
        let layout_features = self.layout_analyzer.analyze_layout(document)?;
        
        Ok(DocumentFeatures {
            text_embeddings: text_features,
            layout_features,
            metadata_features: self.extract_metadata_features(&document.metadata),
        })
    }
}
```

## 5. Custom Output Format Templates

### Template Engine Implementation

```rust
use handlebars::{Handlebars, Helper, RenderContext, Context, Output, HelperResult};
use tera::{Tera, Context as TeraContext};

/// Output formatter with multiple template engines
pub struct OutputFormatter {
    handlebars: Handlebars<'static>,
    tera: Tera,
    custom_helpers: HashMap<String, Box<dyn TemplateHelper>>,
}

impl OutputFormatter {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let mut handlebars = Handlebars::new();
        handlebars.set_strict_mode(true);
        
        // Register custom helpers
        handlebars.register_helper("format_date", Box::new(format_date_helper));
        handlebars.register_helper("format_currency", Box::new(format_currency_helper));
        handlebars.register_helper("table", Box::new(table_helper));
        
        let tera = Tera::new("templates/**/*")?;
        
        Ok(Self {
            handlebars,
            tera,
            custom_helpers: HashMap::new(),
        })
    }
    
    /// Format extracted data using template
    pub fn format(
        &self,
        data: &ExtractedData,
        template: &OutputTemplate
    ) -> Result<String, Box<dyn std::error::Error>> {
        match template.engine {
            TemplateEngine::Handlebars => self.format_handlebars(data, &template.content),
            TemplateEngine::Tera => self.format_tera(data, &template.name),
            TemplateEngine::Custom => self.format_custom(data, template),
        }
    }
    
    fn format_handlebars(
        &self,
        data: &ExtractedData,
        template: &str
    ) -> Result<String, Box<dyn std::error::Error>> {
        let context = self.create_template_context(data);
        Ok(self.handlebars.render_template(template, &context)?)
    }
    
    fn format_tera(
        &self,
        data: &ExtractedData,
        template_name: &str
    ) -> Result<String, Box<dyn std::error::Error>> {
        let mut context = TeraContext::new();
        
        for (key, value) in &data.data {
            context.insert(key, value);
        }
        
        context.insert("metadata", &data.metadata);
        context.insert("schema", &data.schema_name);
        
        Ok(self.tera.render(template_name, &context)?)
    }
    
    fn create_template_context(&self, data: &ExtractedData) -> serde_json::Value {
        json!({
            "data": data.data,
            "metadata": data.metadata,
            "schema": {
                "name": data.schema_name,
                "version": data.schema_version,
            },
            "generated_at": chrono::Utc::now().to_rfc3339(),
        })
    }
}

/// Custom template helper implementations
fn format_date_helper(
    h: &Helper,
    _: &Handlebars,
    _: &Context,
    _: &mut RenderContext,
    out: &mut dyn Output
) -> HelperResult {
    let date_str = h.param(0)
        .and_then(|v| v.value().as_str())
        .unwrap_or("");
    
    let format = h.param(1)
        .and_then(|v| v.value().as_str())
        .unwrap_or("%Y-%m-%d");
    
    if let Ok(date) = chrono::NaiveDate::parse_from_str(date_str, "%Y-%m-%d") {
        out.write(&date.format(format).to_string())?;
    }
    
    Ok(())
}

/// Example templates
pub fn create_invoice_templates() -> HashMap<String, OutputTemplate> {
    let mut templates = HashMap::new();
    
    // HTML Invoice Template
    templates.insert("invoice_html".to_string(), OutputTemplate {
        name: "invoice_html".to_string(),
        engine: TemplateEngine::Handlebars,
        content: r#"
<!DOCTYPE html>
<html>
<head>
    <title>Invoice {{data.invoice_number}}</title>
    <style>
        body { font-family: Arial, sans-serif; }
        .header { background: #f0f0f0; padding: 20px; }
        .invoice-details { margin: 20px 0; }
        table { width: 100%; border-collapse: collapse; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        .total { font-weight: bold; font-size: 1.2em; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Invoice {{data.invoice_number}}</h1>
        <p>Date: {{format_date data.date "%B %d, %Y"}}</p>
    </div>
    
    <div class="invoice-details">
        <h2>Bill To:</h2>
        <p>{{data.customer_name}}<br>
        {{data.customer_address}}</p>
    </div>
    
    <h2>Items:</h2>
    {{#table data.line_items}}
        <tr>
            <td>{{description}}</td>
            <td>{{quantity}}</td>
            <td>{{format_currency unit_price}}</td>
            <td>{{format_currency total}}</td>
        </tr>
    {{/table}}
    
    <div class="total">
        Total: {{format_currency data.total_amount}}
    </div>
</body>
</html>
"#.to_string(),
        variables: vec![
            "invoice_number".to_string(),
            "date".to_string(),
            "customer_name".to_string(),
            "customer_address".to_string(),
            "line_items".to_string(),
            "total_amount".to_string(),
        ],
    });
    
    // Markdown Report Template
    templates.insert("invoice_markdown".to_string(), OutputTemplate {
        name: "invoice_markdown".to_string(),
        engine: TemplateEngine::Tera,
        content: r#"
# Invoice {{ data.invoice_number }}

**Date:** {{ data.date | date(format="%B %d, %Y") }}

## Customer Information
- **Name:** {{ data.customer_name }}
- **Address:** {{ data.customer_address }}

## Invoice Items

| Description | Quantity | Unit Price | Total |
|------------|----------|------------|-------|
{% for item in data.line_items -%}
| {{ item.description }} | {{ item.quantity }} | ${{ item.unit_price | round(2) }} | ${{ item.total | round(2) }} |
{% endfor %}

**Total Amount:** ${{ data.total_amount | round(2) }}

---
*Generated on {{ generated_at }}*
"#.to_string(),
        variables: vec![],
    });
    
    templates
}

/// Dynamic template builder
pub struct TemplateBuilder {
    sections: Vec<TemplateSection>,
    engine: TemplateEngine,
}

impl TemplateBuilder {
    pub fn new(engine: TemplateEngine) -> Self {
        Self {
            sections: Vec::new(),
            engine,
        }
    }
    
    pub fn add_header(mut self, title: &str) -> Self {
        self.sections.push(TemplateSection::Header {
            title: title.to_string(),
            level: 1,
        });
        self
    }
    
    pub fn add_field(mut self, label: &str, field: &str, formatter: Option<String>) -> Self {
        self.sections.push(TemplateSection::Field {
            label: label.to_string(),
            field: field.to_string(),
            formatter,
        });
        self
    }
    
    pub fn add_table(mut self, items_field: &str, columns: Vec<TableColumn>) -> Self {
        self.sections.push(TemplateSection::Table {
            items_field: items_field.to_string(),
            columns,
        });
        self
    }
    
    pub fn build(self) -> OutputTemplate {
        let content = self.generate_template_content();
        
        OutputTemplate {
            name: "dynamic".to_string(),
            engine: self.engine,
            content,
            variables: self.extract_variables(),
        }
    }
    
    fn generate_template_content(&self) -> String {
        let mut content = String::new();
        
        for section in &self.sections {
            match section {
                TemplateSection::Header { title, level } => {
                    match self.engine {
                        TemplateEngine::Handlebars => {
                            content.push_str(&format!("<h{}>{{{{data.{}}}}}</h{}>\n", level, title, level));
                        }
                        TemplateEngine::Tera => {
                            content.push_str(&format!("{} {{{{ data.{} }}}}\n", "#".repeat(*level), title));
                        }
                        _ => {}
                    }
                }
                TemplateSection::Field { label, field, formatter } => {
                    match self.engine {
                        TemplateEngine::Handlebars => {
                            if let Some(fmt) = formatter {
                                content.push_str(&format!("{}: {{{{{}}} data.{}}}\n", label, fmt, field));
                            } else {
                                content.push_str(&format!("{}: {{{{data.{}}}}}\n", label, field));
                            }
                        }
                        TemplateEngine::Tera => {
                            if let Some(fmt) = formatter {
                                content.push_str(&format!("{}: {{{{ data.{} | {} }}}}\n", label, field, fmt));
                            } else {
                                content.push_str(&format!("{}: {{{{ data.{} }}}}\n", label, field));
                            }
                        }
                        _ => {}
                    }
                }
                _ => {} // Table handling would be more complex
            }
        }
        
        content
    }
}
```

## 6. Domain Configuration Examples

### Financial Document Configuration

```rust
use neuraldocflow::{DomainConfig, ExtractionProfile, ValidationRule};

/// Financial document processing configuration
pub fn create_financial_config() -> DomainConfig {
    DomainConfig {
        name: "financial".to_string(),
        description: "Configuration for financial document processing".to_string(),
        
        // Document types supported
        document_types: vec![
            DocumentTypeConfig {
                name: "invoice".to_string(),
                source_priority: vec!["pdf", "docx", "image"],
                extraction_profile: ExtractionProfile {
                    enable_ocr: true,
                    ocr_languages: vec!["eng".to_string()],
                    extract_tables: true,
                    extract_images: false,
                    preserve_formatting: true,
                    confidence_threshold: 0.85,
                },
                validation_rules: vec![
                    ValidationRule::Required(vec!["invoice_number", "date", "total"]),
                    ValidationRule::Format("date", r"\d{4}-\d{2}-\d{2}"),
                    ValidationRule::Range("total", 0.0, 10_000_000.0),
                ],
            },
            DocumentTypeConfig {
                name: "bank_statement".to_string(),
                source_priority: vec!["pdf", "csv"],
                extraction_profile: ExtractionProfile {
                    enable_ocr: true,
                    ocr_languages: vec!["eng".to_string()],
                    extract_tables: true,
                    extract_images: false,
                    preserve_formatting: false,
                    confidence_threshold: 0.90,
                },
                validation_rules: vec![
                    ValidationRule::Required(vec!["account_number", "statement_date"]),
                    ValidationRule::TableStructure("transactions", vec!["date", "description", "amount"]),
                ],
            },
        ],
        
        // Neural enhancement settings
        neural_config: NeuralConfig {
            enable_enhancement: true,
            models: vec![
                ModelConfig {
                    name: "financial_ner".to_string(),
                    path: "models/financial/ner.pb",
                    purpose: ModelPurpose::EntityRecognition,
                    confidence_threshold: 0.8,
                },
                ModelConfig {
                    name: "amount_extractor".to_string(),
                    path: "models/financial/amounts.pb",
                    purpose: ModelPurpose::ValueExtraction,
                    confidence_threshold: 0.85,
                },
            ],
            ensemble_strategy: EnsembleStrategy::Voting,
        },
        
        // Domain-specific extractors
        custom_extractors: vec![
            CustomExtractorConfig {
                name: "iban_extractor".to_string(),
                pattern: r"[A-Z]{2}\d{2}[A-Z0-9]{1,30}".to_string(),
                validator: Some("validate_iban".to_string()),
            },
            CustomExtractorConfig {
                name: "swift_extractor".to_string(),
                pattern: r"[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?".to_string(),
                validator: None,
            },
        ],
        
        // Post-processing rules
        post_processing: vec![
            PostProcessingRule::Normalize("currency_amounts", "standardize_currency"),
            PostProcessingRule::Enrich("company_names", "company_database_lookup"),
            PostProcessingRule::Validate("tax_calculations", "verify_tax_math"),
        ],
    }
}

/// Legal document configuration
pub fn create_legal_config() -> DomainConfig {
    DomainConfig {
        name: "legal".to_string(),
        description: "Configuration for legal document processing".to_string(),
        
        document_types: vec![
            DocumentTypeConfig {
                name: "contract".to_string(),
                source_priority: vec!["pdf", "docx"],
                extraction_profile: ExtractionProfile {
                    enable_ocr: false, // Legal docs usually have selectable text
                    ocr_languages: vec![],
                    extract_tables: true,
                    extract_images: true, // Signatures
                    preserve_formatting: true,
                    confidence_threshold: 0.95, // High accuracy required
                },
                validation_rules: vec![
                    ValidationRule::Required(vec!["parties", "effective_date", "signatures"]),
                    ValidationRule::CrossReference("party_names", "signature_blocks"),
                ],
            },
        ],
        
        neural_config: NeuralConfig {
            enable_enhancement: true,
            models: vec![
                ModelConfig {
                    name: "clause_classifier".to_string(),
                    path: "models/legal/clauses.pb",
                    purpose: ModelPurpose::Classification,
                    confidence_threshold: 0.9,
                },
                ModelConfig {
                    name: "entity_extractor".to_string(),
                    path: "models/legal/entities.pb",
                    purpose: ModelPurpose::EntityRecognition,
                    confidence_threshold: 0.85,
                },
            ],
            ensemble_strategy: EnsembleStrategy::Weighted,
        },
        
        custom_extractors: vec![
            CustomExtractorConfig {
                name: "clause_extractor".to_string(),
                pattern: r"(?i)(whereas|therefore|provided that|subject to).*?\.".to_string(),
                validator: Some("validate_clause_structure".to_string()),
            },
        ],
        
        post_processing: vec![
            PostProcessingRule::Index("clauses", "legal_clause_index"),
            PostProcessingRule::CrossReference("definitions", "usage_locations"),
        ],
    }
}

/// Medical document configuration
pub fn create_medical_config() -> DomainConfig {
    DomainConfig {
        name: "medical".to_string(),
        description: "Configuration for medical document processing".to_string(),
        
        document_types: vec![
            DocumentTypeConfig {
                name: "lab_report".to_string(),
                source_priority: vec!["pdf", "image"],
                extraction_profile: ExtractionProfile {
                    enable_ocr: true,
                    ocr_languages: vec!["eng", "lat"], // Medical terms in Latin
                    extract_tables: true,
                    extract_images: true, // Charts and graphs
                    preserve_formatting: true,
                    confidence_threshold: 0.95, // Critical accuracy
                },
                validation_rules: vec![
                    ValidationRule::Required(vec!["patient_id", "test_date", "results"]),
                    ValidationRule::ReferenceRange("test_results", "normal_ranges"),
                    ValidationRule::PHICompliance(vec!["patient_name", "dob", "ssn"]),
                ],
            },
        ],
        
        neural_config: NeuralConfig {
            enable_enhancement: true,
            models: vec![
                ModelConfig {
                    name: "medical_ner".to_string(),
                    path: "models/medical/ner.pb",
                    purpose: ModelPurpose::EntityRecognition,
                    confidence_threshold: 0.9,
                },
                ModelConfig {
                    name: "icd_classifier".to_string(),
                    path: "models/medical/icd10.pb",
                    purpose: ModelPurpose::Classification,
                    confidence_threshold: 0.85,
                },
            ],
            ensemble_strategy: EnsembleStrategy::Consensus,
        },
        
        custom_extractors: vec![
            CustomExtractorConfig {
                name: "medication_extractor".to_string(),
                pattern: r"(?i)(medication|drug|rx):\s*([A-Za-z\s]+)\s*(\d+\s*mg)".to_string(),
                validator: Some("validate_medication_name".to_string()),
            },
        ],
        
        post_processing: vec![
            PostProcessingRule::Anonymize("phi_fields", "hipaa_compliant_hash"),
            PostProcessingRule::Standardize("drug_names", "rxnorm_lookup"),
            PostProcessingRule::Alert("abnormal_results", "flag_critical_values"),
        ],
    }
}
```

### Domain-Specific Pipeline Configuration

```rust
/// Complete domain processing pipeline
pub struct DomainProcessor {
    config: DomainConfig,
    source_manager: Arc<SourcePluginManager>,
    schema_registry: SchemaRegistry,
    neural_enhancer: Option<DomainNeuralEnhancer>,
    output_formatter: OutputFormatter,
}

impl DomainProcessor {
    pub async fn new(
        domain: &str
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let config = match domain {
            "financial" => create_financial_config(),
            "legal" => create_legal_config(),
            "medical" => create_medical_config(),
            _ => return Err(format!("Unknown domain: {}", domain).into()),
        };
        
        let source_manager = Arc::new(SourcePluginManager::new(Default::default())?);
        
        let schema_registry = SchemaRegistry::new();
        schema_registry.load_domain_schemas(&config.name)?;
        
        let neural_enhancer = if config.neural_config.enable_enhancement {
            Some(DomainNeuralEnhancer::new(&config.neural_config)?)
        } else {
            None
        };
        
        let output_formatter = OutputFormatter::new()?;
        
        Ok(Self {
            config,
            source_manager,
            schema_registry,
            neural_enhancer,
            output_formatter,
        })
    }
    
    pub async fn process_document(
        &self,
        input: SourceInput,
        document_type: &str,
        output_format: &str,
    ) -> Result<ProcessedOutput, Box<dyn std::error::Error>> {
        // Get document type config
        let doc_config = self.config.document_types
            .iter()
            .find(|dt| dt.name == document_type)
            .ok_or("Unknown document type")?;
        
        // Find best source
        let sources = self.source_manager
            .find_compatible_sources(&input)
            .await?;
        
        let source = self.select_source_by_priority(&sources, &doc_config.source_priority)?;
        
        // Extract with profile settings
        let mut extracted = source.extract(input).await?;
        
        // Apply neural enhancement
        if let Some(enhancer) = &self.neural_enhancer {
            enhancer.enhance(&mut extracted, document_type).await?;
        }
        
        // Apply schema extraction
        let schema = self.schema_registry.get_schema(document_type)?;
        let schema_extractor = SchemaBasedExtractor::new(schema)?;
        let extracted_data = schema_extractor.extract(&extracted).await?;
        
        // Apply post-processing
        let processed_data = self.apply_post_processing(extracted_data).await?;
        
        // Format output
        let template = self.get_output_template(document_type, output_format)?;
        let formatted = self.output_formatter.format(&processed_data, &template)?;
        
        Ok(ProcessedOutput {
            document_type: document_type.to_string(),
            data: processed_data,
            formatted_output: formatted,
            metadata: json!({
                "domain": self.config.name,
                "processing_time": start.elapsed(),
                "confidence": extracted.confidence,
            }),
        })
    }
}
```

## 7. Python Usage via PyO3 Bindings

### Python Module Definition

```rust
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::exceptions::PyValueError;

/// Python module for NeuralDocFlow
#[pymodule]
fn neuraldocflow_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyDocumentProcessor>()?;
    m.add_class::<PyExtractionSchema>()?;
    m.add_class::<PyExtractedDocument>()?;
    m.add_class::<PyNeuralEnhancer>()?;
    m.add_function(wrap_pyfunction!(process_document, m)?)?;
    m.add_function(wrap_pyfunction!(create_schema, m)?)?;
    Ok(())
}

/// Python wrapper for document processor
#[pyclass]
struct PyDocumentProcessor {
    processor: Arc<Mutex<DistributedDocumentProcessor>>,
}

#[pymethods]
impl PyDocumentProcessor {
    #[new]
    fn new(agent_count: Option<usize>) -> PyResult<Self> {
        let agent_count = agent_count.unwrap_or(4);
        
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| PyValueError::new_err(format!("Failed to create runtime: {}", e)))?;
        
        let processor = runtime.block_on(async {
            DistributedDocumentProcessor::new(agent_count).await
        }).map_err(|e| PyValueError::new_err(format!("Failed to create processor: {}", e)))?;
        
        Ok(Self {
            processor: Arc::new(Mutex::new(processor)),
        })
    }
    
    /// Process single document
    fn process_file(&self, py: Python, file_path: &str) -> PyResult<PyObject> {
        let processor = self.processor.clone();
        let path = PathBuf::from(file_path);
        
        py.allow_threads(|| {
            let runtime = tokio::runtime::Runtime::new()?;
            
            runtime.block_on(async {
                let proc = processor.lock().unwrap();
                let results = proc.process_documents(vec![path]).await
                    .map_err(|e| PyValueError::new_err(format!("Processing failed: {}", e)))?;
                
                if let Some(result) = results.first() {
                    // Convert to Python dict
                    Python::with_gil(|py| {
                        let dict = PyDict::new(py);
                        dict.set_item("document_id", &result.document_id)?;
                        dict.set_item("blocks_extracted", result.blocks_extracted)?;
                        dict.set_item("confidence", result.confidence)?;
                        dict.set_item("processing_time", result.processing_time.as_secs_f64())?;
                        Ok(dict.into())
                    })
                } else {
                    Err(PyValueError::new_err("No results returned"))
                }
            })
        })
    }
    
    /// Process multiple documents
    fn process_files(&self, py: Python, file_paths: Vec<String>) -> PyResult<PyObject> {
        let processor = self.processor.clone();
        let paths: Vec<PathBuf> = file_paths.into_iter().map(PathBuf::from).collect();
        
        py.allow_threads(|| {
            let runtime = tokio::runtime::Runtime::new()?;
            
            runtime.block_on(async {
                let proc = processor.lock().unwrap();
                let results = proc.process_documents(paths).await
                    .map_err(|e| PyValueError::new_err(format!("Processing failed: {}", e)))?;
                
                Python::with_gil(|py| {
                    let list = PyList::empty(py);
                    
                    for result in results {
                        let dict = PyDict::new(py);
                        dict.set_item("document_id", &result.document_id)?;
                        dict.set_item("blocks_extracted", result.blocks_extracted)?;
                        dict.set_item("confidence", result.confidence)?;
                        dict.set_item("processing_time", result.processing_time.as_secs_f64())?;
                        list.append(dict)?;
                    }
                    
                    Ok(list.into())
                })
            })
        })
    }
    
    /// Process with custom schema
    fn process_with_schema(
        &self,
        py: Python,
        file_path: &str,
        schema: &PyExtractionSchema
    ) -> PyResult<PyObject> {
        let processor = self.processor.clone();
        let path = PathBuf::from(file_path);
        let rust_schema = schema.to_rust_schema()?;
        
        py.allow_threads(|| {
            let runtime = tokio::runtime::Runtime::new()?;
            
            runtime.block_on(async {
                // Process document
                let input = SourceInput::File {
                    path,
                    metadata: None,
                };
                
                // Extract using schema
                let schema_extractor = SchemaBasedExtractor::new(rust_schema)?;
                let manager = SourcePluginManager::new(Default::default())?;
                let sources = manager.find_compatible_sources(&input).await?;
                
                if let Some(source) = sources.first() {
                    let document = source.extract(input).await?;
                    let extracted_data = schema_extractor.extract(&document).await?;
                    
                    // Convert to Python
                    Python::with_gil(|py| {
                        let dict = PyDict::new(py);
                        dict.set_item("schema_name", &extracted_data.schema_name)?;
                        dict.set_item("schema_version", &extracted_data.schema_version)?;
                        dict.set_item("data", pythonize::pythonize(py, &extracted_data.data)?)?;
                        Ok(dict.into())
                    })
                } else {
                    Err(PyValueError::new_err("No compatible source found"))
                }
            })
        })
    }
}

/// Python wrapper for extraction schema
#[pyclass]
struct PyExtractionSchema {
    schema: ExtractionSchema,
}

#[pymethods]
impl PyExtractionSchema {
    #[new]
    fn new(name: String, version: String) -> Self {
        Self {
            schema: ExtractionSchema {
                name,
                version,
                description: None,
                fields: vec![],
                rules: vec![],
                output_format: OutputFormat::Json,
            },
        }
    }
    
    fn add_field(
        &mut self,
        name: String,
        field_type: String,
        required: bool,
        multiple: bool
    ) -> PyResult<()> {
        let field_type = match field_type.as_str() {
            "text" => FieldType::Text { max_length: None },
            "number" => FieldType::Number { min: None, max: None },
            "date" => FieldType::Date { format: "%Y-%m-%d".to_string() },
            "email" => FieldType::Email,
            "phone" => FieldType::Phone,
            "currency" => FieldType::Currency { currency: None },
            _ => return Err(PyValueError::new_err("Invalid field type")),
        };
        
        self.schema.fields.push(FieldDefinition {
            name,
            field_type,
            required,
            multiple,
            validators: vec![],
            extractors: vec![],
        });
        
        Ok(())
    }
    
    fn add_regex_extractor(&mut self, field_name: &str, pattern: &str, group: usize) -> PyResult<()> {
        if let Some(field) = self.schema.fields.iter_mut().find(|f| f.name == field_name) {
            field.extractors.push(ExtractorConfig::Regex {
                pattern: pattern.to_string(),
                group,
            });
            Ok(())
        } else {
            Err(PyValueError::new_err("Field not found"))
        }
    }
    
    fn to_rust_schema(&self) -> PyResult<ExtractionSchema> {
        Ok(self.schema.clone())
    }
}

/// Python-friendly function interface
#[pyfunction]
fn process_document(
    file_path: &str,
    output_format: Option<&str>,
    domain: Option<&str>
) -> PyResult<PyObject> {
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|e| PyValueError::new_err(format!("Failed to create runtime: {}", e)))?;
    
    runtime.block_on(async {
        let domain = domain.unwrap_or("general");
        let output_format = output_format.unwrap_or("json");
        
        // Create processor
        let processor = DomainProcessor::new(domain).await
            .map_err(|e| PyValueError::new_err(format!("Failed to create processor: {}", e)))?;
        
        // Process document
        let input = SourceInput::File {
            path: PathBuf::from(file_path),
            metadata: None,
        };
        
        let result = processor.process_document(
            input,
            "auto", // Auto-detect document type
            output_format
        ).await.map_err(|e| PyValueError::new_err(format!("Processing failed: {}", e)))?;
        
        // Convert to Python
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("document_type", &result.document_type)?;
            dict.set_item("formatted_output", &result.formatted_output)?;
            dict.set_item("metadata", pythonize::pythonize(py, &result.metadata)?)?;
            Ok(dict.into())
        })
    })
}

/// Python usage example
const PYTHON_EXAMPLE: &str = r#"
import neuraldocflow as ndf

# Basic usage
processor = ndf.PyDocumentProcessor(agent_count=4)
result = processor.process_file("invoice.pdf")
print(f"Extracted {result['blocks_extracted']} blocks with {result['confidence']:.2%} confidence")

# Batch processing
results = processor.process_files(["doc1.pdf", "doc2.pdf", "doc3.pdf"])
for result in results:
    print(f"Document {result['document_id']}: {result['blocks_extracted']} blocks")

# Custom schema extraction
schema = ndf.PyExtractionSchema("invoice", "1.0.0")
schema.add_field("invoice_number", "text", required=True, multiple=False)
schema.add_field("date", "date", required=True, multiple=False)
schema.add_field("total", "currency", required=True, multiple=False)
schema.add_field("line_items", "text", required=False, multiple=True)

schema.add_regex_extractor("invoice_number", r"INV-(\d{6})", 1)
schema.add_regex_extractor("date", r"Date:\s*(\d{4}-\d{2}-\d{2})", 1)

result = processor.process_with_schema("invoice.pdf", schema)
print(f"Invoice Number: {result['data']['invoice_number']}")
print(f"Total: {result['data']['total']}")

# Domain-specific processing
financial_result = ndf.process_document(
    "bank_statement.pdf",
    output_format="html",
    domain="financial"
)
print(financial_result['formatted_output'])
"#;
```

## 8. WASM Usage in Browser

### WASM Module Build Configuration

```rust
// Cargo.toml for WASM build
[package]
name = "neuraldocflow-wasm"
version = "1.0.0"

[lib]
crate-type = ["cdylib"]

[dependencies]
wasm-bindgen = "0.2"
wasm-bindgen-futures = "0.4"
web-sys = "0.3"
js-sys = "0.3"
serde-wasm-bindgen = "0.4"
console_error_panic_hook = "0.1"

[dependencies.neuraldocflow]
path = "../neuraldocflow"
default-features = false
features = ["wasm"]
```

### WASM Implementation

```rust
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{File, FileReader, HtmlInputElement};
use js_sys::{ArrayBuffer, Uint8Array, Promise};
use wasm_bindgen_futures::JsFuture;

#[wasm_bindgen]
pub struct WasmDocumentProcessor {
    processor: DocumentProcessor,
}

#[wasm_bindgen]
impl WasmDocumentProcessor {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<WasmDocumentProcessor, JsValue> {
        console_error_panic_hook::set_once();
        
        let processor = DocumentProcessor::new()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        Ok(WasmDocumentProcessor { processor })
    }
    
    /// Process file from File API
    pub async fn process_file(&self, file: File) -> Result<JsValue, JsValue> {
        let array_buffer = Self::read_file_as_array_buffer(file).await?;
        let uint8_array = Uint8Array::new(&array_buffer);
        let bytes = uint8_array.to_vec();
        
        // Process document
        let result = self.processor
            .process_bytes(bytes)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        // Convert to JS object
        serde_wasm_bindgen::to_value(&result)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
    
    /// Process with custom schema
    pub async fn process_with_schema(
        &self,
        file: File,
        schema_json: &str
    ) -> Result<JsValue, JsValue> {
        let schema: ExtractionSchema = serde_json::from_str(schema_json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        let array_buffer = Self::read_file_as_array_buffer(file).await?;
        let uint8_array = Uint8Array::new(&array_buffer);
        let bytes = uint8_array.to_vec();
        
        let result = self.processor
            .process_with_schema(bytes, schema)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        serde_wasm_bindgen::to_value(&result)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
    
    /// Get supported file types
    pub fn supported_types(&self) -> Vec<JsValue> {
        vec![
            JsValue::from_str("pdf"),
            JsValue::from_str("docx"),
            JsValue::from_str("txt"),
            JsValue::from_str("html"),
        ]
    }
    
    async fn read_file_as_array_buffer(file: File) -> Result<ArrayBuffer, JsValue> {
        let file_reader = FileReader::new()?;
        
        let promise = Promise::new(&mut |resolve, reject| {
            let file_reader_clone = file_reader.clone();
            
            let onload = Closure::once(move || {
                let result = file_reader_clone.result().unwrap();
                resolve.call1(&JsValue::undefined(), &result).unwrap();
            });
            
            let onerror = Closure::once(move || {
                reject.call1(&JsValue::undefined(), 
                    &JsValue::from_str("Failed to read file")).unwrap();
            });
            
            file_reader.set_onload(Some(onload.as_ref().unchecked_ref()));
            file_reader.set_onerror(Some(onerror.as_ref().unchecked_ref()));
            
            onload.forget();
            onerror.forget();
        });
        
        file_reader.read_as_array_buffer(&file)?;
        
        let result = JsFuture::from(promise).await?;
        Ok(result.dyn_into::<ArrayBuffer>()?)
    }
}

/// Simplified processor for WASM
struct DocumentProcessor {
    extractor: BasicExtractor,
}

impl DocumentProcessor {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            extractor: BasicExtractor::new(),
        })
    }
    
    async fn process_bytes(
        &self,
        bytes: Vec<u8>
    ) -> Result<ProcessingResult, Box<dyn std::error::Error>> {
        // Simplified processing for WASM
        let doc_type = self.detect_document_type(&bytes)?;
        
        let blocks = match doc_type {
            DocumentType::PDF => self.extractor.extract_pdf(bytes).await?,
            DocumentType::Text => self.extractor.extract_text(bytes).await?,
            _ => return Err("Unsupported document type".into()),
        };
        
        Ok(ProcessingResult {
            document_type: format!("{:?}", doc_type),
            blocks_extracted: blocks.len(),
            content: blocks,
            metadata: HashMap::new(),
        })
    }
}

// JavaScript usage example
#[wasm_bindgen]
pub fn get_js_example() -> String {
    r#"
// Initialize processor
const processor = new WasmDocumentProcessor();

// File input handler
async function handleFileSelect(event) {
    const file = event.target.files[0];
    
    if (!file) return;
    
    try {
        // Process document
        const result = await processor.process_file(file);
        
        console.log(`Document type: ${result.document_type}`);
        console.log(`Blocks extracted: ${result.blocks_extracted}`);
        
        // Display results
        displayResults(result);
    } catch (error) {
        console.error('Processing failed:', error);
    }
}

// Process with schema
async function processWithSchema(file, schema) {
    try {
        const schemaJson = JSON.stringify(schema);
        const result = await processor.process_with_schema(file, schemaJson);
        
        console.log('Extracted data:', result.data);
        return result;
    } catch (error) {
        console.error('Schema extraction failed:', error);
    }
}

// Example schema
const invoiceSchema = {
    name: "invoice",
    version: "1.0.0",
    fields: [
        {
            name: "invoice_number",
            field_type: { type: "Text", max_length: 50 },
            required: true,
            multiple: false,
            extractors: [{
                type: "Regex",
                pattern: "Invoice\\s*#?\\s*:?\\s*(\\w+)",
                group: 1
            }]
        },
        {
            name: "total",
            field_type: { type: "Currency", currency: "USD" },
            required: true,
            multiple: false,
            extractors: [{
                type: "Regex",
                pattern: "Total\\s*:?\\s*\\$?([\\d,]+\\.?\\d*)",
                group: 1
            }]
        }
    ],
    output_format: "Json"
};

// Initialize file input
document.getElementById('file-input').addEventListener('change', handleFileSelect);
"#.to_string()
}
```

### Advanced WASM Features

```rust
/// Streaming processor for large files
#[wasm_bindgen]
pub struct StreamingProcessor {
    buffer: Vec<u8>,
    chunk_size: usize,
}

#[wasm_bindgen]
impl StreamingProcessor {
    #[wasm_bindgen(constructor)]
    pub fn new(chunk_size: Option<usize>) -> Self {
        Self {
            buffer: Vec::new(),
            chunk_size: chunk_size.unwrap_or(1024 * 1024), // 1MB chunks
        }
    }
    
    /// Process file in chunks
    pub async fn process_stream(&mut self, chunk: &[u8]) -> Result<JsValue, JsValue> {
        self.buffer.extend_from_slice(chunk);
        
        if self.buffer.len() >= self.chunk_size {
            // Process accumulated data
            let result = self.process_buffer().await?;
            self.buffer.clear();
            Ok(result)
        } else {
            Ok(JsValue::null())
        }
    }
    
    /// Finalize processing
    pub async fn finalize(&mut self) -> Result<JsValue, JsValue> {
        if !self.buffer.is_empty() {
            self.process_buffer().await
        } else {
            Ok(JsValue::null())
        }
    }
    
    async fn process_buffer(&self) -> Result<JsValue, JsValue> {
        // Process accumulated buffer
        // Implementation depends on document type
        Ok(JsValue::from_str("Processed chunk"))
    }
}

/// Web Worker support for background processing
#[wasm_bindgen]
pub fn create_worker_processor() -> String {
    r#"
// worker.js
importScripts('neuraldocflow_wasm.js');

let processor;

self.onmessage = async function(e) {
    const { type, data } = e.data;
    
    switch(type) {
        case 'init':
            processor = new wasm.WasmDocumentProcessor();
            self.postMessage({ type: 'ready' });
            break;
            
        case 'process':
            try {
                const result = await processor.process_file(data.file);
                self.postMessage({ type: 'result', data: result });
            } catch (error) {
                self.postMessage({ type: 'error', error: error.toString() });
            }
            break;
    }
};
"#.to_string()
}
```

## Performance Optimization Examples

### Memory-Efficient Processing

```rust
/// Zero-copy processing for optimal performance
pub struct ZeroCopyProcessor {
    buffer_pool: BufferPool,
    pinned_memory: PinnedMemory,
}

impl ZeroCopyProcessor {
    pub async fn process_efficient(
        &self,
        input: &[u8]
    ) -> Result<ExtractedDocument, Box<dyn std::error::Error>> {
        // Use memory mapping for large files
        let mapped = unsafe {
            MmapOptions::new()
                .len(input.len())
                .map_raw(input.as_ptr() as *mut libc::c_void)?
        };
        
        // Process without copying
        let document = self.extract_from_mmap(&mapped).await?;
        
        Ok(document)
    }
}

/// SIMD-accelerated text processing
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn find_pattern_simd(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    unsafe {
        if needle.len() > 16 {
            return find_pattern_scalar(haystack, needle);
        }
        
        let needle_vec = _mm_loadu_si128(needle.as_ptr() as *const __m128i);
        let first_byte = _mm_set1_epi8(needle[0] as i8);
        
        for i in 0..haystack.len().saturating_sub(needle.len()) {
            let hay_vec = _mm_loadu_si128(haystack[i..].as_ptr() as *const __m128i);
            let cmp = _mm_cmpeq_epi8(hay_vec, first_byte);
            let mask = _mm_movemask_epi8(cmp);
            
            if mask != 0 {
                // Found potential match, verify
                if &haystack[i..i + needle.len()] == needle {
                    return Some(i);
                }
            }
        }
        
        None
    }
}
```

## Error Handling and Recovery

```rust
/// Comprehensive error handling
#[derive(Debug, thiserror::Error)]
pub enum ProcessingError {
    #[error("Source error: {0}")]
    Source(#[from] SourceError),
    
    #[error("Neural processing error: {0}")]
    Neural(String),
    
    #[error("Schema validation error: {0}")]
    Schema(String),
    
    #[error("Timeout after {0:?}")]
    Timeout(Duration),
    
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),
}

/// Resilient processor with retry and fallback
pub struct ResilientProcessor {
    primary: Box<dyn DocumentSource>,
    fallback: Option<Box<dyn DocumentSource>>,
    retry_policy: RetryPolicy,
}

impl ResilientProcessor {
    pub async fn process_with_fallback(
        &self,
        input: SourceInput
    ) -> Result<ExtractedDocument, ProcessingError> {
        // Try primary processor with retries
        match self.try_with_retries(&self.primary, input.clone()).await {
            Ok(doc) => Ok(doc),
            Err(primary_error) => {
                log::warn!("Primary processor failed: {}", primary_error);
                
                // Try fallback if available
                if let Some(fallback) = &self.fallback {
                    log::info!("Attempting fallback processor");
                    self.try_with_retries(fallback, input).await
                        .map_err(|e| {
                            log::error!("Fallback also failed: {}", e);
                            primary_error
                        })
                } else {
                    Err(primary_error)
                }
            }
        }
    }
    
    async fn try_with_retries(
        &self,
        source: &Box<dyn DocumentSource>,
        input: SourceInput
    ) -> Result<ExtractedDocument, ProcessingError> {
        let mut last_error = None;
        
        for attempt in 0..self.retry_policy.max_attempts {
            match source.extract(input.clone()).await {
                Ok(doc) => return Ok(doc),
                Err(e) => {
                    last_error = Some(ProcessingError::Source(e));
                    
                    if attempt < self.retry_policy.max_attempts - 1 {
                        let delay = self.retry_policy.get_delay(attempt);
                        tokio::time::sleep(delay).await;
                    }
                }
            }
        }
        
        last_error.unwrap_or_else(|| ProcessingError::Source(
            SourceError::ProcessingFailed("Unknown error".to_string())
        ))
    }
}
```

## Conclusion

These examples demonstrate the flexibility and power of NeuralDocFlow's modular architecture. The system can be used for:

1. **Simple PDF extraction** with minimal configuration
2. **Advanced neural enhancement** using ruv-FANN integration
3. **Distributed processing** with DAA coordination
4. **Custom schema extraction** for domain-specific needs
5. **Flexible output formatting** with template support
6. **Domain-specific configurations** for specialized processing
7. **Cross-language support** via PyO3 Python bindings
8. **Browser deployment** using WASM compilation

Each example includes proper error handling, performance optimization considerations, and extensibility patterns to build production-ready document processing systems.