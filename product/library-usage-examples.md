# Library Usage Examples for NeuralDocFlow

## Table of Contents
1. [claude-flow@alpha Examples](#claude-flowalpha-examples)
2. [ruv-FANN Examples](#ruv-fann-examples)
3. [DAA Examples](#daa-examples)
4. [Complete Integration Example](#complete-integration-example)

---

## claude-flow@alpha Examples

### 1. Basic Swarm Initialization and Agent Spawning

```rust
use claude_flow::{SwarmInit, AgentSpawn, TaskOrchestrate, MemoryUsage};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct DocumentProcessingConfig {
    doc_type: String,
    extraction_rules: Vec<String>,
    output_format: String,
}

pub async fn initialize_document_swarm() -> Result<String> {
    // Step 1: Initialize swarm with claude-flow
    let swarm_config = SwarmInitConfig {
        topology: Topology::Hierarchical,
        max_agents: 8,
        strategy: Strategy::Auto,
    };
    
    let swarm_id = claude_flow::swarm_init(swarm_config).await?;
    println!("Swarm initialized: {}", swarm_id);
    
    // Step 2: Spawn specialized agents
    let agents = vec![
        ("doc-classifier", AgentType::Analyst),
        ("text-extractor", AgentType::Coder),
        ("layout-analyzer", AgentType::Researcher),
        ("quality-checker", AgentType::Tester),
        ("orchestrator", AgentType::Coordinator),
    ];
    
    for (name, agent_type) in agents {
        let agent_id = claude_flow::agent_spawn(AgentSpawnConfig {
            agent_type,
            name: name.to_string(),
            swarm_id: swarm_id.clone(),
            capabilities: vec!["document-processing", "neural-inference"],
        }).await?;
        
        println!("Spawned agent {}: {}", name, agent_id);
    }
    
    // Step 3: Store configuration in memory
    let config = DocumentProcessingConfig {
        doc_type: "financial".to_string(),
        extraction_rules: vec!["revenue", "expenses", "ratios"].map(String::from).to_vec(),
        output_format: "json".to_string(),
    };
    
    claude_flow::memory_usage(MemoryConfig {
        action: MemoryAction::Store,
        key: format!("swarm/{}/config", swarm_id),
        value: serde_json::to_string(&config)?,
        namespace: "neuraldocflow",
        ttl: Some(86400), // 24 hours
    }).await?;
    
    Ok(swarm_id)
}
```

### 2. Task Orchestration with Dependencies

```rust
use claude_flow::{TaskOrchestrate, TaskStatus, SwarmMonitor};

pub async fn orchestrate_document_pipeline(
    swarm_id: &str,
    document_path: &str
) -> Result<ProcessingResult> {
    // Define task pipeline with dependencies
    let tasks = vec![
        TaskDefinition {
            id: "load-document",
            description: "Load and validate document",
            priority: Priority::High,
            dependencies: vec![],
        },
        TaskDefinition {
            id: "classify-type",
            description: "Classify document type using neural model",
            priority: Priority::High,
            dependencies: vec!["load-document"],
        },
        TaskDefinition {
            id: "extract-text",
            description: "Extract text content from document",
            priority: Priority::High,
            dependencies: vec!["load-document"],
        },
        TaskDefinition {
            id: "analyze-layout",
            description: "Analyze document layout and structure",
            priority: Priority::Medium,
            dependencies: vec!["load-document"],
        },
        TaskDefinition {
            id: "extract-entities",
            description: "Extract named entities and values",
            priority: Priority::High,
            dependencies: vec!["extract-text", "classify-type"],
        },
        TaskDefinition {
            id: "validate-results",
            description: "Validate extraction results",
            priority: Priority::Medium,
            dependencies: vec!["extract-entities"],
        },
    ];
    
    // Orchestrate tasks
    for task in tasks {
        let task_id = claude_flow::task_orchestrate(TaskConfig {
            task: task.description.clone(),
            dependencies: task.dependencies,
            priority: task.priority,
            strategy: Strategy::Adaptive,
        }).await?;
        
        // Monitor task progress
        loop {
            let status = claude_flow::task_status(task_id.clone()).await?;
            match status {
                TaskStatus::Completed => break,
                TaskStatus::Failed(err) => return Err(err.into()),
                TaskStatus::InProgress => {
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
                _ => {}
            }
        }
    }
    
    // Get final results
    let results = claude_flow::task_results(task_id).await?;
    Ok(results.into())
}
```

### 3. Memory Patterns and Session Management

```rust
use claude_flow::{MemoryUsage, MemorySearch, SessionManagement};

pub struct DocumentMemoryManager {
    namespace: String,
}

impl DocumentMemoryManager {
    pub fn new(namespace: &str) -> Self {
        Self {
            namespace: namespace.to_string(),
        }
    }
    
    // Store document processing results
    pub async fn store_extraction(&self, doc_id: &str, extraction: &ExtractionResult) -> Result<()> {
        // Store main result
        claude_flow::memory_usage(MemoryConfig {
            action: MemoryAction::Store,
            key: format!("docs/{}/extraction", doc_id),
            value: serde_json::to_string(extraction)?,
            namespace: self.namespace.clone(),
            ttl: Some(604800), // 7 days
        }).await?;
        
        // Store metadata for searching
        claude_flow::memory_usage(MemoryConfig {
            action: MemoryAction::Store,
            key: format!("docs/{}/metadata", doc_id),
            value: json!({
                "doc_type": extraction.doc_type,
                "timestamp": extraction.timestamp,
                "confidence": extraction.confidence,
                "entity_count": extraction.entities.len(),
            }).to_string(),
            namespace: self.namespace.clone(),
            ttl: None, // Permanent
        }).await?;
        
        Ok(())
    }
    
    // Search for similar documents
    pub async fn find_similar_docs(&self, pattern: &str) -> Result<Vec<String>> {
        let results = claude_flow::memory_search(SearchConfig {
            pattern: pattern.to_string(),
            namespace: Some(self.namespace.clone()),
            limit: 20,
        }).await?;
        
        Ok(results.into_iter()
            .filter(|r| r.key.contains("docs/") && r.key.contains("/metadata"))
            .map(|r| r.key.split('/').nth(1).unwrap().to_string())
            .collect())
    }
    
    // Session persistence
    pub async fn save_session(&self, session_id: &str, state: &SessionState) -> Result<()> {
        claude_flow::session_save(SessionConfig {
            session_id: session_id.to_string(),
            state: serde_json::to_value(state)?,
            compress: true,
        }).await?;
        
        Ok(())
    }
    
    pub async fn restore_session(&self, session_id: &str) -> Result<SessionState> {
        let saved = claude_flow::session_restore(session_id).await?;
        Ok(serde_json::from_value(saved.state)?)
    }
}
```

---

## ruv-FANN Examples

### 1. Document Classification Neural Network

```rust
use ruv_fann::{Network, ActivationFunc, TrainingData, TrainingParams};
use ruv_fann::wasm::{WasmOptimizer, SimdFeatures};

pub struct DocumentClassifier {
    network: ruv_fann::Network,
    label_map: HashMap<usize, String>,
}

impl DocumentClassifier {
    pub fn new() -> Result<Self> {
        // Create network with ruv-FANN
        // Input: 768 (BERT embeddings), Hidden: 256, 128, Output: 10 (doc types)
        let mut network = ruv_fann::Network::new(&[768, 256, 128, 10])?;
        
        // Set activation functions
        network.set_activation_func_hidden(ActivationFunc::ReLU);
        network.set_activation_func_output(ActivationFunc::Softmax);
        
        // Configure for optimal performance
        network.set_training_algorithm(TrainingAlgorithm::Adam);
        network.set_learning_rate(0.001);
        
        let label_map = HashMap::from([
            (0, "financial_statement".to_string()),
            (1, "legal_contract".to_string()),
            (2, "scientific_paper".to_string()),
            (3, "medical_record".to_string()),
            (4, "invoice".to_string()),
            (5, "resume".to_string()),
            (6, "technical_manual".to_string()),
            (7, "news_article".to_string()),
            (8, "email".to_string()),
            (9, "other".to_string()),
        ]);
        
        Ok(Self { network, label_map })
    }
    
    pub async fn train(&mut self, training_data: Vec<(Vec<f32>, String)>) -> Result<()> {
        // Prepare training data in ruv-FANN format
        let mut fann_data = TrainingData::new()?;
        
        for (features, doc_type) in training_data {
            let label_idx = self.label_map.iter()
                .find(|(_, v)| *v == &doc_type)
                .map(|(k, _)| *k)
                .ok_or("Unknown document type")?;
            
            let mut output = vec![0.0; 10];
            output[label_idx] = 1.0;
            
            fann_data.add_sample(&features, &output)?;
        }
        
        // Train with ruv-FANN
        let params = TrainingParams {
            max_epochs: 1000,
            epochs_between_reports: 100,
            desired_error: 0.001,
            ..Default::default()
        };
        
        self.network.train_on_data(&fann_data, params)?;
        
        Ok(())
    }
    
    pub async fn classify(&self, document_features: &[f32]) -> Result<(String, f32)> {
        // Use ruv-FANN for inference
        let output = self.network.run(document_features)?;
        
        // Find highest confidence class
        let (class_idx, confidence) = output.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        
        let doc_type = self.label_map[&class_idx].clone();
        Ok((doc_type, *confidence))
    }
    
    pub async fn classify_batch_wasm(&self, documents: Vec<Vec<f32>>) -> Result<Vec<(String, f32)>> {
        // Use WASM optimization for batch processing
        let optimizer = WasmOptimizer::new(SimdFeatures::Auto)?;
        
        let batch_output = optimizer.batch_inference(&self.network, documents).await?;
        
        let results = batch_output.into_iter()
            .map(|output| {
                let (idx, conf) = output.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap();
                (self.label_map[&idx].clone(), *conf)
            })
            .collect();
        
        Ok(results)
    }
}
```

### 2. Entity Extraction with Pattern Recognition

```rust
use ruv_fann::{Network, PatternMatcher, NeuralPatterns};

pub struct EntityExtractor {
    ner_network: ruv_fann::Network,
    pattern_matcher: ruv_fann::PatternMatcher,
    entity_types: Vec<String>,
}

impl EntityExtractor {
    pub fn new() -> Result<Self> {
        // Network for Named Entity Recognition
        // Input: 1024 (context window), Hidden: 512, 256, Output: 50 (entity types)
        let ner_network = ruv_fann::Network::new(&[1024, 512, 256, 50])?;
        
        // Pattern matcher for rule-based extraction
        let mut pattern_matcher = ruv_fann::PatternMatcher::new();
        
        // Add financial patterns
        pattern_matcher.add_pattern("revenue", r"\b(?:revenue|sales|income)\s*:?\s*\$?[\d,]+\.?\d*[MBK]?\b")?;
        pattern_matcher.add_pattern("date", r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b")?;
        pattern_matcher.add_pattern("percentage", r"\b\d+\.?\d*\s*%\b")?;
        pattern_matcher.add_pattern("currency", r"\$\s*[\d,]+\.?\d*[MBK]?\b")?;
        
        let entity_types = vec![
            "PERSON", "ORGANIZATION", "LOCATION", "DATE", "MONEY",
            "PERCENTAGE", "PRODUCT", "EVENT", "FACILITY", "GPE",
        ].into_iter().map(String::from).collect();
        
        Ok(Self {
            ner_network,
            pattern_matcher,
            entity_types,
        })
    }
    
    pub async fn extract_entities(&self, text: &str) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();
        
        // Neural entity recognition
        let tokens = self.tokenize(text)?;
        for window in tokens.windows(20) {
            let features = self.encode_context(window)?;
            let predictions = self.ner_network.run(&features)?;
            
            for (i, &score) in predictions.iter().enumerate() {
                if score > 0.8 {
                    entities.push(Entity {
                        text: window.join(" "),
                        entity_type: self.entity_types[i].clone(),
                        confidence: score,
                        method: "neural".to_string(),
                    });
                }
            }
        }
        
        // Pattern-based extraction
        let pattern_matches = self.pattern_matcher.find_all(text)?;
        for (pattern_name, matches) in pattern_matches {
            for m in matches {
                entities.push(Entity {
                    text: m.text.clone(),
                    entity_type: pattern_name.to_uppercase(),
                    confidence: 0.95, // High confidence for pattern matches
                    method: "pattern".to_string(),
                });
            }
        }
        
        // Deduplicate and merge
        self.merge_entities(entities)
    }
    
    pub async fn train_custom_patterns(&mut self, examples: Vec<(String, String)>) -> Result<()> {
        // Use ruv-FANN's pattern learning
        let patterns = ruv_fann::learn_patterns(examples)?;
        
        for (name, pattern) in patterns {
            self.pattern_matcher.add_pattern(&name, &pattern)?;
        }
        
        Ok(())
    }
}
```

### 3. Layout Analysis with Visual Features

```rust
use ruv_fann::{Network, ConvolutionalLayer, VisualFeatureExtractor};

pub struct LayoutAnalyzer {
    layout_network: ruv_fann::Network,
    feature_extractor: ruv_fann::VisualFeatureExtractor,
}

impl LayoutAnalyzer {
    pub fn new() -> Result<Self> {
        // CNN-style network for layout analysis
        let mut network = ruv_fann::Network::new_convolutional()?;
        
        // Add convolutional layers
        network.add_layer(ConvolutionalLayer {
            filters: 32,
            kernel_size: 3,
            stride: 1,
            activation: ActivationFunc::ReLU,
        })?;
        
        network.add_layer(ConvolutionalLayer {
            filters: 64,
            kernel_size: 3,
            stride: 2,
            activation: ActivationFunc::ReLU,
        })?;
        
        // Feature extractor for visual elements
        let feature_extractor = ruv_fann::VisualFeatureExtractor::new()?;
        
        Ok(Self {
            layout_network: network,
            feature_extractor,
        })
    }
    
    pub async fn analyze_layout(&self, page_image: &[u8]) -> Result<LayoutAnalysis> {
        // Extract visual features
        let features = self.feature_extractor.extract(page_image)?;
        
        // Run through layout network
        let predictions = self.layout_network.run(&features)?;
        
        // Interpret predictions as layout elements
        Ok(LayoutAnalysis {
            headers: self.extract_regions(&predictions, 0..10)?,
            paragraphs: self.extract_regions(&predictions, 10..20)?,
            tables: self.extract_regions(&predictions, 20..25)?,
            images: self.extract_regions(&predictions, 25..30)?,
            confidence: predictions[30],
        })
    }
}
```

---

## DAA Examples

### 1. Distributed Document Processing

```rust
use daa::{Agent, AgentCreate, ResourceAlloc, Communication, Consensus};

pub struct DistributedDocumentProcessor {
    coordinator: daa::Agent,
    workers: Vec<daa::Agent>,
    consensus_engine: daa::ConsensusEngine,
}

impl DistributedDocumentProcessor {
    pub async fn new(worker_count: usize) -> Result<Self> {
        // Create coordinator agent
        let coordinator = daa::agent_create(AgentConfig {
            agent_type: "coordinator",
            capabilities: vec!["orchestration", "monitoring", "aggregation"],
            resources: ResourceSpec {
                cpu: CpuQuota::Cores(2),
                memory: MemoryLimit::GB(4),
                network: NetworkBandwidth::Mbps(1000),
            },
        }).await?;
        
        // Create worker agents
        let mut workers = Vec::new();
        for i in 0..worker_count {
            let worker = daa::agent_create(AgentConfig {
                agent_type: "document_worker",
                capabilities: vec!["pdf_processing", "text_extraction", "neural_inference"],
                resources: ResourceSpec {
                    cpu: CpuQuota::Cores(4),
                    memory: MemoryLimit::GB(8),
                    gpu: Some(GpuAllocation::Shared),
                },
            }).await?;
            
            workers.push(worker);
        }
        
        // Initialize consensus engine
        let consensus_engine = daa::ConsensusEngine::new(
            ConsensusProtocol::Raft,
            vec![coordinator.id()] + workers.iter().map(|w| w.id()).collect(),
        )?;
        
        Ok(Self {
            coordinator,
            workers,
            consensus_engine,
        })
    }
    
    pub async fn process_document_batch(&self, documents: Vec<Document>) -> Result<Vec<ProcessingResult>> {
        // Distribute documents among workers
        let chunks = documents.chunks(documents.len() / self.workers.len() + 1);
        let mut tasks = Vec::new();
        
        for (worker, chunk) in self.workers.iter().zip(chunks) {
            // Send work to agent
            let task = daa::communication(CommunicationRequest {
                from: self.coordinator.id(),
                to: worker.id(),
                message: Message::ProcessDocuments(chunk.to_vec()),
                reliability: Reliability::AtLeastOnce,
            }).await?;
            
            tasks.push(task);
        }
        
        // Collect results with consensus
        let mut results = Vec::new();
        for task in tasks {
            let result = task.await?;
            
            // Verify result through consensus
            let verified = self.consensus_engine.verify(ConsensusRequest {
                proposal: result.clone(),
                validators: self.workers.iter().map(|w| w.id()).collect(),
                threshold: 0.66, // 2/3 majority
            }).await?;
            
            if verified {
                results.extend(result.into_results());
            }
        }
        
        Ok(results)
    }
    
    pub async fn handle_agent_failure(&mut self, failed_agent: AgentId) -> Result<()> {
        // Use DAA's fault tolerance
        let recovery_plan = daa::fault_tolerance(FaultRequest {
            failed_agent,
            strategy: RecoveryStrategy::Redistribute,
            timeout: Duration::from_secs(30),
        }).await?;
        
        // Execute recovery
        match recovery_plan {
            RecoveryPlan::Redistribute(tasks) => {
                for (task, new_agent) in tasks {
                    daa::communication(CommunicationRequest {
                        from: self.coordinator.id(),
                        to: new_agent,
                        message: Message::ProcessDocuments(task),
                        reliability: Reliability::Guaranteed,
                    }).await?;
                }
            }
            RecoveryPlan::ReplaceAgent => {
                // Create replacement worker
                let new_worker = daa::agent_create(/* same config */).await?;
                
                // Update workers list
                self.workers.retain(|w| w.id() != failed_agent);
                self.workers.push(new_worker);
                
                // Update consensus engine
                self.consensus_engine.update_members(
                    vec![self.coordinator.id()] + 
                    self.workers.iter().map(|w| w.id()).collect()
                )?;
            }
        }
        
        Ok(())
    }
}
```

### 2. Resource-Aware Processing

```rust
use daa::{ResourceMonitor, ResourceAlloc, DynamicScaling};

pub struct ResourceAwareProcessor {
    resource_monitor: daa::ResourceMonitor,
    scaling_engine: daa::DynamicScaling,
}

impl ResourceAwareProcessor {
    pub async fn new() -> Result<Self> {
        let resource_monitor = daa::ResourceMonitor::new()?;
        let scaling_engine = daa::DynamicScaling::new(ScalingPolicy {
            min_agents: 2,
            max_agents: 16,
            scale_up_threshold: 0.8,  // 80% resource usage
            scale_down_threshold: 0.2, // 20% resource usage
            cooldown: Duration::from_secs(300),
        })?;
        
        Ok(Self {
            resource_monitor,
            scaling_engine,
        })
    }
    
    pub async fn process_with_auto_scaling(&self, workload: Workload) -> Result<()> {
        // Monitor current resources
        let metrics = self.resource_monitor.get_metrics().await?;
        
        // Check if scaling needed
        if let Some(scaling_action) = self.scaling_engine.evaluate(metrics)? {
            match scaling_action {
                ScalingAction::ScaleUp(count) => {
                    for _ in 0..count {
                        let agent = daa::agent_create(/* config */).await?;
                        daa::resource_alloc(ResourceRequest {
                            agents: vec![agent],
                            resources: Resources::auto(),
                        }).await?;
                    }
                }
                ScalingAction::ScaleDown(agents) => {
                    for agent in agents {
                        daa::lifecycle_manage(LifecycleRequest {
                            agent_id: agent,
                            action: LifecycleAction::Graceful Shutdown,
                        }).await?;
                    }
                }
            }
        }
        
        Ok(())
    }
}
```

### 3. Consensus-Based Validation

```rust
use daa::{Consensus, ValidationProtocol, AgreementMechanism};

pub struct ConsensusValidator {
    validators: Vec<daa::Agent>,
    consensus_protocol: daa::ConsensusProtocol,
}

impl ConsensusValidator {
    pub async fn new(validator_count: usize) -> Result<Self> {
        let mut validators = Vec::new();
        
        for i in 0..validator_count {
            let validator = daa::agent_create(AgentConfig {
                agent_type: "validator",
                capabilities: vec!["verification", "quality_check"],
                resources: ResourceSpec::minimal(),
            }).await?;
            
            validators.push(validator);
        }
        
        let consensus_protocol = daa::ConsensusProtocol::new(
            AgreementMechanism::ByzantineFaultTolerant,
            validators.iter().map(|v| v.id()).collect(),
        )?;
        
        Ok(Self {
            validators,
            consensus_protocol,
        })
    }
    
    pub async fn validate_extraction(&self, extraction: ExtractionResult) -> Result<ValidationResult> {
        // Send extraction to all validators
        let mut proposals = Vec::new();
        
        for validator in &self.validators {
            let proposal = daa::communication(CommunicationRequest {
                from: "consensus_coordinator",
                to: validator.id(),
                message: Message::ValidateExtraction(extraction.clone()),
                reliability: Reliability::BestEffort,
            }).await?;
            
            proposals.push(proposal);
        }
        
        // Collect validation results
        let validations: Vec<_> = futures::future::join_all(proposals).await;
        
        // Achieve consensus
        let consensus_result = self.consensus_protocol.reach_agreement(
            validations.into_iter().filter_map(Result::ok).collect()
        ).await?;
        
        Ok(ValidationResult {
            is_valid: consensus_result.agreement,
            confidence: consensus_result.confidence,
            validators_agreed: consensus_result.participant_count,
        })
    }
}
```

---

## Complete Integration Example

### Full Document Processing Pipeline

```rust
use claude_flow::{SwarmInit, AgentSpawn, TaskOrchestrate, MemoryUsage};
use ruv_fann::{Network, PatternMatcher};
use daa::{Agent, ResourceAlloc, Consensus};

pub struct NeuralDocFlowPipeline {
    // Claude Flow components
    swarm_id: String,
    memory_manager: claude_flow::MemoryManager,
    
    // RUV-FANN components
    classifier: ruv_fann::Network,
    extractor: ruv_fann::Network,
    pattern_matcher: ruv_fann::PatternMatcher,
    
    // DAA components
    workers: Vec<daa::Agent>,
    consensus_engine: daa::ConsensusEngine,
}

impl NeuralDocFlowPipeline {
    pub async fn new() -> Result<Self> {
        // Initialize Claude Flow swarm
        let swarm_id = claude_flow::swarm_init(SwarmConfig {
            topology: Topology::Mesh,
            max_agents: 12,
            strategy: Strategy::Adaptive,
        }).await?;
        
        // Spawn specialized agents
        let agent_types = vec![
            ("neural-classifier", AgentType::Analyst),
            ("entity-extractor", AgentType::Researcher),
            ("layout-analyzer", AgentType::Architect),
            ("quality-validator", AgentType::Tester),
            ("result-aggregator", AgentType::Coordinator),
        ];
        
        for (name, agent_type) in agent_types {
            claude_flow::agent_spawn(AgentSpawnConfig {
                swarm_id: swarm_id.clone(),
                agent_type,
                name: name.to_string(),
                capabilities: vec!["document-processing"],
            }).await?;
        }
        
        // Initialize RUV-FANN networks
        let classifier = ruv_fann::Network::new(&[768, 256, 128, 10])?;
        let extractor = ruv_fann::Network::new(&[1024, 512, 256, 50])?;
        let pattern_matcher = ruv_fann::PatternMatcher::new();
        
        // Initialize DAA distributed workers
        let mut workers = Vec::new();
        for i in 0..4 {
            let worker = daa::agent_create(AgentConfig {
                agent_type: "processing_worker",
                capabilities: vec!["parallel_processing"],
                resources: ResourceSpec::default(),
            }).await?;
            workers.push(worker);
        }
        
        let consensus_engine = daa::ConsensusEngine::new(
            ConsensusProtocol::SimpleMajority,
            workers.iter().map(|w| w.id()).collect(),
        )?;
        
        Ok(Self {
            swarm_id,
            memory_manager: claude_flow::MemoryManager::new("neuraldocflow"),
            classifier,
            extractor,
            pattern_matcher,
            workers,
            consensus_engine,
        })
    }
    
    pub async fn process_document(&self, document: Document) -> Result<ProcessingResult> {
        // Step 1: Orchestrate processing with Claude Flow
        let task_id = claude_flow::task_orchestrate(TaskConfig {
            task: format!("Process document: {}", document.id),
            strategy: Strategy::Parallel,
            dependencies: vec![],
            priority: Priority::High,
        }).await?;
        
        // Step 2: Classify document with RUV-FANN
        let features = self.extract_features(&document)?;
        let (doc_type, confidence) = self.classifier.run(&features)
            .map(|output| self.interpret_classification(output))?;
        
        // Store classification result
        claude_flow::memory_usage(MemoryConfig {
            action: MemoryAction::Store,
            key: format!("doc/{}/classification", document.id),
            value: json!({ "type": doc_type, "confidence": confidence }).to_string(),
            namespace: "neuraldocflow",
            ttl: Some(3600),
        }).await?;
        
        // Step 3: Distribute extraction across DAA workers
        let chunks = self.split_document(&document, self.workers.len());
        let mut extraction_tasks = Vec::new();
        
        for (worker, chunk) in self.workers.iter().zip(chunks) {
            let task = daa::communication(CommunicationRequest {
                from: self.swarm_id.clone(),
                to: worker.id(),
                message: Message::ExtractFromChunk {
                    chunk,
                    doc_type: doc_type.clone(),
                    extractor_model: self.extractor.export(),
                },
                reliability: Reliability::AtLeastOnce,
            }).await?;
            
            extraction_tasks.push(task);
        }
        
        // Step 4: Collect and validate results with consensus
        let mut all_entities = Vec::new();
        for task in extraction_tasks {
            let result = task.await?;
            all_entities.extend(result.entities);
        }
        
        // Validate through consensus
        let validated = self.consensus_engine.verify(ConsensusRequest {
            proposal: all_entities.clone(),
            validators: self.workers.iter().map(|w| w.id()).collect(),
            threshold: 0.75,
        }).await?;
        
        // Step 5: Store final results
        let final_result = ProcessingResult {
            document_id: document.id,
            doc_type,
            entities: validated.data,
            confidence,
            processing_time: task.elapsed(),
        };
        
        claude_flow::memory_usage(MemoryConfig {
            action: MemoryAction::Store,
            key: format!("doc/{}/result", document.id),
            value: serde_json::to_string(&final_result)?,
            namespace: "neuraldocflow",
            ttl: None,
        }).await?;
        
        // Complete task
        claude_flow::task_complete(task_id).await?;
        
        Ok(final_result)
    }
}

// Example usage
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize pipeline
    let pipeline = NeuralDocFlowPipeline::new().await?;
    
    // Process document
    let document = Document::load("financial_report.pdf")?;
    let result = pipeline.process_document(document).await?;
    
    println!("Processing complete:");
    println!("  Document Type: {}", result.doc_type);
    println!("  Entities Found: {}", result.entities.len());
    println!("  Confidence: {:.2}%", result.confidence * 100.0);
    println!("  Processing Time: {:?}", result.processing_time);
    
    Ok(())
}
```

## Key Takeaways

1. **Always use claude-flow@alpha for:**
   - Swarm initialization and management
   - Agent spawning and coordination
   - Task orchestration
   - Memory persistence
   - Session management

2. **Always use ruv-FANN for:**
   - Neural network creation and training
   - Pattern matching and recognition
   - WASM/SIMD acceleration
   - Visual feature extraction
   - Model inference

3. **Always use DAA for:**
   - Distributed agent creation
   - Resource allocation and monitoring
   - Consensus and validation
   - Fault tolerance
   - Inter-agent communication

4. **Never reimplement:**
   - Custom coordination systems
   - Neural network architectures
   - Distributed computing logic
   - Memory/persistence layers
   - Consensus protocols

By following these examples, all NeuralDocFlow functionality leverages existing, tested libraries rather than reimplementing core functionality.