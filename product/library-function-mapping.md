# Library Function Mapping for NeuralDocFlow

## Quick Reference: What Library Function to Use

This document provides a direct mapping from architectural needs to specific library functions. **If you need to do X, use library function Y.**

## claude-flow@alpha Function Mapping

### Swarm & Coordination

| Need | Use This Function | Example |
|------|------------------|---------|
| Initialize a swarm | `mcp__claude-flow__swarm_init` | `swarm_init({ topology: "hierarchical", maxAgents: 8 })` |
| Create an agent | `mcp__claude-flow__agent_spawn` | `agent_spawn({ type: "coder", name: "extractor" })` |
| Coordinate tasks | `mcp__claude-flow__task_orchestrate` | `task_orchestrate({ task: "process PDF", strategy: "parallel" })` |
| Check swarm status | `mcp__claude-flow__swarm_status` | `swarm_status({ swarmId: "swarm-123" })` |
| List active agents | `mcp__claude-flow__agent_list` | `agent_list({ swarmId: "swarm-123" })` |
| Monitor performance | `mcp__claude-flow__agent_metrics` | `agent_metrics({ agentId: "agent-456" })` |
| Scale swarm | `mcp__claude-flow__swarm_scale` | `swarm_scale({ swarmId: "swarm-123", targetSize: 12 })` |
| Optimize topology | `mcp__claude-flow__topology_optimize` | `topology_optimize({ swarmId: "swarm-123" })` |

### Memory & Persistence

| Need | Use This Function | Example |
|------|------------------|---------|
| Store data | `mcp__claude-flow__memory_usage` | `memory_usage({ action: "store", key: "doc/123", value: data })` |
| Retrieve data | `mcp__claude-flow__memory_usage` | `memory_usage({ action: "retrieve", key: "doc/123" })` |
| Search memory | `mcp__claude-flow__memory_search` | `memory_search({ pattern: "financial*", limit: 10 })` |
| List stored items | `mcp__claude-flow__memory_usage` | `memory_usage({ action: "list", namespace: "docs" })` |
| Backup memory | `mcp__claude-flow__memory_backup` | `memory_backup({ path: "/backups/memory.db" })` |
| Restore memory | `mcp__claude-flow__memory_restore` | `memory_restore({ backupPath: "/backups/memory.db" })` |
| Persist session | `mcp__claude-flow__memory_persist` | `memory_persist({ sessionId: "session-789" })` |
| Manage namespaces | `mcp__claude-flow__memory_namespace` | `memory_namespace({ action: "create", namespace: "extraction" })` |

### Performance & Monitoring

| Need | Use This Function | Example |
|------|------------------|---------|
| Get performance report | `mcp__claude-flow__performance_report` | `performance_report({ format: "detailed", timeframe: "24h" })` |
| Analyze bottlenecks | `mcp__claude-flow__bottleneck_analyze` | `bottleneck_analyze({ component: "neural-inference" })` |
| Track token usage | `mcp__claude-flow__token_usage` | `token_usage({ operation: "extraction", timeframe: "7d" })` |
| Run benchmarks | `mcp__claude-flow__benchmark_run` | `benchmark_run({ suite: "document-processing" })` |
| Health check | `mcp__claude-flow__health_check` | `health_check({ components: ["swarm", "memory", "neural"] })` |

### Workflow & Automation

| Need | Use This Function | Example |
|------|------------------|---------|
| Create workflow | `mcp__claude-flow__workflow_create` | `workflow_create({ name: "pdf-pipeline", steps: [...] })` |
| Execute workflow | `mcp__claude-flow__workflow_execute` | `workflow_execute({ workflowId: "wf-123", params: {} })` |
| Setup automation | `mcp__claude-flow__automation_setup` | `automation_setup({ rules: [triggerRules] })` |
| Create pipeline | `mcp__claude-flow__pipeline_create` | `pipeline_create({ config: pipelineConfig })` |
| Batch process | `mcp__claude-flow__batch_process` | `batch_process({ items: documents, operation: "extract" })` |

## ruv-FANN Function Mapping

### Neural Networks

| Need | Use This Function | Example |
|------|------------------|---------|
| Create network | `Network::new` | `Network::new(&[768, 256, 128, 10])` |
| Set activation | `set_activation_func` | `network.set_activation_func(ActivationFunc::ReLU)` |
| Train network | `train_on_data` | `network.train_on_data(&training_data, params)` |
| Run inference | `run` / `predict` | `network.run(&input_features)` |
| Save model | `save` | `network.save("model.fann")` |
| Load model | `Network::from_file` | `Network::from_file("model.fann")` |
| Export ONNX | `export_onnx` | `network.export_onnx("model.onnx")` |

### Pattern Recognition

| Need | Use This Function | Example |
|------|------------------|---------|
| Create matcher | `PatternMatcher::new` | `PatternMatcher::new()` |
| Add pattern | `add_pattern` | `matcher.add_pattern("revenue", r"\$[\d,]+")` |
| Find matches | `find_all` | `matcher.find_all(text)` |
| Learn patterns | `learn_patterns` | `learn_patterns(examples)` |
| Extract entities | `extract_entities` | `extract_entities(text, patterns)` |

### WASM Acceleration

| Need | Use This Function | Example |
|------|------------------|---------|
| Initialize WASM | `WasmOptimizer::new` | `WasmOptimizer::new(SimdFeatures::Auto)` |
| Batch inference | `batch_inference` | `optimizer.batch_inference(&network, batch_data)` |
| SIMD processing | `SimdProcessor::new` | `SimdProcessor::with_features(vec!["avx2"])` |
| Optimize ops | `optimize_operations` | `optimizer.optimize_operations(&operations)` |

### Training & Learning

| Need | Use This Function | Example |
|------|------------------|---------|
| Create training data | `TrainingData::new` | `TrainingData::new()` |
| Add samples | `add_sample` | `data.add_sample(&inputs, &outputs)` |
| Set parameters | `TrainingParams` | `TrainingParams { max_epochs: 1000, ... }` |
| Train with params | `train_on_data` | `network.train_on_data(&data, params)` |
| Cross-validation | `cross_validate` | `cross_validate(&network, &data, folds)` |

## DAA Function Mapping

### Agent Management

| Need | Use This Function | Example |
|------|------------------|---------|
| Create agent | `mcp__claude-flow__daa_agent_create` | `daa_agent_create({ agent_type: "processor", capabilities: [...] })` |
| Manage lifecycle | `mcp__claude-flow__daa_lifecycle_manage` | `daa_lifecycle_manage({ agentId: "123", action: "restart" })` |
| Match capabilities | `mcp__claude-flow__daa_capability_match` | `daa_capability_match({ task_requirements: ["ocr", "ml"] })` |
| Agent communication | `mcp__claude-flow__daa_communication` | `daa_communication({ from: "a1", to: "a2", message: {} })` |

### Resource Management

| Need | Use This Function | Example |
|------|------------------|---------|
| Allocate resources | `mcp__claude-flow__daa_resource_alloc` | `daa_resource_alloc({ resources: { cpu: 4, memory: "8GB" } })` |
| Monitor resources | `ResourceMonitor::new` | `ResourceMonitor::new().get_metrics()` |
| Auto-scaling | `DynamicScaling::new` | `DynamicScaling::new(policy).evaluate(metrics)` |
| Load balancing | `mcp__claude-flow__load_balance` | `load_balance({ swarmId: "123", tasks: taskList })` |

### Consensus & Fault Tolerance

| Need | Use This Function | Example |
|------|------------------|---------|
| Reach consensus | `mcp__claude-flow__daa_consensus` | `daa_consensus({ agents: ["a1", "a2"], proposal: {} })` |
| Handle failures | `mcp__claude-flow__daa_fault_tolerance` | `daa_fault_tolerance({ agentId: "123", strategy: "redistribute" })` |
| Validate results | `ConsensusEngine::verify` | `consensus.verify({ proposal: result, threshold: 0.66 })` |
| Recovery planning | `RecoveryPlan::create` | `RecoveryPlan::create(failed_agents, strategy)` |

## Complete Feature-to-Library Mapping

### Document Processing Pipeline

| Pipeline Stage | Primary Library | Functions to Use |
|----------------|----------------|------------------|
| **Initialization** | claude-flow | `swarm_init`, `agent_spawn` × N |
| **Document Loading** | Native Rust | `std::fs`, `memmap2` |
| **Format Detection** | ruv-FANN | `Network::run` on format classifier |
| **Text Extraction** | Native libs | `pdf-rs`, `lopdf` |
| **Classification** | ruv-FANN | `Network::run` on doc classifier |
| **Entity Extraction** | ruv-FANN | `PatternMatcher::find_all`, `Network::run` |
| **Distributed Processing** | DAA | `daa_agent_create`, `daa_communication` |
| **Validation** | DAA | `daa_consensus` |
| **Result Storage** | claude-flow | `memory_usage` with action: "store" |
| **Performance Tracking** | claude-flow | `performance_report`, `agent_metrics` |

### Common Patterns

#### Pattern 1: Parallel Document Processing
```rust
// Step 1: Initialize with claude-flow
let swarm_id = claude_flow::swarm_init(...);

// Step 2: Spawn workers with claude-flow  
for i in 0..8 {
    claude_flow::agent_spawn({ type: "worker" });
}

// Step 3: Distribute with DAA
daa::load_balance({ tasks: documents });

// Step 4: Process with ruv-FANN
let results = ruv_fann::batch_inference(...);

// Step 5: Store with claude-flow
claude_flow::memory_usage({ action: "store", ... });
```

#### Pattern 2: Neural Document Analysis
```rust
// Step 1: Load model with ruv-FANN
let classifier = ruv_fann::Network::from_file("doc_classifier.fann");

// Step 2: Extract features (native)
let features = extract_bert_embeddings(document);

// Step 3: Classify with ruv-FANN
let doc_type = classifier.run(&features);

// Step 4: Extract entities with ruv-FANN
let entities = pattern_matcher.find_all(&document.text);

// Step 5: Validate with DAA consensus
let validated = daa::consensus({ proposal: entities });
```

#### Pattern 3: Fault-Tolerant Processing
```rust
// Step 1: Monitor with claude-flow
let health = claude_flow::health_check({ components: ["workers"] });

// Step 2: Detect issues with DAA
if let Some(failed) = health.failed_agents {
    // Step 3: Handle with DAA
    daa::fault_tolerance({ 
        agentId: failed[0],
        strategy: "redistribute" 
    });
    
    // Step 4: Scale with claude-flow
    claude_flow::swarm_scale({ targetSize: current + 2 });
}

// Step 5: Resume with claude-flow
claude_flow::task_orchestrate({ task: "resume processing" });
```

## Anti-Pattern Detection

### ❌ NEVER Write This → ✅ Use This Instead

| Anti-Pattern | Correct Library Usage |
|--------------|---------------------|
| `struct MySwarmManager` | `claude_flow::swarm_init()` |
| `impl NeuralNetwork` | `ruv_fann::Network::new()` |
| `fn reach_consensus()` | `daa::consensus()` |
| `struct MemoryStore` | `claude_flow::memory_usage()` |
| `async fn spawn_agent()` | `claude_flow::agent_spawn()` |
| `fn train_network()` | `ruv_fann::train_on_data()` |
| `struct DistributedSystem` | `daa::agent_create()` |
| `fn coordinate_tasks()` | `claude_flow::task_orchestrate()` |

## Quick Decision Tree

```
Need to coordinate multiple operations?
  → Use claude-flow (swarm_init, agent_spawn, task_orchestrate)

Need neural networks or ML?
  → Use ruv-FANN (Network::new, train_on_data, run)

Need distributed processing?
  → Use DAA (agent_create, consensus, fault_tolerance)

Need to store/retrieve data?
  → Use claude-flow (memory_usage with store/retrieve)

Need pattern matching?
  → Use ruv-FANN (PatternMatcher)

Need performance monitoring?
  → Use claude-flow (performance_report, agent_metrics)

Need WASM acceleration?
  → Use ruv-FANN (WasmOptimizer)

Need consensus validation?
  → Use DAA (consensus, ConsensusEngine)
```

## Summary

This mapping ensures that for every architectural need in NeuralDocFlow, there is a clear library function to use. No custom implementation should ever be necessary - the libraries provide all required functionality.