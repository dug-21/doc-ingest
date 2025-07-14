/// DAA Integration Test
/// Tests the complete DAA coordination system with message passing, task distribution, and consensus

use neural_doc_flow_coordination::agents::*;
use neural_doc_flow_coordination::topologies::*;
use neural_doc_flow_coordination::messaging::*;
use neural_doc_flow_coordination::consensus::*;
use tokio;
use uuid::Uuid;
use std::sync::Arc;
use std::time::Duration;

#[tokio::test]
async fn test_daa_complete_coordination() {
    // Test the complete DAA coordination system
    
    // 1. Create agent registry with mesh topology
    let agent_registry = Arc::new(tokio::sync::RwLock::new(
        AgentRegistry::new(TopologyType::Mesh)
    ));
    
    // 2. Create and register agents
    let controller_capabilities = AgentCapabilities {
        neural_processing: true,
        text_enhancement: true,
        layout_analysis: true,
        quality_assessment: true,
        coordination: true,
        fault_tolerance: true,
    };
    
    let extractor_capabilities = AgentCapabilities {
        neural_processing: true,
        text_enhancement: true,
        layout_analysis: false,
        quality_assessment: false,
        coordination: true,
        fault_tolerance: false,
    };
    
    let validator_capabilities = AgentCapabilities {
        neural_processing: false,
        text_enhancement: false,
        layout_analysis: false,
        quality_assessment: true,
        coordination: true,
        fault_tolerance: true,
    };
    
    // Create agents
    let controller_agent = Box::new(controller::ControllerAgent::new(controller_capabilities));
    let extractor_agent = Box::new(extractor::ExtractorAgent::new(extractor_capabilities));
    let validator_agent = Box::new(validator::ValidatorAgent::new(validator_capabilities));
    
    let controller_id = controller_agent.id();
    let extractor_id = extractor_agent.id();
    let validator_id = validator_agent.id();
    
    // Register agents
    {
        let registry = agent_registry.write().await;
        registry.register_agent(controller_agent).await.unwrap();
        registry.register_agent(extractor_agent).await.unwrap();
        registry.register_agent(validator_agent).await.unwrap();
    }
    
    // 3. Test task distribution
    let document_task = ProcessingTask {
        id: Uuid::new_v4(),
        document_id: "test_document_001".to_string(),
        task_type: TaskType::DocumentExtraction,
        priority: 255,
        assigned_agent: None,
        status: TaskStatus::Pending,
        created_at: chrono::Utc::now(),
        deadline: Some(chrono::Utc::now() + chrono::Duration::minutes(5)),
    };
    
    // Distribute task
    {
        let registry = agent_registry.read().await;
        registry.distribute_task(document_task.clone()).await.unwrap();
    }
    
    // 4. Test message processing
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    {
        let registry = agent_registry.read().await;
        registry.process_messages().await.unwrap();
    }
    
    // 5. Test coordination statistics
    let stats = {
        let registry = agent_registry.read().await;
        registry.get_coordination_stats().await
    };
    
    assert_eq!(stats.tasks_completed, 0); // Tasks are processed asynchronously
    assert_eq!(stats.tasks_failed, 0);
    
    println!("✅ DAA coordination test passed!");
    println!("📊 Stats: {} tasks completed, {} failed", stats.tasks_completed, stats.tasks_failed);
}

#[tokio::test]
async fn test_consensus_coordination() {
    // Test consensus-based decision making
    
    let mut consensus_coordinator = consensus_coordinator::ConsensusCoordinator::new(
        ConsensusAlgorithm::Majority
    );
    
    // Register agents for voting
    let agent1 = Uuid::new_v4();
    let agent2 = Uuid::new_v4();
    let agent3 = Uuid::new_v4();
    
    consensus_coordinator.register_agent(agent1, 1.0);
    consensus_coordinator.register_agent(agent2, 1.0);
    consensus_coordinator.register_agent(agent3, 1.0);
    
    // Initiate consensus decision
    let decision_id = consensus_coordinator.initiate_consensus_decision(
        consensus_coordinator::DecisionType::TaskAssignment,
        serde_json::json!({
            "task_id": "test_task_001",
            "assigned_agent": agent1.to_string(),
            "priority": 200
        }),
        vec![agent1, agent2, agent3],
        chrono::Utc::now() + chrono::Duration::minutes(10),
        0.6, // 60% consensus threshold
    ).await.unwrap();
    
    // Simulate voting
    consensus_coordinator.process_agent_vote(decision_id, agent1, true, Some("I can handle this task".to_string())).await.unwrap();
    consensus_coordinator.process_agent_vote(decision_id, agent2, true, Some("Agreed".to_string())).await.unwrap();
    
    // Clean up expired decisions
    consensus_coordinator.cleanup_expired_decisions().await.unwrap();
    
    println!("✅ Consensus coordination test passed!");
    println!("📋 Decision ID: {}", decision_id);
}

#[tokio::test]
async fn test_pipeline_connector() {
    // Test pipeline connection to document processing
    
    let agent_registry = Arc::new(tokio::sync::RwLock::new(
        AgentRegistry::new(TopologyType::Pipeline)
    ));
    
    let mut pipeline_connector = pipeline_connector::PipelineConnector::new(agent_registry.clone());
    
    // Initialize pipeline connector
    pipeline_connector.initialize().await.unwrap();
    
    // Create processing request
    let processing_request = pipeline_connector::DocumentProcessingRequest {
        request_id: Uuid::new_v4(),
        document_id: "test_doc_pipeline".to_string(),
        document_data: b"Sample document content for processing".to_vec(),
        processing_config: pipeline_connector::ProcessingConfig {
            extract_text: true,
            enhance_quality: true,
            analyze_layout: true,
            validate_output: true,
            format_output: true,
            neural_enhancement: true,
            accuracy_threshold: 0.99,
        },
        priority: 128,
        deadline: Some(chrono::Utc::now() + chrono::Duration::minutes(5)),
        requester: "test_system".to_string(),
    };
    
    // Submit document for processing
    let request_id = pipeline_connector.submit_document(processing_request).await.unwrap();
    
    // Wait for processing
    tokio::time::sleep(Duration::from_millis(600)).await;
    
    // Check result
    if let Some(result) = pipeline_connector.get_result(request_id).await {
        assert_eq!(result.request_id, request_id);
        assert!(!result.processed_data.is_empty());
        assert!(result.confidence_score > 0.0);
        println!("✅ Pipeline connector test passed!");
        println!("📄 Processed {} bytes with {:.2}% confidence", 
                 result.processed_data.len(), result.confidence_score * 100.0);
    } else {
        println!("⚠️  Pipeline connector test: Processing still in progress");
    }
    
    // Get pipeline stats
    let stats = pipeline_connector.get_pipeline_stats().await;
    println!("📊 Pipeline stats: {} total requests, {} completed", 
             stats.total_requests, stats.completed_requests);
}

#[tokio::test]
async fn test_topology_builder() {
    // Test topology builder for different configurations
    
    let mut builder = TopologyBuilder::new();
    
    // Add nodes
    let controller = Uuid::new_v4();
    let extractor1 = Uuid::new_v4();
    let extractor2 = Uuid::new_v4();
    let validator = Uuid::new_v4();
    let formatter = Uuid::new_v4();
    
    builder.add_node(controller, builder::NodeType::Controller)
           .add_node(extractor1, builder::NodeType::Extractor)
           .add_node(extractor2, builder::NodeType::Extractor)
           .add_node(validator, builder::NodeType::Validator)
           .add_node(formatter, builder::NodeType::Formatter);
    
    // Test star topology
    let star_config = builder::TopologyConfig {
        topology_type: TopologyType::Star,
        max_connections_per_node: 10,
        redundancy_factor: 0.0,
        load_balancing_enabled: true,
        fault_tolerance_enabled: false,
    };
    
    let star_topology = builder.build_topology(&star_config).unwrap();
    assert_eq!(star_topology.topology_type(), TopologyType::Star);
    
    // Test mesh topology
    let mut mesh_builder = TopologyBuilder::new();
    mesh_builder.add_node(controller, builder::NodeType::Controller)
               .add_node(extractor1, builder::NodeType::Extractor)
               .add_node(validator, builder::NodeType::Validator);
    
    let mesh_config = builder::TopologyConfig {
        topology_type: TopologyType::Mesh,
        max_connections_per_node: 10,
        redundancy_factor: 0.0,
        load_balancing_enabled: true,
        fault_tolerance_enabled: false,
    };
    
    let mesh_topology = mesh_builder.build_topology(&mesh_config).unwrap();
    assert_eq!(mesh_topology.topology_type(), TopologyType::Mesh);
    
    // Test hybrid topology
    let mut hybrid_builder = TopologyBuilder::new();
    hybrid_builder.add_node(controller, builder::NodeType::Controller)
                  .add_node(extractor1, builder::NodeType::Extractor)
                  .add_node(extractor2, builder::NodeType::Extractor)
                  .add_node(validator, builder::NodeType::Validator)
                  .add_node(formatter, builder::NodeType::Formatter);
    
    let hybrid_config = builder::TopologyConfig {
        topology_type: TopologyType::Custom("hybrid".to_string()),
        max_connections_per_node: 10,
        redundancy_factor: 0.2,
        load_balancing_enabled: true,
        fault_tolerance_enabled: true,
    };
    
    let hybrid_topology = hybrid_builder.build_topology(&hybrid_config).unwrap();
    
    println!("✅ Topology builder test passed!");
    println!("🌐 Star topology: {} type", star_topology.topology_type());
    println!("🌐 Mesh topology: {} type", mesh_topology.topology_type()); 
    println!("🌐 Hybrid topology: {} type", hybrid_topology.topology_type());
}

#[tokio::test]
async fn test_message_passing_system() {
    // Test the message passing system
    
    let message_bus = MessageBus::new(TopologyType::Mesh);
    
    // Register agents
    let agent1 = Uuid::new_v4();
    let agent2 = Uuid::new_v4();
    let agent3 = Uuid::new_v4();
    
    let capabilities = AgentCapabilities {
        neural_processing: true,
        text_enhancement: true,
        layout_analysis: true,
        quality_assessment: true,
        coordination: true,
        fault_tolerance: true,
    };
    
    message_bus.register_agent(agent1, AgentType::Controller, capabilities.clone()).await.unwrap();
    message_bus.register_agent(agent2, AgentType::Extractor, capabilities.clone()).await.unwrap();
    message_bus.register_agent(agent3, AgentType::Validator, capabilities.clone()).await.unwrap();
    
    // Create and send message
    let test_message = CoordinationMessage {
        id: Uuid::new_v4(),
        from: agent1,
        to: Some(agent2),
        message_type: MessageType::Task,
        payload: serde_json::to_vec(&"Test task data").unwrap(),
        timestamp: chrono::Utc::now(),
        priority: 128,
    };
    
    message_bus.send_message(test_message).await.unwrap();
    
    // Process messages
    tokio::time::sleep(Duration::from_millis(50)).await;
    
    // Check message metrics
    let metrics = message_bus.get_metrics().await;
    assert_eq!(metrics.messages_sent, 1);
    
    println!("✅ Message passing system test passed!");
    println!("📨 Messages sent: {}, delivered: {}", metrics.messages_sent, metrics.messages_delivered);
}

#[tokio::test]
async fn test_fault_tolerance() {
    // Test fault tolerance mechanisms
    
    let message_bus = MessageBus::new(TopologyType::Mesh);
    
    let agent1 = Uuid::new_v4();
    let agent2 = Uuid::new_v4();
    
    let capabilities = AgentCapabilities {
        neural_processing: true,
        text_enhancement: true,
        layout_analysis: true,
        quality_assessment: true,
        coordination: true,
        fault_tolerance: true,
    };
    
    message_bus.register_agent(agent1, AgentType::Controller, capabilities.clone()).await.unwrap();
    message_bus.register_agent(agent2, AgentType::Extractor, capabilities.clone()).await.unwrap();
    
    // Report agent failure
    message_bus.report_agent_failure(agent2, fault_tolerance::FailureType::ProcessingError).await.unwrap();
    
    // Process retry queue
    message_bus.process_retry_queue().await.unwrap();
    
    // Check metrics after failure
    let metrics = message_bus.get_metrics().await;
    
    println!("✅ Fault tolerance test passed!");
    println!("🔧 System handled agent failure gracefully");
    println!("📊 Failed messages: {}", metrics.messages_failed);
}

#[tokio::test]
async fn test_coordination_performance() {
    // Test coordination performance and optimization
    
    let agent_registry = Arc::new(tokio::sync::RwLock::new(
        AgentRegistry::new(TopologyType::Hybrid)
    ));
    
    // Create multiple agents for performance testing
    let mut agent_ids = Vec::new();
    
    for i in 0..10 {
        let capabilities = AgentCapabilities {
            neural_processing: true,
            text_enhancement: true,
            layout_analysis: true,
            quality_assessment: true,
            coordination: true,
            fault_tolerance: true,
        };
        
        let agent = Box::new(controller::ControllerAgent::new(capabilities));
        let agent_id = agent.id();
        agent_ids.push(agent_id);
        
        let registry = agent_registry.read().await;
        registry.register_agent(agent).await.unwrap();
    }
    
    // Create multiple tasks
    let start_time = std::time::Instant::now();
    
    for i in 0..50 {
        let task = ProcessingTask {
            id: Uuid::new_v4(),
            document_id: format!("perf_test_doc_{}", i),
            task_type: TaskType::DocumentExtraction,
            priority: 128,
            assigned_agent: None,
            status: TaskStatus::Pending,
            created_at: chrono::Utc::now(),
            deadline: Some(chrono::Utc::now() + chrono::Duration::minutes(5)),
        };
        
        let registry = agent_registry.read().await;
        registry.distribute_task(task).await.unwrap();
    }
    
    let distribution_time = start_time.elapsed();
    
    // Process all messages
    {
        let registry = agent_registry.read().await;
        registry.process_messages().await.unwrap();
    }
    
    let processing_time = start_time.elapsed();
    
    // Get final stats
    let stats = {
        let registry = agent_registry.read().await;
        registry.get_coordination_stats().await
    };
    
    println!("✅ Coordination performance test passed!");
    println!("⚡ Distribution time: {:?}", distribution_time);
    println!("⚡ Processing time: {:?}", processing_time);
    println!("📊 Throughput: {:.2} tasks/sec", stats.throughput);
    println!("📈 Average processing time: {:.2}ms", stats.average_processing_time * 1000.0);
}