use neural_doc_flow_coordination::*;
use std::sync::Arc;
use tokio::time::{sleep, Duration};
use uuid::Uuid;

#[tokio::test]
async fn test_daa_coordination_system_creation() {
    let config = CoordinationConfig::default();
    let result = DaaCoordinationSystem::new(config).await;
    
    assert!(result.is_ok());
    let system = result.unwrap();
    
    assert_eq!(system.config.max_agents, 12);
    assert_eq!(system.config.topology_type, TopologyType::Mesh);
    assert!(system.config.enable_fault_tolerance);
    assert!(system.config.enable_load_balancing);
    assert!(system.config.neural_coordination);
    assert!(system.config.auto_scaling);
    assert!(system.config.performance_monitoring);
}

#[tokio::test]
async fn test_custom_coordination_config() {
    let config = CoordinationConfig {
        max_agents: 20,
        topology_type: TopologyType::Hierarchical,
        enable_fault_tolerance: false,
        enable_load_balancing: false,
        neural_coordination: true,
        auto_scaling: false,
        performance_monitoring: true,
    };
    
    let system = DaaCoordinationSystem::new(config.clone()).await.unwrap();
    
    assert_eq!(system.config.max_agents, 20);
    assert_eq!(system.config.topology_type, TopologyType::Hierarchical);
    assert!(!system.config.enable_fault_tolerance);
    assert!(!system.config.enable_load_balancing);
}

#[tokio::test]
async fn test_auto_spawn_agent() {
    let system = initialize_daa_neural_system().await.unwrap();
    
    let capabilities = AgentCapabilities {
        neural_processing: true,
        text_enhancement: true,
        layout_analysis: false,
        quality_assessment: true,
        coordination: true,
        fault_tolerance: true,
    };
    
    let agent_id = system.auto_spawn_agent(AgentType::Enhancer, capabilities).await.unwrap();
    
    // Verify agent was spawned
    assert_ne!(agent_id, Uuid::nil());
    
    // Check that agent is registered in performance monitor
    let performance = system.get_performance_metrics().await;
    assert!(performance.agent_utilization.contains_key(&agent_id));
}

#[tokio::test]
async fn test_coordinate_task_sequential() {
    let system = initialize_daa_neural_system().await.unwrap();
    let test_data = b"Test document data".to_vec();
    
    let result = system.coordinate_task(
        test_data.clone(),
        CoordinationStrategy::Sequential
    ).await;
    
    assert!(result.is_ok());
    let processed_data = result.unwrap();
    
    // Sequential processing should add markers from each agent
    let processed_str = String::from_utf8_lossy(&processed_data);
    assert!(processed_str.contains("[controlled]"));
    assert!(processed_str.contains("[extracted]"));
    assert!(processed_str.contains("[enhanced]"));
    assert!(processed_str.contains("[validated]"));
    assert!(processed_str.contains("[formatted]"));
}

#[tokio::test]
async fn test_coordinate_task_parallel() {
    let system = initialize_daa_neural_system().await.unwrap();
    let test_data = b"Parallel processing test".to_vec();
    
    let result = system.coordinate_task(
        test_data.clone(),
        CoordinationStrategy::Parallel
    ).await;
    
    assert!(result.is_ok());
    // Parallel processing completes successfully
}

#[tokio::test]
async fn test_coordinate_task_pipeline() {
    let system = initialize_daa_neural_system().await.unwrap();
    let test_data = b"Pipeline processing test".to_vec();
    
    let result = system.coordinate_task(
        test_data.clone(),
        CoordinationStrategy::Pipeline
    ).await;
    
    assert!(result.is_ok());
    let processed_data = result.unwrap();
    
    // Pipeline processing should have stages applied
    let processed_str = String::from_utf8_lossy(&processed_data);
    assert!(processed_str.contains("[extracted]"));
    assert!(processed_str.contains("[enhanced]"));
    assert!(processed_str.contains("[validated]"));
    assert!(processed_str.contains("[formatted]"));
}

#[tokio::test]
async fn test_coordinate_task_adaptive() {
    let system = initialize_daa_neural_system().await.unwrap();
    let test_data = b"Adaptive processing test".to_vec();
    
    // Set initial performance metrics
    {
        let mut monitor = system.performance_monitor.write().await;
        monitor.coordination_efficiency = 0.9; // High efficiency
        monitor.error_rate = 0.01; // Low error rate
    }
    
    let result = system.coordinate_task(
        test_data.clone(),
        CoordinationStrategy::Adaptive
    ).await;
    
    assert!(result.is_ok());
    // Adaptive should choose appropriate strategy based on metrics
}

#[tokio::test]
async fn test_performance_metrics() {
    let system = initialize_daa_neural_system().await.unwrap();
    
    // Get initial metrics
    let initial_metrics = system.get_performance_metrics().await;
    assert_eq!(initial_metrics.coordination_efficiency, 0.0);
    assert_eq!(initial_metrics.error_rate, 0.0);
    assert_eq!(initial_metrics.auto_scaling_events, 0);
    
    // Process some tasks to update metrics
    let test_data = b"Performance test".to_vec();
    system.coordinate_task(test_data, CoordinationStrategy::Sequential).await.unwrap();
    
    // Optimize performance
    system.optimize_performance().await.unwrap();
    
    // Check updated metrics
    let updated_metrics = system.get_performance_metrics().await;
    assert_eq!(updated_metrics.coordination_efficiency, 0.95);
}

#[tokio::test]
async fn test_auto_scaling() {
    let system = initialize_daa_neural_system().await.unwrap();
    
    // Set high utilization to trigger auto-scaling
    {
        let mut monitor = system.performance_monitor.write().await;
        for (_, utilization) in monitor.agent_utilization.iter_mut() {
            *utilization = 0.9; // High utilization
        }
    }
    
    // Trigger auto-scaling
    let result = system.auto_scale().await;
    assert!(result.is_ok());
    
    // Check that auto-scaling event was recorded
    let metrics = system.get_performance_metrics().await;
    assert!(metrics.auto_scaling_events > 0);
}

#[tokio::test]
async fn test_optimize_performance() {
    let system = initialize_daa_neural_system().await.unwrap();
    
    let result = system.optimize_performance().await;
    assert!(result.is_ok());
    
    // Check that optimization improved efficiency
    let metrics = system.get_performance_metrics().await;
    assert_eq!(metrics.coordination_efficiency, 0.95);
}

#[tokio::test]
async fn test_multiple_agent_spawn() {
    let system = initialize_daa_neural_system().await.unwrap();
    
    let agent_types = vec![
        AgentType::Extractor,
        AgentType::Enhancer,
        AgentType::Validator,
        AgentType::Formatter,
    ];
    
    let mut agent_ids = Vec::new();
    
    for agent_type in agent_types {
        let capabilities = AgentCapabilities {
            neural_processing: true,
            text_enhancement: true,
            layout_analysis: true,
            quality_assessment: true,
            coordination: true,
            fault_tolerance: true,
        };
        
        let agent_id = system.auto_spawn_agent(agent_type, capabilities).await.unwrap();
        agent_ids.push(agent_id);
    }
    
    // Verify all agents have unique IDs
    let unique_ids: std::collections::HashSet<_> = agent_ids.iter().collect();
    assert_eq!(unique_ids.len(), agent_ids.len());
}

#[tokio::test]
async fn test_large_document_coordination() {
    let system = initialize_daa_neural_system().await.unwrap();
    
    // Create a large document (1MB)
    let large_data = vec![b'A'; 1024 * 1024];
    
    let result = system.coordinate_task(
        large_data.clone(),
        CoordinationStrategy::Pipeline
    ).await;
    
    assert!(result.is_ok());
    let processed_data = result.unwrap();
    assert!(processed_data.len() > large_data.len()); // Should have added processing markers
}

#[tokio::test]
async fn test_concurrent_task_coordination() {
    let system = Arc::new(initialize_daa_neural_system().await.unwrap());
    
    let mut handles = vec![];
    
    // Spawn multiple concurrent coordination tasks
    for i in 0..5 {
        let sys = Arc::clone(&system);
        let handle = tokio::spawn(async move {
            let data = format!("Concurrent task {}", i).into_bytes();
            sys.coordinate_task(data, CoordinationStrategy::Parallel).await
        });
        handles.push(handle);
    }
    
    // Wait for all tasks to complete
    let mut success_count = 0;
    for handle in handles {
        if let Ok(Ok(_)) = handle.await {
            success_count += 1;
        }
    }
    
    assert_eq!(success_count, 5);
}

#[tokio::test]
async fn test_system_shutdown() {
    let system = initialize_daa_neural_system().await.unwrap();
    
    // Process a task
    let test_data = b"Shutdown test".to_vec();
    system.coordinate_task(test_data, CoordinationStrategy::Sequential).await.unwrap();
    
    // Shutdown system
    let result = system.shutdown().await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_empty_data_coordination() {
    let system = initialize_daa_neural_system().await.unwrap();
    
    let empty_data = vec![];
    let result = system.coordinate_task(
        empty_data,
        CoordinationStrategy::Sequential
    ).await;
    
    assert!(result.is_ok());
    let processed_data = result.unwrap();
    // Should have processing markers even for empty data
    assert!(!processed_data.is_empty());
}

#[tokio::test]
async fn test_topology_types() {
    let topologies = vec![
        TopologyType::Mesh,
        TopologyType::Hierarchical,
        TopologyType::Star,
        TopologyType::Ring,
        TopologyType::Hybrid,
    ];
    
    for topology in topologies {
        let config = CoordinationConfig {
            topology_type: topology.clone(),
            ..Default::default()
        };
        
        let result = DaaCoordinationSystem::new(config).await;
        assert!(result.is_ok());
        
        let system = result.unwrap();
        assert_eq!(system.config.topology_type, topology);
    }
}

#[tokio::test]
async fn test_agent_capabilities_combinations() {
    let system = initialize_daa_neural_system().await.unwrap();
    
    let capability_sets = vec![
        AgentCapabilities {
            neural_processing: true,
            text_enhancement: false,
            layout_analysis: false,
            quality_assessment: false,
            coordination: true,
            fault_tolerance: true,
        },
        AgentCapabilities {
            neural_processing: false,
            text_enhancement: true,
            layout_analysis: true,
            quality_assessment: false,
            coordination: true,
            fault_tolerance: false,
        },
        AgentCapabilities {
            neural_processing: true,
            text_enhancement: true,
            layout_analysis: true,
            quality_assessment: true,
            coordination: true,
            fault_tolerance: true,
        },
    ];
    
    for (i, capabilities) in capability_sets.into_iter().enumerate() {
        let agent_id = system.auto_spawn_agent(
            AgentType::Enhancer,
            capabilities
        ).await.unwrap();
        
        assert_ne!(agent_id, Uuid::nil());
    }
}

#[tokio::test]
async fn test_adaptive_strategy_switching() {
    let system = initialize_daa_neural_system().await.unwrap();
    let test_data = b"Adaptive test".to_vec();
    
    // Test with high efficiency - should use parallel
    {
        let mut monitor = system.performance_monitor.write().await;
        monitor.coordination_efficiency = 0.9;
        monitor.error_rate = 0.01;
    }
    
    let result1 = system.coordinate_task(
        test_data.clone(),
        CoordinationStrategy::Adaptive
    ).await;
    assert!(result1.is_ok());
    
    // Test with high error rate - should use sequential
    {
        let mut monitor = system.performance_monitor.write().await;
        monitor.coordination_efficiency = 0.5;
        monitor.error_rate = 0.2;
    }
    
    let result2 = system.coordinate_task(
        test_data.clone(),
        CoordinationStrategy::Adaptive
    ).await;
    assert!(result2.is_ok());
    
    // Test with medium performance - should use pipeline
    {
        let mut monitor = system.performance_monitor.write().await;
        monitor.coordination_efficiency = 0.7;
        monitor.error_rate = 0.03;
    }
    
    let result3 = system.coordinate_task(
        test_data,
        CoordinationStrategy::Adaptive
    ).await;
    assert!(result3.is_ok());
}

#[tokio::test]
async fn test_agent_capability_enum() {
    let capabilities = vec![
        AgentCapability::ValidationExpert,
        AgentCapability::ContentEnhancement,
        AgentCapability::MetadataExtraction,
        AgentCapability::QualityImprovement,
        AgentCapability::FormatConversion,
        AgentCapability::StyleApplication,
        AgentCapability::TemplateProcessing,
        AgentCapability::PatternRecognition,
        AgentCapability::ErrorDetection,
    ];
    
    // Test that all capabilities are distinct
    let unique_caps: std::collections::HashSet<_> = capabilities.iter().collect();
    assert_eq!(unique_caps.len(), capabilities.len());
    
    // Test serialization
    for cap in capabilities {
        let serialized = serde_json::to_string(&cap).unwrap();
        let deserialized: AgentCapability = serde_json::from_str(&serialized).unwrap();
        assert_eq!(cap, deserialized);
    }
}
