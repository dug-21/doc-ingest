mod common;

use common::*;
use neural_doc_flow_api::{
    models::ProcessingStatus,
    state::{AppState, ProcessingQueue},
};
use std::sync::Arc;
use tokio::sync::RwLock;

#[tokio::test]
async fn test_app_state_creation() {
    let (_app, state) = create_test_app().await;

    // Verify state components are initialized
    assert!(!state.config.jwt_secret.is_empty());
    assert_eq!(state.config.rate_limit_requests_per_second, 10);
    
    // Check metrics are initialized
    let metrics = state.metrics.read().await;
    assert_eq!(metrics.get("requests_total").unwrap_or(&0.0), &0.0);
}

#[tokio::test]
async fn test_processing_queue() {
    let (_app, state) = create_test_app().await;

    // Add jobs to queue
    let job1 = fixtures::test_processing_job();
    let job2 = fixtures::test_processing_job();

    state.processing_queue.push(job1.id.clone()).await;
    state.processing_queue.push(job2.id.clone()).await;

    // Pop jobs from queue
    let popped1 = state.processing_queue.pop().await;
    let popped2 = state.processing_queue.pop().await;
    let popped3 = state.processing_queue.pop().await;

    assert_eq!(popped1, Some(job1.id));
    assert_eq!(popped2, Some(job2.id));
    assert_eq!(popped3, None);
}

#[tokio::test]
async fn test_concurrent_queue_access() {
    let (_app, state) = create_test_app().await;

    // Spawn multiple tasks adding to queue
    let mut handles = vec![];

    for i in 0..10 {
        let queue = state.processing_queue.clone();
        let handle = tokio::spawn(async move {
            queue.push(format!("job-{}", i)).await;
        });
        handles.push(handle);
    }

    // Wait for all tasks
    for handle in handles {
        handle.await.unwrap();
    }

    // Verify all jobs are in queue
    let mut jobs = vec![];
    while let Some(job) = state.processing_queue.pop().await {
        jobs.push(job);
    }

    assert_eq!(jobs.len(), 10);
}

#[tokio::test]
async fn test_metrics_tracking() {
    let (_app, state) = create_test_app().await;

    // Update metrics
    {
        let mut metrics = state.metrics.write().await;
        metrics.insert("requests_total".to_string(), 100.0);
        metrics.insert("errors_total".to_string(), 5.0);
    }

    // Read metrics
    {
        let metrics = state.metrics.read().await;
        assert_eq!(metrics.get("requests_total"), Some(&100.0));
        assert_eq!(metrics.get("errors_total"), Some(&5.0));
    }
}

#[tokio::test]
async fn test_state_persistence() {
    let (_app, state) = create_test_app().await;

    // Insert test data
    let doc = fixtures::test_document();
    sqlx::query!(
        r#"
        INSERT INTO documents (id, filename, content_type, size, hash, storage_path, created_at, updated_at)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
        "#,
        doc.id,
        doc.filename,
        doc.content_type,
        doc.size,
        doc.hash,
        doc.storage_path,
        doc.created_at,
        doc.updated_at
    )
    .execute(&state.db)
    .await
    .unwrap();

    // Verify data persists
    let result = sqlx::query!("SELECT COUNT(*) as count FROM documents")
        .fetch_one(&state.db)
        .await
        .unwrap();

    assert_eq!(result.count, 1);
}

#[tokio::test]
async fn test_connection_pool() {
    let (_app, state) = create_test_app().await;

    // Spawn multiple tasks using the database
    let mut handles = vec![];

    for i in 0..5 {
        let db = state.db.clone();
        let handle = tokio::spawn(async move {
            let result = sqlx::query!("SELECT ?1 as value", i)
                .fetch_one(&db)
                .await
                .unwrap();
            result.value
        });
        handles.push(handle);
    }

    // Wait for all tasks
    let mut results = vec![];
    for handle in handles {
        results.push(handle.await.unwrap());
    }

    assert_eq!(results.len(), 5);
}

#[tokio::test]
async fn test_rate_limiter_state() {
    let (_app, state) = create_test_app().await;

    // Create test API key
    let api_key = MockAuth::create_test_api_key(&state, "rate-test").await;

    // Check rate limit state
    let limiter = state.rate_limiter.read().await;
    let key_state = limiter.check(&api_key.key);
    
    // Should allow initial requests
    assert!(key_state.is_none() || key_state.unwrap().requests_remaining > 0);
}

#[tokio::test]
async fn test_background_job_scheduling() {
    let (_app, state) = create_test_app().await;

    // Schedule a cleanup job
    let job_id = uuid::Uuid::new_v4().to_string();
    
    // Add to processing queue
    state.processing_queue.push(job_id.clone()).await;

    // Verify job is scheduled
    let popped = state.processing_queue.pop().await;
    assert_eq!(popped, Some(job_id));
}

#[tokio::test]
async fn test_config_reload() {
    let (_app, state) = create_test_app().await;

    // Original config values
    let original_rate_limit = state.config.rate_limit_requests_per_second;
    let original_timeout = state.config.processing_timeout;

    // Verify config is loaded correctly
    assert_eq!(original_rate_limit, 10);
    assert_eq!(original_timeout, 300);

    // In a real app, you might have a config reload mechanism
    // For now, just verify the values are accessible
    assert!(!state.config.jwt_secret.is_empty());
    assert!(state.config.enable_metrics);
}

#[tokio::test]
async fn test_graceful_shutdown() {
    let (_app, state) = create_test_app().await;

    // Add some jobs to queue
    for i in 0..5 {
        state.processing_queue.push(format!("shutdown-test-{}", i)).await;
    }

    // Simulate shutdown by clearing queue
    let mut remaining_jobs = vec![];
    while let Some(job) = state.processing_queue.pop().await {
        remaining_jobs.push(job);
    }

    // In production, these would be persisted or handled gracefully
    assert_eq!(remaining_jobs.len(), 5);
}

#[tokio::test]
async fn test_memory_usage_tracking() {
    let (_app, state) = create_test_app().await;

    // Track memory usage in metrics
    {
        let mut metrics = state.metrics.write().await;
        
        // Simulate memory tracking
        metrics.insert("memory_used_mb".to_string(), 256.0);
        metrics.insert("memory_available_mb".to_string(), 1024.0);
    }

    // Verify memory metrics
    {
        let metrics = state.metrics.read().await;
        assert!(metrics.get("memory_used_mb").is_some());
        assert!(metrics.get("memory_available_mb").is_some());
    }
}

#[tokio::test]
async fn test_database_migration_state() {
    let (_app, state) = create_test_app().await;

    // Check migration status
    let result = sqlx::query!("SELECT COUNT(*) as count FROM _sqlx_migrations")
        .fetch_one(&state.db)
        .await
        .unwrap();

    // Should have at least one migration
    assert!(result.count > 0);
}

#[tokio::test]
async fn test_concurrent_state_updates() {
    let (_app, state) = create_test_app().await;

    // Spawn multiple tasks updating metrics
    let mut handles = vec![];

    for i in 0..10 {
        let metrics = state.metrics.clone();
        let handle = tokio::spawn(async move {
            let mut m = metrics.write().await;
            let current = m.get("concurrent_test").unwrap_or(&0.0);
            m.insert("concurrent_test".to_string(), current + 1.0);
        });
        handles.push(handle);
    }

    // Wait for all updates
    for handle in handles {
        handle.await.unwrap();
    }

    // Verify final value
    let metrics = state.metrics.read().await;
    assert_eq!(metrics.get("concurrent_test"), Some(&10.0));
}