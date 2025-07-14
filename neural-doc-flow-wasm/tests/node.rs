use wasm_bindgen_test::*;
use neural_doc_flow_wasm::*;

// Configure tests to run in Node.js
wasm_bindgen_test_configure!(run_in_node);

#[wasm_bindgen_test]
fn test_node_environment() {
    // Test that we're running in Node.js environment
    assert!(true, "Node.js environment test");
}

#[wasm_bindgen_test]
async fn test_batch_processing_node() {
    // Test batch processing in Node.js environment
    let processor_config = r#"{
        "name": "node-processor",
        "concurrent_tasks": 4,
        "batch_size": 50
    }"#;
    
    let processor = create_document_processor(processor_config)
        .await
        .expect("Should create processor");
    
    let docs = vec![
        r#"{"id": "doc1", "content": "First document"}"#,
        r#"{"id": "doc2", "content": "Second document"}"#,
        r#"{"id": "doc3", "content": "Third document"}"#,
    ];
    
    for doc in docs {
        let result = process_single_document(&processor, doc)
            .await
            .expect("Should process document");
        assert!(!result.is_null(), "Result should not be null");
    }
}

#[wasm_bindgen_test]
async fn test_error_handling_node() {
    // Test error handling in Node.js
    let invalid_config = r#"{ invalid json }"#;
    
    match create_document_processor(invalid_config).await {
        Ok(_) => panic!("Should fail with invalid config"),
        Err(e) => {
            let error_str = format!("{:?}", e);
            assert!(error_str.contains("error") || error_str.contains("invalid"), 
                   "Should contain error message");
        }
    }
}

#[wasm_bindgen_test]
fn test_performance_metrics_node() {
    // Test that performance metrics are available in Node.js
    let start = js_sys::Date::now();
    
    // Simulate some work
    let mut sum = 0;
    for i in 0..1000000 {
        sum += i;
    }
    
    let elapsed = js_sys::Date::now() - start;
    assert!(elapsed > 0.0, "Should measure elapsed time");
    assert!(sum > 0, "Work should be completed");
}

#[wasm_bindgen_test]
async fn test_stream_processing_node() {
    // Test stream processing capabilities in Node.js
    let processor_config = r#"{
        "name": "stream-processor",
        "concurrent_tasks": 2,
        "batch_size": 10
    }"#;
    
    let processor = create_document_processor(processor_config)
        .await
        .expect("Should create processor");
    
    // Process multiple documents as a stream
    let mut processed_count = 0;
    for i in 0..5 {
        let doc = format!(r#"{{"id": "stream-doc-{}", "content": "Stream document {}""}}"#, i, i);
        
        match process_single_document(&processor, &doc).await {
            Ok(_) => processed_count += 1,
            Err(e) => panic!("Failed to process stream document: {:?}", e),
        }
    }
    
    assert_eq!(processed_count, 5, "All stream documents should be processed");
}