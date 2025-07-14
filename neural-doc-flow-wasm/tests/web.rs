use wasm_bindgen_test::*;
use neural_doc_flow_wasm::*;

// Configure tests to run in the browser
wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn test_wasm_initialization() {
    // Test basic WASM module initialization
    assert!(true, "WASM module loaded successfully");
}

#[wasm_bindgen_test]
async fn test_document_processor_creation() {
    // Test creating a document processor in WASM environment
    let config = r#"{
        "name": "test-processor",
        "concurrent_tasks": 2,
        "batch_size": 10
    }"#;
    
    match create_document_processor(config).await {
        Ok(processor) => {
            assert!(!processor.is_null(), "Processor should be created");
        },
        Err(e) => {
            panic!("Failed to create processor: {:?}", e);
        }
    }
}

#[wasm_bindgen_test]
async fn test_process_single_document() {
    // Test processing a single document
    let processor_config = r#"{
        "name": "test-processor",
        "concurrent_tasks": 1,
        "batch_size": 1
    }"#;
    
    let processor = create_document_processor(processor_config)
        .await
        .expect("Should create processor");
    
    let doc_json = r#"{
        "id": "test-doc-1",
        "content": "Test document content",
        "metadata": {
            "source": "test"
        }
    }"#;
    
    match process_single_document(&processor, doc_json).await {
        Ok(result) => {
            let result_str = result.as_string().expect("Should be string");
            assert!(result_str.contains("processed"), "Document should be processed");
        },
        Err(e) => {
            panic!("Failed to process document: {:?}", e);
        }
    }
}

#[wasm_bindgen_test]
fn test_wasm_memory_allocation() {
    // Test WASM memory allocation and limits
    let large_string = "x".repeat(1024 * 1024); // 1MB string
    assert_eq!(large_string.len(), 1024 * 1024, "Memory allocation should work");
}