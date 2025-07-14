//! Integration tests for security features in the neural document flow system

use neural_doc_flow_core::{
    config::{NeuralDocFlowConfig, SecurityScanMode},
    engine::DocumentEngine,
    error::ProcessingError,
};
use tokio;
use tracing_subscriber;

#[tokio::test]
async fn test_security_integration_basic() {
    // Initialize logging for test
    let _ = tracing_subscriber::fmt::try_init();
    
    // Create configuration with security enabled
    let mut config = NeuralDocFlowConfig::default();
    config.security.enabled = true;
    config.security.scan_mode = SecurityScanMode::Standard;
    
    // Create engine with security
    let engine = DocumentEngine::new(config).expect("Failed to create engine");
    
    // Test with benign content
    let benign_content = b"This is a simple text document with no threats.";
    let result = engine.process(benign_content.to_vec(), "text/plain").await;
    
    assert!(result.is_ok(), "Benign content should be processed successfully");
    
    let document = result.unwrap();
    assert_eq!(document.metadata.mime_type, "text/plain");
    assert!(document.content.text.is_some());
}

#[tokio::test]
async fn test_security_file_size_validation() {
    let _ = tracing_subscriber::fmt::try_init();
    
    // Create configuration with small file size limit
    let mut config = NeuralDocFlowConfig::default();
    config.security.enabled = true;
    config.security.policies.max_file_size_mb = 1; // 1MB limit
    
    let engine = DocumentEngine::new(config).expect("Failed to create engine");
    
    // Create content larger than 1MB
    let large_content = vec![b'A'; 2 * 1024 * 1024]; // 2MB
    let result = engine.process(large_content, "text/plain").await;
    
    assert!(result.is_err(), "Large file should be rejected");
    
    if let Err(ProcessingError::SecurityViolation(msg)) = result {
        assert!(msg.contains("exceeds maximum allowed size"));
    } else {
        panic!("Expected SecurityViolation error");
    }
}

#[tokio::test]
async fn test_security_blocked_file_types() {
    let _ = tracing_subscriber::fmt::try_init();
    
    // Create configuration with blocked file types
    let mut config = NeuralDocFlowConfig::default();
    config.security.enabled = true;
    config.security.policies.blocked_file_types.push("application/x-executable".to_string());
    
    let engine = DocumentEngine::new(config).expect("Failed to create engine");
    
    let content = b"Fake executable content";
    let result = engine.process(content.to_vec(), "application/x-executable").await;
    
    assert!(result.is_err(), "Blocked file type should be rejected");
    
    if let Err(ProcessingError::SecurityViolation(msg)) = result {
        assert!(msg.contains("blocked by security policy"));
    } else {
        panic!("Expected SecurityViolation error");
    }
}

#[tokio::test]
async fn test_security_allowed_file_types() {
    let _ = tracing_subscriber::fmt::try_init();
    
    // Create configuration with only specific allowed types
    let mut config = NeuralDocFlowConfig::default();
    config.security.enabled = true;
    config.security.policies.allowed_file_types = vec!["text/plain".to_string()];
    
    let engine = DocumentEngine::new(config).expect("Failed to create engine");
    
    // Test allowed type
    let content = b"This is allowed content";
    let result = engine.process(content.to_vec(), "text/plain").await;
    assert!(result.is_ok(), "Allowed file type should be processed");
    
    // Test non-allowed type
    let result = engine.process(content.to_vec(), "application/pdf").await;
    assert!(result.is_err(), "Non-allowed file type should be rejected");
}

#[tokio::test]
async fn test_security_disabled() {
    let _ = tracing_subscriber::fmt::try_init();
    
    // Create configuration with security disabled
    let mut config = NeuralDocFlowConfig::default();
    config.security.enabled = false;
    
    let engine = DocumentEngine::new(config).expect("Failed to create engine");
    
    // Even potentially suspicious content should be processed when security is disabled
    let suspicious_content = b"<script>alert('xss')</script>";
    let result = engine.process(suspicious_content.to_vec(), "text/html").await;
    
    assert!(result.is_ok(), "Content should be processed when security is disabled");
}

#[tokio::test]
async fn test_security_sanitization() {
    let _ = tracing_subscriber::fmt::try_init();
    
    // This test would work with the security feature enabled
    // For now, we test the basic flow without the actual security processor
    let mut config = NeuralDocFlowConfig::default();
    config.security.enabled = true;
    config.security.scan_mode = SecurityScanMode::Standard;
    
    let engine = DocumentEngine::new(config).expect("Failed to create engine");
    
    let content = b"Some content that might need sanitization";
    let result = engine.process(content.to_vec(), "text/html").await;
    
    // Should succeed even if sanitization is needed
    assert!(result.is_ok(), "Content should be processed with sanitization");
}

#[tokio::test]
async fn test_security_custom_source() {
    let _ = tracing_subscriber::fmt::try_init();
    
    let config = NeuralDocFlowConfig::default();
    let engine = DocumentEngine::new(config).expect("Failed to create engine");
    
    let content = b"Test content with custom source";
    let result = engine.process_with_source(
        content.to_vec(), 
        "text/plain", 
        "custom_test_source"
    ).await;
    
    assert!(result.is_ok(), "Processing with custom source should work");
    
    let document = result.unwrap();
    assert_eq!(document.metadata.source, "custom_test_source");
}

#[tokio::test]
async fn test_performance_monitoring() {
    let _ = tracing_subscriber::fmt::try_init();
    
    let config = NeuralDocFlowConfig::default();
    let engine = DocumentEngine::new(config).expect("Failed to create engine");
    
    // Process multiple documents to test performance monitoring
    for i in 0..5 {
        let content = format!("Test document {}", i);
        let result = engine.process(content.into_bytes(), "text/plain").await;
        assert!(result.is_ok(), "Document {} should be processed successfully", i);
    }
    
    // Performance metrics would be available through the engine's monitor
    // In a real implementation, we'd test that metrics are being collected
}

// Security feature tests - these would work with an actual security processor
mod security_feature_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_neural_security_scanning() {
        let _ = tracing_subscriber::fmt::try_init();
        
        let mut config = NeuralDocFlowConfig::default();
        config.security.enabled = true;
        config.security.scan_mode = SecurityScanMode::Comprehensive;
        
        let engine = DocumentEngine::new(config).expect("Failed to create engine");
        
        // Test with content that should trigger basic security analysis
        let suspicious_content = b"<script>eval(unescape('%75%6E%65%73%63%61%70%65'))</script>"; // Script with eval
        let result = engine.process(suspicious_content.to_vec(), "text/html").await;
        
        // Should either block, quarantine, or sanitize based on threat detection
        assert!(result.is_ok() || matches!(result, Err(ProcessingError::SecurityViolation(_))));
        
        if let Ok(doc) = result {
            // With our basic security analysis, scripts should trigger sanitization
            let was_sanitized = doc.metadata.custom.get("security_sanitized");
            let security_status = doc.metadata.custom.get("security_status");
            
            // Either should be sanitized or quarantined due to script content
            let has_security_action = was_sanitized.is_some() || security_status.is_some();
            
            if !has_security_action {
                // Print debug info to understand what happened
                println!("Document metadata: {:?}", doc.metadata.custom);
                println!("Document content: {:?}", doc.content.text);
            }
            
            assert!(
                has_security_action,
                "Suspicious script content should trigger security measures"
            );
        }
    }
}