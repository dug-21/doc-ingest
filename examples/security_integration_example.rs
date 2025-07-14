//! Example demonstrating the integrated security features in Neural Document Flow
//! 
//! This example shows how security scanning is integrated into the main processing pipeline
//! and how different security policies and scanning modes work.

use neural_doc_flow_core::{
    config::{
        NeuralDocFlowConfig, SecurityConfig, SecurityScanMode, SecurityAction,
        ThreatDetectionConfig, SecurityPolicies, AuditConfig, AuditLogLevel,
        AuditLogDestination, AuditEventType,
    },
    engine::DocumentEngine,
    error::ProcessingError,
};
use std::path::PathBuf;
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    println!("🔒 Neural Document Flow - Security Integration Example");
    println!("====================================================");
    
    // Example 1: Basic security configuration
    println!("\n📋 Example 1: Basic Security Configuration");
    basic_security_example().await?;
    
    // Example 2: Comprehensive security scanning
    println!("\n🔍 Example 2: Comprehensive Security Scanning");
    comprehensive_security_example().await?;
    
    // Example 3: Custom security policies
    println!("\n⚙️ Example 3: Custom Security Policies");
    custom_policies_example().await?;
    
    // Example 4: Security audit logging
    println!("\n📊 Example 4: Security Audit Logging");
    audit_logging_example().await?;
    
    // Example 5: Error handling with security violations
    println!("\n❌ Example 5: Security Violation Handling");
    security_violation_example().await?;
    
    println!("\n✅ All security integration examples completed successfully!");
    
    Ok(())
}

async fn basic_security_example() -> Result<(), ProcessingError> {
    println!("Creating engine with basic security settings...");
    
    // Create a config with standard security
    let mut config = NeuralDocFlowConfig::default();
    config.security = SecurityConfig {
        enabled: true,
        scan_mode: SecurityScanMode::Standard,
        threat_detection: ThreatDetectionConfig {
            malware_detection: true,
            behavioral_analysis: true,
            anomaly_detection: false, // Disable for basic mode
            confidence_threshold: 0.8,
            neural_models: std::collections::HashMap::new(),
            custom_patterns: Vec::new(),
        },
        ..Default::default()
    };
    
    let engine = DocumentEngine::new(config)?;
    
    // Process a benign document
    let safe_content = b"This is a completely safe text document.\nIt contains no threats whatsoever.";
    println!("Processing safe content...");
    
    let result = engine.process(safe_content.to_vec(), "text/plain").await?;
    println!("✅ Safe document processed successfully");
    println!("   Document ID: {}", result.id);
    println!("   Content length: {} characters", 
             result.content.text.as_ref().map(|t| t.len()).unwrap_or(0));
    
    // Process content with potential security concerns
    let suspicious_content = b"<html><script>console.log('potential threat')</script></html>";
    println!("\nProcessing content with JavaScript...");
    
    let result = engine.process(suspicious_content.to_vec(), "text/html").await?;
    println!("✅ HTML content processed with security scanning");
    
    // Check if content was sanitized
    if let Some(sanitized) = result.metadata.custom.get("security_sanitized") {
        println!("   ⚠️ Content was sanitized: {}", sanitized);
    }
    
    Ok(())
}

async fn comprehensive_security_example() -> Result<(), ProcessingError> {
    println!("Creating engine with comprehensive security scanning...");
    
    let mut config = NeuralDocFlowConfig::default();
    config.security.scan_mode = SecurityScanMode::Comprehensive;
    config.security.threat_detection.confidence_threshold = 0.6; // Lower threshold for demo
    
    let engine = DocumentEngine::new(config)?;
    
    // Test various types of content
    let test_cases = vec![
        ("Safe PDF", b"PDF content here" as &[u8], "application/pdf"),
        ("Text with URLs", b"Check out https://example.com and http://test.com", "text/plain"),
        ("HTML with forms", b"<html><form action='/submit'><input type='password'></form></html>", "text/html"),
        ("JSON data", b"{\"user\": \"test\", \"action\": \"login\"}", "application/json"),
    ];
    
    for (name, content, mime_type) in test_cases {
        println!("\nTesting: {}", name);
        
        match engine.process(content.to_vec(), mime_type).await {
            Ok(doc) => {
                println!("   ✅ Processed successfully");
                
                // Check security metadata
                if let Some(status) = doc.metadata.custom.get("security_status") {
                    println!("   🔒 Security status: {}", status);
                }
                
                if doc.metadata.custom.contains_key("security_sanitized") {
                    println!("   🧹 Content was sanitized");
                }
            }
            Err(e) => {
                println!("   ❌ Processing failed: {}", e);
            }
        }
    }
    
    Ok(())
}

async fn custom_policies_example() -> Result<(), ProcessingError> {
    println!("Creating engine with custom security policies...");
    
    let mut config = NeuralDocFlowConfig::default();
    
    // Configure strict security policies
    config.security.policies = SecurityPolicies {
        max_file_size_mb: 10, // 10MB limit
        allowed_file_types: vec![
            "text/plain".to_string(),
            "application/json".to_string(),
            "application/pdf".to_string(),
        ],
        blocked_file_types: vec![
            "application/x-executable".to_string(),
            "application/javascript".to_string(),
        ],
        require_source_validation: true,
        encrypt_content_at_rest: false,
        action_overrides: {
            let mut overrides = std::collections::HashMap::new();
            overrides.insert("large_file".to_string(), SecurityAction::Quarantine);
            overrides.insert("suspicious_script".to_string(), SecurityAction::Sanitize);
            overrides
        },
    };
    
    let engine = DocumentEngine::new(config)?;
    
    println!("\nTesting allowed file type (text/plain):");
    let text_content = b"This is plain text and should be allowed.";
    match engine.process(text_content.to_vec(), "text/plain").await {
        Ok(_) => println!("   ✅ Plain text processed successfully"),
        Err(e) => println!("   ❌ Unexpected error: {}", e),
    }
    
    println!("\nTesting blocked file type (application/javascript):");
    let js_content = b"function maliciousCode() { /* something bad */ }";
    match engine.process(js_content.to_vec(), "application/javascript").await {
        Ok(_) => println!("   ⚠️ JavaScript unexpectedly processed"),
        Err(ProcessingError::SecurityViolation(msg)) => {
            println!("   ✅ JavaScript correctly blocked: {}", msg);
        }
        Err(e) => println!("   ❌ Unexpected error type: {}", e),
    }
    
    println!("\nTesting file size limit:");
    let large_content = vec![b'X'; 15 * 1024 * 1024]; // 15MB
    match engine.process(large_content, "text/plain").await {
        Ok(_) => println!("   ⚠️ Large file unexpectedly processed"),
        Err(ProcessingError::SecurityViolation(msg)) => {
            println!("   ✅ Large file correctly rejected: {}", msg);
        }
        Err(e) => println!("   ❌ Unexpected error type: {}", e),
    }
    
    Ok(())
}

async fn audit_logging_example() -> Result<(), ProcessingError> {
    println!("Creating engine with security audit logging...");
    
    let mut config = NeuralDocFlowConfig::default();
    config.security.audit = AuditConfig {
        enabled: true,
        log_level: AuditLogLevel::All,
        log_destination: AuditLogDestination::File(PathBuf::from("./security-audit-example.log")),
        retention_days: 30,
        include_sensitive_data: false,
        logged_events: vec![
            AuditEventType::ScanStart,
            AuditEventType::ScanComplete,
            AuditEventType::ThreatDetected,
            AuditEventType::DocumentBlocked,
            AuditEventType::PolicyViolation,
        ],
    };
    
    let engine = DocumentEngine::new(config)?;
    
    println!("Processing documents with audit logging enabled...");
    
    // Process several documents to generate audit events
    let test_documents = vec![
        ("document1.txt", b"Normal document content", "text/plain"),
        ("document2.html", b"<html><body>Web content</body></html>", "text/html"),
        ("document3.json", b"{\"data\": \"test\"}", "application/json"),
    ];
    
    for (filename, content, mime_type) in test_documents {
        println!("   Processing: {}", filename);
        
        match engine.process_with_source(content.to_vec(), mime_type, filename).await {
            Ok(doc) => {
                println!("     ✅ Processed - ID: {}", doc.id);
            }
            Err(e) => {
                println!("     ❌ Failed: {}", e);
            }
        }
    }
    
    println!("   📝 Security events logged to: ./security-audit-example.log");
    
    Ok(())
}

async fn security_violation_example() -> Result<(), ProcessingError> {
    println!("Demonstrating security violation handling...");
    
    let mut config = NeuralDocFlowConfig::default();
    config.security.enabled = true;
    config.security.policies.max_file_size_mb = 1; // Very small limit for demo
    
    let engine = DocumentEngine::new(config)?;
    
    // Try to process a file that's too large
    let oversized_content = vec![b'A'; 2 * 1024 * 1024]; // 2MB
    
    println!("Attempting to process oversized content (2MB with 1MB limit)...");
    
    match engine.process(oversized_content, "text/plain").await {
        Ok(_) => {
            println!("   ⚠️ Oversized content was unexpectedly processed");
        }
        Err(ProcessingError::SecurityViolation(msg)) => {
            println!("   ✅ Security violation correctly detected:");
            println!("      {}", msg);
            
            // Demonstrate proper error handling
            println!("   🔧 Recommended actions:");
            println!("      - Check file size before processing");
            println!("      - Increase size limit if appropriate");
            println!("      - Contact administrator for large file processing");
        }
        Err(other_error) => {
            println!("   ❌ Unexpected error type: {}", other_error);
        }
    }
    
    // Try to process a blocked file type
    println!("\nAttempting to process blocked executable content...");
    
    let mut config = NeuralDocFlowConfig::default();
    config.security.policies.blocked_file_types.push("application/x-executable".to_string());
    
    let engine = DocumentEngine::new(config)?;
    let executable_content = b"FAKE_EXECUTABLE_CONTENT";
    
    match engine.process(executable_content.to_vec(), "application/x-executable").await {
        Ok(_) => {
            println!("   ⚠️ Executable content was unexpectedly processed");
        }
        Err(ProcessingError::SecurityViolation(msg)) => {
            println!("   ✅ Blocked file type correctly rejected:");
            println!("      {}", msg);
        }
        Err(other_error) => {
            println!("   ❌ Unexpected error type: {}", other_error);
        }
    }
    
    Ok(())
}