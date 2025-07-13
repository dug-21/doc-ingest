//! Simple test to validate the security module builds

use crate::{SecurityProcessor, ThreatLevel};
use neural_doc_flow_core::DocumentBuilder;

#[tokio::test]
async fn test_security_processor_basic() {
    // Create a test document
    let document = DocumentBuilder::new()
        .title("Test Document")
        .source("test.pdf")
        .mime_type("application/pdf")
        .text_content("This is a test document with some content.")
        .build();

    // Create security processor
    let processor = SecurityProcessor::new();
    assert!(processor.is_ok());
    
    let mut processor = processor.unwrap();
    
    // Test scan
    let result = processor.scan(&document).await;
    assert!(result.is_ok());
    
    let analysis = result.unwrap();
    assert_eq!(analysis.threat_level, ThreatLevel::Low);
}

#[test]
fn test_security_types() {
    use crate::{ThreatCategory, SecurityAction, BehavioralRisk};
    
    // Test enum serialization works
    let category = ThreatCategory::JavaScriptExploit;
    let action = SecurityAction::Allow;
    let risk = BehavioralRisk {
        risk_type: "test".to_string(),
        severity: 0.5,
        description: "test risk".to_string(),
    };
    
    assert_eq!(format!("{:?}", category), "JavaScriptExploit");
    assert_eq!(format!("{:?}", action), "Allow");
    assert_eq!(risk.severity, 0.5);
}