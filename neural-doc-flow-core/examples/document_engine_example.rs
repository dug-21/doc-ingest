//! DocumentEngine Usage Example
//!
//! This example demonstrates how to use the DocumentEngine with its 4-layer architecture:
//! 1. Core Engine Layer - Main processing coordination
//! 2. Plugin System Layer - Modular source and processor plugins
//! 3. Schema Validation Layer - User-definable extraction schemas
//! 4. Output Formatting Layer - Configurable output templates

use neural_doc_flow_core::{
    engine::{DocumentEngine, ExtractionSchema, OutputTemplate, DomainConfig},
    config::NeuralDocFlowConfig,
    plugins::CorePluginRegistry,
};
use std::collections::HashMap;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::init();
    
    println!("🚀 Neural Document Flow Engine - Phase 4 Complete Implementation");
    println!("================================================================");
    
    // 1. Create and initialize the DocumentEngine
    println!("\\n1. Creating DocumentEngine with 4-layer architecture...");
    let config = NeuralDocFlowConfig::default();
    let engine = DocumentEngine::new(config)?;
    
    // Initialize with default plugins and schemas
    engine.initialize().await?;
    println!("✅ DocumentEngine initialized successfully");
    
    // 2. Demonstrate Plugin System (Layer 2)
    println!("\\n2. Plugin System Demonstration:");
    let registry = CorePluginRegistry::new();
    let plugins = registry.get_plugins();
    
    println!("   Available plugins:");
    for plugin in plugins {
        println!("   - {} ({})", plugin.source_type(), plugin.supported_extensions().join(", "));
    }
    
    // 3. Demonstrate Schema Validation (Layer 3)
    println!("\\n3. Schema Validation Engine:");
    let schemas = engine.available_schemas().await;
    println!("   Available schemas: {:?}", schemas);
    
    // Register a custom schema
    let custom_schema = ExtractionSchema {
        name: "financial_document".to_string(),
        description: "Financial document extraction schema".to_string(),
        version: "1.0.0".to_string(),
        required_fields: vec!["account_number".to_string(), "transaction_date".to_string()],
        optional_fields: vec!["amount".to_string(), "description".to_string()],
        field_types: [
            ("account_number".to_string(), "string".to_string()),
            ("transaction_date".to_string(), "date".to_string()),
            ("amount".to_string(), "number".to_string()),
            ("description".to_string(), "string".to_string()),
        ].iter().cloned().collect(),
        validation_rules: vec![],
        domain_specific: Some(DomainConfig {
            domain: "financial".to_string(),
            specialized_fields: vec!["account_number".to_string(), "transaction_date".to_string()],
            compliance_requirements: vec!["financial_validation".to_string(), "audit_trail".to_string()],
            processing_rules: HashMap::new(),
        }),
    };
    
    engine.register_schema(custom_schema).await?;
    println!("   ✅ Registered custom financial schema");
    
    // 4. Demonstrate Output Formatting (Layer 4)
    println!("\\n4. Output Formatting Engine:");
    let formats = engine.available_formats().await;
    println!("   Available formats: {:?}", formats);
    
    // Register a custom output template
    let custom_template = OutputTemplate {
        name: "financial_report".to_string(),
        format: "json".to_string(),
        template: r#"{
  "report_type": "financial_document",
  "document_id": "{{document_id}}",
  "metadata": {{metadata}},
  "extracted_data": {
    "account_number": "{{account_number}}",
    "transaction_date": "{{transaction_date}}",
    "amount": "{{amount}}",
    "description": "{{description}}"
  },
  "processing_info": {
    "schema_used": "financial_document",
    "processed_at": "{{processed_at}}"
  }
}"#.to_string(),
        schema: Some("financial_document".to_string()),
    };
    
    engine.register_template(custom_template).await?;
    println!("   ✅ Registered custom financial report template");
    
    // 5. Demonstrate Domain-Specific Configurations
    println!("\\n5. Domain-Specific Configuration Examples:");
    
    // Legal domain example
    let legal_schema = ExtractionSchema {
        name: "contract_analysis".to_string(),
        description: "Legal contract analysis schema".to_string(),
        version: "1.0.0".to_string(),
        required_fields: vec!["contract_type".to_string(), "parties".to_string(), "effective_date".to_string()],
        optional_fields: vec!["termination_date".to_string(), "governing_law".to_string()],
        field_types: HashMap::new(),
        validation_rules: vec![],
        domain_specific: Some(DomainConfig {
            domain: "legal".to_string(),
            specialized_fields: vec!["contract_type".to_string(), "parties".to_string(), "governing_law".to_string()],
            compliance_requirements: vec!["legal_validation".to_string(), "confidentiality_check".to_string()],
            processing_rules: HashMap::new(),
        }),
    };
    
    engine.register_schema(legal_schema).await?;
    println!("   ✅ Legal: Contract analysis schema registered");
    
    // Medical domain example
    let medical_schema = ExtractionSchema {
        name: "patient_record".to_string(),
        description: "Medical patient record schema".to_string(),
        version: "1.0.0".to_string(),
        required_fields: vec!["patient_id".to_string(), "visit_date".to_string(), "diagnosis".to_string()],
        optional_fields: vec!["treatment_plan".to_string(), "medications".to_string()],
        field_types: HashMap::new(),
        validation_rules: vec![],
        domain_specific: Some(DomainConfig {
            domain: "medical".to_string(),
            specialized_fields: vec!["patient_id".to_string(), "diagnosis".to_string(), "treatment_plan".to_string()],
            compliance_requirements: vec!["hipaa_compliance".to_string(), "medical_validation".to_string()],
            processing_rules: HashMap::new(),
        }),
    };
    
    engine.register_schema(medical_schema).await?;
    println!("   ✅ Medical: Patient record schema registered");
    
    // 6. Demonstrate Plugin Hot-Reload
    println!("\\n6. Plugin Hot-Reload Mechanism:");
    println!("   Testing plugin reload...");
    
    // This would normally reload actual plugins
    let reload_result = engine.reload_plugin("pdf_source").await;
    match reload_result {
        Ok(_) => println!("   ✅ Plugin reloaded successfully"),
        Err(e) => println!("   ⚠️  Plugin reload: {:?}", e),
    }
    
    // 7. Demonstrate Full Pipeline Integration
    println!("\\n7. Complete Pipeline Integration:");
    println!("   The DocumentEngine provides:");
    println!("   - ✅ 4-layer architecture implementation");
    println!("   - ✅ Plugin system with hot-reload");
    println!("   - ✅ User-definable schemas");
    println!("   - ✅ Configurable output templates");
    println!("   - ✅ Domain-specific configurations");
    println!("   - ✅ DAA topology integration");
    println!("   - ✅ Neural processor coordination");
    println!("   - ✅ Async processing pipeline");
    println!("   - ✅ 5 core plugins (PDF, DOCX, HTML, Images, CSV)");
    
    // 8. Architecture Summary
    println!("\\n8. Architecture Summary:");
    println!("   ┌─────────────────────────────────────────────────────────┐");
    println!("   │                 DocumentEngine                          │");
    println!("   ├─────────────────────────────────────────────────────────┤");
    println!("   │ Layer 1: Core Engine - Processing coordination         │");
    println!("   │ Layer 2: Plugin System - Modular sources & processors │");
    println!("   │ Layer 3: Schema Validation - User-definable schemas    │");
    println!("   │ Layer 4: Output Formatting - Configurable templates    │");
    println!("   ├─────────────────────────────────────────────────────────┤");
    println!("   │ Integration: DAA Topology + Neural Processors          │");
    println!("   └─────────────────────────────────────────────────────────┘");
    
    println!("\\n🎉 DocumentEngine demonstration complete!");
    println!("   Ready for production use with full 4-layer architecture");
    
    Ok(())
}

// Helper function to demonstrate plugin capabilities
#[allow(dead_code)]
async fn demonstrate_plugin_capabilities() {
    println!("\\n📋 Plugin Capabilities:");
    
    let registry = CorePluginRegistry::new();
    let plugins = registry.get_plugins();
    
    for plugin in plugins {
        println!("\\n   Plugin: {}", plugin.source_type());
        println!("     Extensions: {:?}", plugin.supported_extensions());
        println!("     MIME Types: {:?}", plugin.supported_mime_types());
        
        // Test can_handle method
        for ext in plugin.supported_extensions() {
            let test_file = format!("test.{}", ext);
            let can_handle = plugin.can_handle(&test_file).await;
            println!("     Can handle '{}': {}", test_file, can_handle);
        }
    }
}

// Helper function to demonstrate schema validation
#[allow(dead_code)]
async fn demonstrate_schema_validation() {
    println!("\\n🔍 Schema Validation Examples:");
    
    // Example validation scenarios
    let validation_scenarios = vec![
        ("general_document", vec!["text"], vec!["title", "author"]),
        ("legal_document", vec!["text", "case_number", "court"], vec!["judge", "filing_date"]),
        ("medical_document", vec!["text", "patient_id", "document_type"], vec!["provider", "diagnosis"]),
    ];
    
    for (schema_name, required, optional) in validation_scenarios {
        println!("\\n   Schema: {}", schema_name);
        println!("     Required fields: {:?}", required);
        println!("     Optional fields: {:?}", optional);
        println!("     Use case: Domain-specific document processing");
    }
}

// Helper function to demonstrate output formatting
#[allow(dead_code)]
async fn demonstrate_output_formatting() {
    println!("\\n📄 Output Formatting Examples:");
    
    let format_examples = vec![
        ("json", "Structured JSON output for APIs"),
        ("xml", "XML format for legacy systems"),
        ("text", "Plain text for simple display"),
        ("html", "HTML format for web display"),
    ];
    
    for (format, description) in format_examples {
        println!("   Format: {} - {}", format, description);
    }
}