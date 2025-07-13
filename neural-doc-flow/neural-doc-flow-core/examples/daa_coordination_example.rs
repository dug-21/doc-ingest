//! Example demonstrating DAA coordination for document processing
//! 
//! This example shows how to use the neural-doc-flow-core library with DAA
//! coordination to process documents in parallel using distributed agents.
//! 
//! Run with: cargo run --example daa_coordination_example

use std::path::PathBuf;
use uuid::Uuid;
use neural_doc_flow_core::{
    coordination::{DocumentCoordinator, CoordinationConfig, BatchConfig, AgentType},
    types::{Document, DocumentSource, DocumentMetadata},
    pipeline::{DocFlow, PipelineConfig},
    traits::{DocumentSource as SourceTrait, ValidationResult},
    error::Result,
};

/// Example PDF source plugin (simplified implementation)
struct PdfSource;

#[async_trait::async_trait]
impl SourceTrait for PdfSource {
    async fn extract(&self, document: &Document) -> Result<neural_doc_flow_core::types::ExtractedContent> {
        println!("🔍 Extracting content from PDF: {}", document.id);
        
        // Simulate PDF extraction
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        let extracted_content = neural_doc_flow_core::types::ExtractedContent {
            blocks: vec![
                neural_doc_flow_core::types::ContentBlock::new(
                    neural_doc_flow_core::types::BlockType::Text,
                    neural_doc_flow_core::types::BlockContent::Text {
                        text: "Sample extracted text from PDF".to_string(),
                    },
                ).with_confidence(0.95),
            ],
            metadata: document.metadata.clone(),
            confidence: neural_doc_flow_core::types::Confidence {
                overall: 0.92,
                text_extraction: 0.95,
                structure_detection: 0.88,
                table_extraction: 0.0,
                metadata_extraction: 0.94,
            },
            extracted_at: chrono::Utc::now(),
        };
        
        println!("✅ PDF extraction complete with {:.1}% confidence", 
                extracted_content.confidence.overall * 100.0);
        
        Ok(extracted_content)
    }
    
    fn can_handle(&self, document: &Document) -> bool {
        match &document.source {
            DocumentSource::File { path } => {
                path.extension().and_then(|ext| ext.to_str()) == Some("pdf")
            },
            _ => false,
        }
    }
    
    fn supported_mime_types(&self) -> Vec<&'static str> {
        vec!["application/pdf"]
    }
    
    fn name(&self) -> &'static str {
        "pdf_source"
    }
    
    fn version(&self) -> &'static str {
        "1.0.0"
    }
}

/// Example validator plugin
struct QualityValidator;

#[async_trait::async_trait]
impl neural_doc_flow_core::traits::ContentValidator for QualityValidator {
    async fn validate(&self, content: &neural_doc_flow_core::types::ExtractedContent) -> Result<ValidationResult> {
        println!("🔍 Validating content quality...");
        
        // Simulate validation
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        
        let is_valid = content.confidence.overall > 0.8;
        let confidence_score = content.confidence.overall;
        
        println!("✅ Validation complete: {} (confidence: {:.1}%)", 
                if is_valid { "VALID" } else { "INVALID" },
                confidence_score * 100.0);
        
        Ok(ValidationResult {
            is_valid,
            confidence_score,
            errors: vec![],
            warnings: vec![],
            suggestions: vec![],
        })
    }
    
    fn name(&self) -> &'static str {
        "quality_validator"
    }
    
    fn validation_rules(&self) -> Vec<neural_doc_flow_core::traits::ValidationRule> {
        vec![
            neural_doc_flow_core::traits::ValidationRule {
                rule_id: "min_confidence".to_string(),
                description: "Minimum confidence threshold".to_string(),
                severity: neural_doc_flow_core::traits::ValidationSeverity::Medium,
                rule_type: neural_doc_flow_core::traits::ValidationRuleType::Accuracy,
            }
        ]
    }
    
    fn can_validate(&self, _content: &neural_doc_flow_core::types::ExtractedContent) -> bool {
        true
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize the neural doc flow system
    neural_doc_flow_core::init();
    
    println!("🚀 Starting Neural Doc Flow with DAA Coordination Example");
    println!("=" .repeat(60));
    
    // Step 1: Create DAA coordinator with custom configuration
    println!("\n📋 Step 1: Initializing DAA Coordinator");
    let coordination_config = CoordinationConfig {
        topology: ruv_swarm_daa::TopologyType::Hierarchical,
        max_agents: 6,
        neural_config: Some(neural_doc_flow_core::coordination::NeuralCoordinationConfig {
            enable_neural_coordination: true,
            learning_rate: 0.001,
            shared_model_state: true,
            enable_simd: true,
        }),
        ..Default::default()
    };
    
    let coordinator = DocumentCoordinator::with_config(coordination_config).await?;
    println!("✅ DocumentCoordinator initialized: {}", coordinator.id);
    
    // Step 2: Create processing pipeline with coordinator
    println!("\n🔧 Step 2: Setting up Processing Pipeline");
    let pipeline_config = PipelineConfig {
        enable_parallel_processing: true,
        enable_neural_enhancement: true,
        enable_validation: true,
        max_concurrent_documents: 5,
        ..Default::default()
    };
    
    let mut docflow = DocFlow::with_coordinator(std::sync::Arc::new(coordinator))?;
    
    // Register plugins
    docflow.register_source(Box::new(PdfSource));
    docflow.register_validator(Box::new(QualityValidator));
    
    // Initialize pipeline
    docflow.initialize().await?;
    println!("✅ Pipeline initialized with DAA coordination");
    
    // Step 3: Create sample documents for processing
    println!("\n📄 Step 3: Creating Sample Documents");
    let documents = vec![
        Document::new(DocumentSource::File { 
            path: PathBuf::from("sample1.pdf") 
        }).with_metadata(DocumentMetadata {
            title: Some("Financial Report Q1".to_string()),
            author: Some("Finance Team".to_string()),
            mime_type: Some("application/pdf".to_string()),
            ..Default::default()
        }),
        Document::new(DocumentSource::File { 
            path: PathBuf::from("sample2.pdf") 
        }).with_metadata(DocumentMetadata {
            title: Some("Technical Documentation".to_string()),
            author: Some("Engineering Team".to_string()),
            mime_type: Some("application/pdf".to_string()),
            ..Default::default()
        }),
        Document::new(DocumentSource::File { 
            path: PathBuf::from("sample3.pdf") 
        }).with_metadata(DocumentMetadata {
            title: Some("Legal Contract".to_string()),
            author: Some("Legal Department".to_string()),
            mime_type: Some("application/pdf".to_string()),
            ..Default::default()
        }),
    ];
    
    println!("✅ Created {} documents for processing", documents.len());
    for (i, doc) in documents.iter().enumerate() {
        println!("   {}. {} ({})", i + 1, 
                doc.metadata.title.as_ref().unwrap_or(&"Untitled".to_string()),
                doc.id);
    }
    
    // Step 4: Process documents with DAA coordination
    println!("\n⚡ Step 4: Processing Documents with DAA Coordination");
    println!("Using parallel processing with distributed agents...");
    
    let start_time = std::time::Instant::now();
    let results = docflow.process_batch(documents).await?;
    let processing_time = start_time.elapsed();
    
    // Step 5: Display results
    println!("\n📊 Step 5: Processing Results");
    println!("=" .repeat(40));
    println!("📋 Total documents processed: {}", results.len());
    println!("⏱️  Total processing time: {:.2}s", processing_time.as_secs_f32());
    println!("🚀 Average time per document: {:.2}ms", 
            processing_time.as_millis() as f32 / results.len() as f32);
    
    println!("\n📄 Individual Results:");
    for (i, result) in results.iter().enumerate() {
        println!("\n   Document {} (ID: {})", i + 1, result.document_id);
        println!("   ├─ Processing time: {}ms", result.processing_time_ms);
        println!("   ├─ Overall confidence: {:.1}%", 
                result.extracted_content.confidence.overall * 100.0);
        println!("   ├─ Content blocks: {}", result.extracted_content.blocks.len());
        
        if let Some(agent_id) = result.agent_id {
            println!("   ├─ Processed by agent: {}", agent_id);
        }
        
        if !result.neural_enhancements.is_empty() {
            println!("   ├─ Neural enhancements: {}", result.neural_enhancements.len());
        }
        
        if result.errors.is_empty() {
            println!("   └─ Status: ✅ SUCCESS");
        } else {
            println!("   ├─ Errors: {}", result.errors.len());
            println!("   └─ Status: ❌ WITH ERRORS");
        }
    }
    
    // Step 6: Show coordination statistics
    println!("\n🤖 Step 6: DAA Coordination Statistics");
    let agent_status = docflow.get_stats().await;
    println!("📊 Pipeline Statistics:");
    println!("   ├─ Registered sources: {}", agent_status.registered_sources);
    println!("   ├─ Registered processors: {}", agent_status.registered_processors);
    println!("   ├─ Registered enhancers: {}", agent_status.registered_enhancers);
    println!("   ├─ Registered validators: {}", agent_status.registered_validators);
    println!("   ├─ Active agents: {}", agent_status.active_agents);
    println!("   └─ Coordinator ID: {}", agent_status.coordinator_id);
    
    // Step 7: Performance analysis
    println!("\n📈 Step 7: Performance Analysis");
    let total_confidence = results.iter()
        .map(|r| r.extracted_content.confidence.overall)
        .sum::<f32>() / results.len() as f32;
    
    let total_blocks = results.iter()
        .map(|r| r.extracted_content.blocks.len())
        .sum::<usize>();
    
    println!("✨ Performance Metrics:");
    println!("   ├─ Average confidence: {:.1}%", total_confidence * 100.0);
    println!("   ├─ Total content blocks extracted: {}", total_blocks);
    println!("   ├─ Processing rate: {:.1} docs/second", 
            results.len() as f32 / processing_time.as_secs_f32());
    println!("   └─ Success rate: {:.1}%", 
            results.iter().filter(|r| r.errors.is_empty()).count() as f32 / results.len() as f32 * 100.0);
    
    // Step 8: Cleanup
    println!("\n🧹 Step 8: Cleanup");
    docflow.shutdown().await?;
    println!("✅ Pipeline shutdown complete");
    
    println!("\n🎉 DAA Coordination Example Complete!");
    println!("   This example demonstrated:");
    println!("   • DAA coordinator initialization");
    println!("   • Parallel document processing");
    println!("   • Plugin registration and usage");
    println!("   • Coordination statistics");
    println!("   • Performance monitoring");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_daa_coordination_basic() {
        let coordinator = DocumentCoordinator::new().await.unwrap();
        let docflow = DocFlow::with_coordinator(std::sync::Arc::new(coordinator)).unwrap();
        
        let stats = docflow.get_stats().await;
        assert_eq!(stats.registered_sources, 0);
        assert_eq!(stats.active_agents, 0);
    }
    
    #[test]
    fn test_pdf_source_can_handle() {
        let source = PdfSource;
        let document = Document::new(DocumentSource::File { 
            path: PathBuf::from("test.pdf") 
        });
        
        assert!(source.can_handle(&document));
        
        let document2 = Document::new(DocumentSource::File { 
            path: PathBuf::from("test.txt") 
        });
        
        assert!(!source.can_handle(&document2));
    }
}