//! DAA Integration for neural processors
//!
//! This module provides integration with the Decentralized Autonomous Agents
//! system when the DAA coordination feature is enabled.

#[cfg(feature = "daa-coordination")]
use neural_doc_flow_coordination as coordination;

use crate::{
    config::NeuralConfig,
    error::{NeuralError, Result},
    types::{EnhancedContent, ContentBlock},
};
use neural_doc_flow_core::Document;

// Placeholder types when DAA coordination is not available
#[cfg(not(feature = "daa-coordination"))]
#[derive(Debug, Clone)]
pub struct EnhancerAgent {
    pub id: String,
}

#[cfg(not(feature = "daa-coordination"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancementRequest {
    pub content: ContentBlock,
    pub enhancement_type: String,
    pub parameters: HashMap<String, serde_json::Value>,
}

#[cfg(not(feature = "daa-coordination"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancementResponse {
    pub enhanced_content: ContentBlock,
    pub confidence: f32,
    pub metadata: HashMap<String, String>,
}

/// DAA-enhanced neural processor
#[derive(Debug)]
pub struct DaaEnhancedNeuralProcessor {
    config: NeuralConfig,
    #[cfg(feature = "daa-coordination")]
    coordinator: coordination::DaaCoordinator,
}

impl DaaEnhancedNeuralProcessor {
    /// Create new DAA-enhanced neural processor
    pub async fn new(config: NeuralConfig) -> Result<Self> {
        #[cfg(feature = "daa-coordination")]
        {
            let coordinator = coordination::DaaCoordinator::new(config.clone()).await
                .map_err(|e| NeuralError::Configuration(format!("DAA coordination setup failed: {}", e)))?;
            
            Ok(Self {
                config,
                coordinator,
            })
        }
        
        #[cfg(not(feature = "daa-coordination"))]
        {
            Ok(Self {
                config,
            })
        }
    }
    
    /// Process document with DAA coordination
    pub async fn process_document(&self, document: Document) -> Result<EnhancedContent> {
        #[cfg(feature = "daa-coordination")]
        {
            // Use DAA coordination for processing
            self.coordinator.process_document(document).await
                .map_err(|e| NeuralError::ProcessingFailed(format!("DAA processing failed: {}", e)))
        }
        
        #[cfg(not(feature = "daa-coordination"))]
        {
            // Fallback to basic processing
            let content_block = ContentBlock::new("document")
                .with_metadata("source", "daa_fallback")
                .with_confidence(0.8);
            
            Ok(EnhancedContent::new(vec![content_block]))
        }
    }
    
    /// Shutdown DAA coordinator
    pub async fn shutdown(&self) -> Result<()> {
        #[cfg(feature = "daa-coordination")]
        {
            self.coordinator.shutdown().await
                .map_err(|e| NeuralError::Configuration(format!("DAA shutdown failed: {}", e)))
        }
        
        #[cfg(not(feature = "daa-coordination"))]
        {
            Ok(())
        }
    }
}