use async_trait::async_trait;
use anyhow::{Result, anyhow};
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};
use uuid::Uuid;

use crate::{
    agents::base::{Agent, BaseAgent, AgentStatus, AgentState, CoordinationMessage},
    agents::{DaaAgent, AgentType, AgentCapabilities, MessageType},
    messaging::{Message, MessagePriority},
    resources::ResourceRequirement,
    AgentCapability,
};

/// Enhancer agent for document enhancement and enrichment
pub struct EnhancerAgent {
    base: BaseAgent,
    enhancement_strategies: Arc<RwLock<Vec<Box<dyn EnhancementStrategy>>>>,
    metadata_extractors: Arc<RwLock<Vec<Box<dyn MetadataExtractor>>>>,
}

#[async_trait]
pub trait EnhancementStrategy: Send + Sync {
    async fn enhance(&self, content: &str, metadata: &serde_json::Value) -> Result<EnhancedContent>;
    fn strategy_name(&self) -> &str;
}

#[async_trait]
pub trait MetadataExtractor: Send + Sync {
    async fn extract(&self, content: &str) -> Result<serde_json::Value>;
    fn extractor_name(&self) -> &str;
}

#[derive(Debug, Clone)]
pub struct EnhancedContent {
    pub original: String,
    pub enhanced: String,
    pub metadata: serde_json::Value,
    pub enhancements_applied: Vec<String>,
    pub quality_score: f64,
}

/// Keywords enhancement strategy
pub struct KeywordEnhancer {
    keyword_database: Arc<RwLock<std::collections::HashMap<String, Vec<String>>>>,
}

#[async_trait]
impl EnhancementStrategy for KeywordEnhancer {
    async fn enhance(&self, content: &str, _metadata: &serde_json::Value) -> Result<EnhancedContent> {
        let keywords = self.keyword_database.read().await;
        let mut enhanced = content.to_string();
        let mut enhancements = Vec::new();
        
        // Simple keyword expansion
        for (keyword, expansions) in keywords.iter() {
            if content.contains(keyword) {
                for expansion in expansions {
                    if !content.contains(expansion) {
                        enhanced.push_str(&format!("\nRelated: {}", expansion));
                        enhancements.push(format!("keyword_expansion:{}", keyword));
                    }
                }
            }
        }
        
        Ok(EnhancedContent {
            original: content.to_string(),
            enhanced,
            metadata: serde_json::json!({
                "enhancement_type": "keyword",
                "timestamp": chrono::Utc::now().to_rfc3339(),
            }),
            enhancements_applied: enhancements,
            quality_score: 0.85,
        })
    }
    
    fn strategy_name(&self) -> &str {
        "keyword_enhancer"
    }
}

/// Structure enhancement strategy
pub struct StructureEnhancer;

#[async_trait]
impl EnhancementStrategy for StructureEnhancer {
    async fn enhance(&self, content: &str, _metadata: &serde_json::Value) -> Result<EnhancedContent> {
        let mut enhanced = String::new();
        let mut enhancements = Vec::new();
        
        // Add structure to unstructured content
        let lines: Vec<&str> = content.lines().collect();
        let mut in_paragraph = false;
        
        for line in lines {
            if line.trim().is_empty() {
                if in_paragraph {
                    enhanced.push_str("</p>\n");
                    in_paragraph = false;
                }
                enhanced.push('\n');
            } else {
                if !in_paragraph {
                    enhanced.push_str("<p>");
                    in_paragraph = true;
                    enhancements.push("paragraph_structure".to_string());
                }
                enhanced.push_str(line);
                enhanced.push(' ');
            }
        }
        
        if in_paragraph {
            enhanced.push_str("</p>");
        }
        
        Ok(EnhancedContent {
            original: content.to_string(),
            enhanced,
            metadata: serde_json::json!({
                "enhancement_type": "structure",
                "paragraphs_added": enhancements.len(),
            }),
            enhancements_applied: enhancements,
            quality_score: 0.90,
        })
    }
    
    fn strategy_name(&self) -> &str {
        "structure_enhancer"
    }
}

/// Basic metadata extractor
pub struct BasicMetadataExtractor;

#[async_trait]
impl MetadataExtractor for BasicMetadataExtractor {
    async fn extract(&self, content: &str) -> Result<serde_json::Value> {
        let word_count = content.split_whitespace().count();
        let char_count = content.chars().count();
        let line_count = content.lines().count();
        
        Ok(serde_json::json!({
            "word_count": word_count,
            "char_count": char_count,
            "line_count": line_count,
            "average_word_length": if word_count > 0 { 
                char_count as f64 / word_count as f64 
            } else { 
                0.0 
            },
            "extracted_at": chrono::Utc::now().to_rfc3339(),
        }))
    }
    
    fn extractor_name(&self) -> &str {
        "basic_metadata"
    }
}

impl EnhancerAgent {
    pub fn new(capabilities: AgentCapabilities) -> Self {
        let (sender, _) = broadcast::channel(1000);
        
        Self {
            base: BaseAgent::new(
                "enhancer".to_string(),
                vec![
                    AgentCapability::ContentEnhancement,
                    AgentCapability::MetadataExtraction,
                    AgentCapability::QualityImprovement,
                ],
                sender,
            ),
            enhancement_strategies: Arc::new(RwLock::new(Vec::new())),
            metadata_extractors: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    pub fn new_with_sender(message_sender: broadcast::Sender<Message>) -> Self {
        let capabilities = vec![
            AgentCapability::ContentEnhancement,
            AgentCapability::MetadataExtraction,
            AgentCapability::QualityImprovement,
        ];
        
        Self {
            base: BaseAgent::new("enhancer".to_string(), capabilities, message_sender),
            enhancement_strategies: Arc::new(RwLock::new(Vec::new())),
            metadata_extractors: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    pub async fn add_enhancement_strategy(&self, strategy: Box<dyn EnhancementStrategy>) {
        let mut strategies = self.enhancement_strategies.write().await;
        strategies.push(strategy);
    }
    
    pub async fn add_metadata_extractor(&self, extractor: Box<dyn MetadataExtractor>) {
        let mut extractors = self.metadata_extractors.write().await;
        extractors.push(extractor);
    }
    
    async fn enhance_document(&self, doc_id: String, content: &str) -> Result<EnhancedContent> {
        self.base.increment_task().await;
        
        // Extract metadata first
        let mut all_metadata = serde_json::json!({});
        let extractors = self.metadata_extractors.read().await;
        
        for extractor in extractors.iter() {
            match extractor.extract(content).await {
                Ok(metadata) => {
                    if let serde_json::Value::Object(map) = metadata {
                        for (k, v) in map {
                            all_metadata[k] = v;
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Metadata extraction failed: {}", e);
                }
            }
        }
        
        // Apply enhancement strategies
        let strategies = self.enhancement_strategies.read().await;
        let mut final_content = content.to_string();
        let mut all_enhancements = Vec::new();
        let mut quality_scores = Vec::new();
        
        for strategy in strategies.iter() {
            match strategy.enhance(&final_content, &all_metadata).await {
                Ok(enhanced) => {
                    final_content = enhanced.enhanced.clone();
                    all_enhancements.extend(enhanced.enhancements_applied);
                    quality_scores.push(enhanced.quality_score);
                }
                Err(e) => {
                    tracing::warn!("Enhancement strategy {} failed: {}", strategy.strategy_name(), e);
                }
            }
        }
        
        let avg_quality = if quality_scores.is_empty() {
            0.0
        } else {
            quality_scores.iter().sum::<f64>() / quality_scores.len() as f64
        };
        
        self.base.complete_task().await;
        
        Ok(EnhancedContent {
            original: content.to_string(),
            enhanced: final_content,
            metadata: all_metadata,
            enhancements_applied: all_enhancements,
            quality_score: avg_quality,
        })
    }
    
    /// Extract metadata from content using registered extractors
    async fn extract_metadata(&self, content: &str) -> Result<serde_json::Value, Box<dyn std::error::Error + Send + Sync>> {
        let extractors = self.metadata_extractors.read().await;
        let mut combined_metadata = serde_json::Map::new();
        
        for extractor in extractors.iter() {
            if let Ok(metadata) = extractor.extract(content).await {
                if let Some(obj) = metadata.as_object() {
                    for (key, value) in obj {
                        combined_metadata.insert(key.clone(), value.clone());
                    }
                }
            }
        }
        
        if combined_metadata.is_empty() {
            // Default metadata if no extractors or all failed
            Ok(serde_json::json!({
                "word_count": content.split_whitespace().count(),
                "line_count": content.lines().count(),
                "timestamp": chrono::Utc::now().to_rfc3339()
            }))
        } else {
            Ok(serde_json::Value::Object(combined_metadata))
        }
    }

    /// Apply all enhancement strategies to content
    async fn apply_enhancements(&self, content: &str, metadata: &serde_json::Value) -> Result<EnhancedContent, Box<dyn std::error::Error + Send + Sync>> {
        let strategies = self.enhancement_strategies.read().await;
        let mut current_content = content.to_string();
        let mut all_enhancements = Vec::new();
        let mut quality_scores = Vec::new();
        
        for strategy in strategies.iter() {
            if let Ok(enhanced) = strategy.enhance(&current_content, metadata).await {
                current_content = enhanced.enhanced;
                all_enhancements.extend(enhanced.enhancements_applied);
                quality_scores.push(enhanced.quality_score);
            }
        }
        
        let average_quality = if quality_scores.is_empty() {
            0.5
        } else {
            quality_scores.iter().sum::<f64>() / quality_scores.len() as f64
        };
        
        Ok(EnhancedContent {
            original: content.to_string(),
            enhanced: current_content,
            metadata: metadata.clone(),
            enhancements_applied: all_enhancements,
            quality_score: average_quality,
        })
    }
}

#[async_trait]
impl DaaAgent for EnhancerAgent {
    fn id(&self) -> Uuid {
        self.base.id
    }
    
    fn agent_type(&self) -> AgentType {
        AgentType::Enhancer
    }
    
    fn state(&self) -> super::AgentState {
        // Since BaseAgent state is Arc<RwLock<AgentState>>, we need to clone a default for now
        // This is a simplified implementation for compilation
        super::AgentState::Ready
    }
    
    fn capabilities(&self) -> AgentCapabilities {
        // Convert Vec<AgentCapability> to AgentCapabilities
        AgentCapabilities {
            neural_processing: self.base.capabilities.contains(&AgentCapability::ContentEnhancement),
            text_enhancement: self.base.capabilities.contains(&AgentCapability::QualityImprovement),
            layout_analysis: false,
            quality_assessment: self.base.capabilities.contains(&AgentCapability::QualityImprovement),
            coordination: true,
            fault_tolerance: false,
        }
    }
    
    async fn initialize(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Set state through BaseAgent's async method
        self.base.set_state(crate::agents::base::AgentState::Ready).await;
        Ok(())
    }
    
    async fn process(&mut self, input: Vec<u8>) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
        self.base.set_state(crate::agents::base::AgentState::Processing).await;
        
        // Convert input to string for processing
        let input_text = String::from_utf8_lossy(&input);
        
        // Extract metadata
        let metadata = self.extract_metadata(&input_text).await?;
        
        // Apply enhancements
        let enhanced_content = self.apply_enhancements(&input_text, &metadata).await?;
        
        self.base.set_state(crate::agents::base::AgentState::Ready).await;
        
        // Return enhanced content as bytes
        Ok(enhanced_content.enhanced.into_bytes())
    }
    
    async fn coordinate(&mut self, message: super::CoordinationMessage) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        match message.message_type {
            super::MessageType::Task => {
                // Handle task assignment
                let result = self.process(message.payload).await?;
                // In a real implementation, would send result back
                Ok(())
            }
            super::MessageType::Status => {
                // Handle status updates
                Ok(())
            }
            super::MessageType::Result => {
                // Handle results from other agents
                Ok(())
            }
            _ => Ok(())
        }
    }
    
    async fn shutdown(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.base.set_state(crate::agents::base::AgentState::Ready).await; // Use Ready since there's no Completed state
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_enhancer_agent_creation() {
        let (tx, _rx) = broadcast::channel(100);
        let agent = EnhancerAgent::new(tx);
        
        assert_eq!(agent.base.agent_type, "Enhancer");
        assert!(agent.base.capabilities.contains(&AgentCapability::TextAnalysis));
        assert!(agent.base.capabilities.contains(&AgentCapability::QualityControl));
    }
}