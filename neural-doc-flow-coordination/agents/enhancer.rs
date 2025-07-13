use async_trait::async_trait;
use anyhow::{Result, anyhow};
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};
use uuid::Uuid;

use crate::{
    agents::base::{Agent, BaseAgent, AgentStatus, AgentState, CoordinationMessage},
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
    pub fn new(message_sender: broadcast::Sender<Message>) -> Self {
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
}

// TODO: Implement DaaAgent trait for EnhancerAgent
/*
impl Agent for EnhancerAgent {
    fn id(&self) -> Uuid {
        self.base.id
    }
    
    fn agent_type(&self) -> &str {
        &self.base.agent_type
    }
    
    fn capabilities(&self) -> Vec<AgentCapability> {
        self.base.capabilities.clone()
    }
    
    async fn initialize(&mut self) -> Result<()> {
        self.base.set_state(AgentState::Ready).await;
        
        // Add default enhancement strategies
        self.add_enhancement_strategy(Box::new(StructureEnhancer)).await;
        self.add_enhancement_strategy(Box::new(KeywordEnhancer {
            keyword_database: Arc::new(RwLock::new(std::collections::HashMap::new())),
        })).await;
        
        // Add default metadata extractors
        self.add_metadata_extractor(Box::new(BasicMetadataExtractor)).await;
        
        Ok(())
    }
    
    async fn process_message(&self, message: Message) -> Result<()> {
        self.base.set_state(AgentState::Processing).await;
        
        let result = match &message.content {
            serde_json::Value::Object(obj) => {
                if let Some(serde_json::Value::String(action)) = obj.get("action") {
                    match action.as_str() {
                        "enhance" => {
                            if let (Some(serde_json::Value::String(doc_id)), 
                                    Some(serde_json::Value::String(content))) = 
                                (obj.get("doc_id"), obj.get("content")) {
                                self.enhance_document(doc_id.clone(), content).await
                            } else {
                                Err(anyhow!("Invalid enhance message format"))
                            }
                        }
                        _ => Err(anyhow!("Unknown action: {}", action))
                    }
                } else {
                    Err(anyhow!("No action specified in message"))
                }
            }
            _ => Err(anyhow!("Invalid message format"))
        };
        
        match result {
            Ok(enhanced_content) => {
                // Send enhancement result back
                let response = Message {
                    id: Uuid::new_v4(),
                    from: self.base.id,
                    to: message.from,
                    priority: MessagePriority::Normal,
                    content: serde_json::to_value(&enhanced_content)?,
                    timestamp: chrono::Utc::now(),
                    correlation_id: Some(message.id),
                };
                
                let _ = self.base.message_sender.send(response);
            }
            Err(e) => {
                self.base.record_error().await;
                return Err(e);
            }
        }
        
        self.base.set_state(AgentState::Ready).await;
        Ok(())
    }
    
    async fn status(&self) -> AgentStatus {
        let state = *self.base.state.read().await;
        let counter = self.base.task_counter.read().await;
        
        AgentStatus {
            id: self.base.id,
            agent_type: self.base.agent_type.clone(),
            state,
            capabilities: self.base.capabilities.clone(),
            current_tasks: counter.current,
            completed_tasks: counter.completed,
            error_count: counter.errors,
            last_heartbeat: chrono::Utc::now(),
        }
    }
    
    async fn shutdown(&mut self) -> Result<()> {
        self.base.set_state(AgentState::Shutting_down).await;
        Ok(())
    }
    
    fn resource_requirements(&self) -> Vec<ResourceRequirement> {
        vec![
            ResourceRequirement::Memory(1024), // 1GB for enhancement operations
            ResourceRequirement::CPU(1.0),     // 1 full core
        ]
    }
    
    async fn handle_coordination(&self, message: CoordinationMessage) -> Result<()> {
        match message {
            CoordinationMessage::StatusRequest => {
                // Status is handled by the status() method
                Ok(())
            }
            CoordinationMessage::Shutdown => {
                self.base.set_state(AgentState::Shutting_down).await;
                Ok(())
            }
            _ => Ok(())
        }
    }
}
*/