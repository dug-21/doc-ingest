use anyhow::{Result, anyhow};
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock, Mutex};
use uuid::Uuid;
use regex::Regex;

use super::{DaaAgent, AgentState, AgentCapabilities, AgentType, CoordinationMessage, MessageType};
use crate::{
    agents::base::{Agent, BaseAgent, AgentStatus},
    messaging::{Message, MessagePriority},
    resources::ResourceRequirement,
    AgentCapability,
};

/// Validator agent for document validation
pub struct ValidatorAgent {
    base: BaseAgent,
    validation_rules: Arc<RwLock<ValidationRules>>,
    validation_cache: Arc<Mutex<lru::LruCache<String, ValidationResult>>>,
}

#[derive(Clone)]
pub struct ValidationRules {
    pub format_rules: Vec<FormatRule>,
    pub content_rules: Vec<ContentRule>,
    pub structural_rules: Vec<StructuralRule>,
}

#[derive(Clone)]
pub struct FormatRule {
    pub name: String,
    pub pattern: Regex,
    pub severity: ValidationSeverity,
}

#[derive(Clone)]
pub struct ContentRule {
    pub name: String,
    pub validator: Arc<dyn ContentValidator>,
}

#[derive(Clone)]
pub struct StructuralRule {
    pub name: String,
    pub validator: Arc<dyn StructuralValidator>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationSeverity {
    Error,
    Warning,
    Info,
}

#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct ValidationError {
    pub rule: String,
    pub message: String,
    pub location: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ValidationWarning {
    pub rule: String,
    pub message: String,
    pub location: Option<String>,
}

pub trait ContentValidator: Send + Sync {
    fn validate(&self, content: &str) -> Result<ValidationResult>;
}

pub trait StructuralValidator: Send + Sync {
    fn validate(&self, structure: &serde_json::Value) -> Result<ValidationResult>;
}

impl ValidatorAgent {
    pub fn new(message_sender: broadcast::Sender<Message>) -> Self {
        let capabilities = vec![
            AgentCapability::ValidationExpert,
            AgentCapability::PatternRecognition,
            AgentCapability::ErrorDetection,
        ];
        
        Self {
            base: BaseAgent::new("validator".to_string(), capabilities, message_sender),
            validation_rules: Arc::new(RwLock::new(ValidationRules {
                format_rules: vec![],
                content_rules: vec![],
                structural_rules: vec![],
            })),
            validation_cache: Arc::new(Mutex::new(lru::LruCache::new(
                std::num::NonZeroUsize::new(1000).unwrap()
            ))),
        }
    }
    
    pub async fn add_format_rule(&self, rule: FormatRule) {
        let mut rules = self.validation_rules.write().await;
        rules.format_rules.push(rule);
    }
    
    pub async fn add_content_rule(&self, rule: ContentRule) {
        let mut rules = self.validation_rules.write().await;
        rules.content_rules.push(rule);
    }
    
    pub async fn add_structural_rule(&self, rule: StructuralRule) {
        let mut rules = self.validation_rules.write().await;
        rules.structural_rules.push(rule);
    }
    
    async fn validate_document(&self, doc_id: String, content: &str) -> Result<ValidationResult> {
        // Check cache first
        {
            let mut cache = self.validation_cache.lock().await;
            if let Some(cached_result) = cache.get(&doc_id) {
                return Ok(cached_result.clone());
            }
        }
        
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let rules = self.validation_rules.read().await;
        
        // Apply format rules
        for rule in &rules.format_rules {
            if !rule.pattern.is_match(content) {
                match rule.severity {
                    ValidationSeverity::Error => {
                        errors.push(ValidationError {
                            rule: rule.name.clone(),
                            message: format!("Format validation failed for rule: {}", rule.name),
                            location: None,
                        });
                    }
                    ValidationSeverity::Warning => {
                        warnings.push(ValidationWarning {
                            rule: rule.name.clone(),
                            message: format!("Format validation warning for rule: {}", rule.name),
                            location: None,
                        });
                    }
                    _ => {}
                }
            }
        }
        
        // Apply content rules
        for rule in &rules.content_rules {
            match rule.validator.validate(content) {
                Ok(result) => {
                    errors.extend(result.errors);
                    warnings.extend(result.warnings);
                }
                Err(e) => {
                    errors.push(ValidationError {
                        rule: rule.name.clone(),
                        message: format!("Content validation error: {}", e),
                        location: None,
                    });
                }
            }
        }
        
        let result = ValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
            metadata: serde_json::json!({
                "validated_at": chrono::Utc::now().to_rfc3339(),
                "agent_id": self.base.id.to_string(),
            }),
        };
        
        // Cache the result
        {
            let mut cache = self.validation_cache.lock().await;
            cache.put(doc_id, result.clone());
        }
        
        Ok(result)
    }
}

// TODO: Implement DaaAgent trait for ValidatorAgent
/*
#[async_trait]
#[async_trait::async_trait]
impl DaaAgent for ValidatorAgent {
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
        
        // Initialize default validation rules
        self.add_format_rule(FormatRule {
            name: "non_empty".to_string(),
            pattern: Regex::new(r".+").unwrap(),
            severity: ValidationSeverity::Error,
        }).await;
        
        Ok(())
    }
    
    async fn process_message(&self, message: Message) -> Result<()> {
        self.base.increment_task().await;
        self.base.set_state(AgentState::Processing).await;
        
        let result = match &message.content {
            serde_json::Value::Object(obj) => {
                if let Some(serde_json::Value::String(action)) = obj.get("action") {
                    match action.as_str() {
                        "validate" => {
                            if let (Some(serde_json::Value::String(doc_id)), 
                                    Some(serde_json::Value::String(content))) = 
                                (obj.get("doc_id"), obj.get("content")) {
                                self.validate_document(doc_id.clone(), content).await
                            } else {
                                Err(anyhow!("Invalid validate message format"))
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
            Ok(validation_result) => {
                // Send validation result back
                let response = Message {
                    id: Uuid::new_v4(),
                    from: self.base.id,
                    to: message.from,
                    priority: MessagePriority::Normal,
                    content: serde_json::to_value(&validation_result)?,
                    timestamp: chrono::Utc::now(),
                    correlation_id: Some(message.id),
                };
                
                let _ = self.base.message_sender.send(response);
                self.base.complete_task().await;
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
            ResourceRequirement::Memory(512), // 512MB
            ResourceRequirement::CPU(0.5),    // 50% of one core
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