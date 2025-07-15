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

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ValidationSeverity {
    Error,
    Warning,
    Info,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
    pub suggestions: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ValidationError {
    pub rule_name: String,
    pub message: String,
    pub severity: ValidationSeverity,
    pub line: Option<usize>,
    pub column: Option<usize>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ValidationWarning {
    pub rule_name: String,
    pub message: String,
    pub line: Option<usize>,
    pub column: Option<usize>,
}

pub trait ContentValidator: Send + Sync {
    fn validate(&self, content: &str) -> Result<ValidationResult>;
}

pub trait StructuralValidator: Send + Sync {
    fn validate(&self, structure: &serde_json::Value) -> Result<ValidationResult>;
}

impl ValidatorAgent {
    pub fn new(capabilities: AgentCapabilities) -> Self {
        let (sender, _) = broadcast::channel(1000);
        
        Self {
            base: BaseAgent::new(
                "validator".to_string(),
                vec![
                    AgentCapability::ValidationExpert,
                    AgentCapability::PatternRecognition,
                    AgentCapability::ErrorDetection,
                ],
                sender,
            ),
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
    
    pub fn new_with_sender(message_sender: broadcast::Sender<Message>) -> Self {
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
    
    pub async fn validate_document(&self, content: &str) -> Result<ValidationResult> {
        // Check cache first
        let cache_key = format!("{:x}", md5::compute(content));
        {
            let mut cache = self.validation_cache.lock().await;
            if let Some(cached_result) = cache.get(&cache_key) {
                return Ok(cached_result.clone());
            }
        }
        
        let mut result = ValidationResult {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            suggestions: Vec::new(),
        };
        
        let rules = self.validation_rules.read().await;
        
        // Apply format rules
        for rule in &rules.format_rules {
            if !rule.pattern.is_match(content) {
                let error = ValidationError {
                    rule_name: rule.name.clone(),
                    message: format!("Content does not match required format: {}", rule.name),
                    severity: rule.severity,
                    line: None,
                    column: None,
                };
                
                match rule.severity {
                    ValidationSeverity::Error => {
                        result.is_valid = false;
                        result.errors.push(error);
                    }
                    ValidationSeverity::Warning => {
                        result.warnings.push(ValidationWarning {
                            rule_name: error.rule_name,
                            message: error.message,
                            line: error.line,
                            column: error.column,
                        });
                    }
                    ValidationSeverity::Info => {
                        result.suggestions.push(error.message);
                    }
                }
            }
        }
        
        // Apply content rules
        for rule in &rules.content_rules {
            match rule.validator.validate(content) {
                Ok(rule_result) => {
                    result.errors.extend(rule_result.errors);
                    result.warnings.extend(rule_result.warnings);
                    result.suggestions.extend(rule_result.suggestions);
                    if !rule_result.is_valid {
                        result.is_valid = false;
                    }
                }
                Err(e) => {
                    result.is_valid = false;
                    result.errors.push(ValidationError {
                        rule_name: rule.name.clone(),
                        message: format!("Content validation failed: {}", e),
                        severity: ValidationSeverity::Error,
                        line: None,
                        column: None,
                    });
                }
            }
        }
        
        // Cache the result
        {
            let mut cache = self.validation_cache.lock().await;
            cache.put(cache_key, result.clone());
        }
        
        Ok(result)
    }
}

#[async_trait::async_trait]
impl DaaAgent for ValidatorAgent {
    fn id(&self) -> Uuid {
        self.base.id
    }
    
    fn agent_type(&self) -> AgentType {
        AgentType::Validator
    }
    
    fn state(&self) -> AgentState {
        // Since BaseAgent state is Arc<RwLock<AgentState>>, we need to clone a default for now
        // This is a simplified implementation for compilation
        AgentState::Ready
    }
    
    fn capabilities(&self) -> AgentCapabilities {
        // Convert Vec<AgentCapability> to AgentCapabilities
        AgentCapabilities {
            neural_processing: false,
            text_enhancement: false,
            layout_analysis: false,
            quality_assessment: self.base.capabilities.contains(&AgentCapability::ValidationExpert),
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
        
        // Validate the document
        let validation_result = self.validate_document(&input_text).await?;
        
        self.base.set_state(crate::agents::base::AgentState::Ready).await;
        
        // Return validation result as bytes
        Ok(serde_json::to_vec(&validation_result)?)
    }
    
    async fn coordinate(&mut self, message: CoordinationMessage) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        match message.message_type {
            MessageType::Task => {
                // Handle task assignment
                let result = self.process(message.payload).await?;
                // In a real implementation, would send result back
                Ok(())
            }
            MessageType::Status => {
                // Handle status updates
                Ok(())
            }
            MessageType::Result => {
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