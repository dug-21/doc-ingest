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

/// Formatter agent for document formatting and styling
pub struct FormatterAgent {
    base: BaseAgent,
    format_handlers: Arc<RwLock<std::collections::HashMap<String, Arc<dyn FormatHandler>>>>,
    style_registry: Arc<RwLock<StyleRegistry>>,
}

#[async_trait]
pub trait FormatHandler: Send + Sync {
    async fn format(&self, content: &str, style: &FormatStyle) -> Result<FormattedContent>;
    fn supported_formats(&self) -> Vec<String>;
    fn handler_name(&self) -> &str;
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FormattedContent {
    pub original: String,
    pub formatted: String,
    pub format: String,
    pub style_applied: String,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct FormatStyle {
    pub name: String,
    pub rules: serde_json::Value,
    pub template: Option<String>,
}

pub struct StyleRegistry {
    styles: std::collections::HashMap<String, FormatStyle>,
}

impl StyleRegistry {
    pub fn new() -> Self {
        Self {
            styles: std::collections::HashMap::new(),
        }
    }
    
    pub fn register_style(&mut self, style: FormatStyle) {
        self.styles.insert(style.name.clone(), style);
    }
    
    pub fn get_style(&self, name: &str) -> Option<&FormatStyle> {
        self.styles.get(name)
    }
}

/// Markdown format handler
pub struct MarkdownFormatter;

#[async_trait]
impl FormatHandler for MarkdownFormatter {
    async fn format(&self, content: &str, style: &FormatStyle) -> Result<FormattedContent> {
        let mut formatted = String::new();
        
        // Apply basic markdown formatting
        let lines: Vec<&str> = content.lines().collect();
        let mut in_list = false;
        
        for line in lines {
            let trimmed = line.trim();
            
            // Detect headers (lines that look like titles)
            if trimmed.len() > 0 && trimmed.chars().all(|c| c.is_alphabetic() || c.is_whitespace()) 
               && trimmed.len() < 50 && !in_list {
                formatted.push_str(&format!("## {}\n\n", trimmed));
            }
            // Detect list items
            else if trimmed.starts_with("- ") || trimmed.starts_with("* ") || trimmed.starts_with("+ ") {
                formatted.push_str(&format!("{}\n", line));
                in_list = true;
            }
            // Regular paragraph
            else if !trimmed.is_empty() {
                formatted.push_str(&format!("{}\n", line));
                in_list = false;
            }
            // Empty line
            else {
                formatted.push_str("\n");
                in_list = false;
            }
        }
        
        Ok(FormattedContent {
            original: content.to_string(),
            formatted,
            format: "markdown".to_string(),
            style_applied: style.name.clone(),
            metadata: serde_json::json!({
                "formatter": "markdown",
                "timestamp": chrono::Utc::now().to_rfc3339(),
            }),
        })
    }
    
    fn supported_formats(&self) -> Vec<String> {
        vec!["markdown".to_string(), "md".to_string()]
    }
    
    fn handler_name(&self) -> &str {
        "markdown_formatter"
    }
}

/// HTML format handler
pub struct HtmlFormatter;

#[async_trait]
impl FormatHandler for HtmlFormatter {
    async fn format(&self, content: &str, style: &FormatStyle) -> Result<FormattedContent> {
        let mut formatted = String::from("<!DOCTYPE html>\n<html>\n<head>\n");
        
        // Add style if provided
        if let Some(template) = &style.template {
            formatted.push_str(&format!("<style>\n{}\n</style>\n", template));
        }
        
        formatted.push_str("</head>\n<body>\n");
        
        // Convert content to HTML
        let lines: Vec<&str> = content.lines().collect();
        let mut in_paragraph = false;
        
        for line in lines {
            let trimmed = line.trim();
            
            if trimmed.is_empty() {
                if in_paragraph {
                    formatted.push_str("</p>\n");
                    in_paragraph = false;
                }
            } else {
                if !in_paragraph {
                    formatted.push_str("<p>");
                    in_paragraph = true;
                }
                // Escape HTML characters
                let escaped = trimmed
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                    .replace("\"", "&quot;");
                formatted.push_str(&escaped);
                formatted.push(' ');
            }
        }
        
        if in_paragraph {
            formatted.push_str("</p>\n");
        }
        
        formatted.push_str("</body>\n</html>");
        
        Ok(FormattedContent {
            original: content.to_string(),
            formatted,
            format: "html".to_string(),
            style_applied: style.name.clone(),
            metadata: serde_json::json!({
                "formatter": "html",
                "timestamp": chrono::Utc::now().to_rfc3339(),
            }),
        })
    }
    
    fn supported_formats(&self) -> Vec<String> {
        vec!["html".to_string(), "htm".to_string()]
    }
    
    fn handler_name(&self) -> &str {
        "html_formatter"
    }
}

/// JSON format handler
pub struct JsonFormatter;

#[async_trait]
impl FormatHandler for JsonFormatter {
    async fn format(&self, content: &str, style: &FormatStyle) -> Result<FormattedContent> {
        // Try to parse as JSON first
        let formatted = if let Ok(value) = serde_json::from_str::<serde_json::Value>(content) {
            serde_json::to_string_pretty(&value)?
        } else {
            // If not valid JSON, create a JSON object with the content
            let json_obj = serde_json::json!({
                "content": content,
                "formatted_at": chrono::Utc::now().to_rfc3339(),
                "style": style.name,
            });
            serde_json::to_string_pretty(&json_obj)?
        };
        
        Ok(FormattedContent {
            original: content.to_string(),
            formatted,
            format: "json".to_string(),
            style_applied: style.name.clone(),
            metadata: serde_json::json!({
                "formatter": "json",
                "timestamp": chrono::Utc::now().to_rfc3339(),
            }),
        })
    }
    
    fn supported_formats(&self) -> Vec<String> {
        vec!["json".to_string()]
    }
    
    fn handler_name(&self) -> &str {
        "json_formatter"
    }
}

impl FormatterAgent {
    pub fn new(capabilities: AgentCapabilities) -> Self {
        let (sender, _) = broadcast::channel(1000);
        
        Self {
            base: BaseAgent::new(
                "formatter".to_string(),
                vec![
                    AgentCapability::FormatConversion,
                    AgentCapability::StyleApplication,
                    AgentCapability::TemplateProcessing,
                ],
                sender,
            ),
            format_handlers: Arc::new(RwLock::new(std::collections::HashMap::new())),
            style_registry: Arc::new(RwLock::new(StyleRegistry::new())),
        }
    }
    
    pub fn new_with_sender(message_sender: broadcast::Sender<Message>) -> Self {
        let capabilities = vec![
            AgentCapability::FormatConversion,
            AgentCapability::StyleApplication,
            AgentCapability::TemplateProcessing,
        ];
        
        Self {
            base: BaseAgent::new("formatter".to_string(), capabilities, message_sender),
            format_handlers: Arc::new(RwLock::new(std::collections::HashMap::new())),
            style_registry: Arc::new(RwLock::new(StyleRegistry::new())),
        }
    }
    
    pub async fn register_format_handler(&self, handler: Arc<dyn FormatHandler>) {
        let mut handlers = self.format_handlers.write().await;
        for format in handler.supported_formats() {
            handlers.insert(format, handler.clone());
        }
    }
    
    pub async fn register_style(&self, style: FormatStyle) {
        let mut registry = self.style_registry.write().await;
        registry.register_style(style);
    }
    
    async fn format_document(&self, doc_id: String, content: &str, format: &str, style_name: &str) -> Result<FormattedContent> {
        self.base.increment_task().await;
        
        // Get the appropriate handler
        let handlers = self.format_handlers.read().await;
        let handler = handlers.get(format)
            .ok_or_else(|| anyhow!("No handler for format: {}", format))?;
        
        // Get the style
        let registry = self.style_registry.read().await;
        let style = registry.get_style(style_name)
            .ok_or_else(|| anyhow!("Style not found: {}", style_name))?
            .clone();
        
        let result = handler.format(content, &style).await;
        self.base.complete_task().await;
        
        result
    }
}

#[async_trait]
impl DaaAgent for FormatterAgent {
    fn id(&self) -> Uuid {
        self.base.id
    }
    
    fn agent_type(&self) -> AgentType {
        AgentType::Formatter
    }
    
    fn state(&self) -> super::AgentState {
        // Since BaseAgent state is Arc<RwLock<AgentState>>, we need to clone a default for now
        // This is a simplified implementation for compilation
        super::AgentState::Ready
    }
    
    fn capabilities(&self) -> AgentCapabilities {
        // Convert Vec<AgentCapability> to AgentCapabilities
        AgentCapabilities {
            neural_processing: false,
            text_enhancement: false,
            layout_analysis: false,
            quality_assessment: false,
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
        
        // Apply formatting (default to markdown with default style)
        let formatted_content = self.format_document("doc".to_string(), &input_text, "markdown", "default").await?;
        
        self.base.set_state(crate::agents::base::AgentState::Ready).await;
        
        // Return formatted content as bytes
        Ok(formatted_content.formatted.into_bytes())
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
    
impl FormatterAgent {
    async fn process_message(&self, message: Message) -> Result<()> {
        self.base.set_state(AgentState::Processing).await;
        
        let result = match &message.content {
            serde_json::Value::Object(obj) => {
                if let Some(serde_json::Value::String(action)) = obj.get("action") {
                    match action.as_str() {
                        "format" => {
                            if let (
                                Some(serde_json::Value::String(doc_id)),
                                Some(serde_json::Value::String(content)),
                                Some(serde_json::Value::String(format)),
                                Some(serde_json::Value::String(style))
                            ) = (
                                obj.get("doc_id"),
                                obj.get("content"), 
                                obj.get("format"),
                                obj.get("style")
                            ) {
                                self.format_document(doc_id.clone(), content, format, style).await
                            } else {
                                Err(anyhow!("Invalid format message format"))
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
            Ok(formatted_content) => {
                // Send formatting result back
                let response = Message {
                    id: Uuid::new_v4(),
                    from: self.base.id,
                    to: message.from,
                    priority: MessagePriority::Normal,
                    content: serde_json::to_value(&formatted_content)?,
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
        self.base.set_state(AgentState::ShuttingDown).await;
        Ok(())
    }
    
    fn resource_requirements(&self) -> Vec<ResourceRequirement> {
        vec![
            ResourceRequirement::Memory(256), // 256MB
            ResourceRequirement::CPU(0.25),   // 25% of one core
        ]
    }
    
    async fn handle_coordination(&self, message: CoordinationMessage) -> Result<()> {
        match message {
            CoordinationMessage::StatusRequest => {
                // Status is handled by the status() method
                Ok(())
            }
            CoordinationMessage::Shutdown => {
                self.base.set_state(AgentState::ShuttingDown).await;
                Ok(())
            }
            _ => Ok(())
        }
    }
}