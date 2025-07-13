//! Message passing system

use neural_doc_flow_core::Result;
use async_trait::async_trait;
use uuid::Uuid;
use serde::{Serialize, Deserialize};
use std::collections::VecDeque;
use tokio::sync::{mpsc, Mutex};
use std::sync::Arc;

/// Message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageType {
    Task,
    Status,
    Data,
    Control,
    Heartbeat,
    Custom(String),
}

/// Core message structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: Uuid,
    pub from: Uuid,
    pub to: Uuid,
    pub message_type: MessageType,
    pub payload: serde_json::Value,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl Message {
    pub fn new(from: Uuid, to: Uuid, message_type: MessageType, payload: serde_json::Value) -> Self {
        Self {
            id: Uuid::new_v4(),
            from,
            to,
            message_type,
            payload,
            timestamp: chrono::Utc::now(),
        }
    }
}

/// Message bus for agent communication
#[derive(Debug)]
pub struct MessageBus {
    message_queue: Arc<Mutex<VecDeque<Message>>>,
    sender: Option<mpsc::UnboundedSender<Message>>,
    receiver: Option<mpsc::UnboundedReceiver<Message>>,
    running: Arc<Mutex<bool>>,
}

impl MessageBus {
    pub fn new() -> Self {
        let (sender, receiver) = mpsc::unbounded_channel();
        
        Self {
            message_queue: Arc::new(Mutex::new(VecDeque::new())),
            sender: Some(sender),
            receiver: Some(receiver),
            running: Arc::new(Mutex::new(false)),
        }
    }
    
    /// Send a message
    pub async fn send(&self, from: Uuid, to: Uuid, message: Message) -> Result<()> {
        if let Some(sender) = &self.sender {
            sender.send(message)?;
        }
        Ok(())
    }
    
    /// Get queue size
    pub fn queue_size(&self) -> usize {
        // This is a simplified implementation
        0
    }
    
    /// Start message processing
    pub async fn start(&mut self) -> Result<()> {
        *self.running.lock().await = true;
        
        // In a real implementation, this would start background message processing
        Ok(())
    }
    
    /// Stop message processing
    pub async fn stop(&mut self) -> Result<()> {
        *self.running.lock().await = false;
        Ok(())
    }
}

/// Message handler trait
#[async_trait]
pub trait MessageHandler: Send + Sync {
    /// Handle incoming message
    async fn handle_message(&mut self, message: Message) -> Result<Option<Message>>;
    
    /// Get handler ID
    fn handler_id(&self) -> Uuid;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_message_creation() {
        let from = Uuid::new_v4();
        let to = Uuid::new_v4();
        let payload = serde_json::json!({"test": "data"});
        
        let message = Message::new(from, to, MessageType::Task, payload.clone());
        
        assert_eq!(message.from, from);
        assert_eq!(message.to, to);
        assert_eq!(message.payload, payload);
        assert!(matches!(message.message_type, MessageType::Task));
    }
    
    #[tokio::test]
    async fn test_message_bus() {
        let mut bus = MessageBus::new();
        
        assert_eq!(bus.queue_size(), 0);
        
        bus.start().await.unwrap();
        bus.stop().await.unwrap();
    }
}