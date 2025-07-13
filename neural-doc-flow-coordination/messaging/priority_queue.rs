use std::{
    cmp::Ordering,
    collections::BinaryHeap,
    sync::Arc,
};
use tokio::sync::Mutex;
use uuid::Uuid;
use chrono::{DateTime, Utc};

use crate::messaging::{Message, MessagePriority};

/// Priority wrapper for messages in the queue
#[derive(Debug, Clone)]
struct PriorityMessage {
    message: Message,
    enqueued_at: DateTime<Utc>,
}

impl PriorityMessage {
    fn new(message: Message) -> Self {
        Self {
            message,
            enqueued_at: Utc::now(),
        }
    }
    
    fn priority_value(&self) -> i32 {
        match self.message.priority {
            MessagePriority::Critical => 100,
            MessagePriority::High => 75,
            MessagePriority::Normal => 50,
            MessagePriority::Low => 25,
        }
    }
}

impl PartialEq for PriorityMessage {
    fn eq(&self, other: &Self) -> bool {
        self.priority_value() == other.priority_value() &&
        self.enqueued_at == other.enqueued_at
    }
}

impl Eq for PriorityMessage {}

impl PartialOrd for PriorityMessage {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityMessage {
    fn cmp(&self, other: &Self) -> Ordering {
        // First compare by priority
        let priority_cmp = self.priority_value().cmp(&other.priority_value());
        if priority_cmp != Ordering::Equal {
            return priority_cmp;
        }
        
        // Then by age (older messages first for same priority)
        other.enqueued_at.cmp(&self.enqueued_at)
    }
}

/// Thread-safe priority message queue
pub struct PriorityMessageQueue {
    queue: Arc<Mutex<BinaryHeap<PriorityMessage>>>,
    max_size: Option<usize>,
    stats: Arc<Mutex<QueueStats>>,
}

#[derive(Debug, Default)]
pub struct QueueStats {
    pub total_enqueued: u64,
    pub total_dequeued: u64,
    pub total_dropped: u64,
    pub critical_count: u64,
    pub high_count: u64,
    pub normal_count: u64,
    pub low_count: u64,
}

impl PriorityMessageQueue {
    pub fn new(max_size: Option<usize>) -> Self {
        Self {
            queue: Arc::new(Mutex::new(BinaryHeap::new())),
            max_size,
            stats: Arc::new(Mutex::new(QueueStats::default())),
        }
    }
    
    /// Enqueue a message with priority handling
    pub async fn enqueue(&self, message: Message) -> Result<(), QueueError> {
        let mut queue = self.queue.lock().await;
        
        // Check capacity
        if let Some(max) = self.max_size {
            if queue.len() >= max {
                // Drop lowest priority message if new message has higher priority
                if let Some(lowest) = queue.peek() {
                    if message.priority as i32 > lowest.message.priority as i32 {
                        queue.pop();
                        let mut stats = self.stats.lock().await;
                        stats.total_dropped += 1;
                    } else {
                        return Err(QueueError::QueueFull);
                    }
                }
            }
        }
        
        // Update stats
        {
            let mut stats = self.stats.lock().await;
            stats.total_enqueued += 1;
            match message.priority {
                MessagePriority::Critical => stats.critical_count += 1,
                MessagePriority::High => stats.high_count += 1,
                MessagePriority::Normal => stats.normal_count += 1,
                MessagePriority::Low => stats.low_count += 1,
            }
        }
        
        queue.push(PriorityMessage::new(message));
        Ok(())
    }
    
    /// Dequeue the highest priority message
    pub async fn dequeue(&self) -> Option<Message> {
        let mut queue = self.queue.lock().await;
        if let Some(priority_msg) = queue.pop() {
            let mut stats = self.stats.lock().await;
            stats.total_dequeued += 1;
            Some(priority_msg.message)
        } else {
            None
        }
    }
    
    /// Peek at the highest priority message without removing it
    pub async fn peek(&self) -> Option<Message> {
        let queue = self.queue.lock().await;
        queue.peek().map(|pm| pm.message.clone())
    }
    
    /// Get current queue size
    pub async fn size(&self) -> usize {
        let queue = self.queue.lock().await;
        queue.len()
    }
    
    /// Check if queue is empty
    pub async fn is_empty(&self) -> bool {
        let queue = self.queue.lock().await;
        queue.is_empty()
    }
    
    /// Clear all messages from the queue
    pub async fn clear(&self) {
        let mut queue = self.queue.lock().await;
        queue.clear();
    }
    
    /// Get queue statistics
    pub async fn stats(&self) -> QueueStats {
        let stats = self.stats.lock().await;
        QueueStats {
            total_enqueued: stats.total_enqueued,
            total_dequeued: stats.total_dequeued,
            total_dropped: stats.total_dropped,
            critical_count: stats.critical_count,
            high_count: stats.high_count,
            normal_count: stats.normal_count,
            low_count: stats.low_count,
        }
    }
    
    /// Remove a specific message by ID
    pub async fn remove_by_id(&self, message_id: Uuid) -> bool {
        let mut queue = self.queue.lock().await;
        let original_len = queue.len();
        
        // Rebuild queue without the target message
        let messages: Vec<PriorityMessage> = queue.drain().collect();
        for msg in messages {
            if msg.message.id != message_id {
                queue.push(msg);
            }
        }
        
        queue.len() < original_len
    }
    
    /// Get all messages with a specific priority
    pub async fn get_by_priority(&self, priority: MessagePriority) -> Vec<Message> {
        let queue = self.queue.lock().await;
        queue.iter()
            .filter(|pm| pm.message.priority == priority)
            .map(|pm| pm.message.clone())
            .collect()
    }
    
    /// Requeue a message with updated priority
    pub async fn requeue_with_priority(&self, mut message: Message, new_priority: MessagePriority) -> Result<(), QueueError> {
        // First remove the message
        if self.remove_by_id(message.id).await {
            // Update priority and re-enqueue
            message.priority = new_priority;
            self.enqueue(message).await
        } else {
            Err(QueueError::MessageNotFound)
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum QueueError {
    #[error("Queue is full")]
    QueueFull,
    
    #[error("Message not found")]
    MessageNotFound,
}

/// Batch operations for efficiency
impl PriorityMessageQueue {
    /// Enqueue multiple messages at once
    pub async fn enqueue_batch(&self, messages: Vec<Message>) -> Result<usize, QueueError> {
        let mut enqueued = 0;
        for message in messages {
            if self.enqueue(message).await.is_ok() {
                enqueued += 1;
            }
        }
        Ok(enqueued)
    }
    
    /// Dequeue multiple messages at once
    pub async fn dequeue_batch(&self, count: usize) -> Vec<Message> {
        let mut messages = Vec::with_capacity(count);
        for _ in 0..count {
            if let Some(msg) = self.dequeue().await {
                messages.push(msg);
            } else {
                break;
            }
        }
        messages
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_priority_ordering() {
        let queue = PriorityMessageQueue::new(None);
        
        // Add messages with different priorities
        let low_msg = Message {
            id: Uuid::new_v4(),
            from: Uuid::new_v4(),
            to: Uuid::new_v4(),
            priority: MessagePriority::Low,
            content: serde_json::json!({"test": "low"}),
            timestamp: Utc::now(),
            correlation_id: None,
        };
        
        let high_msg = Message {
            id: Uuid::new_v4(),
            from: Uuid::new_v4(),
            to: Uuid::new_v4(),
            priority: MessagePriority::High,
            content: serde_json::json!({"test": "high"}),
            timestamp: Utc::now(),
            correlation_id: None,
        };
        
        let critical_msg = Message {
            id: Uuid::new_v4(),
            from: Uuid::new_v4(),
            to: Uuid::new_v4(),
            priority: MessagePriority::Critical,
            content: serde_json::json!({"test": "critical"}),
            timestamp: Utc::now(),
            correlation_id: None,
        };
        
        queue.enqueue(low_msg.clone()).await.unwrap();
        queue.enqueue(high_msg.clone()).await.unwrap();
        queue.enqueue(critical_msg.clone()).await.unwrap();
        
        // Should dequeue in priority order
        assert_eq!(queue.dequeue().await.unwrap().priority, MessagePriority::Critical);
        assert_eq!(queue.dequeue().await.unwrap().priority, MessagePriority::High);
        assert_eq!(queue.dequeue().await.unwrap().priority, MessagePriority::Low);
    }
}