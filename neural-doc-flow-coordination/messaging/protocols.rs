use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::{RwLock, broadcast, mpsc, Mutex};
use uuid::Uuid;
use serde::{Serialize, Deserialize};
use anyhow::{Result, anyhow};
use chrono::{DateTime, Utc};

use crate::messaging::{Message, MessagePriority};

/// Communication protocol types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProtocolType {
    RequestResponse,
    PubSub,
    Streaming,
    Gossip,
}

/// Protocol handler trait
#[async_trait::async_trait]
pub trait ProtocolHandler: Send + Sync {
    async fn handle_message(&self, message: Message) -> Result<Option<Message>>;
    fn protocol_type(&self) -> ProtocolType;
}

/// Request-Response protocol implementation
pub struct RequestResponseProtocol {
    pending_requests: Arc<Mutex<HashMap<Uuid, ResponseCallback>>>,
    #[allow(dead_code)]
    timeout: std::time::Duration,
}

type ResponseCallback = Box<dyn FnOnce(Message) + Send>;

impl RequestResponseProtocol {
    pub fn new(timeout: std::time::Duration) -> Self {
        Self {
            pending_requests: Arc::new(Mutex::new(HashMap::new())),
            timeout,
        }
    }
    
    pub async fn send_request<F>(&self, request: Message, callback: F) -> Result<()>
    where
        F: FnOnce(Message) + Send + 'static,
    {
        let mut pending = self.pending_requests.lock().await;
        pending.insert(request.id, Box::new(callback));
        
        // Set up timeout
        let request_id = request.id;
        let pending_requests = self.pending_requests.clone();
        let timeout = self.timeout;
        
        tokio::spawn(async move {
            tokio::time::sleep(timeout).await;
            let mut pending = pending_requests.lock().await;
            if pending.remove(&request_id).is_some() {
                tracing::warn!("Request {} timed out", request_id);
            }
        });
        
        Ok(())
    }
    
    pub async fn handle_response(&self, response: Message) -> Result<()> {
        if let Some(correlation_id) = response.correlation_id {
            let mut pending = self.pending_requests.lock().await;
            if let Some(callback) = pending.remove(&correlation_id) {
                callback(response);
            }
        }
        Ok(())
    }
}

#[async_trait::async_trait]
impl ProtocolHandler for RequestResponseProtocol {
    async fn handle_message(&self, message: Message) -> Result<Option<Message>> {
        // Check if this is a response to a pending request
        if message.correlation_id.is_some() {
            self.handle_response(message).await?;
            Ok(None)
        } else {
            // This is a new request - return it for processing
            Ok(Some(message))
        }
    }
    
    fn protocol_type(&self) -> ProtocolType {
        ProtocolType::RequestResponse
    }
}

/// Pub-Sub protocol implementation
pub struct PubSubProtocol {
    topics: Arc<RwLock<HashMap<String, TopicSubscribers>>>,
}

struct TopicSubscribers {
    subscribers: Vec<SubscriberInfo>,
}

struct SubscriberInfo {
    id: Uuid,
    filter: Option<Box<dyn MessageFilter>>,
}

pub trait MessageFilter: Send + Sync {
    fn matches(&self, message: &Message) -> bool;
}

impl PubSubProtocol {
    pub fn new() -> Self {
        Self {
            topics: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn subscribe(&self, topic: String, subscriber_id: Uuid, filter: Option<Box<dyn MessageFilter>>) -> Result<()> {
        let mut topics = self.topics.write().await;
        let topic_subs = topics.entry(topic).or_insert_with(|| TopicSubscribers {
            subscribers: Vec::new(),
        });
        
        // Check if already subscribed
        if topic_subs.subscribers.iter().any(|s| s.id == subscriber_id) {
            return Err(anyhow!("Already subscribed to topic"));
        }
        
        topic_subs.subscribers.push(SubscriberInfo {
            id: subscriber_id,
            filter,
        });
        
        Ok(())
    }
    
    pub async fn unsubscribe(&self, topic: &str, subscriber_id: Uuid) -> Result<()> {
        let mut topics = self.topics.write().await;
        if let Some(topic_subs) = topics.get_mut(topic) {
            topic_subs.subscribers.retain(|s| s.id != subscriber_id);
            Ok(())
        } else {
            Err(anyhow!("Topic not found"))
        }
    }
    
    pub async fn publish(&self, topic: &str, message: Message) -> Result<Vec<Uuid>> {
        let topics = self.topics.read().await;
        if let Some(topic_subs) = topics.get(topic) {
            let mut recipients = Vec::new();
            
            for subscriber in &topic_subs.subscribers {
                // Apply filter if present
                if let Some(filter) = &subscriber.filter {
                    if !filter.matches(&message) {
                        continue;
                    }
                }
                recipients.push(subscriber.id);
            }
            
            Ok(recipients)
        } else {
            Ok(vec![])
        }
    }
}

#[async_trait::async_trait]
impl ProtocolHandler for PubSubProtocol {
    async fn handle_message(&self, message: Message) -> Result<Option<Message>> {
        // Extract topic from message
        if let Some(topic) = message.content.get("topic").and_then(|v| v.as_str()) {
            let recipients = self.publish(topic, message.clone()).await?;
            if recipients.is_empty() {
                tracing::warn!("No subscribers for topic: {}", topic);
            }
        }
        Ok(Some(message))
    }
    
    fn protocol_type(&self) -> ProtocolType {
        ProtocolType::PubSub
    }
}

/// Streaming protocol implementation
pub struct StreamingProtocol {
    streams: Arc<RwLock<HashMap<Uuid, StreamInfo>>>,
}

struct StreamInfo {
    stream_id: Uuid,
    sender_id: Uuid,
    receiver_id: Uuid,
    buffer: mpsc::Sender<Message>,
    created_at: DateTime<Utc>,
    chunk_size: usize,
}

impl StreamingProtocol {
    pub fn new() -> Self {
        Self {
            streams: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn create_stream(
        &self,
        sender_id: Uuid,
        receiver_id: Uuid,
        buffer_size: usize,
        chunk_size: usize,
    ) -> Result<(Uuid, mpsc::Receiver<Message>)> {
        let stream_id = Uuid::new_v4();
        let (tx, rx) = mpsc::channel(buffer_size);
        
        let stream_info = StreamInfo {
            stream_id,
            sender_id,
            receiver_id,
            buffer: tx,
            created_at: Utc::now(),
            chunk_size,
        };
        
        let mut streams = self.streams.write().await;
        streams.insert(stream_id, stream_info);
        
        Ok((stream_id, rx))
    }
    
    pub async fn send_chunk(&self, stream_id: Uuid, chunk: Message) -> Result<()> {
        let streams = self.streams.read().await;
        if let Some(stream) = streams.get(&stream_id) {
            stream.buffer.send(chunk).await
                .map_err(|e| anyhow!("Failed to send chunk: {}", e))?;
            Ok(())
        } else {
            Err(anyhow!("Stream not found"))
        }
    }
    
    pub async fn close_stream(&self, stream_id: Uuid) -> Result<()> {
        let mut streams = self.streams.write().await;
        if streams.remove(&stream_id).is_some() {
            Ok(())
        } else {
            Err(anyhow!("Stream not found"))
        }
    }
}

#[async_trait::async_trait]
impl ProtocolHandler for StreamingProtocol {
    async fn handle_message(&self, message: Message) -> Result<Option<Message>> {
        // Check if this is a stream control message
        if let Some(stream_op) = message.content.get("stream_operation").and_then(|v| v.as_str()) {
            match stream_op {
                "create" => {
                    // Handle stream creation
                    Ok(None)
                }
                "chunk" => {
                    // Handle stream chunk
                    if let Some(stream_id) = message.content.get("stream_id")
                        .and_then(|v| v.as_str())
                        .and_then(|s| Uuid::parse_str(s).ok()) {
                        self.send_chunk(stream_id, message).await?;
                    }
                    Ok(None)
                }
                "close" => {
                    // Handle stream closure
                    if let Some(stream_id) = message.content.get("stream_id")
                        .and_then(|v| v.as_str())
                        .and_then(|s| Uuid::parse_str(s).ok()) {
                        self.close_stream(stream_id).await?;
                    }
                    Ok(None)
                }
                _ => Ok(Some(message))
            }
        } else {
            Ok(Some(message))
        }
    }
    
    fn protocol_type(&self) -> ProtocolType {
        ProtocolType::Streaming
    }
}

/// Gossip protocol implementation for distributed state synchronization
pub struct GossipProtocol {
    node_id: Uuid,
    peers: Arc<RwLock<Vec<Uuid>>>,
    state: Arc<RwLock<GossipState>>,
    fanout: usize,
}

#[derive(Clone, Serialize, Deserialize)]
struct GossipState {
    values: HashMap<String, GossipValue>,
}

#[derive(Clone, Serialize, Deserialize)]
struct GossipValue {
    data: serde_json::Value,
    version: u64,
    timestamp: DateTime<Utc>,
    source: Uuid,
}

impl GossipProtocol {
    pub fn new(node_id: Uuid, fanout: usize) -> Self {
        Self {
            node_id,
            peers: Arc::new(RwLock::new(Vec::new())),
            state: Arc::new(RwLock::new(GossipState {
                values: HashMap::new(),
            })),
            fanout,
        }
    }
    
    pub async fn add_peer(&self, peer_id: Uuid) -> Result<()> {
        let mut peers = self.peers.write().await;
        if !peers.contains(&peer_id) {
            peers.push(peer_id);
        }
        Ok(())
    }
    
    pub async fn remove_peer(&self, peer_id: Uuid) -> Result<()> {
        let mut peers = self.peers.write().await;
        peers.retain(|p| *p != peer_id);
        Ok(())
    }
    
    pub async fn update_value(&self, key: String, value: serde_json::Value) -> Result<()> {
        let mut state = self.state.write().await;
        let version = state.values.get(&key)
            .map(|v| v.version + 1)
            .unwrap_or(1);
        
        state.values.insert(key, GossipValue {
            data: value,
            version,
            timestamp: Utc::now(),
            source: self.node_id,
        });
        
        Ok(())
    }
    
    pub async fn get_value(&self, key: &str) -> Option<serde_json::Value> {
        let state = self.state.read().await;
        state.values.get(key).map(|v| v.data.clone())
    }
    
    async fn select_gossip_targets(&self) -> Vec<Uuid> {
        let peers = self.peers.read().await;
        let mut targets = peers.clone();
        
        // Randomly select fanout peers
        use rand::seq::SliceRandom;
        targets.shuffle(&mut rand::thread_rng());
        targets.truncate(self.fanout);
        
        targets
    }
    
    pub async fn create_gossip_message(&self) -> Message {
        let state = self.state.read().await;
        Message {
            id: Uuid::new_v4(),
            from: self.node_id,
            to: Uuid::nil(), // Will be set per target
            priority: MessagePriority::Low,
            content: serde_json::json!({
                "type": "gossip",
                "state": serde_json::to_value(&state.values).unwrap(),
            }),
            timestamp: Utc::now(),
            correlation_id: None,
        }
    }
    
    async fn merge_gossip_state(&self, remote_state: &HashMap<String, GossipValue>) -> Result<()> {
        let mut state = self.state.write().await;
        
        for (key, remote_value) in remote_state {
            let should_update = if let Some(local_value) = state.values.get(key) {
                // Update if remote version is newer or same version but newer timestamp
                remote_value.version > local_value.version ||
                (remote_value.version == local_value.version && 
                 remote_value.timestamp > local_value.timestamp)
            } else {
                true
            };
            
            if should_update {
                state.values.insert(key.clone(), remote_value.clone());
            }
        }
        
        Ok(())
    }
}

#[async_trait::async_trait]
impl ProtocolHandler for GossipProtocol {
    async fn handle_message(&self, message: Message) -> Result<Option<Message>> {
        if let Some(msg_type) = message.content.get("type").and_then(|v| v.as_str()) {
            if msg_type == "gossip" {
                // Extract and merge remote state
                if let Some(remote_state) = message.content.get("state") {
                    if let Ok(state) = serde_json::from_value::<HashMap<String, GossipValue>>(remote_state.clone()) {
                        self.merge_gossip_state(&state).await?;
                    }
                }
                return Ok(None);
            }
        }
        
        Ok(Some(message))
    }
    
    fn protocol_type(&self) -> ProtocolType {
        ProtocolType::Gossip
    }
}

/// Protocol manager for handling multiple protocols
pub struct ProtocolManager {
    handlers: Arc<RwLock<HashMap<ProtocolType, Box<dyn ProtocolHandler>>>>,
}

impl ProtocolManager {
    pub fn new() -> Self {
        Self {
            handlers: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn register_handler(&self, handler: Box<dyn ProtocolHandler>) {
        let mut handlers = self.handlers.write().await;
        handlers.insert(handler.protocol_type(), handler);
    }
    
    pub async fn handle_message(&self, message: Message, protocol: ProtocolType) -> Result<Option<Message>> {
        let handlers = self.handlers.read().await;
        if let Some(handler) = handlers.get(&protocol) {
            handler.handle_message(message).await
        } else {
            Err(anyhow!("No handler for protocol: {:?}", protocol))
        }
    }
}