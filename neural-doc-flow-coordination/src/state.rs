//! Shared state management for coordination

use neural_doc_flow_core::Result;
use uuid::Uuid;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use tokio::sync::RwLock;
use std::sync::Arc;

/// Coordination state entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateEntry {
    pub key: String,
    pub value: serde_json::Value,
    pub owner: Uuid,
    pub version: u64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub ttl: Option<chrono::Duration>,
}

/// Coordination state manager
#[derive(Debug, Clone)]
pub struct CoordinationState {
    entries: Arc<RwLock<HashMap<String, StateEntry>>>,
    version_counter: Arc<RwLock<u64>>,
}

impl CoordinationState {
    pub fn new() -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
            version_counter: Arc::new(RwLock::new(0)),
        }
    }
    
    /// Set a state value
    pub async fn set(&self, key: String, value: serde_json::Value, owner: Uuid, ttl: Option<chrono::Duration>) -> Result<u64> {
        let mut counter = self.version_counter.write().await;
        *counter += 1;
        let version = *counter;
        
        let entry = StateEntry {
            key: key.clone(),
            value,
            owner,
            version,
            timestamp: chrono::Utc::now(),
            ttl,
        };
        
        let mut entries = self.entries.write().await;
        entries.insert(key, entry);
        
        Ok(version)
    }
    
    /// Get a state value
    pub async fn get(&self, key: &str) -> Option<StateEntry> {
        let entries = self.entries.read().await;
        entries.get(key).cloned()
    }
    
    /// Remove a state value
    pub async fn remove(&self, key: &str, owner: Uuid) -> Result<bool> {
        let mut entries = self.entries.write().await;
        
        if let Some(entry) = entries.get(key) {
            if entry.owner == owner {
                entries.remove(key);
                Ok(true)
            } else {
                Err(anyhow::anyhow!("Only the owner can remove this entry"))
            }
        } else {
            Ok(false)
        }
    }
    
    /// List all keys
    pub async fn keys(&self) -> Vec<String> {
        let entries = self.entries.read().await;
        entries.keys().cloned().collect()
    }
    
    /// List keys owned by a specific agent
    pub async fn keys_by_owner(&self, owner: Uuid) -> Vec<String> {
        let entries = self.entries.read().await;
        entries.iter()
            .filter(|(_, entry)| entry.owner == owner)
            .map(|(key, _)| key.clone())
            .collect()
    }
    
    /// Clean up expired entries
    pub async fn cleanup_expired(&self) -> Result<usize> {
        let mut entries = self.entries.write().await;
        let now = chrono::Utc::now();
        let mut removed_count = 0;
        
        entries.retain(|_, entry| {
            if let Some(ttl) = entry.ttl {
                let expires_at = entry.timestamp + ttl;
                if now > expires_at {
                    removed_count += 1;
                    false
                } else {
                    true
                }
            } else {
                true
            }
        });
        
        Ok(removed_count)
    }
    
    /// Get state statistics
    pub async fn stats(&self) -> StateStats {
        let entries = self.entries.read().await;
        let total_entries = entries.len();
        let version_counter = *self.version_counter.read().await;
        
        let mut owners = std::collections::HashSet::new();
        let mut expired_count = 0;
        let now = chrono::Utc::now();
        
        for entry in entries.values() {
            owners.insert(entry.owner);
            
            if let Some(ttl) = entry.ttl {
                let expires_at = entry.timestamp + ttl;
                if now > expires_at {
                    expired_count += 1;
                }
            }
        }
        
        StateStats {
            total_entries,
            unique_owners: owners.len(),
            expired_entries: expired_count,
            current_version: version_counter,
        }
    }
}

impl Default for CoordinationState {
    fn default() -> Self {
        Self::new()
    }
}

/// State statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateStats {
    pub total_entries: usize,
    pub unique_owners: usize,
    pub expired_entries: usize,
    pub current_version: u64,
}

/// Shared state type alias
pub type SharedState = CoordinationState;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_coordination_state() {
        let state = CoordinationState::new();
        let owner = Uuid::new_v4();
        
        // Set a value
        let version = state.set(
            "test_key".to_string(),
            serde_json::json!({"value": 42}),
            owner,
            None
        ).await.unwrap();
        
        assert_eq!(version, 1);
        
        // Get the value
        let entry = state.get("test_key").await.unwrap();
        assert_eq!(entry.value, serde_json::json!({"value": 42}));
        assert_eq!(entry.owner, owner);
        assert_eq!(entry.version, 1);
        
        // List keys
        let keys = state.keys().await;
        assert!(keys.contains(&"test_key".to_string()));
        
        // Remove the value
        let removed = state.remove("test_key", owner).await.unwrap();
        assert!(removed);
        
        // Verify removal
        let entry = state.get("test_key").await;
        assert!(entry.is_none());
    }
    
    #[tokio::test]
    async fn test_state_expiration() {
        let state = CoordinationState::new();
        let owner = Uuid::new_v4();
        
        // Set a value with short TTL
        state.set(
            "temp_key".to_string(),
            serde_json::json!({"temp": true}),
            owner,
            Some(chrono::Duration::milliseconds(1))
        ).await.unwrap();
        
        // Wait for expiration
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        
        // Clean up expired entries
        let removed_count = state.cleanup_expired().await.unwrap();
        assert_eq!(removed_count, 1);
        
        // Verify removal
        let entry = state.get("temp_key").await;
        assert!(entry.is_none());
    }
    
    #[tokio::test]
    async fn test_state_stats() {
        let state = CoordinationState::new();
        let owner1 = Uuid::new_v4();
        let owner2 = Uuid::new_v4();
        
        state.set("key1".to_string(), serde_json::json!(1), owner1, None).await.unwrap();
        state.set("key2".to_string(), serde_json::json!(2), owner2, None).await.unwrap();
        
        let stats = state.stats().await;
        assert_eq!(stats.total_entries, 2);
        assert_eq!(stats.unique_owners, 2);
        assert_eq!(stats.current_version, 2);
    }
}