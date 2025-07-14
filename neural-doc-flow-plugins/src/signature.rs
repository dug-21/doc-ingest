//! Plugin signature verification for security
//!
//! This module provides cryptographic signature verification for plugins
//! to ensure only trusted plugins can be loaded into the system.

use neural_doc_flow_core::ProcessingError;
use sha2::{Sha256, Digest};
use ed25519_dalek::{Verifier, PublicKey, Signature, VerifyingKey};
use std::path::Path;
use std::fs;
use serde::{Serialize, Deserialize};
use std::time::SystemTime;
use tracing::{info, warn, error};

/// Plugin signature verification
pub struct PluginSignatureVerifier {
    trusted_keys: Vec<VerifyingKey>,
    require_signature: bool,
    signature_cache: std::collections::HashMap<String, CachedSignature>,
}

/// Plugin signature metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginSignature {
    pub plugin_hash: String,
    pub signature: String,
    pub signer_key: String,
    pub timestamp: u64,
    pub version: String,
    pub metadata: SignatureMetadata,
}

/// Additional signature metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignatureMetadata {
    pub plugin_name: String,
    pub plugin_version: String,
    pub author: String,
    pub build_timestamp: u64,
    pub capabilities: Vec<String>,
    pub dependencies: Vec<String>,
    pub minimum_api_version: String,
}

/// Cached signature for performance
#[derive(Debug, Clone)]
struct CachedSignature {
    signature: PluginSignature,
    verified_at: SystemTime,
    file_size: u64,
    file_modified: SystemTime,
}

/// Signature verification result
#[derive(Debug)]
pub struct VerificationResult {
    pub is_valid: bool,
    pub signer: Option<String>,
    pub timestamp: Option<SystemTime>,
    pub capabilities: Vec<String>,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

impl PluginSignatureVerifier {
    /// Create a new signature verifier
    pub fn new(require_signature: bool) -> Self {
        Self {
            trusted_keys: Vec::new(),
            require_signature,
            signature_cache: std::collections::HashMap::new(),
        }
    }
    
    /// Add a trusted public key
    pub fn add_trusted_key(&mut self, public_key_hex: &str) -> Result<(), ProcessingError> {
        let key_bytes = hex::decode(public_key_hex)
            .map_err(|e| ProcessingError::PluginLoadError(format!("Invalid key format: {}", e)))?;
        
        if key_bytes.len() != 32 {
            return Err(ProcessingError::PluginLoadError(
                "Public key must be 32 bytes".to_string()
            ));
        }
        
        let mut key_array = [0u8; 32];
        key_array.copy_from_slice(&key_bytes);
        
        let verifying_key = VerifyingKey::from_bytes(&key_array)
            .map_err(|e| ProcessingError::PluginLoadError(format!("Invalid public key: {}", e)))?;
        
        self.trusted_keys.push(verifying_key);
        info!("Added trusted key: {}", &public_key_hex[..16]);
        Ok(())
    }
    
    /// Load trusted keys from file
    pub fn load_trusted_keys(&mut self, keys_file: &Path) -> Result<(), ProcessingError> {
        if !keys_file.exists() {
            if self.require_signature {
                return Err(ProcessingError::PluginLoadError(
                    format!("Trusted keys file not found: {:?}", keys_file)
                ));
            } else {
                warn!("No trusted keys file found, signature verification disabled");
                return Ok(());
            }
        }
        
        let keys_content = fs::read_to_string(keys_file)
            .map_err(|e| ProcessingError::PluginLoadError(format!("Failed to read keys file: {}", e)))?;
        
        for line in keys_content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            
            self.add_trusted_key(line)?;
        }
        
        info!("Loaded {} trusted keys", self.trusted_keys.len());
        Ok(())
    }
    
    /// Verify plugin signature
    pub fn verify_plugin(&mut self, plugin_path: &Path) -> Result<VerificationResult, ProcessingError> {
        if !self.require_signature && self.trusted_keys.is_empty() {
            return Ok(VerificationResult {
                is_valid: true,
                signer: None,
                timestamp: None,
                capabilities: Vec::new(),
                warnings: vec!["Signature verification disabled".to_string()],
                errors: Vec::new(),
            });
        }
        
        let plugin_hash = self.calculate_file_hash(plugin_path)?;
        
        // Check cache first
        if let Some(cached) = self.check_signature_cache(&plugin_hash, plugin_path)? {
            info!("Using cached signature verification for {}", plugin_hash);
            return Ok(VerificationResult {
                is_valid: true,
                signer: Some(cached.signature.signer_key.clone()),
                timestamp: Some(SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(cached.signature.timestamp)),
                capabilities: cached.signature.metadata.capabilities.clone(),
                warnings: Vec::new(),
                errors: Vec::new(),
            });
        }
        
        // Find signature file
        let signature_path = plugin_path.with_extension("sig");
        if !signature_path.exists() {
            if self.require_signature {
                return Ok(VerificationResult {
                    is_valid: false,
                    signer: None,
                    timestamp: None,
                    capabilities: Vec::new(),
                    warnings: Vec::new(),
                    errors: vec!["No signature file found".to_string()],
                });
            } else {
                return Ok(VerificationResult {
                    is_valid: true,
                    signer: None,
                    timestamp: None,
                    capabilities: Vec::new(),
                    warnings: vec!["No signature verification performed".to_string()],
                    errors: Vec::new(),
                });
            }
        }
        
        // Load and verify signature
        let signature = self.load_signature(&signature_path)?;
        let result = self.verify_signature(&signature, &plugin_hash)?;
        
        // Cache successful verification
        if result.is_valid {
            self.cache_signature(&plugin_hash, signature, plugin_path)?;
        }
        
        Ok(result)
    }
    
    /// Calculate SHA256 hash of plugin file
    fn calculate_file_hash(&self, plugin_path: &Path) -> Result<String, ProcessingError> {
        let plugin_data = fs::read(plugin_path)
            .map_err(|e| ProcessingError::PluginLoadError(format!("Failed to read plugin: {}", e)))?;
        
        let mut hasher = Sha256::new();
        hasher.update(&plugin_data);
        let hash = hasher.finalize();
        
        Ok(hex::encode(hash))
    }
    
    /// Check signature cache for valid entry
    fn check_signature_cache(
        &self,
        plugin_hash: &str,
        plugin_path: &Path,
    ) -> Result<Option<&CachedSignature>, ProcessingError> {
        if let Some(cached) = self.signature_cache.get(plugin_hash) {
            // Check if file has been modified
            let metadata = fs::metadata(plugin_path)
                .map_err(|e| ProcessingError::PluginLoadError(format!("Failed to read file metadata: {}", e)))?;
            
            let file_size = metadata.len();
            let file_modified = metadata.modified()
                .map_err(|e| ProcessingError::PluginLoadError(format!("Failed to get modification time: {}", e)))?;
            
            if cached.file_size == file_size && cached.file_modified == file_modified {
                // Check cache expiry (24 hours)
                if cached.verified_at.elapsed().unwrap_or(std::time::Duration::MAX).as_secs() < 86400 {
                    return Ok(Some(cached));
                }
            }
        }
        
        Ok(None)
    }
    
    /// Load signature from file
    fn load_signature(&self, signature_path: &Path) -> Result<PluginSignature, ProcessingError> {
        let signature_data = fs::read_to_string(signature_path)
            .map_err(|e| ProcessingError::PluginLoadError(format!("Failed to read signature: {}", e)))?;
        
        let signature: PluginSignature = serde_json::from_str(&signature_data)
            .map_err(|e| ProcessingError::PluginLoadError(format!("Invalid signature format: {}", e)))?;
        
        Ok(signature)
    }
    
    /// Verify signature against plugin hash
    fn verify_signature(
        &self,
        signature: &PluginSignature,
        plugin_hash: &str,
    ) -> Result<VerificationResult, ProcessingError> {
        let mut result = VerificationResult {
            is_valid: false,
            signer: Some(signature.signer_key.clone()),
            timestamp: Some(SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(signature.timestamp)),
            capabilities: signature.metadata.capabilities.clone(),
            warnings: Vec::new(),
            errors: Vec::new(),
        };
        
        // Verify hash matches
        if signature.plugin_hash != plugin_hash {
            result.errors.push("Plugin hash mismatch".to_string());
            return Ok(result);
        }
        
        // Check signature age (warn if older than 30 days)
        let signature_age = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(signature.timestamp))
            .unwrap_or(std::time::Duration::ZERO);
        
        if signature_age.as_secs() > 30 * 24 * 3600 {
            result.warnings.push("Signature is older than 30 days".to_string());
        }
        
        // Find matching trusted key
        let signer_key_bytes = hex::decode(&signature.signer_key)
            .map_err(|e| ProcessingError::PluginLoadError(format!("Invalid signer key format: {}", e)))?;
        
        if signer_key_bytes.len() != 32 {
            result.errors.push("Invalid signer key length".to_string());
            return Ok(result);
        }
        
        let mut key_array = [0u8; 32];
        key_array.copy_from_slice(&signer_key_bytes);
        
        let signer_key = VerifyingKey::from_bytes(&key_array)
            .map_err(|e| ProcessingError::PluginLoadError(format!("Invalid signer key: {}", e)))?;
        
        // Check if key is trusted
        let is_trusted = self.trusted_keys.iter().any(|trusted| trusted.as_bytes() == signer_key.as_bytes());
        if !is_trusted {
            result.errors.push("Signer key not trusted".to_string());
            return Ok(result);
        }
        
        // Verify signature
        let signature_bytes = hex::decode(&signature.signature)
            .map_err(|e| ProcessingError::PluginLoadError(format!("Invalid signature format: {}", e)))?;
        
        if signature_bytes.len() != 64 {
            result.errors.push("Invalid signature length".to_string());
            return Ok(result);
        }
        
        let sig = Signature::from_slice(&signature_bytes)
            .map_err(|e| ProcessingError::PluginLoadError(format!("Invalid signature: {}", e)))?;
        
        // Create message to verify (hash + metadata)
        let message = format!("{}|{}|{}|{}", 
            plugin_hash,
            signature.timestamp,
            signature.metadata.plugin_name,
            signature.metadata.plugin_version
        );
        
        match signer_key.verify(message.as_bytes(), &sig) {
            Ok(()) => {
                result.is_valid = true;
                info!("Plugin signature verified: {}", signature.metadata.plugin_name);
            }
            Err(e) => {
                result.errors.push(format!("Signature verification failed: {}", e));
                error!("Signature verification failed for {}: {}", signature.metadata.plugin_name, e);
            }
        }
        
        Ok(result)
    }
    
    /// Cache successful signature verification
    fn cache_signature(
        &mut self,
        plugin_hash: &str,
        signature: PluginSignature,
        plugin_path: &Path,
    ) -> Result<(), ProcessingError> {
        let metadata = fs::metadata(plugin_path)
            .map_err(|e| ProcessingError::PluginLoadError(format!("Failed to read file metadata: {}", e)))?;
        
        let cached = CachedSignature {
            signature,
            verified_at: SystemTime::now(),
            file_size: metadata.len(),
            file_modified: metadata.modified()
                .map_err(|e| ProcessingError::PluginLoadError(format!("Failed to get modification time: {}", e)))?,
        };
        
        self.signature_cache.insert(plugin_hash.to_string(), cached);
        Ok(())
    }
    
    /// Clear signature cache
    pub fn clear_cache(&mut self) {
        self.signature_cache.clear();
        info!("Signature cache cleared");
    }
    
    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        let total = self.signature_cache.len();
        let expired = self.signature_cache.values()
            .filter(|cached| {
                cached.verified_at.elapsed().unwrap_or(std::time::Duration::MAX).as_secs() >= 86400
            })
            .count();
        
        (total, expired)
    }
}

/// Plugin signature generator for development
pub struct PluginSignatureGenerator {
    signing_key: ed25519_dalek::SigningKey,
}

impl PluginSignatureGenerator {
    /// Create new signature generator with random key
    pub fn new() -> Self {
        let signing_key = ed25519_dalek::SigningKey::generate(&mut rand::thread_rng());
        Self { signing_key }
    }
    
    /// Create signature generator from existing key
    pub fn from_key(key_bytes: &[u8; 32]) -> Result<Self, ProcessingError> {
        let signing_key = ed25519_dalek::SigningKey::from_bytes(key_bytes);
        Ok(Self { signing_key })
    }
    
    /// Get public key for verification
    pub fn public_key_hex(&self) -> String {
        hex::encode(self.signing_key.verifying_key().as_bytes())
    }
    
    /// Generate signature for plugin
    pub fn sign_plugin(
        &self,
        plugin_path: &Path,
        metadata: SignatureMetadata,
    ) -> Result<PluginSignature, ProcessingError> {
        // Calculate plugin hash
        let plugin_data = fs::read(plugin_path)
            .map_err(|e| ProcessingError::PluginLoadError(format!("Failed to read plugin: {}", e)))?;
        
        let mut hasher = Sha256::new();
        hasher.update(&plugin_data);
        let hash = hasher.finalize();
        let plugin_hash = hex::encode(hash);
        
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        // Create message to sign
        let message = format!("{}|{}|{}|{}", 
            plugin_hash,
            timestamp,
            metadata.plugin_name,
            metadata.plugin_version
        );
        
        // Generate signature
        let signature = self.signing_key.sign(message.as_bytes());
        
        Ok(PluginSignature {
            plugin_hash,
            signature: hex::encode(signature.to_bytes()),
            signer_key: self.public_key_hex(),
            timestamp,
            version: "1.0".to_string(),
            metadata,
        })
    }
    
    /// Save signature to file
    pub fn save_signature(
        &self,
        signature: &PluginSignature,
        signature_path: &Path,
    ) -> Result<(), ProcessingError> {
        let signature_json = serde_json::to_string_pretty(signature)
            .map_err(|e| ProcessingError::PluginLoadError(format!("Failed to serialize signature: {}", e)))?;
        
        fs::write(signature_path, signature_json)
            .map_err(|e| ProcessingError::PluginLoadError(format!("Failed to write signature: {}", e)))?;
        
        info!("Signature saved to: {:?}", signature_path);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::{TempDir, NamedTempFile};
    use std::io::Write;
    
    #[test]
    fn test_signature_generation_and_verification() {
        let temp_dir = TempDir::new().unwrap();
        let plugin_path = temp_dir.path().join("test_plugin.so");
        
        // Create test plugin file
        fs::write(&plugin_path, b"test plugin content").unwrap();
        
        // Generate signature
        let generator = PluginSignatureGenerator::new();
        let metadata = SignatureMetadata {
            plugin_name: "test_plugin".to_string(),
            plugin_version: "1.0.0".to_string(),
            author: "test".to_string(),
            build_timestamp: 0,
            capabilities: vec!["pdf".to_string()],
            dependencies: vec![],
            minimum_api_version: "1.0.0".to_string(),
        };
        
        let signature = generator.sign_plugin(&plugin_path, metadata).unwrap();
        
        // Save signature
        let signature_path = plugin_path.with_extension("sig");
        generator.save_signature(&signature, &signature_path).unwrap();
        
        // Verify signature
        let mut verifier = PluginSignatureVerifier::new(true);
        verifier.add_trusted_key(&generator.public_key_hex()).unwrap();
        
        let result = verifier.verify_plugin(&plugin_path).unwrap();
        assert!(result.is_valid);
        assert!(result.errors.is_empty());
    }
    
    #[test]
    fn test_signature_cache() {
        let temp_dir = TempDir::new().unwrap();
        let plugin_path = temp_dir.path().join("test_plugin.so");
        
        // Create test plugin file
        fs::write(&plugin_path, b"test plugin content").unwrap();
        
        // Generate and save signature
        let generator = PluginSignatureGenerator::new();
        let metadata = SignatureMetadata {
            plugin_name: "test_plugin".to_string(),
            plugin_version: "1.0.0".to_string(),
            author: "test".to_string(),
            build_timestamp: 0,
            capabilities: vec!["pdf".to_string()],
            dependencies: vec![],
            minimum_api_version: "1.0.0".to_string(),
        };
        
        let signature = generator.sign_plugin(&plugin_path, metadata).unwrap();
        let signature_path = plugin_path.with_extension("sig");
        generator.save_signature(&signature, &signature_path).unwrap();
        
        // Verify twice to test caching
        let mut verifier = PluginSignatureVerifier::new(true);
        verifier.add_trusted_key(&generator.public_key_hex()).unwrap();
        
        let result1 = verifier.verify_plugin(&plugin_path).unwrap();
        assert!(result1.is_valid);
        
        let result2 = verifier.verify_plugin(&plugin_path).unwrap();
        assert!(result2.is_valid);
        
        let (cache_size, _) = verifier.cache_stats();
        assert_eq!(cache_size, 1);
    }
    
    #[test]
    fn test_untrusted_signature() {
        let temp_dir = TempDir::new().unwrap();
        let plugin_path = temp_dir.path().join("test_plugin.so");
        
        // Create test plugin file
        fs::write(&plugin_path, b"test plugin content").unwrap();
        
        // Generate signature with one key
        let generator1 = PluginSignatureGenerator::new();
        let metadata = SignatureMetadata {
            plugin_name: "test_plugin".to_string(),
            plugin_version: "1.0.0".to_string(),
            author: "test".to_string(),
            build_timestamp: 0,
            capabilities: vec!["pdf".to_string()],
            dependencies: vec![],
            minimum_api_version: "1.0.0".to_string(),
        };
        
        let signature = generator1.sign_plugin(&plugin_path, metadata).unwrap();
        let signature_path = plugin_path.with_extension("sig");
        generator1.save_signature(&signature, &signature_path).unwrap();
        
        // Verify with different trusted key
        let generator2 = PluginSignatureGenerator::new();
        let mut verifier = PluginSignatureVerifier::new(true);
        verifier.add_trusted_key(&generator2.public_key_hex()).unwrap();
        
        let result = verifier.verify_plugin(&plugin_path).unwrap();
        assert!(!result.is_valid);
        assert!(result.errors.iter().any(|e| e.contains("not trusted")));
    }
}