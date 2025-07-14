//! Enhanced document processing engine with integrated security for Phase 3

use crate::{
    config::{NeuralDocFlowConfig, SecurityScanMode, SecurityAction},
    document::{Document, DocumentBuilder},
    error::ProcessingError,
};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{info, warn, error};

use async_trait::async_trait;

/// Security analysis result
#[derive(Debug, Clone)]
pub struct SecurityAnalysis {
    pub threat_level: ThreatLevel,
    pub malware_probability: f32,
    pub threat_categories: Vec<String>,
    pub anomaly_score: f32,
    pub behavioral_risks: Vec<String>,
    pub recommended_action: SecurityAction,
}

/// Threat severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreatLevel {
    Safe,
    Low,
    Medium,
    High,
    Critical,
}

/// Security processor trait for pluggable security implementations
#[async_trait]
pub trait SecurityProcessor: Send + Sync {
    /// Perform security scan on a document
    async fn scan(&mut self, document: &Document) -> Result<SecurityAnalysis, ProcessingError>;
}

/// Enhanced document processing engine with security features
pub struct DocumentEngine {
    // Configuration
    config: NeuralDocFlowConfig,
    
    // Security processor (pluggable)
    security_processor: Option<Arc<RwLock<dyn SecurityProcessor>>>,
    
    // Performance monitor
    performance_monitor: Arc<PerformanceMonitor>,
}

impl DocumentEngine {
    /// Create a new document engine with Phase 3 security integration
    pub fn new(config: NeuralDocFlowConfig) -> Result<Self, ProcessingError> {
        info!("Initializing Phase 3 Document Engine with Security Integration");
        
        let performance_monitor = Arc::new(PerformanceMonitor::new());
        
        Ok(Self {
            config,
            security_processor: None, // Will be set via set_security_processor
            performance_monitor,
        })
    }
    
    /// Set the security processor (pluggable security implementation)
    pub fn set_security_processor(&mut self, processor: Arc<RwLock<dyn SecurityProcessor>>) {
        info!("Security processor attached to engine");
        self.security_processor = Some(processor);
    }
    
    /// Process document with integrated security scanning
    pub async fn process(
        &self,
        input: Vec<u8>,
        mime_type: &str,
    ) -> Result<Document, ProcessingError> {
        self.process_with_options(input, mime_type, None).await
    }
    
    /// Process document with custom source name
    pub async fn process_with_source(
        &self,
        input: Vec<u8>,
        mime_type: &str,
        source: &str,
    ) -> Result<Document, ProcessingError> {
        self.process_with_options(input, mime_type, Some(source)).await
    }
    
    /// Internal process method with full security integration
    async fn process_with_options(
        &self,
        input: Vec<u8>,
        mime_type: &str,
        source: Option<&str>,
    ) -> Result<Document, ProcessingError> {
        let start = Instant::now();
        let doc_id = uuid::Uuid::new_v4().to_string();
        
        info!("Starting document processing with security integration for {}", doc_id);
        
        // Step 1: Validate input according to security policies
        self.validate_input(&input, mime_type)?;
        
        // Step 2: Create initial document
        let mut document = DocumentBuilder::new()
            .mime_type(mime_type)
            .source(source.unwrap_or("phase3_engine"))
            .size(input.len() as u64)
            .build();
        
        // Store raw content for security scanning
        document.raw_content = input;
        
        // Step 3: Perform security scanning (if enabled)
        let security_analysis = self.perform_security_scan(&document).await?;
        
        // Step 4: Apply security action based on analysis
        self.apply_security_action(&security_analysis, &mut document).await?;
        
        // Step 5: Process document content (if allowed)
        let should_process = match &security_analysis {
            Some(analysis) => matches!(analysis.recommended_action, 
                                     SecurityAction::Allow | SecurityAction::Sanitize),
            None => true, // Process if no security analysis
        };
        
        if should_process {
            self.process_document_content(&mut document).await?;
        }
        
        // Step 6: Record performance metrics
        let processing_time = start.elapsed();
        self.performance_monitor.record_processing(
            &doc_id,
            processing_time,
            document.raw_content.len(),
        ).await;
        
        info!("Document processing complete in {:?} for {}", processing_time, doc_id);
        
        Ok(document)
    }
    
    /// Validate input according to security policies
    fn validate_input(&self, input: &[u8], mime_type: &str) -> Result<(), ProcessingError> {
        // Check file size limits
        let file_size_mb = input.len() as f64 / (1024.0 * 1024.0);
        if file_size_mb > self.config.security.policies.max_file_size_mb as f64 {
            return Err(ProcessingError::SecurityViolation(format!(
                "File size {:.2}MB exceeds maximum allowed size {}MB",
                file_size_mb, self.config.security.policies.max_file_size_mb
            )));
        }
        
        // Check blocked file types
        if self.config.security.policies.blocked_file_types.contains(&mime_type.to_string()) {
            return Err(ProcessingError::SecurityViolation(format!(
                "File type {} is blocked by security policy", mime_type
            )));
        }
        
        // Check allowed file types (if allowlist is configured)
        if !self.config.security.policies.allowed_file_types.is_empty() 
           && !self.config.security.policies.allowed_file_types.contains(&mime_type.to_string()) {
            return Err(ProcessingError::SecurityViolation(format!(
                "File type {} is not in the allowed types list", mime_type
            )));
        }
        
        Ok(())
    }
    
    /// Perform security scanning if enabled
    async fn perform_security_scan(&self, document: &Document) -> Result<Option<SecurityAnalysis>, ProcessingError> {
        if let Some(ref security_processor) = self.security_processor {
            match self.config.security.scan_mode {
                SecurityScanMode::Disabled => {
                    info!("Security scanning disabled");
                    return Ok(None);
                },
                SecurityScanMode::Basic | SecurityScanMode::Standard | SecurityScanMode::Comprehensive => {
                    info!("Performing security scan in {:?} mode", self.config.security.scan_mode);
                    
                    let mut processor = security_processor.write().await;
                    match processor.scan(document).await {
                        Ok(analysis) => {
                            info!("Security scan completed - Threat level: {:?}, Action: {:?}", 
                                 analysis.threat_level, analysis.recommended_action);
                            Ok(Some(analysis))
                        },
                        Err(e) => {
                            error!("Security scan failed: {}", e);
                            // Continue processing but log the error
                            warn!("Proceeding without security scan due to error");
                            Ok(None)
                        }
                    }
                },
                SecurityScanMode::Custom(_) => {
                    warn!("Custom security scan mode not yet implemented");
                    Ok(None)
                }
            }
        } else if self.config.security.enabled {
            info!("Security enabled but no processor available - performing basic validation only");
            // Perform basic built-in security checks
            Ok(Some(self.basic_security_analysis(document).await?))
        } else {
            info!("Security scanning disabled");
            Ok(None)
        }
    }
    
    /// Basic security analysis when no external processor is available
    async fn basic_security_analysis(&self, document: &Document) -> Result<SecurityAnalysis, ProcessingError> {
        let mut threat_categories = Vec::new();
        let mut behavioral_risks = Vec::new();
        let mut anomaly_score = 0.0;
        let mut malware_probability = 0.0;
        
        // Basic content analysis - check both raw content and processed text
        let content_to_check = if let Some(text) = &document.content.text {
            text.clone()
        } else if !document.raw_content.is_empty() {
            // Try to convert raw content to string for analysis
            String::from_utf8_lossy(&document.raw_content).to_string()
        } else {
            String::new()
        };
        
        if !content_to_check.is_empty() {
            // Check for script content
            if content_to_check.contains("<script") || content_to_check.contains("javascript:") {
                threat_categories.push("Script Content".to_string());
                behavioral_risks.push("JavaScript execution risk".to_string());
                malware_probability += 0.3;
            }
            
            // Check for suspicious patterns
            let suspicious_patterns = ["eval(", "exec(", "system(", "shell_exec"];
            for pattern in &suspicious_patterns {
                if content_to_check.contains(pattern) {
                    threat_categories.push("Suspicious Function Call".to_string());
                    behavioral_risks.push(format!("Contains {}", pattern));
                    malware_probability += 0.2;
                }
            }
            
            // Check entropy (very basic)
            if content_to_check.len() > 100 {
                let unique_chars = content_to_check.chars().collect::<std::collections::HashSet<_>>().len();
                let entropy = unique_chars as f32 / content_to_check.len() as f32;
                if entropy < 0.01 {
                    anomaly_score += 0.3; // Very low entropy might indicate encoded content
                }
            }
        }
        
        // Check file size anomalies
        let file_size = document.raw_content.len();
        if file_size > 50_000_000 { // 50MB
            anomaly_score += 0.2;
            behavioral_risks.push("Unusually large file size".to_string());
        }
        
        // Determine threat level
        let threat_level = if malware_probability >= 0.8 || anomaly_score >= 0.8 {
            ThreatLevel::High
        } else if malware_probability >= 0.5 || anomaly_score >= 0.5 {
            ThreatLevel::Medium
        } else if malware_probability > 0.0 || anomaly_score > 0.0 || !threat_categories.is_empty() {
            ThreatLevel::Low
        } else {
            ThreatLevel::Safe
        };
        
        // Determine recommended action - be more aggressive with security
        let recommended_action = match threat_level {
            ThreatLevel::Safe => SecurityAction::Allow,
            ThreatLevel::Low => {
                // If we found scripts or suspicious patterns, sanitize even at low level
                if threat_categories.iter().any(|cat| cat.contains("Script") || cat.contains("Function")) {
                    SecurityAction::Sanitize
                } else {
                    SecurityAction::Allow
                }
            },
            ThreatLevel::Medium => SecurityAction::Sanitize,
            ThreatLevel::High | ThreatLevel::Critical => SecurityAction::Quarantine,
        };
        
        // Debug logging
        info!("Security analysis complete - Threat level: {:?}, Categories: {:?}, Action: {:?}", 
              threat_level, threat_categories, recommended_action);
        
        Ok(SecurityAnalysis {
            threat_level,
            malware_probability,
            threat_categories,
            anomaly_score,
            behavioral_risks,
            recommended_action,
        })
    }
    
    /// Apply security action based on analysis
    async fn apply_security_action(
        &self,
        analysis: &Option<SecurityAnalysis>,
        document: &mut Document,
    ) -> Result<(), ProcessingError> {
        if let Some(analysis) = analysis {
            match analysis.recommended_action {
                SecurityAction::Allow => {
                    info!("Security analysis: ALLOW - proceeding with processing");
                },
                SecurityAction::Sanitize => {
                    info!("Security analysis: SANITIZE - cleaning document content");
                    self.sanitize_document(document).await?;
                },
                SecurityAction::Quarantine => {
                    warn!("Security analysis: QUARANTINE - document flagged for review");
                    // Add quarantine metadata
                    document.metadata.custom.insert(
                        "security_status".to_string(),
                        serde_json::json!("quarantined")
                    );
                    document.metadata.custom.insert(
                        "quarantine_reason".to_string(),
                        serde_json::json!(format!("Threat level: {:?}", analysis.threat_level))
                    );
                },
                SecurityAction::Block => {
                    error!("Security analysis: BLOCK - refusing to process document");
                    return Err(ProcessingError::SecurityViolation(format!(
                        "Document blocked due to security threat: {:?}", analysis.threat_level
                    )));
                },
                SecurityAction::Custom(ref action) => {
                    warn!("Custom security action not implemented: {}", action);
                }
            }
        }
        
        Ok(())
    }
    
    /// Sanitize document content to remove potential threats
    async fn sanitize_document(&self, document: &mut Document) -> Result<(), ProcessingError> {
        info!("Sanitizing document content");
        
        // Remove potentially dangerous content from text
        if let Some(ref mut text) = document.content.text {
            // Remove script tags and javascript
            *text = text.replace("<script", "&lt;script")
                       .replace("javascript:", "")
                       .replace("vbscript:", "")
                       .replace("data:", "");
        }
        
        // Clear suspicious metadata
        document.metadata.custom.retain(|key, _| {
            !key.to_lowercase().contains("script") && 
            !key.to_lowercase().contains("exec") &&
            !key.to_lowercase().contains("eval")
        });
        
        // Add sanitization marker
        document.metadata.custom.insert(
            "security_sanitized".to_string(),
            serde_json::json!(true)
        );
        
        Ok(())
    }
    
    /// Process document content (placeholder for actual processing logic)
    async fn process_document_content(&self, document: &mut Document) -> Result<(), ProcessingError> {
        info!("Processing document content");
        
        // Placeholder for actual document processing logic
        // In a full implementation, this would:
        // 1. Extract text from various formats
        // 2. Perform neural enhancement
        // 3. Extract tables and images
        // 4. Structure the content
        
        // For now, just add some basic extracted text if empty
        if document.content.text.is_none() && !document.raw_content.is_empty() {
            // Try to extract basic text (very simplified)
            if let Ok(text) = String::from_utf8(document.raw_content.clone()) {
                if !text.trim().is_empty() {
                    document.content.text = Some(text);
                }
            }
        }
        
        Ok(())
    }
}

/// Performance monitoring (placeholder)
pub struct PerformanceMonitor {
    metrics: Arc<RwLock<PerformanceMetrics>>,
}

#[derive(Default)]
struct PerformanceMetrics {
    total_documents: u64,
    total_processing_time: std::time::Duration,
    total_bytes_processed: usize,
    #[allow(dead_code)] // Used in future error tracking implementations
    error_count: u64,
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(PerformanceMetrics::default())),
        }
    }
    
    pub async fn record_processing(
        &self,
        _doc_id: &str,
        processing_time: std::time::Duration,
        bytes_processed: usize,
    ) {
        let mut metrics = self.metrics.write().await;
        metrics.total_documents += 1;
        metrics.total_processing_time += processing_time;
        metrics.total_bytes_processed += bytes_processed;
        
        // Calculate and log performance stats
        let avg_time = metrics.total_processing_time / metrics.total_documents as u32;
        info!(
            "Performance stats - Docs: {}, Avg time: {:?}, Total bytes: {}",
            metrics.total_documents, avg_time, metrics.total_bytes_processed
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_engine_creation() {
        let config = NeuralDocFlowConfig::default();
        let engine = DocumentEngine::new(config);
        assert!(engine.is_ok());
    }
}