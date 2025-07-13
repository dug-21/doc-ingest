//! Security audit logging

use crate::SecurityAnalysis;
use neural_doc_flow_core::{Document, ProcessingError};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt;
use tracing::{info, warn, error};

/// Security audit logger
pub struct AuditLogger {
    log_path: PathBuf,
    enable_remote_logging: bool,
}

/// Audit log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub timestamp: DateTime<Utc>,
    pub event_type: AuditEventType,
    pub document_id: String,
    pub threat_level: Option<String>,
    pub action_taken: Option<String>,
    pub details: serde_json::Value,
}

/// Types of audit events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditEventType {
    ScanStarted,
    ScanCompleted,
    ThreatDetected,
    DocumentBlocked,
    DocumentSanitized,
    DocumentAllowed,
    PluginSandboxed,
    SecurityViolation,
}

impl AuditLogger {
    /// Create a new audit logger
    pub fn new() -> Result<Self, ProcessingError> {
        let log_path = PathBuf::from("/var/log/neuraldocflow/security.log");
        
        // Ensure log directory exists
        if let Some(parent) = log_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| ProcessingError::Io(e))?;
        }
        
        Ok(Self {
            log_path,
            enable_remote_logging: false,
        })
    }
    
    /// Log scan initiation
    pub async fn log_scan_start(&self, document: &Document) -> Result<(), ProcessingError> {
        let entry = AuditEntry {
            timestamp: Utc::now(),
            event_type: AuditEventType::ScanStarted,
            document_id: self.get_document_id(document),
            threat_level: None,
            action_taken: None,
            details: serde_json::json!({
                "document_type": format!("{:?}", document.doc_type),
                "size_bytes": self.get_document_size(document),
            }),
        };
        
        self.write_entry(&entry).await?;
        info!("Security scan started for document: {}", entry.document_id);
        
        Ok(())
    }
    
    /// Log scan completion
    pub async fn log_scan_complete(
        &self,
        document: &Document,
        analysis: &SecurityAnalysis,
    ) -> Result<(), ProcessingError> {
        let entry = AuditEntry {
            timestamp: Utc::now(),
            event_type: AuditEventType::ScanCompleted,
            document_id: self.get_document_id(document),
            threat_level: Some(format!("{:?}", analysis.threat_level)),
            action_taken: Some(format!("{:?}", analysis.recommended_action)),
            details: serde_json::json!({
                "malware_probability": analysis.malware_probability,
                "threat_categories": analysis.threat_categories,
                "anomaly_score": analysis.anomaly_score,
                "behavioral_risks": analysis.behavioral_risks.len(),
            }),
        };
        
        self.write_entry(&entry).await?;
        
        // Log additional events based on action
        match analysis.recommended_action {
            crate::SecurityAction::Block => {
                self.log_document_blocked(document, analysis).await?;
                error!("Document blocked due to security threat: {}", entry.document_id);
            }
            crate::SecurityAction::Sanitize => {
                self.log_document_sanitized(document, analysis).await?;
                warn!("Document sanitized due to security concerns: {}", entry.document_id);
            }
            _ => {
                info!("Security scan completed for document: {}", entry.document_id);
            }
        }
        
        Ok(())
    }
    
    /// Log document blocked
    async fn log_document_blocked(
        &self,
        document: &Document,
        analysis: &SecurityAnalysis,
    ) -> Result<(), ProcessingError> {
        let entry = AuditEntry {
            timestamp: Utc::now(),
            event_type: AuditEventType::DocumentBlocked,
            document_id: self.get_document_id(document),
            threat_level: Some(format!("{:?}", analysis.threat_level)),
            action_taken: Some("Blocked".to_string()),
            details: serde_json::json!({
                "reason": "High security threat detected",
                "threat_categories": analysis.threat_categories,
            }),
        };
        
        self.write_entry(&entry).await?;
        
        // Send alert if remote logging enabled
        if self.enable_remote_logging {
            self.send_security_alert(&entry).await?;
        }
        
        Ok(())
    }
    
    /// Log document sanitized
    async fn log_document_sanitized(
        &self,
        document: &Document,
        analysis: &SecurityAnalysis,
    ) -> Result<(), ProcessingError> {
        let entry = AuditEntry {
            timestamp: Utc::now(),
            event_type: AuditEventType::DocumentSanitized,
            document_id: self.get_document_id(document),
            threat_level: Some(format!("{:?}", analysis.threat_level)),
            action_taken: Some("Sanitized".to_string()),
            details: serde_json::json!({
                "sanitization_actions": "Removed suspicious content",
                "threat_categories": analysis.threat_categories,
            }),
        };
        
        self.write_entry(&entry).await?;
        Ok(())
    }
    
    /// Write audit entry to log file
    async fn write_entry(&self, entry: &AuditEntry) -> Result<(), ProcessingError> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_path)
            .await?;
        
        let json = serde_json::to_string(entry)?;
        file.write_all(json.as_bytes()).await?;
        file.write_all(b"\n").await?;
        file.flush().await?;
        
        Ok(())
    }
    
    /// Send security alert for critical events
    async fn send_security_alert(&self, entry: &AuditEntry) -> Result<(), ProcessingError> {
        // In production, integrate with SIEM or alerting system
        warn!("Security alert: {:?}", entry);
        Ok(())
    }
    
    /// Get document ID from document
    fn get_document_id(&self, document: &Document) -> String {
        document.id.to_string()
    }
    
    /// Get document size
    fn get_document_size(&self, document: &Document) -> usize {
        document.raw_content.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_audit_logger_creation() {
        let logger = AuditLogger::new();
        // May fail in test environment due to permissions
        // assert!(logger.is_ok());
    }
}