//! Security analysis types for Python bindings

use pyo3::prelude::*;
use std::collections::HashMap;
use serde_json;

use neural_doc_flow_security::{
    SecurityAnalysis as RustSecurityAnalysis,
    ThreatLevel as RustThreatLevel,
    BehavioralRisk as RustBehavioralRisk,
    SecurityAction as RustSecurityAction,
};
use crate::NeuralDocFlowError;

/// Security analysis result from document scanning
/// 
/// Contains threat assessment, risk analysis, and recommended actions
/// for processed documents.
/// 
/// # Example
/// ```python
/// result = processor.process_document("document.pdf")
/// if result.security_analysis:
///     analysis = result.security_analysis
///     print(f"Threat level: {analysis.threat_level}")
///     print(f"Malware probability: {analysis.malware_probability:.2%}")
///     print(f"Recommended action: {analysis.recommended_action}")
/// ```
#[pyclass]
#[derive(Clone)]
pub struct SecurityAnalysis {
    /// Threat severity level
    #[pyo3(get)]
    pub threat_level: ThreatLevel,
    
    /// Probability of malware (0.0 to 1.0)
    #[pyo3(get)]
    pub malware_probability: f32,
    
    /// List of detected threat categories
    #[pyo3(get)]
    pub threat_categories: Vec<String>,
    
    /// Anomaly score (0.0 to 1.0)
    #[pyo3(get)]
    pub anomaly_score: f32,
    
    /// List of behavioral risk indicators
    #[pyo3(get)]
    pub behavioral_risks: Vec<BehavioralRisk>,
    
    /// Recommended security action
    #[pyo3(get)]
    pub recommended_action: SecurityAction,
}

/// Threat severity levels
#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThreatLevel {
    Safe,
    Low,
    Medium,
    High,
    Critical,
}

/// Behavioral risk indicator
#[pyclass]
#[derive(Clone)]
pub struct BehavioralRisk {
    /// Type of risk detected
    #[pyo3(get)]
    pub risk_type: String,
    
    /// Risk severity (0.0 to 1.0)
    #[pyo3(get)]
    pub severity: f32,
    
    /// Human-readable description
    #[pyo3(get)]
    pub description: String,
}

/// Recommended security action
#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SecurityAction {
    Allow,
    Sanitize,
    Quarantine,
    Block,
}

#[pymethods]
impl SecurityAnalysis {
    /// Create a new security analysis
    #[new]
    pub fn new() -> Self {
        Self {
            threat_level: ThreatLevel::Safe,
            malware_probability: 0.0,
            threat_categories: Vec::new(),
            anomaly_score: 0.0,
            behavioral_risks: Vec::new(),
            recommended_action: SecurityAction::Allow,
        }
    }
    
    /// Check if the document is considered safe
    /// 
    /// # Returns
    /// True if threat level is Safe or Low and recommended action is Allow
    #[getter]
    pub fn is_safe(&self) -> bool {
        matches!(self.threat_level, ThreatLevel::Safe | ThreatLevel::Low) &&
        matches!(self.recommended_action, SecurityAction::Allow)
    }
    
    /// Check if the document requires attention
    /// 
    /// # Returns
    /// True if threat level is Medium or higher
    #[getter]
    pub fn requires_attention(&self) -> bool {
        matches!(self.threat_level, ThreatLevel::Medium | ThreatLevel::High | ThreatLevel::Critical)
    }
    
    /// Get a summary of the security analysis
    /// 
    /// # Returns
    /// Dictionary containing key security metrics
    #[getter]
    pub fn summary(&self) -> HashMap<String, serde_json::Value> {
        let mut summary = HashMap::new();
        
        summary.insert("threat_level".to_string(), 
                      serde_json::Value::String(format!("{:?}", self.threat_level)));
        summary.insert("malware_probability".to_string(),
                      serde_json::Value::Number(serde_json::Number::from_f64(self.malware_probability as f64).unwrap()));
        summary.insert("anomaly_score".to_string(),
                      serde_json::Value::Number(serde_json::Number::from_f64(self.anomaly_score as f64).unwrap()));
        summary.insert("threat_count".to_string(),
                      serde_json::Value::Number(serde_json::Number::from(self.threat_categories.len())));
        summary.insert("risk_count".to_string(),
                      serde_json::Value::Number(serde_json::Number::from(self.behavioral_risks.len())));
        summary.insert("recommended_action".to_string(),
                      serde_json::Value::String(format!("{:?}", self.recommended_action)));
        summary.insert("is_safe".to_string(),
                      serde_json::Value::Bool(self.is_safe()));
        summary.insert("requires_attention".to_string(),
                      serde_json::Value::Bool(self.requires_attention()));
        
        summary
    }
    
    /// Get high-severity risks only
    /// 
    /// # Arguments
    /// * `min_severity` - Minimum severity threshold (0.0 to 1.0)
    /// 
    /// # Returns
    /// List of risks above the severity threshold
    #[pyo3(signature = (min_severity = 0.7))]
    pub fn get_high_severity_risks(&self, min_severity: f32) -> Vec<BehavioralRisk> {
        self.behavioral_risks
            .iter()
            .filter(|risk| risk.severity >= min_severity)
            .cloned()
            .collect()
    }
    
    /// Convert to JSON string
    /// 
    /// # Returns
    /// JSON representation of the security analysis
    pub fn to_json(&self) -> PyResult<String> {
        let json_value = self.to_json_value()?;
        serde_json::to_string_pretty(&json_value)
            .map_err(|e| NeuralDocFlowError::new_err(format!("Failed to serialize security analysis: {}", e)))
    }
    
    /// String representation of the security analysis
    fn __repr__(&self) -> String {
        format!(
            "SecurityAnalysis(threat_level={:?}, malware_probability={:.2}, action={:?})",
            self.threat_level,
            self.malware_probability,
            self.recommended_action
        )
    }
}

#[pymethods]
impl ThreatLevel {
    /// Get the numeric value of the threat level
    /// 
    /// # Returns
    /// Integer value: Safe=0, Low=1, Medium=2, High=3, Critical=4
    #[getter]
    pub fn value(&self) -> u8 {
        match self {
            ThreatLevel::Safe => 0,
            ThreatLevel::Low => 1,
            ThreatLevel::Medium => 2,
            ThreatLevel::High => 3,
            ThreatLevel::Critical => 4,
        }
    }
    
    /// Get the color code for UI display
    /// 
    /// # Returns
    /// Color name suitable for UI styling
    #[getter]
    pub fn color(&self) -> &str {
        match self {
            ThreatLevel::Safe => "green",
            ThreatLevel::Low => "yellow",
            ThreatLevel::Medium => "orange",
            ThreatLevel::High => "red",
            ThreatLevel::Critical => "darkred",
        }
    }
    
    /// String representation of threat level
    fn __repr__(&self) -> String {
        format!("ThreatLevel.{:?}", self)
    }
    
    /// Compare threat levels
    fn __richcmp__(&self, other: &Self, op: pyo3::basic::CompareOp) -> bool {
        use pyo3::basic::CompareOp;
        match op {
            CompareOp::Eq => self == other,
            CompareOp::Ne => self != other,
            CompareOp::Lt => self.value() < other.value(),
            CompareOp::Le => self.value() <= other.value(),
            CompareOp::Gt => self.value() > other.value(),
            CompareOp::Ge => self.value() >= other.value(),
        }
    }
}

#[pymethods]
impl BehavioralRisk {
    /// Create a new behavioral risk
    #[new]
    pub fn new(risk_type: String, severity: f32, description: String) -> Self {
        Self {
            risk_type,
            severity: severity.max(0.0).min(1.0), // Clamp to [0.0, 1.0]
            description,
        }
    }
    
    /// Check if this is a high-severity risk
    /// 
    /// # Returns
    /// True if severity >= 0.7
    #[getter]
    pub fn is_high_severity(&self) -> bool {
        self.severity >= 0.7
    }
    
    /// Get the severity category
    /// 
    /// # Returns
    /// String category: "low", "medium", "high"
    #[getter]
    pub fn severity_category(&self) -> &str {
        if self.severity >= 0.7 {
            "high"
        } else if self.severity >= 0.4 {
            "medium"
        } else {
            "low"
        }
    }
    
    /// String representation of behavioral risk
    fn __repr__(&self) -> String {
        format!(
            "BehavioralRisk(type='{}', severity={:.2}, category='{}')",
            self.risk_type,
            self.severity,
            self.severity_category()
        )
    }
}

#[pymethods]
impl SecurityAction {
    /// Check if action allows processing
    /// 
    /// # Returns
    /// True if action is Allow or Sanitize
    #[getter]
    pub fn allows_processing(&self) -> bool {
        matches!(self, SecurityAction::Allow | SecurityAction::Sanitize)
    }
    
    /// Check if action blocks processing
    /// 
    /// # Returns
    /// True if action is Block or Quarantine
    #[getter]
    pub fn blocks_processing(&self) -> bool {
        matches!(self, SecurityAction::Block | SecurityAction::Quarantine)
    }
    
    /// Get the action priority
    /// 
    /// # Returns
    /// Integer priority: Allow=0, Sanitize=1, Quarantine=2, Block=3
    #[getter]
    pub fn priority(&self) -> u8 {
        match self {
            SecurityAction::Allow => 0,
            SecurityAction::Sanitize => 1,
            SecurityAction::Quarantine => 2,
            SecurityAction::Block => 3,
        }
    }
    
    /// String representation of security action
    fn __repr__(&self) -> String {
        format!("SecurityAction.{:?}", self)
    }
}

impl SecurityAnalysis {
    /// Convert from Rust security analysis
    pub fn from_rust(analysis: RustSecurityAnalysis) -> Self {
        Self {
            threat_level: ThreatLevel::from_rust(analysis.threat_level),
            malware_probability: analysis.malware_probability,
            threat_categories: analysis.threat_categories.into_iter()
                .map(|cat| format!("{:?}", cat))
                .collect(),
            anomaly_score: analysis.anomaly_score,
            behavioral_risks: analysis.behavioral_risks.into_iter()
                .map(BehavioralRisk::from_rust)
                .collect(),
            recommended_action: SecurityAction::from_rust(analysis.recommended_action),
        }
    }
    
    /// Convert to JSON value for serialization
    pub fn to_json_value(&self) -> PyResult<serde_json::Value> {
        let mut analysis_map = serde_json::Map::new();
        
        analysis_map.insert("threat_level".to_string(), 
                           serde_json::Value::String(format!("{:?}", self.threat_level)));
        analysis_map.insert("malware_probability".to_string(),
                           serde_json::Value::Number(serde_json::Number::from_f64(self.malware_probability as f64)
                               .ok_or_else(|| NeuralDocFlowError::new_err("Invalid malware probability"))?));
        analysis_map.insert("threat_categories".to_string(),
                           serde_json::Value::Array(self.threat_categories.iter()
                               .map(|cat| serde_json::Value::String(cat.clone()))
                               .collect()));
        analysis_map.insert("anomaly_score".to_string(),
                           serde_json::Value::Number(serde_json::Number::from_f64(self.anomaly_score as f64)
                               .ok_or_else(|| NeuralDocFlowError::new_err("Invalid anomaly score"))?));
        analysis_map.insert("behavioral_risks".to_string(),
                           serde_json::Value::Array(self.behavioral_risks.iter()
                               .map(|risk| {
                                   let mut risk_map = serde_json::Map::new();
                                   risk_map.insert("risk_type".to_string(), serde_json::Value::String(risk.risk_type.clone()));
                                   risk_map.insert("severity".to_string(), 
                                                  serde_json::Value::Number(serde_json::Number::from_f64(risk.severity as f64).unwrap()));
                                   risk_map.insert("description".to_string(), serde_json::Value::String(risk.description.clone()));
                                   serde_json::Value::Object(risk_map)
                               })
                               .collect()));
        analysis_map.insert("recommended_action".to_string(),
                           serde_json::Value::String(format!("{:?}", self.recommended_action)));
        
        Ok(serde_json::Value::Object(analysis_map))
    }
}

impl ThreatLevel {
    /// Convert from Rust threat level
    pub fn from_rust(level: RustThreatLevel) -> Self {
        match level {
            RustThreatLevel::Safe => ThreatLevel::Safe,
            RustThreatLevel::Low => ThreatLevel::Low,
            RustThreatLevel::Medium => ThreatLevel::Medium,
            RustThreatLevel::High => ThreatLevel::High,
            RustThreatLevel::Critical => ThreatLevel::Critical,
        }
    }
}

impl BehavioralRisk {
    /// Convert from Rust behavioral risk
    pub fn from_rust(risk: RustBehavioralRisk) -> Self {
        Self {
            risk_type: risk.risk_type,
            severity: risk.severity,
            description: risk.description,
        }
    }
}

impl SecurityAction {
    /// Convert from Rust security action
    pub fn from_rust(action: RustSecurityAction) -> Self {
        match action {
            RustSecurityAction::Allow => SecurityAction::Allow,
            RustSecurityAction::Sanitize => SecurityAction::Sanitize,
            RustSecurityAction::Quarantine => SecurityAction::Quarantine,
            RustSecurityAction::Block => SecurityAction::Block,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_security_analysis_creation() {
        let analysis = SecurityAnalysis::new();
        assert_eq!(analysis.threat_level, ThreatLevel::Safe);
        assert_eq!(analysis.malware_probability, 0.0);
        assert!(analysis.is_safe());
        assert!(!analysis.requires_attention());
    }
    
    #[test]
    fn test_threat_level_comparison() {
        assert!(ThreatLevel::Safe.value() < ThreatLevel::Critical.value());
        assert_eq!(ThreatLevel::Medium.value(), 2);
        assert_eq!(ThreatLevel::Safe.color(), "green");
        assert_eq!(ThreatLevel::Critical.color(), "darkred");
    }
    
    #[test]
    fn test_behavioral_risk() {
        let risk = BehavioralRisk::new(
            "test_risk".to_string(),
            0.8,
            "Test description".to_string()
        );
        assert!(risk.is_high_severity());
        assert_eq!(risk.severity_category(), "high");
    }
    
    #[test]
    fn test_security_action() {
        assert!(SecurityAction::Allow.allows_processing());
        assert!(SecurityAction::Sanitize.allows_processing());
        assert!(SecurityAction::Block.blocks_processing());
        assert!(SecurityAction::Quarantine.blocks_processing());
        
        assert_eq!(SecurityAction::Allow.priority(), 0);
        assert_eq!(SecurityAction::Block.priority(), 3);
    }
}