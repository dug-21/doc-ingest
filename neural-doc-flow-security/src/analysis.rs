//! Threat analysis and categorization

use crate::{SecurityFeatures, ThreatCategory};
use neural_doc_flow_core::ProcessingError;
use serde::{Deserialize, Serialize};
use regex::Regex;
use aho_corasick::AhoCorasick;

/// Threat analyzer for document security
pub struct ThreatAnalyzer {
    exploit_patterns: Vec<ExploitPattern>,
    keyword_matcher: AhoCorasick,
    js_analyzer: JavaScriptAnalyzer,
}

/// Threat analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatResult {
    pub categories: Vec<ThreatCategory>,
    pub severity_scores: Vec<f32>,
}

/// Exploit pattern definition
struct ExploitPattern {
    name: String,
    pattern: Regex,
    category: ThreatCategory,
    severity: f32,
}

/// JavaScript threat analyzer
struct JavaScriptAnalyzer {
    dangerous_functions: Vec<String>,
    obfuscation_patterns: Vec<Regex>,
}

impl ThreatAnalyzer {
    /// Create a new threat analyzer
    pub fn new() -> Result<Self, ProcessingError> {
        // Initialize exploit patterns
        let exploit_patterns = vec![
            ExploitPattern {
                name: "eval_exploit".to_string(),
                pattern: Regex::new(r"eval\s*\(")
                    .map_err(|e| ProcessingError::Regex(e.to_string()))?,
                category: ThreatCategory::JavaScriptExploit,
                severity: 0.8,
            },
            ExploitPattern {
                name: "embedded_exe".to_string(),
                pattern: Regex::new(r"(?i)(\.exe|\.dll|\.scr)")
                    .map_err(|e| ProcessingError::Regex(e.to_string()))?,
                category: ThreatCategory::EmbeddedExecutable,
                severity: 0.9,
            },
        ];
        
        // Initialize keyword matcher
        let keywords = vec![
            "eval", "exec", "system", "shell", "powershell",
            "cmd.exe", "subprocess", "os.system", "Runtime.exec",
        ];
        let keyword_matcher = AhoCorasick::new(&keywords)
            .map_err(|e| ProcessingError::AhoCorasick(e.to_string()))?;
        
        // Initialize JavaScript analyzer
        let js_analyzer = JavaScriptAnalyzer {
            dangerous_functions: vec![
                "eval".to_string(),
                "Function".to_string(),
                "setTimeout".to_string(),
                "setInterval".to_string(),
            ],
            obfuscation_patterns: vec![
                Regex::new(r"\\x[0-9a-fA-F]{2}")
                    .map_err(|e| ProcessingError::Regex(e.to_string()))?,
                Regex::new(r"\\u[0-9a-fA-F]{4}")
                    .map_err(|e| ProcessingError::Regex(e.to_string()))?,
                Regex::new(r"String\.fromCharCode")
                    .map_err(|e| ProcessingError::Regex(e.to_string()))?,
            ],
        };
        
        Ok(Self {
            exploit_patterns,
            keyword_matcher,
            js_analyzer,
        })
    }
    
    /// Analyze document for threats
    pub async fn analyze(&self, features: &SecurityFeatures) -> Result<ThreatResult, ProcessingError> {
        let mut categories = Vec::new();
        let mut severity_scores = Vec::new();
        
        // Check for JavaScript threats
        if features.javascript_present {
            categories.push(ThreatCategory::JavaScriptExploit);
            severity_scores.push(0.7);
        }
        
        // Check for suspicious keywords
        let keyword_count = self.count_suspicious_keywords(features);
        if keyword_count > 5 {
            categories.push(ThreatCategory::ExploitPattern);
            severity_scores.push(0.6 + (keyword_count as f32 * 0.05).min(0.3));
        }
        
        // Check for obfuscation
        if features.obfuscation_score > 0.7 {
            categories.push(ThreatCategory::ObfuscatedContent);
            severity_scores.push(features.obfuscation_score);
        }
        
        // Check for embedded files
        for file in &features.embedded_files {
            if self.is_suspicious_file_type(&file.file_type) {
                categories.push(ThreatCategory::EmbeddedExecutable);
                severity_scores.push(0.9);
                break;
            }
        }
        
        // Check for anomalies based on entropy
        if features.header_entropy > 7.5 {
            categories.push(ThreatCategory::ZeroDayAnomaly);
            severity_scores.push(0.8);
        }
        
        Ok(ThreatResult {
            categories,
            severity_scores,
        })
    }
    
    /// Count suspicious keywords in features
    fn count_suspicious_keywords(&self, features: &SecurityFeatures) -> usize {
        let mut count = 0;
        
        for keyword in &features.suspicious_keywords {
            if self.keyword_matcher.find(keyword).is_some() {
                count += 1;
            }
        }
        
        count
    }
    
    /// Check if file type is suspicious
    fn is_suspicious_file_type(&self, file_type: &str) -> bool {
        let suspicious_types = ["exe", "dll", "scr", "bat", "cmd", "vbs", "js"];
        suspicious_types.iter().any(|&t| file_type.ends_with(t))
    }
}

impl JavaScriptAnalyzer {
    /// Analyze JavaScript code for threats
    pub fn analyze_javascript(&self, code: &str) -> Vec<ThreatCategory> {
        let mut threats = Vec::new();
        
        // Check for dangerous functions
        for func in &self.dangerous_functions {
            if code.contains(func) {
                threats.push(ThreatCategory::JavaScriptExploit);
                break;
            }
        }
        
        // Check for obfuscation
        for pattern in &self.obfuscation_patterns {
            if pattern.is_match(code) {
                threats.push(ThreatCategory::ObfuscatedContent);
                break;
            }
        }
        
        threats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_threat_analyzer_creation() {
        let analyzer = ThreatAnalyzer::new();
        assert!(analyzer.is_ok());
    }
}