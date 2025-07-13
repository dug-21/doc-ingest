//! Result type utilities and extensions
//!
//! This module provides additional utilities for working with processing results

use crate::types::{ProcessingResultData, ProcessingMetadata, ProcessingIssue, IssueSeverity};
use crate::error::{ProcessingResult, ProcessingError};
use std::collections::HashMap;

/// Result builder for convenient result creation
#[derive(Debug)]
pub struct ResultBuilder<T> {
    data: Option<T>,
    confidence: f64,
    metadata: ProcessingMetadata,
    issues: Vec<ProcessingIssue>,
}

impl<T> ResultBuilder<T> {
    /// Create a new result builder
    pub fn new() -> Self {
        Self {
            data: None,
            confidence: 1.0,
            metadata: ProcessingMetadata {
                duration_ms: 0,
                memory_usage: None,
                processor_version: String::new(),
                parameters: HashMap::new(),
            },
            issues: Vec::new(),
        }
    }
    
    /// Set the result data
    pub fn data(mut self, data: T) -> Self {
        self.data = Some(data);
        self
    }
    
    /// Set confidence score
    pub fn confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }
    
    /// Set processing duration
    pub fn duration_ms(mut self, duration: u64) -> Self {
        self.metadata.duration_ms = duration;
        self
    }
    
    /// Set memory usage
    pub fn memory_usage(mut self, bytes: u64) -> Self {
        self.metadata.memory_usage = Some(bytes);
        self
    }
    
    /// Set processor version
    pub fn processor_version(mut self, version: impl Into<String>) -> Self {
        self.metadata.processor_version = version.into();
        self
    }
    
    /// Add processing parameter
    pub fn parameter(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.parameters.insert(key.into(), value);
        self
    }
    
    /// Add an issue
    pub fn issue(mut self, issue: ProcessingIssue) -> Self {
        self.issues.push(issue);
        self
    }
    
    /// Add a warning
    pub fn warning(mut self, message: impl Into<String>) -> Self {
        self.issues.push(ProcessingIssue {
            severity: IssueSeverity::Warning,
            message: message.into(),
            location: None,
            suggestion: None,
        });
        self
    }
    
    /// Add an error
    pub fn error(mut self, message: impl Into<String>) -> Self {
        self.issues.push(ProcessingIssue {
            severity: IssueSeverity::Error,
            message: message.into(),
            location: None,
            suggestion: None,
        });
        self
    }
    
    /// Build the result
    pub fn build(self) -> ProcessingResult<ProcessingResultData<T>> {
        match self.data {
            Some(data) => Ok(ProcessingResultData {
                data,
                confidence: self.confidence,
                metadata: self.metadata,
                issues: self.issues,
            }),
            None => Err(ProcessingError::ProcessorFailed {
                processor_name: self.metadata.processor_version,
                reason: "No data provided".to_string(),
            }),
        }
    }
}

impl<T> Default for ResultBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Result utilities
pub struct ResultUtils;

impl ResultUtils {
    /// Create a successful result with high confidence
    pub fn success<T>(data: T, processor_name: &str) -> ProcessingResultData<T> {
        ProcessingResultData {
            data,
            confidence: 1.0,
            metadata: ProcessingMetadata {
                duration_ms: 0,
                memory_usage: None,
                processor_version: processor_name.to_string(),
                parameters: HashMap::new(),
            },
            issues: Vec::new(),
        }
    }
    
    /// Create a result with reduced confidence
    pub fn with_confidence<T>(data: T, confidence: f64, processor_name: &str) -> ProcessingResultData<T> {
        ProcessingResultData {
            data,
            confidence: confidence.clamp(0.0, 1.0),
            metadata: ProcessingMetadata {
                duration_ms: 0,
                memory_usage: None,
                processor_version: processor_name.to_string(),
                parameters: HashMap::new(),
            },
            issues: Vec::new(),
        }
    }
    
    /// Combine multiple results into one
    pub fn combine<T, F>(results: Vec<ProcessingResultData<T>>, combiner: F) -> ProcessingResult<ProcessingResultData<T>>
    where
        F: FnOnce(Vec<T>) -> T,
    {
        if results.is_empty() {
            return Err(ProcessingError::ProcessorFailed {
                processor_name: "combine".to_string(),
                reason: "No results to combine".to_string(),
            });
        }
        
        // Calculate combined metrics first (before moving results)
        let results_len = results.len();
        let avg_confidence = results.iter()
            .map(|r| r.confidence)
            .sum::<f64>() / results_len as f64;
        
        let total_duration: u64 = results.iter()
            .map(|r| r.metadata.duration_ms)
            .sum();
        
        let total_memory: u64 = results.iter()
            .filter_map(|r| r.metadata.memory_usage)
            .sum();
        
        // Collect all data and issues (this moves results)
        let mut data_vec = Vec::with_capacity(results_len);
        let mut all_issues = Vec::new();
        
        for result in results {
            data_vec.push(result.data);
            all_issues.extend(result.issues);
        }
        
        let combined_data = combiner(data_vec);
        
        Ok(ProcessingResultData {
            data: combined_data,
            confidence: avg_confidence,
            metadata: ProcessingMetadata {
                duration_ms: total_duration,
                memory_usage: Some(total_memory),
                processor_version: "combined".to_string(),
                parameters: HashMap::new(),
            },
            issues: all_issues,
        })
    }
    
    /// Map result data while preserving metadata
    pub fn map<T, U, F>(result: ProcessingResultData<T>, mapper: F) -> ProcessingResultData<U>
    where
        F: FnOnce(T) -> U,
    {
        ProcessingResultData {
            data: mapper(result.data),
            confidence: result.confidence,
            metadata: result.metadata,
            issues: result.issues,
        }
    }
    
    /// Filter issues by severity
    pub fn filter_issues<T>(result: &ProcessingResultData<T>, severity: IssueSeverity) -> Vec<&ProcessingIssue> {
        result.issues.iter()
            .filter(|issue| std::mem::discriminant(&issue.severity) == std::mem::discriminant(&severity))
            .collect()
    }
    
    /// Calculate overall result quality
    pub fn calculate_quality<T>(result: &ProcessingResultData<T>) -> f64 {
        let base_quality = result.confidence;
        
        // Reduce quality based on issues
        let issue_penalty = result.issues.iter()
            .map(|issue| match issue.severity {
                IssueSeverity::Info => 0.0,
                IssueSeverity::Warning => 0.05,
                IssueSeverity::Error => 0.15,
                IssueSeverity::Critical => 0.30,
            })
            .sum::<f64>();
        
        (base_quality - issue_penalty).max(0.0)
    }
}

/// Result timing utilities
pub struct TimingUtils;

impl TimingUtils {
    /// Measure processing time and update result
    pub async fn measure<T, F, Fut>(processor_name: &str, f: F) -> ProcessingResult<ProcessingResultData<T>>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = ProcessingResult<T>>,
    {
        let start = std::time::Instant::now();
        
        match f().await {
            Ok(data) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                Ok(ProcessingResultData {
                    data,
                    confidence: 1.0,
                    metadata: ProcessingMetadata {
                        duration_ms,
                        memory_usage: None,
                        processor_version: processor_name.to_string(),
                        parameters: HashMap::new(),
                    },
                    issues: Vec::new(),
                })
            },
            Err(e) => Err(e),
        }
    }
    
    /// Create a timed result builder
    pub fn timed_builder<T>() -> TimedResultBuilder<T> {
        TimedResultBuilder::new()
    }
}

/// Timed result builder that automatically tracks processing time
pub struct TimedResultBuilder<T> {
    start_time: std::time::Instant,
    builder: ResultBuilder<T>,
}

impl<T> TimedResultBuilder<T> {
    /// Create a new timed result builder
    pub fn new() -> Self {
        Self {
            start_time: std::time::Instant::now(),
            builder: ResultBuilder::new(),
        }
    }
    
    /// Set the result data
    pub fn data(mut self, data: T) -> Self {
        self.builder = self.builder.data(data);
        self
    }
    
    /// Set confidence score
    pub fn confidence(mut self, confidence: f64) -> Self {
        self.builder = self.builder.confidence(confidence);
        self
    }
    
    /// Set processor version
    pub fn processor_version(mut self, version: impl Into<String>) -> Self {
        self.builder = self.builder.processor_version(version);
        self
    }
    
    /// Add processing parameter
    pub fn parameter(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.builder = self.builder.parameter(key, value);
        self
    }
    
    /// Add a warning
    pub fn warning(mut self, message: impl Into<String>) -> Self {
        self.builder = self.builder.warning(message);
        self
    }
    
    /// Add an error
    pub fn error(mut self, message: impl Into<String>) -> Self {
        self.builder = self.builder.error(message);
        self
    }
    
    /// Build the result with automatic timing
    pub fn build(self) -> ProcessingResult<ProcessingResultData<T>> {
        let duration_ms = self.start_time.elapsed().as_millis() as u64;
        self.builder.duration_ms(duration_ms).build()
    }
}

/// Result validation utilities
pub struct ResultValidator;

impl ResultValidator {
    /// Validate result confidence is within acceptable range
    pub fn validate_confidence<T>(result: &ProcessingResultData<T>, min_confidence: f64) -> ProcessingResult<()> {
        if result.confidence < min_confidence {
            Err(ProcessingError::ProcessorFailed {
                processor_name: result.metadata.processor_version.clone(),
                reason: format!("Confidence {} below minimum threshold {}", result.confidence, min_confidence),
            })
        } else {
            Ok(())
        }
    }
    
    /// Check if result has critical issues
    pub fn check_critical_issues<T>(result: &ProcessingResultData<T>) -> ProcessingResult<()> {
        let critical_issues: Vec<&ProcessingIssue> = result.issues.iter()
            .filter(|issue| matches!(issue.severity, IssueSeverity::Critical))
            .collect();
        
        if !critical_issues.is_empty() {
            let messages: Vec<&str> = critical_issues.iter()
                .map(|issue| issue.message.as_str())
                .collect();
            
            Err(ProcessingError::ProcessorFailed {
                processor_name: result.metadata.processor_version.clone(),
                reason: format!("Critical issues found: {}", messages.join(", ")),
            })
        } else {
            Ok(())
        }
    }
    
    /// Validate processing time is within limits
    pub fn validate_processing_time<T>(result: &ProcessingResultData<T>, max_duration_ms: u64) -> ProcessingResult<()> {
        if result.metadata.duration_ms > max_duration_ms {
            Err(ProcessingError::Timeout)
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_result_builder() {
        let result = ResultBuilder::new()
            .data("test data")
            .confidence(0.95)
            .duration_ms(100)
            .processor_version("test-processor v1.0")
            .warning("Minor issue detected")
            .build()
            .unwrap();
        
        assert_eq!(result.data, "test data");
        assert_eq!(result.confidence, 0.95);
        assert_eq!(result.metadata.duration_ms, 100);
        assert_eq!(result.metadata.processor_version, "test-processor v1.0");
        assert_eq!(result.issues.len(), 1);
        assert!(matches!(result.issues[0].severity, IssueSeverity::Warning));
    }
    
    #[test]
    fn test_quality_calculation() {
        let mut result = ResultUtils::success("data", "test");
        result.confidence = 0.9;
        result.issues.push(ProcessingIssue {
            severity: IssueSeverity::Warning,
            message: "Warning".to_string(),
            location: None,
            suggestion: None,
        });
        result.issues.push(ProcessingIssue {
            severity: IssueSeverity::Error,
            message: "Error".to_string(),
            location: None,
            suggestion: None,
        });
        
        let quality = ResultUtils::calculate_quality(&result);
        assert_eq!(quality, 0.7); // 0.9 - 0.05 - 0.15
    }
}