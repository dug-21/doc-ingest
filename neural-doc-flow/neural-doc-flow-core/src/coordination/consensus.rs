//! Consensus mechanisms for DAA coordination

use std::collections::HashMap;
use uuid::Uuid;
use serde::{Deserialize, Serialize};
use tracing::{info, debug, warn};

use crate::types::{ExtractedContent, Confidence};
use crate::traits::ValidationResult;
use crate::error::{CoreError, Result};

/// Result of consensus operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusResult {
    pub consensus_reached: bool,
    pub final_content: ExtractedContent,
    pub confidence: f32,
    pub participating_agents: Vec<Uuid>,
    pub agreement_score: f32,
    pub iterations: u32,
    pub disagreements: Vec<ConsensusDisagreement>,
}

/// Disagreement found during consensus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusDisagreement {
    pub disagreement_type: DisagreementType,
    pub agents: Vec<Uuid>,
    pub content_blocks: Vec<Uuid>,
    pub severity: DisagreementSeverity,
    pub resolved: bool,
}

/// Types of disagreements that can occur
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DisagreementType {
    ContentMismatch,
    ConfidenceMismatch,
    StructureMismatch,
    MetadataMismatch,
    QualityAssessment,
}

/// Severity of disagreements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DisagreementSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Validation consensus for quality assurance
#[derive(Debug)]
pub struct ValidationConsensus {
    validators: Vec<Uuid>,
    consensus_threshold: f32,
    max_iterations: u32,
}

impl ValidationConsensus {
    /// Create new validation consensus mechanism
    pub fn new(validators: Vec<Uuid>, consensus_threshold: f32) -> Self {
        Self {
            validators,
            consensus_threshold,
            max_iterations: 10,
        }
    }
    
    /// Perform consensus validation on extracted content
    pub async fn validate_consensus(
        &self,
        candidates: Vec<ExtractedContent>,
        agent_results: HashMap<Uuid, ValidationResult>,
    ) -> Result<ConsensusResult> {
        info!("Starting consensus validation with {} candidates", candidates.len());
        debug!("Validators: {:?}", self.validators);
        
        if candidates.is_empty() {
            return Err(CoreError::ValidationError(
                "No candidates provided for consensus".to_string()
            ));
        }
        
        let mut iterations = 0;
        let mut current_candidates = candidates;
        let mut disagreements = Vec::new();
        
        while iterations < self.max_iterations {
            iterations += 1;
            debug!("Consensus iteration {}", iterations);
            
            // Analyze agreements and disagreements
            let analysis = self.analyze_candidates(&current_candidates, &agent_results)?;
            
            if analysis.agreement_score >= self.consensus_threshold {
                info!("Consensus reached after {} iterations", iterations);
                return Ok(ConsensusResult {
                    consensus_reached: true,
                    final_content: analysis.best_candidate,
                    confidence: analysis.confidence,
                    participating_agents: self.validators.clone(),
                    agreement_score: analysis.agreement_score,
                    iterations,
                    disagreements,
                });
            }
            
            // Record disagreements
            disagreements.extend(analysis.disagreements);
            
            // Attempt to resolve disagreements
            current_candidates = self.resolve_disagreements(current_candidates, &analysis)?;
            
            if current_candidates.len() <= 1 {
                break;
            }
        }
        
        warn!("Consensus not reached after {} iterations", iterations);
        
        // Return best candidate even without consensus
        let final_analysis = self.analyze_candidates(&current_candidates, &agent_results)?;
        
        Ok(ConsensusResult {
            consensus_reached: false,
            final_content: final_analysis.best_candidate,
            confidence: final_analysis.confidence * 0.8, // Reduced confidence
            participating_agents: self.validators.clone(),
            agreement_score: final_analysis.agreement_score,
            iterations,
            disagreements,
        })
    }
    
    /// Analyze candidates to find agreements and disagreements
    fn analyze_candidates(
        &self,
        candidates: &[ExtractedContent],
        agent_results: &HashMap<Uuid, ValidationResult>,
    ) -> Result<ConsensusAnalysis> {
        debug!("Analyzing {} candidates", candidates.len());
        
        // Find the candidate with highest average confidence
        let mut best_candidate = candidates[0].clone();
        let mut best_score = 0.0;
        let mut total_confidence = 0.0;
        
        for candidate in candidates {
            // Calculate composite score based on confidence and validation results
            let confidence_score = candidate.confidence.overall;
            
            // Factor in validation results from agents
            let validation_score = agent_results.values()
                .map(|v| v.confidence_score)
                .sum::<f32>() / agent_results.len() as f32;
            
            let composite_score = (confidence_score + validation_score) / 2.0;
            total_confidence += composite_score;
            
            if composite_score > best_score {
                best_score = composite_score;
                best_candidate = candidate.clone();
            }
        }
        
        let average_confidence = total_confidence / candidates.len() as f32;
        
        // Calculate agreement score based on consistency between candidates
        let agreement_score = self.calculate_agreement_score(candidates)?;
        
        // Identify disagreements
        let disagreements = self.identify_disagreements(candidates)?;
        
        Ok(ConsensusAnalysis {
            best_candidate,
            confidence: best_score,
            agreement_score,
            disagreements,
        })
    }
    
    /// Calculate agreement score between candidates
    fn calculate_agreement_score(&self, candidates: &[ExtractedContent]) -> Result<f32> {
        if candidates.len() < 2 {
            return Ok(1.0);
        }
        
        let mut total_agreement = 0.0;
        let mut comparisons = 0;
        
        for i in 0..candidates.len() {
            for j in (i + 1)..candidates.len() {
                let agreement = self.compare_candidates(&candidates[i], &candidates[j])?;
                total_agreement += agreement;
                comparisons += 1;
            }
        }
        
        Ok(total_agreement / comparisons as f32)
    }
    
    /// Compare two candidates and return agreement score (0.0 to 1.0)
    fn compare_candidates(&self, a: &ExtractedContent, b: &ExtractedContent) -> Result<f32> {
        let mut agreement_factors = Vec::new();
        
        // Compare content blocks count
        let block_count_agreement = if a.blocks.len() == b.blocks.len() {
            1.0
        } else {
            let diff = (a.blocks.len() as f32 - b.blocks.len() as f32).abs();
            let max_blocks = a.blocks.len().max(b.blocks.len()) as f32;
            1.0 - (diff / max_blocks)
        };
        agreement_factors.push(block_count_agreement);
        
        // Compare confidence scores
        let confidence_diff = (a.confidence.overall - b.confidence.overall).abs();
        let confidence_agreement = 1.0 - confidence_diff;
        agreement_factors.push(confidence_agreement);
        
        // Compare metadata similarity
        let metadata_agreement = self.compare_metadata(&a.metadata, &b.metadata);
        agreement_factors.push(metadata_agreement);
        
        // Calculate overall agreement
        let overall_agreement = agreement_factors.iter().sum::<f32>() / agreement_factors.len() as f32;
        
        Ok(overall_agreement.max(0.0).min(1.0))
    }
    
    /// Compare metadata between two extracted contents
    fn compare_metadata(&self, a: &crate::types::DocumentMetadata, b: &crate::types::DocumentMetadata) -> f32 {
        let mut matches = 0;
        let mut total_fields = 0;
        
        // Compare title
        total_fields += 1;
        if a.title == b.title {
            matches += 1;
        }
        
        // Compare author
        total_fields += 1;
        if a.author == b.author {
            matches += 1;
        }
        
        // Compare MIME type
        total_fields += 1;
        if a.mime_type == b.mime_type {
            matches += 1;
        }
        
        // Compare language
        total_fields += 1;
        if a.language == b.language {
            matches += 1;
        }
        
        matches as f32 / total_fields as f32
    }
    
    /// Identify specific disagreements between candidates
    fn identify_disagreements(&self, candidates: &[ExtractedContent]) -> Result<Vec<ConsensusDisagreement>> {
        let mut disagreements = Vec::new();
        
        // Check for content mismatches
        if self.has_content_disagreement(candidates) {
            disagreements.push(ConsensusDisagreement {
                disagreement_type: DisagreementType::ContentMismatch,
                agents: self.validators.clone(),
                content_blocks: vec![], // Would identify specific blocks
                severity: DisagreementSeverity::Medium,
                resolved: false,
            });
        }
        
        // Check for confidence mismatches
        if self.has_confidence_disagreement(candidates) {
            disagreements.push(ConsensusDisagreement {
                disagreement_type: DisagreementType::ConfidenceMismatch,
                agents: self.validators.clone(),
                content_blocks: vec![],
                severity: DisagreementSeverity::Low,
                resolved: false,
            });
        }
        
        Ok(disagreements)
    }
    
    /// Check if there are significant content disagreements
    fn has_content_disagreement(&self, candidates: &[ExtractedContent]) -> bool {
        if candidates.len() < 2 {
            return false;
        }
        
        let block_counts: Vec<usize> = candidates.iter().map(|c| c.blocks.len()).collect();
        let min_blocks = *block_counts.iter().min().unwrap();
        let max_blocks = *block_counts.iter().max().unwrap();
        
        // Significant disagreement if block count varies by more than 20%
        if min_blocks > 0 {
            let variation = (max_blocks as f32 - min_blocks as f32) / min_blocks as f32;
            variation > 0.2
        } else {
            max_blocks > 0
        }
    }
    
    /// Check if there are significant confidence disagreements
    fn has_confidence_disagreement(&self, candidates: &[ExtractedContent]) -> bool {
        if candidates.len() < 2 {
            return false;
        }
        
        let confidences: Vec<f32> = candidates.iter().map(|c| c.confidence.overall).collect();
        let min_confidence = confidences.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let max_confidence = confidences.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        
        // Significant disagreement if confidence varies by more than 0.3
        (max_confidence - min_confidence) > 0.3
    }
    
    /// Attempt to resolve disagreements by filtering or merging candidates
    fn resolve_disagreements(
        &self,
        candidates: Vec<ExtractedContent>,
        analysis: &ConsensusAnalysis,
    ) -> Result<Vec<ExtractedContent>> {
        let mut resolved_candidates = candidates;
        
        // For simplicity, we'll just keep the top half of candidates based on confidence
        resolved_candidates.sort_by(|a, b| 
            b.confidence.overall.partial_cmp(&a.confidence.overall).unwrap()
        );
        
        let keep_count = (resolved_candidates.len() + 1) / 2; // Keep at least half
        resolved_candidates.truncate(keep_count);
        
        debug!("Resolved to {} candidates", resolved_candidates.len());
        Ok(resolved_candidates)
    }
}

/// Internal analysis result
struct ConsensusAnalysis {
    best_candidate: ExtractedContent,
    confidence: f32,
    agreement_score: f32,
    disagreements: Vec<ConsensusDisagreement>,
}

impl Default for ValidationConsensus {
    fn default() -> Self {
        Self {
            validators: vec![],
            consensus_threshold: 0.8,
            max_iterations: 10,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DocumentMetadata, Confidence};
    
    #[test]
    fn test_consensus_creation() {
        let validators = vec![Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4()];
        let consensus = ValidationConsensus::new(validators.clone(), 0.8);
        
        assert_eq!(consensus.validators, validators);
        assert_eq!(consensus.consensus_threshold, 0.8);
    }
    
    #[test]
    fn test_metadata_comparison() {
        let consensus = ValidationConsensus::default();
        
        let metadata1 = DocumentMetadata {
            title: Some("Test Document".to_string()),
            author: Some("Test Author".to_string()),
            mime_type: Some("application/pdf".to_string()),
            language: Some("en".to_string()),
            ..Default::default()
        };
        
        let metadata2 = metadata1.clone();
        let similarity = consensus.compare_metadata(&metadata1, &metadata2);
        assert_eq!(similarity, 1.0);
        
        let metadata3 = DocumentMetadata {
            title: Some("Different Title".to_string()),
            ..metadata1.clone()
        };
        let similarity = consensus.compare_metadata(&metadata1, &metadata3);
        assert_eq!(similarity, 0.75); // 3 out of 4 fields match
    }
}