//! Consensus mechanisms for distributed coordination

use neural_doc_flow_core::Result;
use uuid::Uuid;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Consensus algorithm types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsensusAlgorithm {
    Majority,
    Unanimous,
    WeightedVoting,
    RAFT,
    Custom(String),
}

/// Consensus proposal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proposal {
    pub id: Uuid,
    pub proposer: Uuid,
    pub content: serde_json::Value,
    pub votes: HashMap<Uuid, Vote>,
    pub status: ProposalStatus,
}

/// Vote on a proposal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vote {
    pub voter: Uuid,
    pub support: bool,
    pub weight: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Proposal status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProposalStatus {
    Pending,
    Accepted,
    Rejected,
    Timeout,
}

/// Consensus result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusResult {
    pub proposal_id: Uuid,
    pub status: ProposalStatus,
    pub votes_for: usize,
    pub votes_against: usize,
    pub total_weight: f64,
    pub decision_time: chrono::DateTime<chrono::Utc>,
}

/// Consensus engine
#[derive(Debug)]
pub struct ConsensusEngine {
    algorithm: ConsensusAlgorithm,
    active_proposals: HashMap<Uuid, Proposal>,
    threshold: f64,
}

impl ConsensusEngine {
    pub fn new() -> Self {
        Self {
            algorithm: ConsensusAlgorithm::Majority,
            active_proposals: HashMap::new(),
            threshold: 0.5, // 50% for majority
        }
    }
    
    pub fn with_algorithm(algorithm: ConsensusAlgorithm) -> Self {
        let threshold = match algorithm {
            ConsensusAlgorithm::Majority => 0.5,
            ConsensusAlgorithm::Unanimous => 1.0,
            ConsensusAlgorithm::WeightedVoting => 0.6,
            _ => 0.5,
        };
        
        Self {
            algorithm,
            active_proposals: HashMap::new(),
            threshold,
        }
    }
    
    /// Submit a new proposal
    pub fn submit_proposal(&mut self, proposer: Uuid, content: serde_json::Value) -> Uuid {
        let proposal_id = Uuid::new_v4();
        let proposal = Proposal {
            id: proposal_id,
            proposer,
            content,
            votes: HashMap::new(),
            status: ProposalStatus::Pending,
        };
        
        self.active_proposals.insert(proposal_id, proposal);
        proposal_id
    }
    
    /// Cast a vote on a proposal
    pub fn vote(&mut self, proposal_id: Uuid, voter: Uuid, support: bool, weight: Option<f64>) -> Result<()> {
        if let Some(proposal) = self.active_proposals.get_mut(&proposal_id) {
            if proposal.status != ProposalStatus::Pending {
                return Err(anyhow::anyhow!("Proposal is not in pending state"));
            }
            
            let vote = Vote {
                voter,
                support,
                weight: weight.unwrap_or(1.0),
                timestamp: chrono::Utc::now(),
            };
            
            proposal.votes.insert(voter, vote);
            
            // Check if consensus is reached
            self.check_consensus(proposal_id)?;
        } else {
            return Err(anyhow::anyhow!("Proposal not found"));
        }
        
        Ok(())
    }
    
    /// Check if consensus is reached for a proposal
    fn check_consensus(&mut self, proposal_id: Uuid) -> Result<Option<ConsensusResult>> {
        if let Some(proposal) = self.active_proposals.get_mut(&proposal_id) {
            let total_votes = proposal.votes.len();
            let mut votes_for = 0;
            let mut votes_against = 0;
            let mut total_weight = 0.0;
            let mut weight_for = 0.0;
            
            for vote in proposal.votes.values() {
                total_weight += vote.weight;
                if vote.support {
                    votes_for += 1;
                    weight_for += vote.weight;
                } else {
                    votes_against += 1;
                }
            }
            
            let consensus_reached = match self.algorithm {
                ConsensusAlgorithm::Majority => {
                    if total_votes > 0 {
                        (votes_for as f64 / total_votes as f64) > self.threshold
                    } else {
                        false
                    }
                }
                ConsensusAlgorithm::Unanimous => votes_against == 0 && votes_for > 0,
                ConsensusAlgorithm::WeightedVoting => {
                    if total_weight > 0.0 {
                        (weight_for / total_weight) > self.threshold
                    } else {
                        false
                    }
                }
                _ => false,
            };
            
            if consensus_reached {
                proposal.status = ProposalStatus::Accepted;
            } else if self.should_reject(proposal) {
                proposal.status = ProposalStatus::Rejected;
            }
            
            if proposal.status != ProposalStatus::Pending {
                let result = ConsensusResult {
                    proposal_id,
                    status: proposal.status.clone(),
                    votes_for,
                    votes_against,
                    total_weight,
                    decision_time: chrono::Utc::now(),
                };
                return Ok(Some(result));
            }
        }
        
        Ok(None)
    }
    
    /// Check if proposal should be rejected
    fn should_reject(&self, proposal: &Proposal) -> bool {
        // Simple rejection logic - could be more sophisticated
        let total_votes = proposal.votes.len();
        let votes_against = proposal.votes.values().filter(|v| !v.support).count();
        
        if total_votes > 0 {
            (votes_against as f64 / total_votes as f64) > (1.0 - self.threshold)
        } else {
            false
        }
    }
    
    /// Get proposal status
    pub fn get_proposal_status(&self, proposal_id: Uuid) -> Option<ProposalStatus> {
        self.active_proposals.get(&proposal_id).map(|p| p.status.clone())
    }
    
    /// Remove completed proposals
    pub fn cleanup_completed(&mut self) {
        self.active_proposals.retain(|_, proposal| proposal.status == ProposalStatus::Pending);
    }
}

impl Default for ConsensusEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_consensus_engine() {
        let mut engine = ConsensusEngine::new();
        
        let proposer = Uuid::new_v4();
        let voter1 = Uuid::new_v4();
        let voter2 = Uuid::new_v4();
        
        let proposal_id = engine.submit_proposal(proposer, serde_json::json!({"action": "test"}));
        
        assert_eq!(engine.get_proposal_status(proposal_id), Some(ProposalStatus::Pending));
        
        engine.vote(proposal_id, voter1, true, None).unwrap();
        engine.vote(proposal_id, voter2, true, None).unwrap();
        
        assert_eq!(engine.get_proposal_status(proposal_id), Some(ProposalStatus::Accepted));
    }
    
    #[test]
    fn test_weighted_voting() {
        let mut engine = ConsensusEngine::with_algorithm(ConsensusAlgorithm::WeightedVoting);
        
        let proposer = Uuid::new_v4();
        let voter1 = Uuid::new_v4();
        let voter2 = Uuid::new_v4();
        
        let proposal_id = engine.submit_proposal(proposer, serde_json::json!({"action": "test"}));
        
        // High weight vote for
        engine.vote(proposal_id, voter1, true, Some(0.8)).unwrap();
        // Low weight vote against
        engine.vote(proposal_id, voter2, false, Some(0.2)).unwrap();
        
        assert_eq!(engine.get_proposal_status(proposal_id), Some(ProposalStatus::Accepted));
    }
}