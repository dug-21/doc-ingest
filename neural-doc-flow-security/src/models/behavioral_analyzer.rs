//! Behavioral analysis neural model for detecting suspicious document behaviors

use crate::models::base::{BaseNeuralModel, ModelConfig, ModelMetrics, NeuralSecurityModel, TrainingData, TrainingResult, ValidationResult};
use neural_doc_flow_core::ProcessingError;
use std::path::{Path, PathBuf};

/// Neural model for behavioral analysis
/// Detects suspicious behavioral patterns in documents
pub struct BehavioralAnalyzerModel {
    base_model: BaseNeuralModel,
    model_path: PathBuf,
    behavior_categories: Vec<String>,
}

impl BehavioralAnalyzerModel {
    /// Create a new behavioral analyzer model
    pub fn new() -> Result<Self, ProcessingError> {
        let config = ModelConfig {
            name: "BehavioralAnalyzer".to_string(),
            version: "1.0.0".to_string(),
            input_size: 160,  // Behavioral feature vectors
            output_size: 8,   // 8 behavior types
            hidden_layers: vec![256, 128, 64], // Deep enough for pattern recognition
            activation_function: "tanh".to_string(),
            learning_rate: 0.005,
            momentum: 0.92,
            target_error: 0.0015,
            max_epochs: 25000,
            simd_enabled: true,
        };
        
        let base_model = BaseNeuralModel::new(config)?;
        
        let behavior_categories = vec![
            "Dropper Behavior".to_string(),
            "Downloader Activity".to_string(),
            "Code Injection Patterns".to_string(),
            "Persistence Mechanisms".to_string(),
            "Data Collection".to_string(),
            "Network Communication".to_string(),
            "Evasion Techniques".to_string(),
            "Privilege Escalation".to_string(),
        ];
        
        Ok(Self {
            base_model,
            model_path: PathBuf::from("models/behavioral_analyzer.fann"),
            behavior_categories,
        })
    }
    
    /// Load or create model
    pub fn load_or_create(model_dir: &Path) -> Result<Self, ProcessingError> {
        let mut model = Self::new()?;
        let model_path = model_dir.join("behavioral_analyzer.fann");
        model.model_path = model_path.clone();
        
        if model_path.exists() {
            model.load(&model_path)?;
        }
        
        Ok(model)
    }
    
    /// Extract behavioral features from document
    pub fn extract_features(raw_features: &crate::SecurityFeatures) -> Vec<f32> {
        let mut features = Vec::with_capacity(160);
        
        // Dropper behavior indicators
        let dropper_keywords = ["download", "fetch", "get", "retrieve", "pull"];
        for keyword in &dropper_keywords {
            let found = raw_features.suspicious_keywords.iter()
                .any(|k| k.to_lowercase().contains(keyword));
            features.push(if found { 1.0 } else { 0.0 });
        }
        features.push((raw_features.embedded_files.len() as f32) / 10.0);
        
        // Downloader activity indicators
        features.push((raw_features.url_count as f32) / 20.0);
        let download_extensions = ["exe", "dll", "zip", "rar", "7z"];
        for ext in &download_extensions {
            let count = raw_features.embedded_files.iter()
                .filter(|f| f.file_type.to_lowercase().contains(ext)).count();
            features.push((count as f32).min(1.0));
        }
        
        // Code injection patterns
        let injection_keywords = ["eval", "exec", "compile", "assembly", "invoke"];
        for keyword in &injection_keywords {
            let found = raw_features.suspicious_keywords.iter()
                .any(|k| k.to_lowercase().contains(keyword));
            features.push(if found { 1.0 } else { 0.0 });
        }
        features.push(raw_features.obfuscation_score);
        
        // Persistence mechanism indicators
        let persistence_keywords = ["registry", "startup", "autorun", "schedule", "service"];
        for keyword in &persistence_keywords {
            let found = raw_features.suspicious_keywords.iter()
                .any(|k| k.to_lowercase().contains(keyword));
            features.push(if found { 1.0 } else { 0.0 });
        }
        
        // Data collection indicators
        let data_keywords = ["password", "credential", "cookie", "token", "key"];
        for keyword in &data_keywords {
            let found = raw_features.suspicious_keywords.iter()
                .any(|k| k.to_lowercase().contains(keyword));
            features.push(if found { 1.0 } else { 0.0 });
        }
        
        // Network communication patterns
        features.push(if raw_features.url_count > 0 { 1.0 } else { 0.0 });
        let network_keywords = ["http", "ftp", "socket", "connect", "send"];
        for keyword in &network_keywords {
            let found = raw_features.suspicious_keywords.iter()
                .any(|k| k.to_lowercase().contains(keyword));
            features.push(if found { 1.0 } else { 0.0 });
        }
        
        // Evasion technique indicators
        features.push(raw_features.header_entropy / 8.0);
        features.push(if raw_features.obfuscation_score > 0.6 { 1.0 } else { 0.0 });
        let evasion_keywords = ["sleep", "delay", "wait", "detect", "sandbox"];
        for keyword in &evasion_keywords {
            let found = raw_features.suspicious_keywords.iter()
                .any(|k| k.to_lowercase().contains(keyword));
            features.push(if found { 1.0 } else { 0.0 });
        }
        
        // Privilege escalation indicators
        let privilege_keywords = ["admin", "root", "privilege", "elevate", "bypass"];
        for keyword in &privilege_keywords {
            let found = raw_features.suspicious_keywords.iter()
                .any(|k| k.to_lowercase().contains(keyword));
            features.push(if found { 1.0 } else { 0.0 });
        }
        
        // Behavioral combinations
        let multiple_behaviors = {
            let behavior_count = [
                raw_features.embedded_files.len() > 0,
                raw_features.url_count > 5,
                raw_features.obfuscation_score > 0.5,
                raw_features.javascript_present,
            ].iter().filter(|&&x| x).count();
            
            (behavior_count as f32) / 4.0
        };
        features.push(multiple_behaviors);
        
        // Temporal patterns (simulated - in real implementation would track over time)
        features.push(if raw_features.file_size > 10_000_000 { 0.8 } else { 0.2 });
        
        // Complexity metrics
        let behavioral_complexity = (raw_features.suspicious_keywords.len() as f32 * 
            raw_features.obfuscation_score * 
            (raw_features.url_count as f32 / 10.0).min(1.0)).min(1.0);
        features.push(behavioral_complexity);
        
        // Cross-behavior correlations
        let dropper_downloader = raw_features.embedded_files.len() > 0 && raw_features.url_count > 0;
        features.push(if dropper_downloader { 1.0 } else { 0.0 });
        
        let injection_evasion = raw_features.obfuscation_score > 0.5 && 
            raw_features.suspicious_keywords.iter().any(|k| k.contains("eval"));
        features.push(if injection_evasion { 1.0 } else { 0.0 });
        
        // Pad to expected size
        while features.len() < 160 {
            features.push(0.0);
        }
        
        features
    }
    
    /// Analyze document behavior
    pub fn analyze(&self, features: &[f32]) -> Result<Vec<(String, f32)>, ProcessingError> {
        let output = self.base_model.network.lock().unwrap().run(features);
        
        // Apply sigmoid to get probabilities
        let probabilities: Vec<f32> = output.iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect();
        
        // Return behaviors with significant probability
        let mut results = Vec::new();
        for (i, &prob) in probabilities.iter().enumerate() {
            if prob > 0.3 {  // 30% threshold for behavioral indicators
                results.push((self.behavior_categories[i].clone(), prob));
            }
        }
        
        // Sort by probability descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        Ok(results)
    }
    
    /// Get behavioral risk assessment
    pub fn assess_risk(&self, features: &[f32]) -> Result<BehavioralRiskAssessment, ProcessingError> {
        let behaviors = self.analyze(features)?;
        
        // Calculate overall risk score
        let risk_score = behaviors.iter()
            .map(|(_, prob)| prob)
            .sum::<f32>() / self.behavior_categories.len() as f32;
        
        // Identify high-risk combinations
        let mut critical_behaviors = Vec::new();
        
        // Check for dropper + downloader combination
        let has_dropper = behaviors.iter().any(|(b, p)| b.contains("Dropper") && *p > 0.5);
        let has_downloader = behaviors.iter().any(|(b, p)| b.contains("Downloader") && *p > 0.5);
        if has_dropper && has_downloader {
            critical_behaviors.push("Dropper-Downloader Chain".to_string());
        }
        
        // Check for injection + evasion combination
        let has_injection = behaviors.iter().any(|(b, p)| b.contains("Injection") && *p > 0.5);
        let has_evasion = behaviors.iter().any(|(b, p)| b.contains("Evasion") && *p > 0.5);
        if has_injection && has_evasion {
            critical_behaviors.push("Evasive Code Injection".to_string());
        }
        
        // Check for data collection + network communication
        let has_collection = behaviors.iter().any(|(b, p)| b.contains("Data Collection") && *p > 0.5);
        let has_network = behaviors.iter().any(|(b, p)| b.contains("Network") && *p > 0.5);
        if has_collection && has_network {
            critical_behaviors.push("Data Exfiltration Risk".to_string());
        }
        
        let risk_level = match risk_score {
            s if s > 0.8 => RiskLevel::Critical,
            s if s > 0.6 => RiskLevel::High,
            s if s > 0.4 => RiskLevel::Medium,
            s if s > 0.2 => RiskLevel::Low,
            _ => RiskLevel::Minimal,
        };
        
        Ok(BehavioralRiskAssessment {
            risk_score,
            risk_level,
            detected_behaviors: behaviors.clone(),
            critical_combinations: critical_behaviors,
            confidence: self.calculate_confidence(&behaviors),
        })
    }
    
    fn calculate_confidence(&self, behaviors: &[(String, f32)]) -> f32 {
        if behaviors.is_empty() {
            return 0.5;
        }
        
        // Higher confidence when multiple behaviors detected with high probability
        let avg_prob = behaviors.iter().map(|(_, p)| p).sum::<f32>() / behaviors.len() as f32;
        let high_prob_count = behaviors.iter().filter(|(_, p)| *p > 0.7).count();
        
        (avg_prob * 0.6 + (high_prob_count as f32 / behaviors.len() as f32) * 0.4).min(0.99)
    }
    
    /// Advanced behavioral analysis with temporal patterns
    pub fn analyze_advanced_behavior(&self, features: &[f32], raw_features: &crate::SecurityFeatures) -> Result<AdvancedBehavioralAssessment, ProcessingError> {
        let basic_assessment = self.assess_risk(features)?;
        
        // Enhanced behavior pattern detection
        let attack_chain_probability = self.detect_attack_chain_patterns(raw_features);
        let evasion_techniques = self.detect_evasion_techniques(raw_features);
        let persistence_indicators = self.detect_persistence_indicators(raw_features);
        
        // Calculate advanced risk metrics
        let sophistication_score = self.calculate_sophistication_score(raw_features);
        let impact_score = self.calculate_potential_impact(raw_features);
        
        Ok(AdvancedBehavioralAssessment {
            basic_assessment: basic_assessment.clone(),
            attack_chain_probability,
            evasion_techniques,
            persistence_indicators,
            sophistication_score,
            impact_score,
            threat_actor_profile: self.infer_threat_actor_profile(raw_features),
            mitigation_recommendations: self.generate_mitigation_recommendations(&basic_assessment),
        })
    }
    
    fn detect_attack_chain_patterns(&self, features: &crate::SecurityFeatures) -> f32 {
        let mut chain_score: f32 = 0.0;
        
        // Dropper -> Downloader -> Payload pattern
        let has_network = features.url_count > 0;
        let has_executables = features.embedded_files.iter().any(|f| 
            f.file_type.contains("executable") || f.file_type.contains("dll"));
        let has_scripts = features.javascript_present || 
            features.suspicious_keywords.iter().any(|k| k.contains("script"));
        
        if has_network && has_executables && has_scripts {
            chain_score += 0.8;
        } else if (has_network && has_executables) || (has_scripts && has_executables) {
            chain_score += 0.5;
        }
        
        // Obfuscation + persistence pattern
        if features.obfuscation_score > 0.6 && 
           features.suspicious_keywords.iter().any(|k| k.contains("registry") || k.contains("startup")) {
            chain_score += 0.4;
        }
        
        chain_score.min(1.0)
    }
    
    fn detect_evasion_techniques(&self, features: &crate::SecurityFeatures) -> Vec<EvasionTechnique> {
        let mut techniques = Vec::new();
        
        if features.obfuscation_score > 0.7 {
            techniques.push(EvasionTechnique::CodeObfuscation);
        }
        
        if features.header_entropy > 7.5 {
            techniques.push(EvasionTechnique::Encryption);
        }
        
        if features.suspicious_keywords.iter().any(|k| k.contains("sleep") || k.contains("delay")) {
            techniques.push(EvasionTechnique::TimingEvasion);
        }
        
        if features.suspicious_keywords.iter().any(|k| k.contains("detect") || k.contains("sandbox")) {
            techniques.push(EvasionTechnique::SandboxEvasion);
        }
        
        techniques
    }
    
    fn detect_persistence_indicators(&self, features: &crate::SecurityFeatures) -> Vec<PersistenceMethod> {
        let mut methods = Vec::new();
        
        if features.suspicious_keywords.iter().any(|k| k.contains("registry")) {
            methods.push(PersistenceMethod::RegistryModification);
        }
        
        if features.suspicious_keywords.iter().any(|k| k.contains("startup") || k.contains("autorun")) {
            methods.push(PersistenceMethod::StartupFolder);
        }
        
        if features.suspicious_keywords.iter().any(|k| k.contains("service")) {
            methods.push(PersistenceMethod::ServiceInstallation);
        }
        
        if features.suspicious_keywords.iter().any(|k| k.contains("schedule")) {
            methods.push(PersistenceMethod::ScheduledTask);
        }
        
        methods
    }
    
    fn calculate_sophistication_score(&self, features: &crate::SecurityFeatures) -> f32 {
        let mut score = 0.0;
        
        // High obfuscation indicates sophistication
        score += features.obfuscation_score * 0.3;
        
        // Multiple evasion techniques
        let evasion_count = self.detect_evasion_techniques(features).len() as f32;
        score += (evasion_count / 4.0) * 0.2;
        
        // Complex file structure
        if features.embedded_files.len() > 10 {
            score += 0.2;
        }
        
        // Advanced persistence methods
        let persistence_count = self.detect_persistence_indicators(features).len() as f32;
        score += (persistence_count / 4.0) * 0.2;
        
        // High entropy content
        if features.header_entropy > 7.5 {
            score += 0.1;
        }
        
        score.min(1.0)
    }
    
    fn calculate_potential_impact(&self, features: &crate::SecurityFeatures) -> f32 {
        let mut impact: f32 = 0.0;
        
        // Data collection capabilities
        if features.suspicious_keywords.iter().any(|k| 
            k.contains("password") || k.contains("credential") || k.contains("cookie")) {
            impact += 0.3;
        }
        
        // Network communication capabilities
        if features.url_count > 0 {
            impact += 0.2;
        }
        
        // System modification capabilities
        if features.suspicious_keywords.iter().any(|k| 
            k.contains("registry") || k.contains("system")) {
            impact += 0.3;
        }
        
        // Executable dropping capabilities
        if features.embedded_files.iter().any(|f| f.file_type.contains("executable")) {
            impact += 0.2;
        }
        
        impact.min(1.0)
    }
    
    fn infer_threat_actor_profile(&self, features: &crate::SecurityFeatures) -> ThreatActorProfile {
        let sophistication = self.calculate_sophistication_score(features);
        
        match sophistication {
            s if s > 0.8 => ThreatActorProfile::AdvancedPersistentThreat,
            s if s > 0.6 => ThreatActorProfile::OrganizedCrime,
            s if s > 0.4 => ThreatActorProfile::ExperiencedMalware,
            s if s > 0.2 => ThreatActorProfile::ScriptKiddie,
            _ => ThreatActorProfile::Opportunistic,
        }
    }
    
    fn generate_mitigation_recommendations(&self, assessment: &BehavioralRiskAssessment) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        match assessment.risk_level {
            RiskLevel::Critical | RiskLevel::High => {
                recommendations.push("Immediate isolation of affected systems".to_string());
                recommendations.push("Full incident response activation".to_string());
                recommendations.push("Forensic analysis and threat hunting".to_string());
            },
            RiskLevel::Medium => {
                recommendations.push("Enhanced monitoring and logging".to_string());
                recommendations.push("Application sandboxing".to_string());
                recommendations.push("User awareness training".to_string());
            },
            RiskLevel::Low => {
                recommendations.push("Regular security scans".to_string());
                recommendations.push("Update security policies".to_string());
            },
            RiskLevel::Minimal => {
                recommendations.push("Continue normal monitoring".to_string());
            },
        }
        
        // Specific recommendations based on detected behaviors
        for (behavior, _) in &assessment.detected_behaviors {
            match behavior.as_str() {
                s if s.contains("Network") => {
                    recommendations.push("Monitor network traffic for C&C communication".to_string());
                }
                s if s.contains("Data Collection") => {
                    recommendations.push("Implement data loss prevention controls".to_string());
                }
                s if s.contains("Code Injection") => {
                    recommendations.push("Deploy advanced endpoint protection".to_string());
                }
                _ => {}
            }
        }
        
        recommendations.into_iter().collect::<std::collections::HashSet<_>>().into_iter().collect()
    }
}

impl NeuralSecurityModel for BehavioralAnalyzerModel {
    fn name(&self) -> &str {
        &self.base_model.config.name
    }
    
    fn version(&self) -> &str {
        &self.base_model.config.version
    }
    
    fn input_size(&self) -> usize {
        self.base_model.config.input_size
    }
    
    fn output_size(&self) -> usize {
        self.base_model.config.output_size
    }
    
    fn predict(&self, features: &[f32]) -> Result<Vec<f32>, ProcessingError> {
        Ok(self.base_model.network.lock().unwrap().run(features))
    }
    
    fn train(&mut self, data: &TrainingData) -> Result<TrainingResult, ProcessingError> {
        self.base_model.train_network(data)
    }
    
    fn save(&self, path: &Path) -> Result<(), ProcessingError> {
        self.base_model.save_network(path)
    }
    
    fn load(&mut self, path: &Path) -> Result<(), ProcessingError> {
        self.base_model.load_network(path)
    }
    
    fn get_metrics(&self) -> ModelMetrics {
        self.base_model.metrics.lock().unwrap().clone()
    }
    
    fn validate(&self, test_data: &TrainingData) -> Result<ValidationResult, ProcessingError> {
        // Multi-label validation (behaviors can co-occur)
        let mut true_positives = vec![0u32; 8];
        let mut false_positives = vec![0u32; 8];
        let mut false_negatives = vec![0u32; 8];
        
        for (input, expected) in test_data.inputs.iter().zip(&test_data.outputs) {
            let output = self.predict(input)?;
            
            for i in 0..8 {
                let predicted = output[i] > 0.5;
                let actual = expected[i] > 0.5;
                
                match (predicted, actual) {
                    (true, true) => true_positives[i] += 1,
                    (true, false) => false_positives[i] += 1,
                    (false, true) => false_negatives[i] += 1,
                    _ => {} // true negatives not tracked for multi-label
                }
            }
        }
        
        // Calculate micro-averaged metrics
        let total_tp: u32 = true_positives.iter().sum();
        let total_fp: u32 = false_positives.iter().sum();
        let total_fn: u32 = false_negatives.iter().sum();
        
        let precision = if total_tp + total_fp > 0 {
            total_tp as f32 / (total_tp + total_fp) as f32
        } else {
            0.0
        };
        
        let recall = if total_tp + total_fn > 0 {
            total_tp as f32 / (total_tp + total_fn) as f32
        } else {
            0.0
        };
        
        let f1_score = if precision + recall > 0.0 {
            2.0 * (precision * recall) / (precision + recall)
        } else {
            0.0
        };
        
        // Accuracy for multi-label is challenging, using Jaccard similarity
        let mut jaccard_sum = 0.0;
        for (input, expected) in test_data.inputs.iter().zip(&test_data.outputs) {
            let output = self.predict(input)?;
            
            let predicted_set: Vec<bool> = output.iter().map(|&x| x > 0.5).collect();
            let actual_set: Vec<bool> = expected.iter().map(|&x| x > 0.5).collect();
            
            let intersection = predicted_set.iter().zip(&actual_set)
                .filter(|(p, a)| **p && **a).count();
            let union = predicted_set.iter().zip(&actual_set)
                .filter(|(p, a)| **p || **a).count();
            
            if union > 0 {
                jaccard_sum += intersection as f32 / union as f32;
            }
        }
        
        let accuracy = jaccard_sum / test_data.inputs.len() as f32;
        
        Ok(ValidationResult {
            accuracy,
            precision,
            recall,
            f1_score,
            confusion_matrix: None, // Too complex for multi-label
        })
    }
    
}

/// Behavioral risk assessment result
#[derive(Debug, Clone)]
pub struct BehavioralRiskAssessment {
    pub risk_score: f32,
    pub risk_level: RiskLevel,
    pub detected_behaviors: Vec<(String, f32)>,
    pub critical_combinations: Vec<String>,
    pub confidence: f32,
}

/// Risk level categories
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskLevel {
    Minimal,
    Low,
    Medium,
    High,
    Critical,
}

/// Advanced behavioral assessment result
#[derive(Debug, Clone)]
pub struct AdvancedBehavioralAssessment {
    pub basic_assessment: BehavioralRiskAssessment,
    pub attack_chain_probability: f32,
    pub evasion_techniques: Vec<EvasionTechnique>,
    pub persistence_indicators: Vec<PersistenceMethod>,
    pub sophistication_score: f32,
    pub impact_score: f32,
    pub threat_actor_profile: ThreatActorProfile,
    pub mitigation_recommendations: Vec<String>,
}

/// Evasion techniques detected
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvasionTechnique {
    CodeObfuscation,
    Encryption,
    TimingEvasion,
    SandboxEvasion,
    AntiDebugging,
    Polymorphism,
}

/// Persistence methods detected
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PersistenceMethod {
    RegistryModification,
    StartupFolder,
    ServiceInstallation,
    ScheduledTask,
    DllHijacking,
    BootkitInstallation,
}

/// Threat actor profile inference
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThreatActorProfile {
    Opportunistic,
    ScriptKiddie,
    ExperiencedMalware,
    OrganizedCrime,
    AdvancedPersistentThreat,
    NationState,
}