//! Binary for testing neural security models

use neural_doc_flow_security::models::*;
use neural_doc_flow_security::SecurityFeatures;
use neural_doc_flow_security::EmbeddedFile;
use std::path::PathBuf;
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    
    let models_dir = if args.len() > 1 {
        PathBuf::from(&args[1])
    } else {
        PathBuf::from("./models")
    };
    
    println!("🧪 Neural Security Models Testing Tool");
    println!("======================================");
    println!("📁 Models directory: {:?}", models_dir);
    println!();
    
    // Create test security features
    let test_features = create_test_malware_features();
    let benign_features = create_test_benign_features();
    
    println!("🔍 Testing Malware Detector Model...");
    test_malware_detector(&models_dir, &test_features, &benign_features)?;
    
    println!("\n🎯 Testing Threat Classifier Model...");
    test_threat_classifier(&models_dir, &test_features)?;
    
    println!("\n🚨 Testing Anomaly Detector Model...");
    test_anomaly_detector(&models_dir, &test_features, &benign_features)?;
    
    println!("\n🧠 Testing Behavioral Analyzer Model...");
    test_behavioral_analyzer(&models_dir, &test_features)?;
    
    println!("\n💥 Testing Exploit Detector Model...");
    test_exploit_detector(&models_dir, &test_features)?;
    
    println!("\n✅ All model tests completed!");
    
    Ok(())
}

fn create_test_malware_features() -> SecurityFeatures {
    SecurityFeatures {
        file_size: 2_500_000,
        header_entropy: 8.2,
        stream_count: 15,
        javascript_present: true,
        embedded_files: vec![
            EmbeddedFile {
                name: "payload.exe".to_string(),
                size: 850_000,
                file_type: "application/x-executable".to_string(),
            },
            EmbeddedFile {
                name: "script.js".to_string(),
                size: 45_000,
                file_type: "application/javascript".to_string(),
            }
        ],
        suspicious_keywords: vec![
            "eval".to_string(),
            "exec".to_string(), 
            "powershell".to_string(),
            "VirtualProtect".to_string(),
            "WriteProcessMemory".to_string(),
        ],
        url_count: 12,
        obfuscation_score: 0.89,
    }
}

fn create_test_benign_features() -> SecurityFeatures {
    SecurityFeatures {
        file_size: 125_000,
        header_entropy: 5.2,
        stream_count: 3,
        javascript_present: false,
        embedded_files: vec![
            EmbeddedFile {
                name: "image.jpg".to_string(),
                size: 85_000,
                file_type: "image/jpeg".to_string(),
            }
        ],
        suspicious_keywords: vec![],
        url_count: 2,
        obfuscation_score: 0.05,
    }
}

fn test_malware_detector(models_dir: &PathBuf, malware_features: &SecurityFeatures, benign_features: &SecurityFeatures) -> Result<(), Box<dyn std::error::Error>> {
    let model = MalwareDetectorModel::load_or_create(models_dir)?;
    
    // Test with malware features
    let malware_input = MalwareDetectorModel::extract_features(malware_features);
    let malware_result = model.detect(&malware_input)?;
    
    println!("  🦠 Malware sample:");
    println!("    Probability: {:.3}", malware_result.probability);
    println!("    Is malicious: {}", malware_result.is_malicious);
    println!("    Confidence: {:.3}", malware_result.confidence);
    println!("    Threat level: {:?}", malware_result.threat_level);
    
    // Test with benign features
    let benign_input = MalwareDetectorModel::extract_features(benign_features);
    let benign_result = model.detect(&benign_input)?;
    
    println!("  ✅ Benign sample:");
    println!("    Probability: {:.3}", benign_result.probability);
    println!("    Is malicious: {}", benign_result.is_malicious);
    println!("    Confidence: {:.3}", benign_result.confidence);
    println!("    Threat level: {:?}", benign_result.threat_level);
    
    Ok(())
}

fn test_threat_classifier(models_dir: &PathBuf, test_features: &SecurityFeatures) -> Result<(), Box<dyn std::error::Error>> {
    let model = ThreatClassifierModel::load_or_create(models_dir)?;
    
    let input = ThreatClassifierModel::extract_features(test_features);
    let classification = model.analyze_threat_context(&input, test_features)?;
    
    println!("  🎯 Threat Classification:");
    println!("    Primary category: {}", classification.primary_category);
    println!("    Confidence: {:.3}", classification.confidence);
    println!("    Severity: {:.3}", classification.calculate_severity());
    println!("    Recommended action: {}", classification.get_recommended_action());
    
    println!("    All categories:");
    for (category, prob) in &classification.all_categories {
        println!("      • {}: {:.3}", category, prob);
    }
    
    Ok(())
}

fn test_anomaly_detector(models_dir: &PathBuf, malware_features: &SecurityFeatures, benign_features: &SecurityFeatures) -> Result<(), Box<dyn std::error::Error>> {
    let model = AnomalyDetectorModel::load_or_create(models_dir)?;
    
    // Test with malware features (should be anomalous)
    let malware_input = AnomalyDetectorModel::extract_features(malware_features);
    let malware_analysis = model.detect_advanced_anomalies(&malware_input, malware_features)?;
    
    println!("  🚨 Malware sample anomaly analysis:");
    println!("    Combined score: {:.3}", malware_analysis.combined_score);
    println!("    Severity: {:?}", malware_analysis.severity);
    println!("    Size anomaly: {:.3}", malware_analysis.size_anomaly_score);
    println!("    Entropy anomaly: {:.3}", malware_analysis.entropy_anomaly_score);
    println!("    Structure anomaly: {:.3}", malware_analysis.structure_anomaly_score);
    
    if !malware_analysis.anomaly_indicators.is_empty() {
        println!("    Indicators:");
        for indicator in &malware_analysis.anomaly_indicators {
            println!("      • {}", indicator);
        }
    }
    
    // Test with benign features (should be normal)
    let benign_input = AnomalyDetectorModel::extract_features(benign_features);
    let benign_analysis = model.detect_advanced_anomalies(&benign_input, benign_features)?;
    
    println!("  ✅ Benign sample anomaly analysis:");
    println!("    Combined score: {:.3}", benign_analysis.combined_score);
    println!("    Severity: {:?}", benign_analysis.severity);
    
    Ok(())
}

fn test_behavioral_analyzer(models_dir: &PathBuf, test_features: &SecurityFeatures) -> Result<(), Box<dyn std::error::Error>> {
    let model = BehavioralAnalyzerModel::load_or_create(models_dir)?;
    
    let input = BehavioralAnalyzerModel::extract_features(test_features);
    let assessment = model.analyze_advanced_behavior(&input, test_features)?;
    
    println!("  🧠 Behavioral Analysis:");
    println!("    Risk score: {:.3}", assessment.basic_assessment.risk_score);
    println!("    Risk level: {:?}", assessment.basic_assessment.risk_level);
    println!("    Attack chain probability: {:.3}", assessment.attack_chain_probability);
    println!("    Sophistication score: {:.3}", assessment.sophistication_score);
    println!("    Impact score: {:.3}", assessment.impact_score);
    println!("    Threat actor profile: {:?}", assessment.threat_actor_profile);
    
    if !assessment.evasion_techniques.is_empty() {
        println!("    Evasion techniques:");
        for technique in &assessment.evasion_techniques {
            println!("      • {:?}", technique);
        }
    }
    
    if !assessment.persistence_indicators.is_empty() {
        println!("    Persistence indicators:");
        for method in &assessment.persistence_indicators {
            println!("      • {:?}", method);
        }
    }
    
    if !assessment.mitigation_recommendations.is_empty() {
        println!("    Mitigation recommendations:");
        for recommendation in &assessment.mitigation_recommendations {
            println!("      • {}", recommendation);
        }
    }
    
    Ok(())
}

fn test_exploit_detector(models_dir: &PathBuf, test_features: &SecurityFeatures) -> Result<(), Box<dyn std::error::Error>> {
    let model = ExploitDetectorModel::load_or_create(models_dir)?;
    
    let input = ExploitDetectorModel::extract_features(test_features);
    let analysis = model.analyze_exploit_advanced(&input, test_features)?;
    
    println!("  💥 Exploit Analysis:");
    println!("    Exploit probability: {:.3}", analysis.basic_analysis.exploit_probability);
    println!("    Exploit type: {:?}", analysis.basic_analysis.exploit_type);
    println!("    Severity: {:.3}", analysis.basic_analysis.severity);
    println!("    Confidence: {:.3}", analysis.basic_analysis.confidence);
    println!("    Weaponization level: {:?}", analysis.weaponization_level);
    
    if !analysis.basic_analysis.detected_signatures.is_empty() {
        println!("    Detected signatures:");
        for signature in &analysis.basic_analysis.detected_signatures {
            println!("      • {}", signature);
        }
    }
    
    if !analysis.delivery_mechanisms.is_empty() {
        println!("    Delivery mechanisms:");
        for mechanism in &analysis.delivery_mechanisms {
            println!("      • {:?}", mechanism);
        }
    }
    
    if let Some(threat_group) = &analysis.attribution_data.threat_group {
        println!("    Likely threat group: {}", threat_group);
    }
    
    println!("    Development timeline: {}", analysis.threat_timeline.development_time_estimate);
    println!("    Threat maturity: {}", analysis.threat_timeline.threat_maturity);
    
    if !analysis.countermeasures.is_empty() {
        println!("    Recommended countermeasures:");
        for countermeasure in &analysis.countermeasures {
            println!("      • {}", countermeasure);
        }
    }
    
    Ok(())
}