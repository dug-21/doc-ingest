/// Text Enhancement Module for ruv-FANN Neural Networks
/// Provides specialized text enhancement algorithms using neural processing

use crate::neural_engine::fann_wrapper::{FannWrapper, NetworkType};
use crate::neural_engine::NeuralConfig;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Text Enhancement Processor using ruv-FANN
pub struct TextEnhancer {
    pub neural_network: Arc<RwLock<FannWrapper>>,
    pub config: TextEnhancementConfig,
}

#[derive(Debug, Clone)]
pub struct TextEnhancementConfig {
    pub grammar_correction: bool,
    pub spelling_correction: bool,
    pub readability_enhancement: bool,
    pub vocabulary_enrichment: bool,
    pub style_normalization: bool,
    pub accuracy_threshold: f64,
}

impl Default for TextEnhancementConfig {
    fn default() -> Self {
        Self {
            grammar_correction: true,
            spelling_correction: true,
            readability_enhancement: true,
            vocabulary_enrichment: false,
            style_normalization: true,
            accuracy_threshold: 0.99,
        }
    }
}

impl TextEnhancer {
    /// Create a new text enhancer with ruv-FANN
    pub async fn new(neural_config: &NeuralConfig, enhancement_config: TextEnhancementConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let neural_network = Arc::new(RwLock::new(
            FannWrapper::new_text_enhancement(neural_config).await?
        ));
        
        Ok(Self {
            neural_network,
            config: enhancement_config,
        })
    }
    
    /// Enhance text using neural processing
    pub async fn enhance_text(&self, input_text: &str) -> Result<String, Box<dyn std::error::Error>> {
        // Extract text features
        let features = self.extract_text_features(input_text).await?;
        
        // Process through neural network
        let network = self.neural_network.read().await;
        let enhanced_features = network.process_simd(features).await?;
        
        // Reconstruct enhanced text
        let enhanced_text = self.reconstruct_from_features(enhanced_features, input_text).await?;
        
        Ok(enhanced_text)
    }
    
    /// Extract linguistic features for neural processing
    async fn extract_text_features(&self, text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let mut features = Vec::with_capacity(64);
        
        // Character-level features
        let total_chars = text.len() as f32;
        for ch in 'a'..='z' {
            let count = text.chars().filter(|&c| c.to_ascii_lowercase() == ch).count() as f32;
            features.push(count / total_chars);
        }
        
        // Word-level features
        let words: Vec<&str> = text.split_whitespace().collect();
        let word_count = words.len() as f32;
        
        if word_count > 0.0 {
            // Average word length
            let avg_word_len = words.iter().map(|w| w.len()).sum::<usize>() as f32 / word_count;
            features.push(avg_word_len);
            
            // Vocabulary complexity
            let unique_words = words.iter().collect::<std::collections::HashSet<_>>().len() as f32;
            features.push(unique_words / word_count);
        }
        
        // Sentence-level features
        let sentences = text.split('.').count() as f32;
        features.push(sentences);
        
        // Grammar indicators
        let uppercase_count = text.chars().filter(|c| c.is_uppercase()).count() as f32;
        features.push(uppercase_count / total_chars);
        
        // Punctuation density
        let punctuation_count = text.chars().filter(|c| c.is_ascii_punctuation()).count() as f32;
        features.push(punctuation_count / total_chars);
        
        // Readability metrics (simplified)
        let syllable_estimate = words.iter()
            .map(|w| self.estimate_syllables(w))
            .sum::<usize>() as f32;
        if word_count > 0.0 {
            features.push(syllable_estimate / word_count);
        }
        
        // Pad to exactly 64 features
        features.resize(64, 0.0);
        
        Ok(features)
    }
    
    /// Estimate syllables in a word (simple heuristic)
    fn estimate_syllables(&self, word: &str) -> usize {
        let vowels = ['a', 'e', 'i', 'o', 'u', 'y'];
        let mut count = 0;
        let mut prev_was_vowel = false;
        
        for ch in word.to_lowercase().chars() {
            let is_vowel = vowels.contains(&ch);
            if is_vowel && !prev_was_vowel {
                count += 1;
            }
            prev_was_vowel = is_vowel;
        }
        
        if word.ends_with('e') && count > 1 {
            count -= 1;
        }
        
        std::cmp::max(1, count)
    }
    
    /// Reconstruct enhanced text from neural features
    async fn reconstruct_from_features(&self, features: Vec<f32>, original: &str) -> Result<String, Box<dyn std::error::Error>> {
        let mut enhanced = original.to_string();
        
        // Apply enhancements based on neural network output
        if self.config.grammar_correction {
            enhanced = self.apply_grammar_corrections(&enhanced, &features).await?;
        }
        
        if self.config.spelling_correction {
            enhanced = self.apply_spelling_corrections(&enhanced, &features).await?;
        }
        
        if self.config.readability_enhancement {
            enhanced = self.enhance_readability(&enhanced, &features).await?;
        }
        
        if self.config.style_normalization {
            enhanced = self.normalize_style(&enhanced, &features).await?;
        }
        
        Ok(enhanced)
    }
    
    /// Apply grammar corrections based on neural features
    async fn apply_grammar_corrections(&self, text: &str, features: &[f32]) -> Result<String, Box<dyn std::error::Error>> {
        let mut corrected = text.to_string();
        
        // Neural network confidence threshold for grammar corrections
        let grammar_confidence = features.iter().take(10).sum::<f32>() / 10.0;
        
        if grammar_confidence > 0.7 {
            // Basic grammar corrections
            corrected = corrected.replace(" i ", " I ");
            corrected = corrected.replace("cant", "can't");
            corrected = corrected.replace("wont", "won't");
            corrected = corrected.replace("dont", "don't");
            
            // Sentence capitalization
            let sentences: Vec<&str> = corrected.split('.').collect();
            corrected = sentences.iter()
                .map(|s| {
                    let trimmed = s.trim();
                    if !trimmed.is_empty() {
                        let mut chars = trimmed.chars();
                        if let Some(first) = chars.next() {
                            first.to_uppercase().collect::<String>() + &chars.collect::<String>()
                        } else {
                            trimmed.to_string()
                        }
                    } else {
                        trimmed.to_string()
                    }
                })
                .collect::<Vec<_>>()
                .join(".");
        }
        
        Ok(corrected)
    }
    
    /// Apply spelling corrections based on neural features
    async fn apply_spelling_corrections(&self, text: &str, features: &[f32]) -> Result<String, Box<dyn std::error::Error>> {
        let mut corrected = text.to_string();
        
        // Neural network confidence for spelling
        let spelling_confidence = features.iter().skip(10).take(10).sum::<f32>() / 10.0;
        
        if spelling_confidence > 0.6 {
            // Common spelling corrections
            let corrections = [
                ("recieve", "receive"),
                ("occured", "occurred"),
                ("seperate", "separate"),
                ("definately", "definitely"),
                ("neccessary", "necessary"),
                ("accomodate", "accommodate"),
                ("begining", "beginning"),
                ("beleive", "believe"),
                ("occurance", "occurrence"),
                ("existance", "existence"),
            ];
            
            for (wrong, right) in &corrections {
                corrected = corrected.replace(wrong, right);
                corrected = corrected.replace(&wrong.to_uppercase(), &right.to_uppercase());
            }
        }
        
        Ok(corrected)
    }
    
    /// Enhance readability based on neural features
    async fn enhance_readability(&self, text: &str, features: &[f32]) -> Result<String, Box<dyn std::error::Error>> {
        let mut enhanced = text.to_string();
        
        // Neural network readability assessment
        let readability_score = features.iter().skip(20).take(15).sum::<f32>() / 15.0;
        
        if readability_score < 0.5 {
            // Break up long sentences
            enhanced = enhanced.replace(", and ", ". ");
            enhanced = enhanced.replace(", but ", ". However, ");
            enhanced = enhanced.replace(", so ", ". Therefore, ");
            
            // Simplify complex words (basic examples)
            enhanced = enhanced.replace("utilize", "use");
            enhanced = enhanced.replace("facilitate", "help");
            enhanced = enhanced.replace("demonstrate", "show");
            enhanced = enhanced.replace("approximately", "about");
        }
        
        Ok(enhanced)
    }
    
    /// Normalize writing style based on neural features
    async fn normalize_style(&self, text: &str, features: &[f32]) -> Result<String, Box<dyn std::error::Error>> {
        let mut normalized = text.to_string();
        
        // Style consistency features
        let style_variance = features.iter().skip(35).take(10).sum::<f32>() / 10.0;
        
        if style_variance > 0.3 {
            // Normalize contractions
            normalized = normalized.replace("cannot", "can't");
            normalized = normalized.replace("will not", "won't");
            normalized = normalized.replace("do not", "don't");
            
            // Consistent punctuation spacing
            normalized = normalized.replace(" ,", ",");
            normalized = normalized.replace(" .", ".");
            normalized = normalized.replace("  ", " ");
            
            // Remove extra whitespace
            while normalized.contains("  ") {
                normalized = normalized.replace("  ", " ");
            }
        }
        
        Ok(normalized)
    }
    
    /// Train the text enhancement neural network
    pub async fn train(&mut self, training_data: Vec<(String, String)>) -> Result<(), Box<dyn std::error::Error>> {
        let mut network_training_data = Vec::new();
        
        for (input_text, target_text) in training_data {
            let input_features = self.extract_text_features(&input_text).await?;
            let target_features = self.extract_text_features(&target_text).await?;
            network_training_data.push((input_features, target_features));
        }
        
        let mut network = self.neural_network.write().await;
        network.train(network_training_data).await?;
        
        Ok(())
    }
    
    /// Get text enhancement performance metrics
    pub async fn get_performance_metrics(&self) -> Result<TextEnhancementMetrics, Box<dyn std::error::Error>> {
        let network = self.neural_network.read().await;
        let stats = network.get_stats().await;
        
        Ok(TextEnhancementMetrics {
            accuracy: stats.accuracy,
            processing_speed: 1.0 / stats.inference_time,
            memory_usage: stats.memory_usage,
            enhancement_quality: stats.accuracy * 100.0,
        })
    }
}

#[derive(Debug, Clone)]
pub struct TextEnhancementMetrics {
    pub accuracy: f64,
    pub processing_speed: f64, // enhancements per second
    pub memory_usage: usize,
    pub enhancement_quality: f64, // 0-100 quality score
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_engine::NeuralConfig;
    
    #[tokio::test]
    async fn test_text_enhancement() {
        let neural_config = NeuralConfig::default();
        let enhancement_config = TextEnhancementConfig::default();
        
        let enhancer = TextEnhancer::new(&neural_config, enhancement_config).await.unwrap();
        
        let input = "this is a test sentance with some errors.";
        let result = enhancer.enhance_text(input).await;
        
        assert!(result.is_ok());
        let enhanced = result.unwrap();
        assert!(!enhanced.is_empty());
    }
    
    #[tokio::test]
    async fn test_feature_extraction() {
        let neural_config = NeuralConfig::default();
        let enhancement_config = TextEnhancementConfig::default();
        
        let enhancer = TextEnhancer::new(&neural_config, enhancement_config).await.unwrap();
        
        let text = "This is a sample text for feature extraction testing.";
        let features = enhancer.extract_text_features(text).await.unwrap();
        
        assert_eq!(features.len(), 64);
        assert!(features.iter().all(|&f| f >= 0.0 && f <= 1.0));
    }
}