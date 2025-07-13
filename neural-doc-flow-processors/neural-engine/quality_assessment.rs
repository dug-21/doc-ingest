/// Quality Assessment Module for ruv-FANN Neural Networks
/// Provides document quality evaluation and scoring using neural processing

use crate::neural_engine::fann_wrapper::{FannWrapper, NetworkType};
use crate::neural_engine::NeuralConfig;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Quality Assessment Processor using ruv-FANN
pub struct QualityAssessor {
    pub neural_network: Arc<RwLock<FannWrapper>>,
    pub config: QualityAssessmentConfig,
}

#[derive(Debug, Clone)]
pub struct QualityAssessmentConfig {
    pub grammar_weight: f64,
    pub spelling_weight: f64,
    pub readability_weight: f64,
    pub structure_weight: f64,
    pub content_weight: f64,
    pub formatting_weight: f64,
    pub minimum_score: f64,
    pub accuracy_threshold: f64,
}

impl Default for QualityAssessmentConfig {
    fn default() -> Self {
        Self {
            grammar_weight: 0.2,
            spelling_weight: 0.2,
            readability_weight: 0.15,
            structure_weight: 0.15,
            content_weight: 0.2,
            formatting_weight: 0.1,
            minimum_score: 0.8,
            accuracy_threshold: 0.99,
        }
    }
}

#[derive(Debug, Clone)]
pub struct QualityReport {
    pub overall_score: f64,
    pub grammar_score: f64,
    pub spelling_score: f64,
    pub readability_score: f64,
    pub structure_score: f64,
    pub content_score: f64,
    pub formatting_score: f64,
    pub recommendations: Vec<String>,
    pub confidence: f64,
}

impl QualityAssessor {
    /// Create a new quality assessor with ruv-FANN
    pub async fn new(neural_config: &NeuralConfig, assessment_config: QualityAssessmentConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let neural_network = Arc::new(RwLock::new(
            FannWrapper::new_quality_assessment(neural_config).await?
        ));
        
        Ok(Self {
            neural_network,
            config: assessment_config,
        })
    }
    
    /// Assess document quality using neural processing
    pub async fn assess_quality(&self, input_text: &str) -> Result<QualityReport, Box<dyn std::error::Error>> {
        // Extract quality features
        let features = self.extract_quality_features(input_text).await?;
        
        // Process through neural network
        let network = self.neural_network.read().await;
        let quality_scores = network.process_simd(features).await?;
        
        // Generate comprehensive quality report
        let report = self.generate_quality_report(quality_scores, input_text).await?;
        
        Ok(report)
    }
    
    /// Extract comprehensive quality features for neural processing
    async fn extract_quality_features(&self, text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let mut features = Vec::with_capacity(16);
        
        // Basic text metrics
        let total_chars = text.len() as f32;
        let total_words = text.split_whitespace().count() as f32;
        let total_sentences = text.split('.').count() as f32;
        
        if total_chars == 0.0 {
            return Ok(vec![0.0; 16]);
        }
        
        // Grammar quality indicators
        features.push(self.assess_grammar_quality(text));
        
        // Spelling quality indicators
        features.push(self.assess_spelling_quality(text));
        
        // Readability metrics
        features.push(self.calculate_readability_score(text));
        
        // Structure quality
        features.push(self.assess_structure_quality(text));
        
        // Content coherence
        features.push(self.assess_content_coherence(text));
        
        // Formatting consistency
        features.push(self.assess_formatting_quality(text));
        
        // Vocabulary richness
        features.push(self.calculate_vocabulary_richness(text));
        
        // Sentence complexity
        features.push(self.calculate_sentence_complexity(text));
        
        // Punctuation usage
        features.push(self.assess_punctuation_usage(text));
        
        // Paragraph structure
        features.push(self.assess_paragraph_structure(text));
        
        // Language consistency
        features.push(self.assess_language_consistency(text));
        
        // Content density
        features.push(self.calculate_content_density(text));
        
        // Error density
        features.push(self.calculate_error_density(text));
        
        // Overall coherence
        features.push(self.assess_overall_coherence(text));
        
        // Document completeness
        features.push(self.assess_document_completeness(text));
        
        // Professional quality
        features.push(self.assess_professional_quality(text));
        
        // Ensure exactly 16 features
        features.resize(16, 0.0);
        
        Ok(features)
    }
    
    /// Assess grammar quality
    fn assess_grammar_quality(&self, text: &str) -> f32 {
        let mut score = 1.0;
        let total_words = text.split_whitespace().count() as f32;
        
        if total_words == 0.0 {
            return 0.0;
        }
        
        // Common grammar errors
        let grammar_errors = [
            " i ", " cant ", " wont ", " dont ", " isnt ", " arent ",
            " your welcome", " there house", " its cold", "could of", "should of"
        ];
        
        let mut error_count = 0.0;
        for error in &grammar_errors {
            error_count += text.to_lowercase().matches(error).count() as f32;
        }
        
        // Sentence structure quality
        let sentences: Vec<&str> = text.split('.').collect();
        let mut sentence_quality = 0.0;
        
        for sentence in sentences {
            let words: Vec<&str> = sentence.split_whitespace().collect();
            if words.len() > 2 && words.len() < 30 {
                sentence_quality += 1.0;
            }
        }
        
        if !sentences.is_empty() {
            sentence_quality /= sentences.len() as f32;
        }
        
        score = (sentence_quality * 0.7) + ((1.0 - (error_count / total_words)) * 0.3);
        score.max(0.0).min(1.0)
    }
    
    /// Assess spelling quality
    fn assess_spelling_quality(&self, text: &str) -> f32 {
        let words: Vec<&str> = text.split_whitespace().collect();
        let total_words = words.len() as f32;
        
        if total_words == 0.0 {
            return 0.0;
        }
        
        // Common spelling errors
        let spelling_errors = [
            "recieve", "occured", "seperate", "definately", "neccessary",
            "accomodate", "begining", "beleive", "occurance", "existance",
            "teh", "hte", "adn", "taht", "thier"
        ];
        
        let mut error_count = 0.0;
        for word in words {
            let clean_word = word.trim_matches(|c: char| !c.is_alphabetic()).to_lowercase();
            if spelling_errors.contains(&clean_word.as_str()) {
                error_count += 1.0;
            }
        }
        
        // Word formation quality
        let mut malformed_words = 0.0;
        for word in words {
            let clean_word = word.trim_matches(|c: char| !c.is_alphabetic());
            if clean_word.len() > 2 {
                // Check for repeated characters (potential typos)
                let mut prev_char = ' ';
                let mut repeat_count = 0;
                for ch in clean_word.chars() {
                    if ch == prev_char {
                        repeat_count += 1;
                        if repeat_count > 2 {
                            malformed_words += 0.5;
                            break;
                        }
                    } else {
                        repeat_count = 0;
                    }
                    prev_char = ch;
                }
            }
        }
        
        let error_ratio = (error_count + malformed_words) / total_words;
        (1.0 - error_ratio).max(0.0).min(1.0)
    }
    
    /// Calculate readability score
    fn calculate_readability_score(&self, text: &str) -> f32 {
        let words: Vec<&str> = text.split_whitespace().collect();
        let sentences: Vec<&str> = text.split('.').collect();
        
        if words.is_empty() || sentences.is_empty() {
            return 0.0;
        }
        
        let avg_words_per_sentence = words.len() as f32 / sentences.len() as f32;
        let avg_syllables_per_word = words.iter()
            .map(|w| self.count_syllables(w))
            .sum::<usize>() as f32 / words.len() as f32;
        
        // Simplified Flesch-Kincaid readability
        let readability = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word);
        
        // Normalize to 0-1 scale
        (readability / 100.0).max(0.0).min(1.0)
    }
    
    /// Count syllables in a word
    fn count_syllables(&self, word: &str) -> usize {
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
    
    /// Assess structure quality
    fn assess_structure_quality(&self, text: &str) -> f32 {
        let lines: Vec<&str> = text.lines().collect();
        let paragraphs: Vec<&str> = text.split("\n\n").collect();
        
        if lines.is_empty() {
            return 0.0;
        }
        
        let mut structure_score = 0.0;
        
        // Paragraph length consistency
        let paragraph_lengths: Vec<usize> = paragraphs.iter()
            .map(|p| p.split_whitespace().count())
            .collect();
        
        if !paragraph_lengths.is_empty() {
            let avg_length = paragraph_lengths.iter().sum::<usize>() as f32 / paragraph_lengths.len() as f32;
            let variance = paragraph_lengths.iter()
                .map(|&l| (l as f32 - avg_length).powi(2))
                .sum::<f32>() / paragraph_lengths.len() as f32;
            
            // Lower variance indicates better structure
            structure_score += (1.0 - (variance.sqrt() / 100.0)).max(0.0).min(1.0) * 0.4;
        }
        
        // Heading and list structure
        let potential_headings = lines.iter()
            .filter(|l| {
                let trimmed = l.trim();
                !trimmed.is_empty() && trimmed.len() < 80 && !trimmed.ends_with('.')
            })
            .count() as f32;
        
        structure_score += (potential_headings / lines.len() as f32 * 10.0).min(1.0) * 0.3;
        
        // Consistent indentation
        let indented_lines = lines.iter()
            .filter(|l| l.starts_with(' ') || l.starts_with('\t'))
            .count() as f32;
        
        if indented_lines > 0.0 {
            structure_score += 0.3;
        }
        
        structure_score.min(1.0)
    }
    
    /// Assess content coherence
    fn assess_content_coherence(&self, text: &str) -> f32 {
        let sentences: Vec<&str> = text.split('.').collect();
        
        if sentences.len() < 2 {
            return 0.5;
        }
        
        let mut coherence_score = 0.0;
        let mut word_overlap_score = 0.0;
        
        // Calculate word overlap between consecutive sentences
        for i in 1..sentences.len() {
            let prev_words: std::collections::HashSet<&str> = sentences[i-1]
                .split_whitespace()
                .map(|w| w.trim_matches(|c: char| !c.is_alphabetic()))
                .filter(|w| w.len() > 3)
                .collect();
            
            let curr_words: std::collections::HashSet<&str> = sentences[i]
                .split_whitespace()
                .map(|w| w.trim_matches(|c: char| !c.is_alphabetic()))
                .filter(|w| w.len() > 3)
                .collect();
            
            let intersection = prev_words.intersection(&curr_words).count();
            let union = prev_words.union(&curr_words).count();
            
            if union > 0 {
                word_overlap_score += intersection as f32 / union as f32;
            }
        }
        
        coherence_score = word_overlap_score / (sentences.len() - 1) as f32;
        
        // Topic consistency (simplified)
        let all_words: Vec<&str> = text.split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphabetic()))
            .filter(|w| w.len() > 4)
            .collect();
        
        let unique_words: std::collections::HashSet<&str> = all_words.iter().cloned().collect();
        let vocabulary_density = unique_words.len() as f32 / all_words.len() as f32;
        
        // Balance between variety and repetition
        let topic_consistency = (1.0 - vocabulary_density).max(0.0).min(1.0);
        
        (coherence_score * 0.6 + topic_consistency * 0.4).min(1.0)
    }
    
    /// Assess formatting quality
    fn assess_formatting_quality(&self, text: &str) -> f32 {
        let mut formatting_score = 0.0;
        
        // Consistent punctuation spacing
        let good_spacing = text.matches(". ").count() + text.matches("? ").count() + text.matches("! ").count();
        let bad_spacing = text.matches(" .").count() + text.matches(" ?").count() + text.matches(" !").count();
        
        if good_spacing + bad_spacing > 0 {
            formatting_score += good_spacing as f32 / (good_spacing + bad_spacing) as f32 * 0.3;
        }
        
        // Capitalization after periods
        let sentences: Vec<&str> = text.split('.').collect();
        let mut proper_capitalization = 0;
        
        for sentence in sentences.iter().skip(1) {
            let trimmed = sentence.trim();
            if !trimmed.is_empty() {
                if let Some(first_char) = trimmed.chars().next() {
                    if first_char.is_uppercase() {
                        proper_capitalization += 1;
                    }
                }
            }
        }
        
        if sentences.len() > 1 {
            formatting_score += proper_capitalization as f32 / (sentences.len() - 1) as f32 * 0.4;
        }
        
        // Consistent line breaks and spacing
        let double_spaces = text.matches("  ").count();
        let total_spaces = text.matches(' ').count();
        
        if total_spaces > 0 {
            formatting_score += (1.0 - double_spaces as f32 / total_spaces as f32) * 0.3;
        }
        
        formatting_score.min(1.0)
    }
    
    /// Calculate vocabulary richness
    fn calculate_vocabulary_richness(&self, text: &str) -> f32 {
        let words: Vec<&str> = text.split_whitespace().collect();
        
        if words.is_empty() {
            return 0.0;
        }
        
        let unique_words: std::collections::HashSet<&str> = words.iter().cloned().collect();
        let richness = unique_words.len() as f32 / words.len() as f32;
        
        // Normalize richness score (higher is better, but not too high)
        if richness > 0.7 {
            1.0
        } else if richness > 0.3 {
            richness / 0.7
        } else {
            richness * 2.0
        }
    }
    
    /// Calculate sentence complexity
    fn calculate_sentence_complexity(&self, text: &str) -> f32 {
        let sentences: Vec<&str> = text.split('.').collect();
        
        if sentences.is_empty() {
            return 0.0;
        }
        
        let mut complexity_score = 0.0;
        
        for sentence in sentences {
            let words = sentence.split_whitespace().count();
            let clauses = sentence.matches(',').count() + 1;
            
            // Optimal sentence length is 15-25 words
            let length_score = if words >= 15 && words <= 25 {
                1.0
            } else if words < 15 {
                words as f32 / 15.0
            } else {
                25.0 / words as f32
            };
            
            // Some complexity is good, but not too much
            let clause_score = if clauses >= 2 && clauses <= 4 {
                1.0
            } else if clauses == 1 {
                0.7
            } else {
                4.0 / clauses as f32
            };
            
            complexity_score += (length_score + clause_score) / 2.0;
        }
        
        complexity_score / sentences.len() as f32
    }
    
    /// Assess punctuation usage
    fn assess_punctuation_usage(&self, text: &str) -> f32 {
        let total_chars = text.len() as f32;
        
        if total_chars == 0.0 {
            return 0.0;
        }
        
        let periods = text.matches('.').count() as f32;
        let questions = text.matches('?').count() as f32;
        let exclamations = text.matches('!').count() as f32;
        let commas = text.matches(',').count() as f32;
        
        let total_punctuation = periods + questions + exclamations + commas;
        let punctuation_density = total_punctuation / total_chars;
        
        // Optimal punctuation density is around 3-8%
        if punctuation_density >= 0.03 && punctuation_density <= 0.08 {
            1.0
        } else if punctuation_density < 0.03 {
            punctuation_density / 0.03
        } else {
            0.08 / punctuation_density
        }
    }
    
    /// Assess paragraph structure
    fn assess_paragraph_structure(&self, text: &str) -> f32 {
        let paragraphs: Vec<&str> = text.split("\n\n").collect();
        
        if paragraphs.len() < 2 {
            return 0.5;
        }
        
        let paragraph_lengths: Vec<usize> = paragraphs.iter()
            .map(|p| p.split_whitespace().count())
            .collect();
        
        let avg_length = paragraph_lengths.iter().sum::<usize>() as f32 / paragraph_lengths.len() as f32;
        
        // Optimal paragraph length is 50-150 words
        let length_score = if avg_length >= 50.0 && avg_length <= 150.0 {
            1.0
        } else if avg_length < 50.0 {
            avg_length / 50.0
        } else {
            150.0 / avg_length
        };
        
        length_score.max(0.0).min(1.0)
    }
    
    /// Assess language consistency
    fn assess_language_consistency(&self, text: &str) -> f32 {
        // Simplified language consistency check
        let words: Vec<&str> = text.split_whitespace().collect();
        
        if words.is_empty() {
            return 0.0;
        }
        
        let ascii_words = words.iter()
            .filter(|w| w.chars().all(|c| c.is_ascii()))
            .count() as f32;
        
        ascii_words / words.len() as f32
    }
    
    /// Calculate content density
    fn calculate_content_density(&self, text: &str) -> f32 {
        let total_chars = text.len() as f32;
        
        if total_chars == 0.0 {
            return 0.0;
        }
        
        let content_chars = text.chars()
            .filter(|c| c.is_alphanumeric())
            .count() as f32;
        
        content_chars / total_chars
    }
    
    /// Calculate error density
    fn calculate_error_density(&self, text: &str) -> f32 {
        let total_chars = text.len() as f32;
        
        if total_chars == 0.0 {
            return 1.0;
        }
        
        // Count various types of errors
        let typos = text.matches("teh").count() + text.matches("adn").count() + text.matches("taht").count();
        let double_spaces = text.matches("  ").count();
        let missing_spaces = text.matches(".").count() - text.matches(". ").count();
        
        let total_errors = typos + double_spaces + missing_spaces;
        
        (1.0 - (total_errors as f32 / total_chars * 100.0)).max(0.0)
    }
    
    /// Assess overall coherence
    fn assess_overall_coherence(&self, text: &str) -> f32 {
        // Combine multiple coherence factors
        let content_coherence = self.assess_content_coherence(text);
        let structure_quality = self.assess_structure_quality(text);
        let formatting_quality = self.assess_formatting_quality(text);
        
        (content_coherence * 0.5 + structure_quality * 0.3 + formatting_quality * 0.2)
    }
    
    /// Assess document completeness
    fn assess_document_completeness(&self, text: &str) -> f32 {
        let words = text.split_whitespace().count() as f32;
        let sentences = text.split('.').count() as f32;
        let paragraphs = text.split("\n\n").count() as f32;
        
        // Basic completeness indicators
        let word_completeness = (words / 100.0).min(1.0);
        let sentence_completeness = (sentences / 10.0).min(1.0);
        let paragraph_completeness = (paragraphs / 3.0).min(1.0);
        
        (word_completeness + sentence_completeness + paragraph_completeness) / 3.0
    }
    
    /// Assess professional quality
    fn assess_professional_quality(&self, text: &str) -> f32 {
        let grammar_quality = self.assess_grammar_quality(text);
        let spelling_quality = self.assess_spelling_quality(text);
        let formatting_quality = self.assess_formatting_quality(text);
        let vocabulary_richness = self.calculate_vocabulary_richness(text);
        
        (grammar_quality * 0.3 + spelling_quality * 0.3 + formatting_quality * 0.2 + vocabulary_richness * 0.2)
    }
    
    /// Generate comprehensive quality report
    async fn generate_quality_report(&self, neural_scores: Vec<f32>, text: &str) -> Result<QualityReport, Box<dyn std::error::Error>> {
        let overall_neural_score = neural_scores.iter().sum::<f32>() / neural_scores.len() as f32;
        
        // Individual component scores
        let grammar_score = self.assess_grammar_quality(text) as f64;
        let spelling_score = self.assess_spelling_quality(text) as f64;
        let readability_score = self.calculate_readability_score(text) as f64;
        let structure_score = self.assess_structure_quality(text) as f64;
        let content_score = self.assess_content_coherence(text) as f64;
        let formatting_score = self.assess_formatting_quality(text) as f64;
        
        // Weighted overall score
        let overall_score = (grammar_score * self.config.grammar_weight +
                            spelling_score * self.config.spelling_weight +
                            readability_score * self.config.readability_weight +
                            structure_score * self.config.structure_weight +
                            content_score * self.config.content_weight +
                            formatting_score * self.config.formatting_weight) *
                           overall_neural_score as f64;
        
        // Generate recommendations
        let mut recommendations = Vec::new();
        
        if grammar_score < 0.8 {
            recommendations.push("Improve grammar and sentence structure".to_string());
        }
        if spelling_score < 0.9 {
            recommendations.push("Check spelling and word usage".to_string());
        }
        if readability_score < 0.6 {
            recommendations.push("Simplify language for better readability".to_string());
        }
        if structure_score < 0.7 {
            recommendations.push("Improve document structure and organization".to_string());
        }
        if content_score < 0.7 {
            recommendations.push("Enhance content coherence and flow".to_string());
        }
        if formatting_score < 0.8 {
            recommendations.push("Improve formatting consistency".to_string());
        }
        
        Ok(QualityReport {
            overall_score,
            grammar_score,
            spelling_score,
            readability_score,
            structure_score,
            content_score,
            formatting_score,
            recommendations,
            confidence: overall_neural_score as f64,
        })
    }
    
    /// Train the quality assessment neural network
    pub async fn train(&mut self, training_data: Vec<(String, f64)>) -> Result<(), Box<dyn std::error::Error>> {
        let mut network_training_data = Vec::new();
        
        for (text, target_score) in training_data {
            let input_features = self.extract_quality_features(&text).await?;
            let target_features = vec![target_score as f32];
            network_training_data.push((input_features, target_features));
        }
        
        let mut network = self.neural_network.write().await;
        network.train(network_training_data).await?;
        
        Ok(())
    }
    
    /// Get quality assessment performance metrics
    pub async fn get_performance_metrics(&self) -> Result<QualityAssessmentMetrics, Box<dyn std::error::Error>> {
        let network = self.neural_network.read().await;
        let stats = network.get_stats().await;
        
        Ok(QualityAssessmentMetrics {
            accuracy: stats.accuracy,
            processing_speed: 1.0 / stats.inference_time,
            memory_usage: stats.memory_usage,
            assessment_quality: stats.accuracy * 100.0,
        })
    }
}

#[derive(Debug, Clone)]
pub struct QualityAssessmentMetrics {
    pub accuracy: f64,
    pub processing_speed: f64, // assessments per second
    pub memory_usage: usize,
    pub assessment_quality: f64, // 0-100 quality score
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_engine::NeuralConfig;
    
    #[tokio::test]
    async fn test_quality_assessment() {
        let neural_config = NeuralConfig::default();
        let assessment_config = QualityAssessmentConfig::default();
        
        let assessor = QualityAssessor::new(&neural_config, assessment_config).await.unwrap();
        
        let high_quality_text = "This is a well-written document with proper grammar and spelling. The sentences are clear and concise. The content flows logically from one point to the next.";
        let report = assessor.assess_quality(high_quality_text).await.unwrap();
        
        assert!(report.overall_score > 0.5);
        assert!(report.confidence > 0.0);
    }
    
    #[tokio::test]
    async fn test_feature_extraction() {
        let neural_config = NeuralConfig::default();
        let assessment_config = QualityAssessmentConfig::default();
        
        let assessor = QualityAssessor::new(&neural_config, assessment_config).await.unwrap();
        
        let text = "This is a test document for quality assessment feature extraction.";
        let features = assessor.extract_quality_features(text).await.unwrap();
        
        assert_eq!(features.len(), 16);
        assert!(features.iter().all(|&f| f >= 0.0 && f <= 1.0));
    }
    
    #[tokio::test]
    async fn test_grammar_assessment() {
        let neural_config = NeuralConfig::default();
        let assessment_config = QualityAssessmentConfig::default();
        
        let assessor = QualityAssessor::new(&neural_config, assessment_config).await.unwrap();
        
        let good_grammar = "This is a well-written sentence with proper grammar.";
        let bad_grammar = "this is not written good and have bad grammar";
        
        let good_score = assessor.assess_grammar_quality(good_grammar);
        let bad_score = assessor.assess_grammar_quality(bad_grammar);
        
        assert!(good_score > bad_score);
    }
}