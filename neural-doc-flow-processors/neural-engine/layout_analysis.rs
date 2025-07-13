/// Layout Analysis Module for ruv-FANN Neural Networks
/// Provides document layout analysis and optimization using neural processing

use crate::neural_engine::fann_wrapper::{FannWrapper, NetworkType};
use crate::neural_engine::NeuralConfig;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Layout Analysis Processor using ruv-FANN
pub struct LayoutAnalyzer {
    pub neural_network: Arc<RwLock<FannWrapper>>,
    pub config: LayoutAnalysisConfig,
}

#[derive(Debug, Clone)]
pub struct LayoutAnalysisConfig {
    pub paragraph_optimization: bool,
    pub heading_detection: bool,
    pub table_analysis: bool,
    pub list_formatting: bool,
    pub whitespace_normalization: bool,
    pub column_detection: bool,
    pub accuracy_threshold: f64,
}

impl Default for LayoutAnalysisConfig {
    fn default() -> Self {
        Self {
            paragraph_optimization: true,
            heading_detection: true,
            table_analysis: false,
            list_formatting: true,
            whitespace_normalization: true,
            column_detection: false,
            accuracy_threshold: 0.95,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LayoutFeatures {
    pub line_count: f32,
    pub average_line_length: f32,
    pub paragraph_count: f32,
    pub indentation_ratio: f32,
    pub heading_patterns: f32,
    pub list_patterns: f32,
    pub table_patterns: f32,
    pub whitespace_density: f32,
    pub column_indicators: f32,
    pub structure_complexity: f32,
}

impl LayoutAnalyzer {
    /// Create a new layout analyzer with ruv-FANN
    pub async fn new(neural_config: &NeuralConfig, layout_config: LayoutAnalysisConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let neural_network = Arc::new(RwLock::new(
            FannWrapper::new_layout_analysis(neural_config).await?
        ));
        
        Ok(Self {
            neural_network,
            config: layout_config,
        })
    }
    
    /// Analyze and optimize document layout
    pub async fn analyze_layout(&self, input_text: &str) -> Result<String, Box<dyn std::error::Error>> {
        // Extract layout features
        let features = self.extract_layout_features(input_text).await?;
        
        // Process through neural network
        let network = self.neural_network.read().await;
        let analyzed_features = network.process_simd(features).await?;
        
        // Reconstruct optimized layout
        let optimized_layout = self.reconstruct_layout(analyzed_features, input_text).await?;
        
        Ok(optimized_layout)
    }
    
    /// Extract comprehensive layout features for neural processing
    async fn extract_layout_features(&self, text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let mut features = Vec::with_capacity(32);
        
        let lines: Vec<&str> = text.lines().collect();
        let total_lines = lines.len() as f32;
        
        // Basic line metrics
        features.push(total_lines);
        
        if total_lines > 0.0 {
            // Average line length
            let avg_line_length = lines.iter().map(|l| l.len()).sum::<usize>() as f32 / total_lines;
            features.push(avg_line_length);
            
            // Line length variance
            let variance = lines.iter()
                .map(|l| (l.len() as f32 - avg_line_length).powi(2))
                .sum::<f32>() / total_lines;
            features.push(variance.sqrt());
        } else {
            features.push(0.0);
            features.push(0.0);
        }
        
        // Paragraph analysis
        let paragraphs: Vec<&str> = text.split("\n\n").collect();
        features.push(paragraphs.len() as f32);
        
        // Indentation patterns
        let indented_lines = lines.iter().filter(|l| l.starts_with(' ') || l.starts_with('\t')).count() as f32;
        features.push(indented_lines / total_lines);
        
        // Different indentation levels
        let mut indent_levels = std::collections::HashSet::new();
        for line in &lines {
            let indent = line.chars().take_while(|c| c.is_whitespace()).count();
            if indent > 0 {
                indent_levels.insert(indent);
            }
        }
        features.push(indent_levels.len() as f32);
        
        // Heading detection patterns
        let potential_headings = lines.iter()
            .filter(|l| {
                let trimmed = l.trim();
                !trimmed.is_empty() && 
                (trimmed.chars().all(|c| c.is_uppercase() || c.is_whitespace() || c.is_ascii_punctuation()) ||
                 trimmed.len() < 50)
            })
            .count() as f32;
        features.push(potential_headings / total_lines);
        
        // List pattern detection
        let list_indicators = ["-", "*", "+", "•"];
        let list_lines = lines.iter()
            .filter(|l| {
                let trimmed = l.trim();
                list_indicators.iter().any(|&indicator| trimmed.starts_with(indicator)) ||
                trimmed.chars().take(3).collect::<String>().matches(char::is_numeric).count() > 0
            })
            .count() as f32;
        features.push(list_lines / total_lines);
        
        // Table pattern detection (simplified)
        let table_indicators = ["|", "\t"];
        let table_lines = lines.iter()
            .filter(|l| table_indicators.iter().any(|&indicator| l.contains(indicator)))
            .count() as f32;
        features.push(table_lines / total_lines);
        
        // Whitespace analysis
        let total_chars = text.len() as f32;
        let whitespace_chars = text.chars().filter(|c| c.is_whitespace()).count() as f32;
        features.push(whitespace_chars / total_chars);
        
        // Empty line patterns
        let empty_lines = lines.iter().filter(|l| l.trim().is_empty()).count() as f32;
        features.push(empty_lines / total_lines);
        
        // Line consistency patterns
        let consistent_starts = self.analyze_line_start_consistency(&lines);
        features.push(consistent_starts);
        
        // Column detection (basic)
        let column_indicators = self.detect_column_patterns(&lines);
        features.push(column_indicators);
        
        // Structure complexity
        let complexity = self.calculate_structure_complexity(&lines);
        features.push(complexity);
        
        // Word distribution across lines
        let words_per_line: Vec<usize> = lines.iter().map(|l| l.split_whitespace().count()).collect();
        if !words_per_line.is_empty() {
            let avg_words = words_per_line.iter().sum::<usize>() as f32 / words_per_line.len() as f32;
            features.push(avg_words);
            
            let word_variance = words_per_line.iter()
                .map(|&w| (w as f32 - avg_words).powi(2))
                .sum::<f32>() / words_per_line.len() as f32;
            features.push(word_variance.sqrt());
        } else {
            features.push(0.0);
            features.push(0.0);
        }
        
        // Punctuation distribution
        let punctuation_density = text.chars().filter(|c| c.is_ascii_punctuation()).count() as f32 / total_chars;
        features.push(punctuation_density);
        
        // Capitalization patterns
        let capital_lines = lines.iter()
            .filter(|l| l.chars().filter(|c| c.is_alphabetic()).any(|c| c.is_uppercase()))
            .count() as f32;
        features.push(capital_lines / total_lines);
        
        // Sentence structure within layout
        let sentences_per_paragraph = paragraphs.iter()
            .map(|p| p.split('.').count() as f32)
            .collect::<Vec<_>>();
        if !sentences_per_paragraph.is_empty() {
            let avg_sentences = sentences_per_paragraph.iter().sum::<f32>() / sentences_per_paragraph.len() as f32;
            features.push(avg_sentences);
        } else {
            features.push(0.0);
        }
        
        // Pad to exactly 32 features
        features.resize(32, 0.0);
        
        Ok(features)
    }
    
    /// Analyze consistency of line starting patterns
    fn analyze_line_start_consistency(&self, lines: &[&str]) -> f32 {
        let mut start_patterns = std::collections::HashMap::new();
        
        for line in lines {
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                let start_char = trimmed.chars().next().unwrap_or(' ');
                *start_patterns.entry(start_char).or_insert(0) += 1;
            }
        }
        
        if start_patterns.is_empty() {
            return 0.0;
        }
        
        let max_pattern = *start_patterns.values().max().unwrap_or(&0) as f32;
        max_pattern / lines.len() as f32
    }
    
    /// Detect potential column patterns in text
    fn detect_column_patterns(&self, lines: &[&str]) -> f32 {
        if lines.len() < 3 {
            return 0.0;
        }
        
        let mut consistent_spacing = 0;
        
        for i in 1..lines.len() {
            let prev_words: Vec<&str> = lines[i-1].split_whitespace().collect();
            let curr_words: Vec<&str> = lines[i].split_whitespace().collect();
            
            if prev_words.len() > 1 && curr_words.len() > 1 {
                // Check if words align in columns
                let prev_positions: Vec<usize> = prev_words.iter()
                    .scan(0, |pos, word| {
                        let start = *pos;
                        *pos += word.len() + 1;
                        Some(start)
                    })
                    .collect();
                
                let curr_positions: Vec<usize> = curr_words.iter()
                    .scan(0, |pos, word| {
                        let start = *pos;
                        *pos += word.len() + 1;
                        Some(start)
                    })
                    .collect();
                
                let alignment_score = prev_positions.iter()
                    .zip(curr_positions.iter())
                    .filter(|(p, c)| (*p as i32 - *c as i32).abs() < 3)
                    .count();
                
                if alignment_score > prev_positions.len() / 2 {
                    consistent_spacing += 1;
                }
            }
        }
        
        consistent_spacing as f32 / (lines.len() - 1) as f32
    }
    
    /// Calculate overall structure complexity
    fn calculate_structure_complexity(&self, lines: &[&str]) -> f32 {
        let mut complexity = 0.0;
        
        // Indent level changes
        let mut prev_indent = 0;
        for line in lines {
            let current_indent = line.chars().take_while(|c| c.is_whitespace()).count();
            if current_indent != prev_indent {
                complexity += 0.1;
            }
            prev_indent = current_indent;
        }
        
        // Line length variations
        let lengths: Vec<usize> = lines.iter().map(|l| l.len()).collect();
        if !lengths.is_empty() {
            let avg_length = lengths.iter().sum::<usize>() as f32 / lengths.len() as f32;
            let variance = lengths.iter()
                .map(|&l| (l as f32 - avg_length).powi(2))
                .sum::<f32>() / lengths.len() as f32;
            complexity += variance.sqrt() / 100.0;
        }
        
        complexity.min(1.0)
    }
    
    /// Reconstruct optimized layout from neural features
    async fn reconstruct_layout(&self, features: Vec<f32>, original: &str) -> Result<String, Box<dyn std::error::Error>> {
        let mut optimized = original.to_string();
        
        // Apply optimizations based on neural network output
        if self.config.paragraph_optimization {
            optimized = self.optimize_paragraphs(&optimized, &features).await?;
        }
        
        if self.config.heading_detection {
            optimized = self.enhance_headings(&optimized, &features).await?;
        }
        
        if self.config.list_formatting {
            optimized = self.format_lists(&optimized, &features).await?;
        }
        
        if self.config.whitespace_normalization {
            optimized = self.normalize_whitespace(&optimized, &features).await?;
        }
        
        Ok(optimized)
    }
    
    /// Optimize paragraph structure based on neural analysis
    async fn optimize_paragraphs(&self, text: &str, features: &[f32]) -> Result<String, Box<dyn std::error::Error>> {
        let mut optimized = text.to_string();
        
        // Neural confidence for paragraph optimization
        let paragraph_confidence = features.iter().take(5).sum::<f32>() / 5.0;
        
        if paragraph_confidence > 0.7 {
            // Normalize paragraph spacing
            while optimized.contains("\n\n\n") {
                optimized = optimized.replace("\n\n\n", "\n\n");
            }
            
            // Add paragraph breaks where sentences are long
            let sentences: Vec<&str> = optimized.split('.').collect();
            optimized = sentences.iter()
                .map(|s| {
                    if s.len() > 200 {
                        // Break long sentences into paragraphs
                        s.replace(", ", ".\n\n")
                    } else {
                        s.to_string()
                    }
                })
                .collect::<Vec<_>>()
                .join(".");
        }
        
        Ok(optimized)
    }
    
    /// Enhance heading detection and formatting
    async fn enhance_headings(&self, text: &str, features: &[f32]) -> Result<String, Box<dyn std::error::Error>> {
        let mut enhanced = text.to_string();
        
        // Neural confidence for heading detection
        let heading_confidence = features.iter().skip(5).take(5).sum::<f32>() / 5.0;
        
        if heading_confidence > 0.6 {
            let lines: Vec<&str> = enhanced.lines().collect();
            let mut result_lines = Vec::new();
            
            for line in lines {
                let trimmed = line.trim();
                
                // Detect potential headings
                if !trimmed.is_empty() && 
                   trimmed.len() < 80 && 
                   !trimmed.ends_with('.') &&
                   trimmed.chars().any(|c| c.is_uppercase()) {
                    
                    // Format as heading with proper spacing
                    result_lines.push("".to_string());
                    result_lines.push(format!("# {}", trimmed));
                    result_lines.push("".to_string());
                } else {
                    result_lines.push(line.to_string());
                }
            }
            
            enhanced = result_lines.join("\n");
        }
        
        Ok(enhanced)
    }
    
    /// Format lists based on neural analysis
    async fn format_lists(&self, text: &str, features: &[f32]) -> Result<String, Box<dyn std::error::Error>> {
        let mut formatted = text.to_string();
        
        // Neural confidence for list formatting
        let list_confidence = features.iter().skip(10).take(5).sum::<f32>() / 5.0;
        
        if list_confidence > 0.5 {
            let lines: Vec<&str> = formatted.lines().collect();
            let mut result_lines = Vec::new();
            
            for line in lines {
                let trimmed = line.trim();
                
                // Detect and format list items
                if trimmed.starts_with('-') || trimmed.starts_with('*') || trimmed.starts_with('+') {
                    result_lines.push(format!("- {}", &trimmed[1..].trim()));
                } else if trimmed.chars().take(3).any(|c| c.is_numeric()) && trimmed.contains('.') {
                    // Numbered list item
                    result_lines.push(trimmed.to_string());
                } else {
                    result_lines.push(line.to_string());
                }
            }
            
            formatted = result_lines.join("\n");
        }
        
        Ok(formatted)
    }
    
    /// Normalize whitespace based on neural analysis
    async fn normalize_whitespace(&self, text: &str, features: &[f32]) -> Result<String, Box<dyn std::error::Error>> {
        let mut normalized = text.to_string();
        
        // Neural confidence for whitespace normalization
        let whitespace_confidence = features.iter().skip(15).take(5).sum::<f32>() / 5.0;
        
        if whitespace_confidence > 0.4 {
            // Remove trailing whitespace
            let lines: Vec<&str> = normalized.lines().collect();
            normalized = lines.iter()
                .map(|l| l.trim_end())
                .collect::<Vec<_>>()
                .join("\n");
            
            // Normalize multiple spaces
            while normalized.contains("  ") {
                normalized = normalized.replace("  ", " ");
            }
            
            // Normalize tabs to spaces
            normalized = normalized.replace('\t', "    ");
        }
        
        Ok(normalized)
    }
    
    /// Train the layout analysis neural network
    pub async fn train(&mut self, training_data: Vec<(String, String)>) -> Result<(), Box<dyn std::error::Error>> {
        let mut network_training_data = Vec::new();
        
        for (input_layout, target_layout) in training_data {
            let input_features = self.extract_layout_features(&input_layout).await?;
            let target_features = self.extract_layout_features(&target_layout).await?;
            network_training_data.push((input_features, target_features));
        }
        
        let mut network = self.neural_network.write().await;
        network.train(network_training_data).await?;
        
        Ok(())
    }
    
    /// Get layout analysis performance metrics
    pub async fn get_performance_metrics(&self) -> Result<LayoutAnalysisMetrics, Box<dyn std::error::Error>> {
        let network = self.neural_network.read().await;
        let stats = network.get_stats().await;
        
        Ok(LayoutAnalysisMetrics {
            accuracy: stats.accuracy,
            processing_speed: 1.0 / stats.inference_time,
            memory_usage: stats.memory_usage,
            layout_quality: stats.accuracy * 100.0,
        })
    }
}

#[derive(Debug, Clone)]
pub struct LayoutAnalysisMetrics {
    pub accuracy: f64,
    pub processing_speed: f64, // layouts per second
    pub memory_usage: usize,
    pub layout_quality: f64, // 0-100 quality score
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_engine::NeuralConfig;
    
    #[tokio::test]
    async fn test_layout_analysis() {
        let neural_config = NeuralConfig::default();
        let layout_config = LayoutAnalysisConfig::default();
        
        let analyzer = LayoutAnalyzer::new(&neural_config, layout_config).await.unwrap();
        
        let input = "This is a heading\n\nThis is a paragraph with some text.\n- List item 1\n- List item 2";
        let result = analyzer.analyze_layout(input).await;
        
        assert!(result.is_ok());
        let analyzed = result.unwrap();
        assert!(!analyzed.is_empty());
    }
    
    #[tokio::test]
    async fn test_feature_extraction() {
        let neural_config = NeuralConfig::default();
        let layout_config = LayoutAnalysisConfig::default();
        
        let analyzer = LayoutAnalyzer::new(&neural_config, layout_config).await.unwrap();
        
        let text = "Line 1\n  Indented line\nAnother line\n\nNew paragraph";
        let features = analyzer.extract_layout_features(text).await.unwrap();
        
        assert_eq!(features.len(), 32);
        assert!(features.iter().all(|&f| f >= 0.0));
    }
}