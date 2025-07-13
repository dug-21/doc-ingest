//! Neural enhancement example
//!
//! This example demonstrates how to configure and use neural enhancement
//! features in NeuralDocFlow for improved extraction accuracy.

use neuraldocflow::{
    DocFlow, SourceInput, Config, NeuralConfig, 
    neural::{NeuralModel, TableStructure, ModelLoader},
    core::{ExtractedDocument, ContentBlock, BlockPosition},
    error::Result,
};
use async_trait::async_trait;
use std::path::PathBuf;
use std::time::Duration;

/// Custom neural model for demonstration
pub struct DemoNeuralModel {
    name: String,
    version: String,
    enhancement_strength: f32,
}

impl DemoNeuralModel {
    pub fn new(enhancement_strength: f32) -> Self {
        Self {
            name: "demo_enhancer".to_string(),
            version: "1.0.0".to_string(),
            enhancement_strength: enhancement_strength.clamp(0.0, 1.0),
        }
    }
}

#[async_trait]
impl NeuralModel for DemoNeuralModel {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn version(&self) -> &str {
        &self.version
    }
    
    async fn analyze_layout(&self, block: &ContentBlock) -> Result<BlockPosition> {
        // Simulate neural layout enhancement
        let mut enhanced_position = block.position.clone();
        
        // Apply neural corrections to position
        let correction_factor = self.enhancement_strength * 0.05; // 5% max correction
        
        enhanced_position.x *= 1.0 - correction_factor;
        enhanced_position.y *= 1.0 - correction_factor;
        enhanced_position.width *= 1.0 + correction_factor;
        enhanced_position.height *= 1.0 + correction_factor;
        
        println!("🧠 Neural layout analysis: corrected position by {:.1}%", 
            correction_factor * 100.0);
        
        Ok(enhanced_position)
    }
    
    async fn enhance_text(&self, text: &str) -> Result<String> {
        // Simulate text enhancement with common OCR error corrections
        let mut enhanced = text.to_string();
        
        // Common OCR corrections
        enhanced = enhanced
            .replace("rn", "m")          // Common confusion
            .replace("||", "ll")         // Parallel lines as letters
            .replace("1", "l")           // In text contexts
            .replace("0", "o")           // In text contexts
            .replace("5", "s")           // In certain fonts
            .replace("6", "b")           // Similar shapes
            .replace("8", "B");          // Capital confusion
        
        // Apply enhancement strength
        if self.enhancement_strength > 0.7 {
            // High strength: more aggressive corrections
            enhanced = enhanced
                .replace("vv", "w")
                .replace("VV", "W")
                .replace("ni", "m")
                .replace(")", "j");
        }
        
        // Improve formatting
        enhanced = enhanced
            .lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        
        if enhanced != text {
            println!("🧠 Neural text enhancement: {} characters improved", 
                text.len().abs_diff(enhanced.len()));
        }
        
        Ok(enhanced)
    }
    
    async fn detect_table_structure(&self, block: &ContentBlock) -> Result<Option<TableStructure>> {
        let Some(text) = &block.text else {
            return Ok(None);
        };
        
        // Neural table detection heuristics
        let lines: Vec<&str> = text.lines().collect();
        
        if lines.len() < 2 {
            return Ok(None);
        }
        
        // Look for tabular patterns
        let has_separators = text.contains('\t') || 
                           text.contains('|') || 
                           text.matches(',').count() > lines.len();
        
        let has_alignment = lines.iter()
            .all(|line| line.split_whitespace().count() > 1);
        
        let neural_confidence = self.enhancement_strength * 0.9;
        
        if has_separators || (has_alignment && neural_confidence > 0.6) {
            // Determine separator
            let separator = if text.contains('\t') { '\t' }
                          else if text.contains('|') { '|' }
                          else if text.contains(',') { ',' }
                          else { ' ' };
            
            // Parse table structure
            let headers: Vec<String> = lines[0]
                .split(separator)
                .map(|s| s.trim().to_string())
                .collect();
            
            let cells: Vec<Vec<String>> = lines[1..]
                .iter()
                .map(|line| {
                    line.split(separator)
                        .map(|s| s.trim().to_string())
                        .collect()
                })
                .collect();
            
            let table = TableStructure {
                rows: cells.len() + 1, // Include header
                columns: headers.len(),
                headers,
                cells,
                confidence: neural_confidence,
            };
            
            println!("🧠 Neural table detection: found {}x{} table with {:.1}% confidence", 
                table.rows, table.columns, table.confidence * 100.0);
            
            Ok(Some(table))
        } else {
            Ok(None)
        }
    }
    
    async fn calculate_confidence(&self, document: &ExtractedDocument) -> Result<f32> {
        // Advanced neural confidence calculation
        let mut total_confidence = 0.0;
        let mut weight_sum = 0.0;
        
        for block in &document.content {
            // Block-specific confidence weights
            let weight = match block.block_type {
                neuraldocflow::core::BlockType::Paragraph => 1.0,
                neuraldocflow::core::BlockType::Table => 1.5, // Tables are more valuable
                neuraldocflow::core::BlockType::Heading(_) => 1.3,
                neuraldocflow::core::BlockType::Image => 0.8,
                _ => 1.0,
            };
            
            // Apply neural enhancement to base confidence
            let enhanced_confidence = block.metadata.confidence * 
                (1.0 + self.enhancement_strength * 0.1);
            
            total_confidence += enhanced_confidence * weight;
            weight_sum += weight;
        }
        
        let base_confidence = if weight_sum > 0.0 {
            total_confidence / weight_sum
        } else {
            0.0
        };
        
        // Document-level neural adjustments
        let structure_bonus = if document.structure.sections.len() > 1 { 0.05 } else { 0.0 };
        let content_density_bonus = if document.content.len() > 5 { 0.03 } else { 0.0 };
        let neural_boost = self.enhancement_strength * 0.08;
        
        let final_confidence = (base_confidence + structure_bonus + content_density_bonus + neural_boost)
            .clamp(0.0, 1.0);
        
        println!("🧠 Neural confidence calculation: {:.1}% (boost: +{:.1}%)", 
            final_confidence * 100.0, neural_boost * 100.0);
        
        Ok(final_confidence)
    }
    
    async fn cleanup(&mut self) -> Result<()> {
        println!("🧠 Neural model {} cleaned up", self.name);
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Neural Enhancement Example ===\n");
    
    // Configure neural enhancement
    let neural_config = NeuralConfig {
        enabled: true,
        model_directory: PathBuf::from("./demo_models"),
        max_loaded_models: 3,
        model_load_timeout: Duration::from_secs(30),
        processing: neuraldocflow::config::NeuralProcessingConfig {
            batch_size: 16,
            enable_gpu: false, // Disable GPU for demo
            inference_threads: 2,
            memory_limit: 512 * 1024 * 1024, // 512MB
        },
        models: vec![
            neuraldocflow::config::ModelConfig {
                name: "demo_enhancer".to_string(),
                path: PathBuf::from("demo_enhancer.model"),
                enabled: true,
                confidence_threshold: 0.7,
            },
        ],
    };
    
    let config = Config {
        neural: neural_config,
        ..Default::default()
    };
    
    println!("🧠 Initializing neural-enhanced DocFlow...");
    let docflow = DocFlow::with_config(config)?;
    
    // Create sample document with OCR-like errors
    create_sample_document_with_errors().await?;
    
    // Test without neural enhancement first
    println!("\n📄 Testing WITHOUT neural enhancement...");
    let basic_result = test_basic_extraction().await?;
    
    // Test with neural enhancement
    println!("\n🧠 Testing WITH neural enhancement...");
    let enhanced_result = test_neural_extraction(&docflow).await?;
    
    // Compare results
    compare_extraction_results(&basic_result, &enhanced_result);
    
    // Demonstrate different neural models
    println!("\n🔬 Testing different neural enhancement strengths...");
    test_different_enhancement_levels().await?;
    
    // Cleanup
    cleanup_demo_files().await?;
    
    println!("\n✅ Neural enhancement demonstration completed!");
    
    Ok(())
}

/// Test basic extraction without neural enhancement
async fn test_basic_extraction() -> Result<ExtractedDocument> {
    let basic_config = Config {
        neural: NeuralConfig {
            enabled: false,
            ..Default::default()
        },
        ..Default::default()
    };
    
    let docflow = DocFlow::with_config(basic_config)?;
    
    let input = SourceInput::File {
        path: PathBuf::from("sample_with_errors.txt"),
        metadata: None,
    };
    
    let document = docflow.extract(input).await?;
    
    println!("Basic extraction results:");
    println!("  - Confidence: {:.1}%", document.confidence * 100.0);
    println!("  - Content blocks: {}", document.content.len());
    
    Ok(document)
}

/// Test neural-enhanced extraction
async fn test_neural_extraction(docflow: &DocFlow) -> Result<ExtractedDocument> {
    let input = SourceInput::File {
        path: PathBuf::from("sample_with_errors.txt"),
        metadata: None,
    };
    
    let document = docflow.extract(input).await?;
    
    println!("Neural-enhanced extraction results:");
    println!("  - Confidence: {:.1}%", document.confidence * 100.0);
    println!("  - Content blocks: {}", document.content.len());
    
    Ok(document)
}

/// Compare extraction results
fn compare_extraction_results(basic: &ExtractedDocument, enhanced: &ExtractedDocument) {
    println!("\n=== Comparison Results ===");
    
    let confidence_improvement = enhanced.confidence - basic.confidence;
    println!("Confidence improvement: {:+.1}%", confidence_improvement * 100.0);
    
    let block_difference = enhanced.content.len() as i32 - basic.content.len() as i32;
    println!("Content blocks difference: {:+}", block_difference);
    
    // Compare text quality
    let basic_text = basic.get_text();
    let enhanced_text = enhanced.get_text();
    
    if enhanced_text != basic_text {
        println!("✅ Text content was enhanced by neural processing");
        
        // Show text differences
        let basic_words = basic_text.split_whitespace().count();
        let enhanced_words = enhanced_text.split_whitespace().count();
        let word_difference = enhanced_words as i32 - basic_words as i32;
        
        println!("Word count difference: {:+}", word_difference);
        
        // Count OCR-like errors in original vs enhanced
        let basic_errors = count_ocr_errors(&basic_text);
        let enhanced_errors = count_ocr_errors(&enhanced_text);
        let error_reduction = basic_errors as i32 - enhanced_errors as i32;
        
        println!("OCR errors reduced: {} -> {} ({:+})", 
            basic_errors, enhanced_errors, -error_reduction);
    } else {
        println!("ℹ️  Text content unchanged (no errors detected)");
    }
    
    // Compare table detection
    let basic_tables = basic.get_tables().len();
    let enhanced_tables = enhanced.get_tables().len();
    
    if enhanced_tables > basic_tables {
        println!("✅ Neural enhancement detected {} additional tables", 
            enhanced_tables - basic_tables);
    }
}

/// Test different neural enhancement strength levels
async fn test_different_enhancement_levels() -> Result<()> {
    let enhancement_levels = [0.3, 0.6, 0.9];
    
    for &level in &enhancement_levels {
        println!("\n🔬 Testing enhancement strength: {:.1}", level);
        
        let model = DemoNeuralModel::new(level);
        
        // Test text enhancement
        let sample_text = "Th1s 1s a sarnple w1th 0CR err0rs and rn1stake5.";
        let enhanced = model.enhance_text(sample_text).await?;
        
        println!("  Original: {}", sample_text);
        println!("  Enhanced: {}", enhanced);
        
        // Test layout enhancement
        let sample_block = neuraldocflow::core::ContentBlock::new(
            neuraldocflow::core::BlockType::Paragraph
        );
        
        let enhanced_position = model.analyze_layout(&sample_block).await?;
        println!("  Layout correction applied: {:.3} -> {:.3}", 
            sample_block.position.x, enhanced_position.x);
    }
    
    Ok(())
}

/// Count OCR-like errors in text
fn count_ocr_errors(text: &str) -> usize {
    let error_patterns = ["rn", "||", "1", "0", "5", "6", "8"];
    
    error_patterns.iter()
        .map(|pattern| text.matches(pattern).count())
        .sum()
}

/// Create a sample document with OCR-like errors
async fn create_sample_document_with_errors() -> Result<()> {
    let content = r#"Sarnple D0curnent w1th 0CR Err0rs

Th1s d0curnent c0nta1ns var10us typ1cal 0CR err0rs that neural 
enhancernent can detect and c0rrect. 

Table Exarnple:
Narne    Age    Departrnent
J0hn     3O     Eng1neer1ng
Jane     25     Market1ng
B0b      35     5ales

K3y P01nts:
- Neural pr0cess1ng 1mpr0ves accuracy
- 0CR err0rs are c0mmon 1n scanned d0curnents  
- Mach1ne learn1ng can f1x these 1ssues
- ||ayout detect10n 1s a|s0 1mp0rtant

C0nc|us10n:
Neural enhancernent s1gn1f1cant|y 1mpr0ves d0curnent extract10n 
qua|1ty by c0rrect1ng 0CR err0rs and 1mp0v1ng structure detect10n."#;
    
    std::fs::write("sample_with_errors.txt", content)?;
    println!("📝 Created sample document with OCR errors");
    
    Ok(())
}

/// Cleanup demonstration files
async fn cleanup_demo_files() -> Result<()> {
    let files = ["sample_with_errors.txt"];
    
    for file in &files {
        if std::path::Path::new(file).exists() {
            std::fs::remove_file(file)?;
        }
    }
    
    println!("🧹 Cleaned up demonstration files");
    Ok(())
}

/// Example of configuring multiple neural models
#[allow(dead_code)]
async fn example_multiple_models() -> Result<()> {
    let config = Config {
        neural: NeuralConfig {
            enabled: true,
            model_directory: PathBuf::from("./models"),
            max_loaded_models: 5,
            processing: neuraldocflow::config::NeuralProcessingConfig {
                batch_size: 32,
                enable_gpu: true,
                inference_threads: 4,
                memory_limit: 2 * 1024 * 1024 * 1024, // 2GB
            },
            models: vec![
                neuraldocflow::config::ModelConfig {
                    name: "layout_analyzer".to_string(),
                    path: PathBuf::from("layout_v2.model"),
                    enabled: true,
                    confidence_threshold: 0.8,
                },
                neuraldocflow::config::ModelConfig {
                    name: "text_enhancer".to_string(),
                    path: PathBuf::from("text_enhancement.model"),
                    enabled: true,
                    confidence_threshold: 0.75,
                },
                neuraldocflow::config::ModelConfig {
                    name: "table_detector".to_string(),
                    path: PathBuf::from("table_detection.model"),
                    enabled: true,
                    confidence_threshold: 0.85,
                },
                neuraldocflow::config::ModelConfig {
                    name: "confidence_scorer".to_string(),
                    path: PathBuf::from("confidence_v3.model"),
                    enabled: true,
                    confidence_threshold: 0.7,
                },
            ],
            ..Default::default()
        },
        ..Default::default()
    };
    
    let docflow = DocFlow::with_config(config)?;
    
    println!("✅ Configured DocFlow with multiple neural models");
    
    // The neural engine will automatically load and use all enabled models
    // during the extraction process
    
    Ok(())
}

/// Example of runtime model management
#[allow(dead_code)]
async fn example_runtime_model_management() -> Result<()> {
    let mut docflow = DocFlow::new()?;
    
    // Load additional model at runtime
    docflow.neural_engine.load_model("specialized_model").await?;
    
    println!("✅ Loaded specialized model at runtime");
    
    // Extract with enhanced capabilities
    let input = SourceInput::File {
        path: PathBuf::from("specialized_document.pdf"),
        metadata: None,
    };
    
    let document = docflow.extract(input).await?;
    
    println!("Specialized extraction completed with {:.1}% confidence", 
        document.confidence * 100.0);
    
    Ok(())
}