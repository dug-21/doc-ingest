//! Binary for training all neural security models

use neural_doc_flow_security::models::training::train_all_security_models;
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
    
    let sample_count = if args.len() > 2 {
        args[2].parse::<usize>().unwrap_or(1000)
    } else {
        1000
    };
    
    println!("🚀 Neural Security Models Training Tool");
    println!("========================================");
    println!("📁 Models directory: {:?}", models_dir);
    println!("📊 Training samples: {}", sample_count);
    println!();
    
    // Create models directory if it doesn't exist
    std::fs::create_dir_all(&models_dir)?;
    
    // Train all models
    match train_all_security_models(&models_dir, sample_count) {
        Ok(_) => {
            println!("\n🎉 Training completed successfully!");
            println!("📂 Trained models saved to: {:?}", models_dir);
            
            // List generated files
            println!("\n📋 Generated files:");
            for entry in std::fs::read_dir(&models_dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_file() {
                    println!("  • {:?}", path.file_name().unwrap());
                }
            }
        },
        Err(e) => {
            eprintln!("❌ Training failed: {}", e);
            std::process::exit(1);
        }
    }
    
    Ok(())
}