//! Neural Document Flow CLI
//! 
//! Command-line interface for the Neural Document Flow system.

use anyhow::Result;
use clap::{Parser, Subcommand};
use neural_doc_flow::{NeuralDocumentFlow, FlowConfig};
use std::path::PathBuf;
use tracing::{info, warn, error};

#[derive(Parser)]
#[command(name = "neural-doc-flow")]
#[command(about = "Neural-enhanced document processing and flow management")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    
    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
    
    /// Disable neural processing
    #[arg(long)]
    no_neural: bool,
    
    /// Disable DAA coordination
    #[arg(long)]
    no_coordination: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Process a single document
    Process {
        /// Input document path
        input: PathBuf,
        
        /// Output directory
        #[arg(short, long)]
        output: Option<PathBuf>,
        
        /// Output format
        #[arg(short, long, default_value = "json")]
        format: String,
    },
    
    /// Process multiple documents
    Batch {
        /// Input directory
        input_dir: PathBuf,
        
        /// Output directory
        #[arg(short, long)]
        output_dir: Option<PathBuf>,
        
        /// File pattern to match
        #[arg(short, long, default_value = "*")]
        pattern: String,
    },
    
    /// Start interactive mode
    Interactive,
    
    /// Show system status
    Status,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    
    // Initialize tracing
    let level = if cli.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(format!("neural_doc_flow={}", level))
        .init();
    
    info!("Starting Neural Document Flow CLI");
    
    // Create flow configuration
    let config = FlowConfig {
        coordination_enabled: !cli.no_coordination,
        neural_processing: !cli.no_neural,
        max_concurrent_documents: 10,
        output_formats: vec!["json".to_string()],
    };
    
    let flow = NeuralDocumentFlow::with_config(config);
    
    match cli.command {
        Commands::Process { input, output: _, format: _ } => {
            info!("Processing document: {}", input.display());
            match flow.process_document(&input).await {
                Ok(result) => {
                    println!("✅ Successfully processed: {}", result);
                }
                Err(e) => {
                    error!("❌ Failed to process document: {}", e);
                    std::process::exit(1);
                }
            }
        }
        
        Commands::Batch { input_dir, output_dir: _, pattern: _ } => {
            info!("Processing batch from directory: {}", input_dir.display());
            warn!("Batch processing not yet implemented");
        }
        
        Commands::Interactive => {
            info!("Starting interactive mode");
            println!("🧠 Neural Document Flow - Interactive Mode");
            println!("Type 'help' for available commands or 'quit' to exit");
            
            loop {
                print!("> ");
                use std::io::{self, Write};
                io::stdout().flush()?;
                
                let mut input = String::new();
                io::stdin().read_line(&mut input)?;
                let input = input.trim();
                
                match input {
                    "quit" | "exit" => break,
                    "help" => {
                        println!("Available commands:");
                        println!("  help    - Show this help");
                        println!("  status  - Show system status");
                        println!("  quit    - Exit interactive mode");
                    }
                    "status" => {
                        println!("📊 System Status:");
                        println!("  Neural Processing: {}", if config.neural_processing { "✅ Enabled" } else { "❌ Disabled" });
                        println!("  DAA Coordination: {}", if config.coordination_enabled { "✅ Enabled" } else { "❌ Disabled" });
                    }
                    "" => continue,
                    _ => {
                        println!("Unknown command: {}. Type 'help' for available commands.", input);
                    }
                }
            }
        }
        
        Commands::Status => {
            println!("📊 Neural Document Flow Status");
            println!("  Version: {}", env!("CARGO_PKG_VERSION"));
            println!("  Neural Processing: {}", if config.neural_processing { "✅ Enabled" } else { "❌ Disabled" });
            println!("  DAA Coordination: {}", if config.coordination_enabled { "✅ Enabled" } else { "❌ Disabled" });
            println!("  Max Concurrent: {}", config.max_concurrent_documents);
        }
    }
    
    info!("Neural Document Flow CLI completed");
    Ok(())
}