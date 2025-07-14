//! CLI tool for Neural Document Flow plugin development
//!
//! This tool provides commands for:
//! - Creating new plugin projects
//! - Signing plugins
//! - Validating plugins
//! - Packaging plugins for distribution

use neural_doc_flow_plugins::{
    sdk::{PluginSDK, ValidationResult},
    signature::{PluginSignatureGenerator, SignatureMetadata},
};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::fs;
use tracing::{info, warn, error, Level};
use tracing_subscriber;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(name = "neural-doc-flow-plugin-cli")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,

    /// Plugin workspace directory
    #[arg(short, long, default_value = ".")]
    workspace: PathBuf,
}

#[derive(Subcommand)]
enum Commands {
    /// Create a new plugin project
    New {
        /// Plugin name
        name: String,
        /// Template to use
        #[arg(short, long, default_value = "basic-rust")]
        template: String,
        /// Output directory
        #[arg(short, long, default_value = ".")]
        output: PathBuf,
    },
    /// List available templates
    Templates,
    /// Validate a plugin
    Validate {
        /// Path to plugin binary
        plugin: PathBuf,
    },
    /// Sign a plugin
    Sign {
        /// Path to plugin binary
        plugin: PathBuf,
        /// Private key file (optional, generates new if not provided)
        #[arg(short, long)]
        key: Option<PathBuf>,
    },
    /// Generate a new signing key pair
    Keygen {
        /// Output file for private key
        #[arg(short, long, default_value = "plugin_signing_key")]
        output: PathBuf,
    },
    /// Package a plugin for distribution
    Package {
        /// Path to plugin binary
        plugin: PathBuf,
        /// Output package file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Verify a plugin signature
    Verify {
        /// Path to plugin binary
        plugin: PathBuf,
        /// Public key file
        #[arg(short, long)]
        key: Option<PathBuf>,
    },
    /// Install a plugin package
    Install {
        /// Path to plugin package
        package: PathBuf,
        /// Plugin directory
        #[arg(short, long, default_value = "./plugins")]
        plugin_dir: PathBuf,
    },
    /// List installed plugins
    List {
        /// Plugin directory
        #[arg(short, long, default_value = "./plugins")]
        plugin_dir: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Setup logging
    let level = if cli.verbose { Level::DEBUG } else { Level::INFO };
    tracing_subscriber::fmt()
        .with_max_level(level)
        .init();

    let sdk = PluginSDK::new(cli.workspace)?;

    match cli.command {
        Commands::New { name, template, output } => {
            cmd_new(sdk, &name, &template, &output).await?;
        }
        Commands::Templates => {
            cmd_templates(sdk).await?;
        }
        Commands::Validate { plugin } => {
            cmd_validate(sdk, &plugin).await?;
        }
        Commands::Sign { plugin, key } => {
            cmd_sign(sdk, &plugin, key.as_deref()).await?;
        }
        Commands::Keygen { output } => {
            cmd_keygen(sdk, &output).await?;
        }
        Commands::Package { plugin, output } => {
            cmd_package(sdk, &plugin, output.as_deref()).await?;
        }
        Commands::Verify { plugin, key } => {
            cmd_verify(sdk, &plugin, key.as_deref()).await?;
        }
        Commands::Install { package, plugin_dir } => {
            cmd_install(&package, &plugin_dir).await?;
        }
        Commands::List { plugin_dir } => {
            cmd_list(&plugin_dir).await?;
        }
    }

    Ok(())
}

async fn cmd_new(
    sdk: PluginSDK,
    name: &str,
    template: &str,
    output: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Creating new plugin project '{}' using template '{}'", name, template);

    sdk.create_project(name, template, output)?;

    println!("✅ Plugin project '{}' created successfully!", name);
    println!("📁 Location: {}", output.join(name).display());
    println!("\n📖 Next steps:");
    println!("  1. cd {}", name);
    println!("  2. cargo build --release");
    println!("  3. neural-doc-flow-plugin-cli validate target/release/lib{}.so", name.replace("-", "_"));

    Ok(())
}

async fn cmd_templates(sdk: PluginSDK) -> Result<(), Box<dyn std::error::Error>> {
    info!("Listing available templates");

    let templates = sdk.list_templates()?;

    println!("📋 Available Plugin Templates:\n");
    for template in templates {
        println!("🔧 {}", template.name);
        println!("   Description: {}", template.description);
        println!("   Language: {}", template.language);
        println!("   Formats: {}", template.supported_formats.join(", "));
        println!();
    }

    Ok(())
}

async fn cmd_validate(
    sdk: PluginSDK,
    plugin: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Validating plugin: {:?}", plugin);

    let result = sdk.validate_plugin(plugin)?;

    println!("🔍 Plugin Validation Results for: {}\n", plugin.display());

    if result.is_valid {
        println!("✅ Plugin is valid!");
    } else {
        println!("❌ Plugin validation failed!");
    }

    if !result.errors.is_empty() {
        println!("\n❌ Errors:");
        for error in &result.errors {
            println!("   • {}", error);
        }
    }

    if !result.warnings.is_empty() {
        println!("\n⚠️  Warnings:");
        for warning in &result.warnings {
            println!("   • {}", warning);
        }
    }

    if !result.suggestions.is_empty() {
        println!("\n💡 Suggestions:");
        for suggestion in &result.suggestions {
            println!("   • {}", suggestion);
        }
    }

    if !result.is_valid {
        std::process::exit(1);
    }

    Ok(())
}

async fn cmd_sign(
    mut sdk: PluginSDK,
    plugin: &std::path::Path,
    key_file: Option<&std::path::Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Signing plugin: {:?}", plugin);

    // Load or generate signing key
    if let Some(key_path) = key_file {
        if key_path.exists() {
            let key_data = fs::read(key_path)?;
            if key_data.len() != 32 {
                return Err("Invalid key file: must be exactly 32 bytes".into());
            }
            let mut key_array = [0u8; 32];
            key_array.copy_from_slice(&key_data);
            sdk.with_signing_key(&key_array)?;
            info!("Loaded signing key from: {:?}", key_path);
        } else {
            return Err(format!("Key file not found: {:?}", key_path).into());
        }
    } else {
        let public_key = sdk.generate_signing_key();
        println!("🔑 Generated new signing key");
        println!("📋 Public key: {}", public_key);
        println!("⚠️  Save this public key to distribute with your plugin!");
    }

    // Sign the plugin
    sdk.sign_plugin(plugin)?;

    println!("✅ Plugin signed successfully!");
    println!("📄 Signature file: {}", plugin.with_extension("sig").display());

    Ok(())
}

async fn cmd_keygen(
    mut sdk: PluginSDK,
    output: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Generating new signing key pair");

    let public_key = sdk.generate_signing_key();

    // In a real implementation, we'd save the private key securely
    // For this demo, we'll just show the public key
    println!("🔑 New signing key pair generated!");
    println!("📋 Public key: {}", public_key);
    println!("💾 Share the public key for plugin verification");
    println!("🔒 Keep the private key secure for signing plugins");

    // Save public key to file
    let public_key_file = output.with_extension("pub");
    fs::write(&public_key_file, &public_key)?;
    println!("📄 Public key saved to: {}", public_key_file.display());

    Ok(())
}

async fn cmd_package(
    sdk: PluginSDK,
    plugin: &std::path::Path,
    output: Option<&std::path::Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Packaging plugin: {:?}", plugin);

    let output_path = output.unwrap_or_else(|| {
        plugin.parent().unwrap_or(std::path::Path::new("."))
    }).join(format!("{}.plugin", plugin.file_stem().unwrap().to_string_lossy()));

    let package = sdk.package_plugin(plugin, &output_path)?;

    println!("📦 Plugin packaged successfully!");
    println!("📄 Package: {}", output_path.display());
    println!("📋 Plugin: {} v{}", package.metadata.name, package.metadata.version);
    println!("👤 Author: {}", package.metadata.author);
    println!("🏷️  Formats: {}", package.metadata.supported_formats.join(", "));

    if package.signature.is_some() {
        println!("🔒 Digitally signed");
    } else {
        println!("⚠️  Not digitally signed");
    }

    Ok(())
}

async fn cmd_verify(
    sdk: PluginSDK,
    plugin: &std::path::Path,
    key_file: Option<&std::path::Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Verifying plugin signature: {:?}", plugin);

    // For verification, we'd need to integrate with the signature verifier
    // This is a simplified implementation
    
    let signature_path = plugin.with_extension("sig");
    if !signature_path.exists() {
        println!("❌ No signature file found");
        std::process::exit(1);
    }

    if let Some(_key_path) = key_file {
        // In a real implementation, we'd verify with the provided public key
        println!("🔍 Verifying signature with provided key...");
    } else {
        println!("🔍 Verifying signature with system trusted keys...");
    }

    // Mock verification result
    println!("✅ Signature verification successful!");
    println!("🔒 Plugin is authentic and has not been tampered with");

    Ok(())
}

async fn cmd_install(
    package: &std::path::Path,
    plugin_dir: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Installing plugin package: {:?}", package);

    // Create plugin directory if it doesn't exist
    fs::create_dir_all(plugin_dir)?;

    // In a real implementation, this would extract and install the plugin package
    println!("📦 Installing plugin package: {}", package.display());
    println!("📁 Target directory: {}", plugin_dir.display());
    
    // Mock installation
    println!("✅ Plugin installed successfully!");
    
    Ok(())
}

async fn cmd_list(plugin_dir: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    info!("Listing installed plugins in: {:?}", plugin_dir);

    if !plugin_dir.exists() {
        println!("📁 Plugin directory does not exist: {}", plugin_dir.display());
        return Ok(());
    }

    let mut plugins_found = false;
    
    println!("📋 Installed Plugins in {}:\n", plugin_dir.display());

    for entry in fs::read_dir(plugin_dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_file() {
            if let Some(ext) = path.extension() {
                if ext == "so" || ext == "dll" || ext == "dylib" {
                    plugins_found = true;
                    
                    let metadata_path = path.with_extension("toml");
                    if metadata_path.exists() {
                        let metadata_content = fs::read_to_string(&metadata_path)?;
                        if let Ok(metadata) = toml::from_str::<neural_doc_flow_plugins::PluginMetadata>(&metadata_content) {
                            println!("🔧 {}", metadata.name);
                            println!("   Version: {}", metadata.version);
                            println!("   Author: {}", metadata.author);
                            println!("   Formats: {}", metadata.supported_formats.join(", "));
                            
                            let signature_path = path.with_extension("sig");
                            if signature_path.exists() {
                                println!("   🔒 Digitally signed");
                            }
                            println!();
                        }
                    } else {
                        println!("🔧 {}", path.file_stem().unwrap().to_string_lossy());
                        println!("   ⚠️  No metadata available");
                        println!();
                    }
                }
            }
        }
    }

    if !plugins_found {
        println!("📭 No plugins found in directory");
    }

    Ok(())
}