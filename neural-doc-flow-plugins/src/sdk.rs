//! Plugin SDK for Neural Document Flow
//!
//! This module provides tools and utilities for plugin developers
//! to create, test, and package plugins for the neural document flow system.

use neural_doc_flow_core::{DocumentSource, ProcessingError};
use crate::{
    Plugin, PluginMetadata, PluginCapabilities,
    signature::{PluginSignatureGenerator, SignatureMetadata, PluginSignature},
};
use std::path::{Path, PathBuf};
use std::fs;
use serde::{Serialize, Deserialize};
use tracing::{info, warn, error};

/// Plugin development kit
pub struct PluginSDK {
    workspace_dir: PathBuf,
    signature_generator: Option<PluginSignatureGenerator>,
    config: SDKConfig,
}

/// SDK configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SDKConfig {
    pub default_author: String,
    pub default_capabilities: PluginCapabilities,
    pub minimum_api_version: String,
    pub signing_enabled: bool,
    pub template_dir: Option<PathBuf>,
}

/// Plugin template information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginTemplate {
    pub name: String,
    pub description: String,
    pub language: String,
    pub supported_formats: Vec<String>,
    pub template_files: Vec<TemplateFile>,
    pub build_instructions: Vec<String>,
}

/// Template file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateFile {
    pub path: String,
    pub content: String,
    pub is_binary: bool,
}

/// Plugin validation result
#[derive(Debug)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub suggestions: Vec<String>,
}

/// Plugin package information
#[derive(Debug, Serialize, Deserialize)]
pub struct PluginPackage {
    pub metadata: PluginMetadata,
    pub signature: Option<PluginSignature>,
    pub binary_path: PathBuf,
    pub readme: Option<String>,
    pub changelog: Option<String>,
    pub license: Option<String>,
}

impl PluginSDK {
    /// Create new SDK instance
    pub fn new(workspace_dir: PathBuf) -> Result<Self, ProcessingError> {
        let config_path = workspace_dir.join("sdk_config.toml");
        let config = if config_path.exists() {
            let content = fs::read_to_string(&config_path)
                .map_err(|e| ProcessingError::PluginLoadError(format!("Failed to read SDK config: {}", e)))?;
            toml::from_str(&content)
                .map_err(|e| ProcessingError::PluginLoadError(format!("Invalid SDK config: {}", e)))?
        } else {
            SDKConfig::default()
        };

        Ok(Self {
            workspace_dir,
            signature_generator: None,
            config,
        })
    }

    /// Initialize SDK with signing key
    pub fn with_signing_key(&mut self, private_key: &[u8; 32]) -> Result<(), ProcessingError> {
        self.signature_generator = Some(PluginSignatureGenerator::from_key(private_key)?);
        self.config.signing_enabled = true;
        Ok(())
    }

    /// Generate new signing key pair
    pub fn generate_signing_key(&mut self) -> String {
        let generator = PluginSignatureGenerator::new();
        let public_key = generator.public_key_hex();
        self.signature_generator = Some(generator);
        self.config.signing_enabled = true;
        
        info!("Generated new signing key pair");
        info!("Public key: {}", public_key);
        
        public_key
    }

    /// Create new plugin project from template
    pub fn create_project(
        &self,
        project_name: &str,
        template_name: &str,
        output_dir: &Path,
    ) -> Result<(), ProcessingError> {
        info!("Creating plugin project '{}' from template '{}'", project_name, template_name);

        let template = self.get_template(template_name)?;
        let project_dir = output_dir.join(project_name);

        // Create project directory
        fs::create_dir_all(&project_dir)
            .map_err(|e| ProcessingError::PluginLoadError(format!("Failed to create project dir: {}", e)))?;

        // Generate files from template
        for template_file in &template.template_files {
            let file_path = project_dir.join(&template_file.path);
            
            // Create parent directories
            if let Some(parent) = file_path.parent() {
                fs::create_dir_all(parent)
                    .map_err(|e| ProcessingError::PluginLoadError(format!("Failed to create dir: {}", e)))?;
            }

            // Replace template variables
            let content = self.process_template_content(&template_file.content, project_name, &template);

            if template_file.is_binary {
                // Handle binary files (base64 encoded in template)
                let binary_data = base64::decode(&content)
                    .map_err(|e| ProcessingError::PluginLoadError(format!("Invalid binary template: {}", e)))?;
                fs::write(&file_path, binary_data)
                    .map_err(|e| ProcessingError::PluginLoadError(format!("Failed to write file: {}", e)))?;
            } else {
                fs::write(&file_path, content)
                    .map_err(|e| ProcessingError::PluginLoadError(format!("Failed to write file: {}", e)))?;
            }
        }

        // Create plugin metadata file
        let metadata = PluginMetadata {
            name: project_name.to_string(),
            version: "0.1.0".to_string(),
            author: self.config.default_author.clone(),
            description: format!("Plugin for processing {} documents", template.supported_formats.join(", ")),
            supported_formats: template.supported_formats.clone(),
            capabilities: self.config.default_capabilities.clone(),
        };

        let metadata_path = project_dir.join("plugin_metadata.toml");
        let metadata_toml = toml::to_string_pretty(&metadata)
            .map_err(|e| ProcessingError::PluginLoadError(format!("Failed to serialize metadata: {}", e)))?;
        fs::write(&metadata_path, metadata_toml)
            .map_err(|e| ProcessingError::PluginLoadError(format!("Failed to write metadata: {}", e)))?;

        // Create README with build instructions
        let readme_content = self.generate_readme(project_name, &template);
        let readme_path = project_dir.join("README.md");
        fs::write(&readme_path, readme_content)
            .map_err(|e| ProcessingError::PluginLoadError(format!("Failed to write README: {}", e)))?;

        info!("Plugin project '{}' created successfully", project_name);
        info!("Project location: {:?}", project_dir);
        
        Ok(())
    }

    /// Validate plugin before packaging
    pub fn validate_plugin(&self, plugin_path: &Path) -> Result<ValidationResult, ProcessingError> {
        let mut result = ValidationResult {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            suggestions: Vec::new(),
        };

        // Check if plugin file exists
        if !plugin_path.exists() {
            result.errors.push("Plugin binary does not exist".to_string());
            result.is_valid = false;
            return Ok(result);
        }

        // Check file size (warn if too large)
        let metadata = fs::metadata(plugin_path)
            .map_err(|e| ProcessingError::PluginLoadError(format!("Failed to read plugin metadata: {}", e)))?;
        
        let size_mb = metadata.len() / (1024 * 1024);
        if size_mb > 100 {
            result.warnings.push(format!("Plugin binary is large: {}MB", size_mb));
        }
        if size_mb > 500 {
            result.errors.push("Plugin binary is too large (>500MB)".to_string());
            result.is_valid = false;
        }

        // Check for metadata file
        let metadata_path = plugin_path.with_extension("toml");
        if !metadata_path.exists() {
            result.warnings.push("No metadata file found (.toml)".to_string());
            result.suggestions.push("Create a metadata file with plugin information".to_string());
        } else {
            // Validate metadata
            match self.validate_metadata(&metadata_path) {
                Ok(warnings) => result.warnings.extend(warnings),
                Err(e) => {
                    result.errors.push(format!("Invalid metadata: {}", e));
                    result.is_valid = false;
                }
            }
        }

        // Check for signature file
        let signature_path = plugin_path.with_extension("sig");
        if !signature_path.exists() {
            if self.config.signing_enabled {
                result.warnings.push("No signature file found - plugin won't be trusted".to_string());
                result.suggestions.push("Sign the plugin using 'sdk sign <plugin>'".to_string());
            }
        } else {
            // Validate signature format
            if let Err(e) = self.validate_signature(&signature_path) {
                result.errors.push(format!("Invalid signature: {}", e));
                result.is_valid = false;
            }
        }

        // Check for documentation
        let readme_path = plugin_path.parent().unwrap_or(Path::new(".")).join("README.md");
        if !readme_path.exists() {
            result.suggestions.push("Add README.md with usage instructions".to_string());
        }

        Ok(result)
    }

    /// Sign a plugin
    pub fn sign_plugin(&self, plugin_path: &Path) -> Result<(), ProcessingError> {
        let generator = self.signature_generator.as_ref()
            .ok_or_else(|| ProcessingError::PluginLoadError("No signing key configured".to_string()))?;

        // Load metadata
        let metadata_path = plugin_path.with_extension("toml");
        let plugin_metadata = if metadata_path.exists() {
            let content = fs::read_to_string(&metadata_path)
                .map_err(|e| ProcessingError::PluginLoadError(format!("Failed to read metadata: {}", e)))?;
            toml::from_str::<PluginMetadata>(&content)
                .map_err(|e| ProcessingError::PluginLoadError(format!("Invalid metadata: {}", e)))?
        } else {
            return Err(ProcessingError::PluginLoadError("No metadata file found".to_string()));
        };

        // Create signature metadata
        let signature_metadata = SignatureMetadata {
            plugin_name: plugin_metadata.name.clone(),
            plugin_version: plugin_metadata.version.clone(),
            author: plugin_metadata.author.clone(),
            build_timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            capabilities: plugin_metadata.supported_formats.clone(),
            dependencies: Vec::new(), // TODO: Extract from Cargo.toml if available
            minimum_api_version: self.config.minimum_api_version.clone(),
        };

        // Generate signature
        let signature = generator.sign_plugin(plugin_path, signature_metadata)?;

        // Save signature
        let signature_path = plugin_path.with_extension("sig");
        generator.save_signature(&signature, &signature_path)?;

        info!("Plugin signed successfully: {:?}", signature_path);
        Ok(())
    }

    /// Package plugin for distribution
    pub fn package_plugin(
        &self,
        plugin_path: &Path,
        output_path: &Path,
    ) -> Result<PluginPackage, ProcessingError> {
        // Validate plugin first
        let validation = self.validate_plugin(plugin_path)?;
        if !validation.is_valid {
            return Err(ProcessingError::PluginLoadError(
                format!("Plugin validation failed: {}", validation.errors.join(", "))
            ));
        }

        // Load metadata
        let metadata_path = plugin_path.with_extension("toml");
        let metadata = if metadata_path.exists() {
            let content = fs::read_to_string(&metadata_path)
                .map_err(|e| ProcessingError::PluginLoadError(format!("Failed to read metadata: {}", e)))?;
            toml::from_str(&content)
                .map_err(|e| ProcessingError::PluginLoadError(format!("Invalid metadata: {}", e)))?
        } else {
            return Err(ProcessingError::PluginLoadError("No metadata file found".to_string()));
        };

        // Load signature if available
        let signature_path = plugin_path.with_extension("sig");
        let signature = if signature_path.exists() {
            let content = fs::read_to_string(&signature_path)
                .map_err(|e| ProcessingError::PluginLoadError(format!("Failed to read signature: {}", e)))?;
            Some(serde_json::from_str(&content)
                .map_err(|e| ProcessingError::PluginLoadError(format!("Invalid signature: {}", e)))?)
        } else {
            None
        };

        // Load optional files
        let plugin_dir = plugin_path.parent().unwrap_or(Path::new("."));
        let readme = fs::read_to_string(plugin_dir.join("README.md")).ok();
        let changelog = fs::read_to_string(plugin_dir.join("CHANGELOG.md")).ok();
        let license = fs::read_to_string(plugin_dir.join("LICENSE")).ok();

        let package = PluginPackage {
            metadata,
            signature,
            binary_path: plugin_path.to_path_buf(),
            readme,
            changelog,
            license,
        };

        // Create package archive (tar.gz)
        self.create_package_archive(&package, output_path)?;

        info!("Plugin packaged successfully: {:?}", output_path);
        Ok(package)
    }

    /// Get available templates
    pub fn list_templates(&self) -> Result<Vec<PluginTemplate>, ProcessingError> {
        let templates_dir = self.get_templates_dir()?;
        let mut templates = Vec::new();

        if templates_dir.exists() {
            for entry in fs::read_dir(&templates_dir)
                .map_err(|e| ProcessingError::PluginLoadError(format!("Failed to read templates dir: {}", e)))? {
                let entry = entry
                    .map_err(|e| ProcessingError::PluginLoadError(format!("Failed to read template entry: {}", e)))?;
                
                if entry.file_type()
                    .map_err(|e| ProcessingError::PluginLoadError(format!("Failed to get file type: {}", e)))?
                    .is_dir() {
                    let template_file = entry.path().join("template.toml");
                    if template_file.exists() {
                        let content = fs::read_to_string(&template_file)
                            .map_err(|e| ProcessingError::PluginLoadError(format!("Failed to read template: {}", e)))?;
                        let template: PluginTemplate = toml::from_str(&content)
                            .map_err(|e| ProcessingError::PluginLoadError(format!("Invalid template: {}", e)))?;
                        templates.push(template);
                    }
                }
            }
        }

        // Add built-in templates if no custom ones found
        if templates.is_empty() {
            templates.extend(self.get_builtin_templates());
        }

        Ok(templates)
    }

    /// Helper methods
    fn get_template(&self, name: &str) -> Result<PluginTemplate, ProcessingError> {
        let templates = self.list_templates()?;
        templates.into_iter()
            .find(|t| t.name == name)
            .ok_or_else(|| ProcessingError::PluginLoadError(format!("Template '{}' not found", name)))
    }

    fn get_templates_dir(&self) -> Result<PathBuf, ProcessingError> {
        Ok(self.config.template_dir.clone()
            .unwrap_or_else(|| self.workspace_dir.join("templates")))
    }

    fn process_template_content(
        &self,
        content: &str,
        project_name: &str,
        template: &PluginTemplate,
    ) -> String {
        content
            .replace("{{PROJECT_NAME}}", project_name)
            .replace("{{PROJECT_NAME_UPPER}}", &project_name.to_uppercase())
            .replace("{{PROJECT_NAME_SNAKE}}", &project_name.replace("-", "_"))
            .replace("{{AUTHOR}}", &self.config.default_author)
            .replace("{{FORMATS}}", &template.supported_formats.join(", "))
            .replace("{{DESCRIPTION}}", &template.description)
    }

    fn generate_readme(&self, project_name: &str, template: &PluginTemplate) -> String {
        format!(r#"# {} Plugin

{}

## Supported Formats
{}

## Building

{}

## Installation

1. Build the plugin using the instructions above
2. Copy the resulting binary to your Neural Document Flow plugins directory
3. Restart the application or use hot-reload

## Testing

```bash
cargo test
```

## Signing

To sign your plugin for distribution:

```bash
neural-doc-flow-sdk sign target/release/lib{}.so
```

## Author

{}
"#,
            project_name,
            template.description,
            template.supported_formats.iter().map(|f| format!("- {}", f)).collect::<Vec<_>>().join("\n"),
            template.build_instructions.join("\n"),
            project_name.replace("-", "_"),
            self.config.default_author
        )
    }

    fn validate_metadata(&self, metadata_path: &Path) -> Result<Vec<String>, ProcessingError> {
        let content = fs::read_to_string(metadata_path)
            .map_err(|e| ProcessingError::PluginLoadError(format!("Failed to read metadata: {}", e)))?;
        
        let metadata: PluginMetadata = toml::from_str(&content)
            .map_err(|e| ProcessingError::PluginLoadError(format!("Invalid metadata format: {}", e)))?;

        let mut warnings = Vec::new();

        if metadata.name.is_empty() {
            return Err(ProcessingError::PluginLoadError("Plugin name cannot be empty".to_string()));
        }

        if metadata.supported_formats.is_empty() {
            warnings.push("No supported formats specified".to_string());
        }

        if metadata.capabilities.max_memory_mb > 1000 {
            warnings.push("Plugin requests high memory usage".to_string());
        }

        Ok(warnings)
    }

    fn validate_signature(&self, signature_path: &Path) -> Result<(), ProcessingError> {
        let content = fs::read_to_string(signature_path)
            .map_err(|e| ProcessingError::PluginLoadError(format!("Failed to read signature: {}", e)))?;
        
        let _signature: PluginSignature = serde_json::from_str(&content)
            .map_err(|e| ProcessingError::PluginLoadError(format!("Invalid signature format: {}", e)))?;

        Ok(())
    }

    fn create_package_archive(
        &self,
        package: &PluginPackage,
        output_path: &Path,
    ) -> Result<(), ProcessingError> {
        // For now, just copy files to a directory
        // In a full implementation, this would create a tar.gz archive
        
        let package_dir = output_path.parent()
            .ok_or_else(|| ProcessingError::PluginLoadError("Invalid output path".to_string()))?
            .join(format!("{}-{}", package.metadata.name, package.metadata.version));

        fs::create_dir_all(&package_dir)
            .map_err(|e| ProcessingError::PluginLoadError(format!("Failed to create package dir: {}", e)))?;

        // Copy binary
        let binary_dest = package_dir.join(package.binary_path.file_name().unwrap());
        fs::copy(&package.binary_path, &binary_dest)
            .map_err(|e| ProcessingError::PluginLoadError(format!("Failed to copy binary: {}", e)))?;

        // Write metadata
        let metadata_toml = toml::to_string_pretty(&package.metadata)
            .map_err(|e| ProcessingError::PluginLoadError(format!("Failed to serialize metadata: {}", e)))?;
        fs::write(package_dir.join("metadata.toml"), metadata_toml)
            .map_err(|e| ProcessingError::PluginLoadError(format!("Failed to write metadata: {}", e)))?;

        // Write signature if available
        if let Some(ref signature) = package.signature {
            let signature_json = serde_json::to_string_pretty(signature)
                .map_err(|e| ProcessingError::PluginLoadError(format!("Failed to serialize signature: {}", e)))?;
            fs::write(package_dir.join("signature.json"), signature_json)
                .map_err(|e| ProcessingError::PluginLoadError(format!("Failed to write signature: {}", e)))?;
        }

        // Write optional files
        if let Some(ref readme) = package.readme {
            fs::write(package_dir.join("README.md"), readme)
                .map_err(|e| ProcessingError::PluginLoadError(format!("Failed to write README: {}", e)))?;
        }

        Ok(())
    }

    fn get_builtin_templates(&self) -> Vec<PluginTemplate> {
        vec![
            PluginTemplate {
                name: "basic-rust".to_string(),
                description: "Basic Rust plugin template".to_string(),
                language: "Rust".to_string(),
                supported_formats: vec!["custom".to_string()],
                template_files: vec![
                    TemplateFile {
                        path: "Cargo.toml".to_string(),
                        content: r#"[package]
name = "{{PROJECT_NAME}}"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
neural-doc-flow-core = { path = "../neural-doc-flow-core" }
serde = { version = "1.0", features = ["derive"] }
"#.to_string(),
                        is_binary: false,
                    },
                    TemplateFile {
                        path: "src/lib.rs".to_string(),
                        content: r#"use neural_doc_flow_core::{DocumentSource, ProcessingError, Document};
use neural_doc_flow_plugins::{Plugin, PluginMetadata, PluginCapabilities};

pub struct {{PROJECT_NAME_UPPER}}Plugin {
    metadata: PluginMetadata,
}

impl Plugin for {{PROJECT_NAME_UPPER}}Plugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }
    
    fn initialize(&mut self) -> Result<(), ProcessingError> {
        // Initialize your plugin here
        Ok(())
    }
    
    fn shutdown(&mut self) -> Result<(), ProcessingError> {
        // Cleanup resources here
        Ok(())
    }
    
    fn document_source(&self) -> Box<dyn DocumentSource> {
        Box::new({{PROJECT_NAME_UPPER}}Source)
    }
}

struct {{PROJECT_NAME_UPPER}}Source;

impl DocumentSource for {{PROJECT_NAME_UPPER}}Source {
    fn can_process(&self, path: &std::path::Path) -> bool {
        // Check if this plugin can process the file
        true
    }
    
    fn extract_document(&self, path: &std::path::Path) -> Result<Document, ProcessingError> {
        // Extract document content
        Ok(Document {
            content: "Extracted content".to_string(),
            metadata: Default::default(),
        })
    }
}

#[no_mangle]
pub extern "C" fn create_plugin() -> *mut dyn Plugin {
    let plugin = {{PROJECT_NAME_UPPER}}Plugin {
        metadata: PluginMetadata {
            name: "{{PROJECT_NAME}}".to_string(),
            version: "0.1.0".to_string(),
            author: "{{AUTHOR}}".to_string(),
            description: "{{DESCRIPTION}}".to_string(),
            supported_formats: vec!["{{FORMATS}}".to_string()],
            capabilities: PluginCapabilities {
                requires_network: false,
                requires_filesystem: true,
                max_memory_mb: 100,
                max_cpu_percent: 50.0,
                timeout_seconds: 30,
            },
        },
    };
    
    Box::into_raw(Box::new(plugin))
}
"#.to_string(),
                        is_binary: false,
                    },
                ],
                build_instructions: vec![
                    "cargo build --release".to_string(),
                ],
            }
        ]
    }
}

impl Default for SDKConfig {
    fn default() -> Self {
        Self {
            default_author: "Unknown".to_string(),
            default_capabilities: PluginCapabilities {
                requires_network: false,
                requires_filesystem: true,
                max_memory_mb: 100,
                max_cpu_percent: 50.0,
                timeout_seconds: 30,
            },
            minimum_api_version: "1.0.0".to_string(),
            signing_enabled: false,
            template_dir: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_sdk_creation() {
        let temp_dir = TempDir::new().unwrap();
        let sdk = PluginSDK::new(temp_dir.path().to_path_buf());
        assert!(sdk.is_ok());
    }

    #[test]
    fn test_template_listing() {
        let temp_dir = TempDir::new().unwrap();
        let sdk = PluginSDK::new(temp_dir.path().to_path_buf()).unwrap();
        let templates = sdk.list_templates().unwrap();
        assert!(!templates.is_empty());
        assert!(templates.iter().any(|t| t.name == "basic-rust"));
    }

    #[test]
    fn test_project_creation() {
        let temp_dir = TempDir::new().unwrap();
        let sdk = PluginSDK::new(temp_dir.path().to_path_buf()).unwrap();
        
        let result = sdk.create_project("test_plugin", "basic-rust", temp_dir.path());
        assert!(result.is_ok());
        
        let project_dir = temp_dir.path().join("test_plugin");
        assert!(project_dir.exists());
        assert!(project_dir.join("Cargo.toml").exists());
        assert!(project_dir.join("src/lib.rs").exists());
        assert!(project_dir.join("README.md").exists());
    }
}