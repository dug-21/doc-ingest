//! Model serialization and deserialization utilities

use crate::models::base::{ModelConfig, ModelMetrics, TrainingResult};
use neural_doc_flow_core::ProcessingError;
use serde::{Serialize, Deserialize};
use std::path::Path;
use std::fs;

/// Serializable model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub config: ModelConfig,
    pub metrics: ModelMetrics,
    pub training_history: Vec<TrainingResult>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_updated: chrono::DateTime<chrono::Utc>,
    pub version: String,
    pub checksum: String,
}

impl ModelMetadata {
    /// Create new model metadata
    pub fn new(config: ModelConfig, metrics: ModelMetrics) -> Self {
        let now = chrono::Utc::now();
        Self {
            config,
            metrics,
            training_history: Vec::new(),
            created_at: now,
            last_updated: now,
            version: "1.0.0".to_string(),
            checksum: String::new(),
        }
    }
    
    /// Add training result to history
    pub fn add_training_result(&mut self, result: TrainingResult) {
        self.training_history.push(result);
        self.last_updated = chrono::Utc::now();
    }
    
    /// Update checksum
    pub fn update_checksum(&mut self, checksum: String) {
        self.checksum = checksum;
        self.last_updated = chrono::Utc::now();
    }
}

/// Model serializer for saving models with metadata
pub struct ModelSerializer;

impl ModelSerializer {
    /// Save model with metadata
    pub fn save_model_with_metadata(
        model_path: &Path,
        metadata: &ModelMetadata,
    ) -> Result<(), ProcessingError> {
        // Save metadata
        let metadata_path = model_path.with_extension("json");
        let metadata_json = serde_json::to_string_pretty(metadata)
            .map_err(|e| ProcessingError::SerializationError(e.to_string()))?;
        
        fs::write(&metadata_path, metadata_json)
            .map_err(|e| ProcessingError::SerializationError(e.to_string()))?;
        
        println!("Model metadata saved to {:?}", metadata_path);
        Ok(())
    }
    
    /// Create model package (model + metadata)
    pub fn create_model_package(
        model_path: &Path,
        package_path: &Path,
        metadata: &ModelMetadata,
    ) -> Result<(), ProcessingError> {
        // Create package directory
        fs::create_dir_all(package_path)
            .map_err(|e| ProcessingError::SerializationError(e.to_string()))?;
        
        // Copy model file
        let model_name = model_path.file_name()
            .ok_or_else(|| ProcessingError::SerializationError("Invalid model path".to_string()))?;
        let package_model_path = package_path.join(model_name);
        
        fs::copy(model_path, &package_model_path)
            .map_err(|e| ProcessingError::SerializationError(e.to_string()))?;
        
        // Save metadata
        let metadata_path = package_path.join("metadata.json");
        let metadata_json = serde_json::to_string_pretty(metadata)
            .map_err(|e| ProcessingError::SerializationError(e.to_string()))?;
        
        fs::write(&metadata_path, metadata_json)
            .map_err(|e| ProcessingError::SerializationError(e.to_string()))?;
        
        // Create package info
        let package_info = PackageInfo {
            name: metadata.config.name.clone(),
            version: metadata.version.clone(),
            model_file: model_name.to_string_lossy().to_string(),
            metadata_file: "metadata.json".to_string(),
            created_at: chrono::Utc::now(),
        };
        
        let package_info_json = serde_json::to_string_pretty(&package_info)
            .map_err(|e| ProcessingError::SerializationError(e.to_string()))?;
        
        fs::write(package_path.join("package.json"), package_info_json)
            .map_err(|e| ProcessingError::SerializationError(e.to_string()))?;
        
        println!("Model package created at {:?}", package_path);
        Ok(())
    }
    
    /// Export models for deployment
    pub fn export_for_deployment(
        models: &[(&Path, &ModelMetadata)],
        export_path: &Path,
    ) -> Result<(), ProcessingError> {
        fs::create_dir_all(export_path)
            .map_err(|e| ProcessingError::SerializationError(e.to_string()))?;
        
        let mut deployment_manifest = DeploymentManifest {
            models: Vec::new(),
            export_date: chrono::Utc::now(),
            version: "1.0.0".to_string(),
        };
        
        for (model_path, metadata) in models {
            let model_name = model_path.file_stem()
                .ok_or_else(|| ProcessingError::SerializationError("Invalid model path".to_string()))?
                .to_string_lossy()
                .to_string();
            
            // Copy model
            let export_model_path = export_path.join(format!("{}.fann", model_name));
            fs::copy(model_path, &export_model_path)
                .map_err(|e| ProcessingError::SerializationError(e.to_string()))?;
            
            // Add to manifest
            deployment_manifest.models.push(DeploymentModel {
                name: model_name.clone(),
                file_path: format!("{}.fann", model_name),
                config: metadata.config.clone(),
                metrics: metadata.metrics.clone(),
                checksum: metadata.checksum.clone(),
            });
        }
        
        // Save deployment manifest
        let manifest_json = serde_json::to_string_pretty(&deployment_manifest)
            .map_err(|e| ProcessingError::SerializationError(e.to_string()))?;
        
        fs::write(export_path.join("deployment_manifest.json"), manifest_json)
            .map_err(|e| ProcessingError::SerializationError(e.to_string()))?;
        
        println!("Models exported for deployment to {:?}", export_path);
        Ok(())
    }
}

/// Model deserializer for loading models with metadata
pub struct ModelDeserializer;

impl ModelDeserializer {
    /// Load model metadata
    pub fn load_metadata(metadata_path: &Path) -> Result<ModelMetadata, ProcessingError> {
        let metadata_json = fs::read_to_string(metadata_path)
            .map_err(|e| ProcessingError::DeserializationError(e.to_string()))?;
        
        let metadata: ModelMetadata = serde_json::from_str(&metadata_json)
            .map_err(|e| ProcessingError::DeserializationError(e.to_string()))?;
        
        Ok(metadata)
    }
    
    /// Load model package
    pub fn load_model_package(package_path: &Path) -> Result<(PathBuf, ModelMetadata), ProcessingError> {
        // Load package info
        let package_info_path = package_path.join("package.json");
        let package_info_json = fs::read_to_string(&package_info_path)
            .map_err(|e| ProcessingError::DeserializationError(e.to_string()))?;
        
        let package_info: PackageInfo = serde_json::from_str(&package_info_json)
            .map_err(|e| ProcessingError::DeserializationError(e.to_string()))?;
        
        // Load metadata
        let metadata_path = package_path.join(&package_info.metadata_file);
        let metadata = Self::load_metadata(&metadata_path)?;
        
        // Get model path
        let model_path = package_path.join(&package_info.model_file);
        
        Ok((model_path, metadata))
    }
    
    /// Load deployment manifest
    pub fn load_deployment_manifest(manifest_path: &Path) -> Result<DeploymentManifest, ProcessingError> {
        let manifest_json = fs::read_to_string(manifest_path)
            .map_err(|e| ProcessingError::DeserializationError(e.to_string()))?;
        
        let manifest: DeploymentManifest = serde_json::from_str(&manifest_json)
            .map_err(|e| ProcessingError::DeserializationError(e.to_string()))?;
        
        Ok(manifest)
    }
    
    /// Validate model integrity
    pub fn validate_model_integrity(
        model_path: &Path,
        expected_checksum: &str,
    ) -> Result<bool, ProcessingError> {
        let model_data = fs::read(model_path)
            .map_err(|e| ProcessingError::ValidationError(e.to_string()))?;
        
        let checksum = Self::calculate_checksum(&model_data);
        Ok(checksum == expected_checksum)
    }
    
    /// Calculate model checksum
    fn calculate_checksum(data: &[u8]) -> String {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(data);
        format!("{:x}", hasher.finalize())
    }
}

use std::path::PathBuf;

/// Package information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageInfo {
    pub name: String,
    pub version: String,
    pub model_file: String,
    pub metadata_file: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Deployment manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentManifest {
    pub models: Vec<DeploymentModel>,
    pub export_date: chrono::DateTime<chrono::Utc>,
    pub version: String,
}

/// Deployment model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentModel {
    pub name: String,
    pub file_path: String,
    pub config: ModelConfig,
    pub metrics: ModelMetrics,
    pub checksum: String,
}

/// Model backup utilities
pub struct ModelBackup;

impl ModelBackup {
    /// Create backup of all models
    pub fn backup_models(
        models_dir: &Path,
        backup_dir: &Path,
    ) -> Result<(), ProcessingError> {
        fs::create_dir_all(backup_dir)
            .map_err(|e| ProcessingError::SerializationError(e.to_string()))?;
        
        let backup_timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        let backup_path = backup_dir.join(format!("models_backup_{}", backup_timestamp));
        fs::create_dir_all(&backup_path)
            .map_err(|e| ProcessingError::SerializationError(e.to_string()))?;
        
        // Copy all model files
        for entry in fs::read_dir(models_dir)
            .map_err(|e| ProcessingError::SerializationError(e.to_string()))? {
            let entry = entry.map_err(|e| ProcessingError::SerializationError(e.to_string()))?;
            let path = entry.path();
            
            if path.is_file() {
                let file_name = path.file_name()
                    .ok_or_else(|| ProcessingError::SerializationError("Invalid file name".to_string()))?;
                let backup_file_path = backup_path.join(file_name);
                
                fs::copy(&path, &backup_file_path)
                    .map_err(|e| ProcessingError::SerializationError(e.to_string()))?;
            }
        }
        
        // Create backup manifest
        let backup_manifest = BackupManifest {
            backup_date: chrono::Utc::now(),
            source_dir: models_dir.to_string_lossy().to_string(),
            backup_dir: backup_path.to_string_lossy().to_string(),
            files_count: fs::read_dir(&backup_path)
                .map_err(|e| ProcessingError::SerializationError(e.to_string()))?
                .count(),
        };
        
        let manifest_json = serde_json::to_string_pretty(&backup_manifest)
            .map_err(|e| ProcessingError::SerializationError(e.to_string()))?;
        
        fs::write(backup_path.join("backup_manifest.json"), manifest_json)
            .map_err(|e| ProcessingError::SerializationError(e.to_string()))?;
        
        println!("Models backed up to {:?}", backup_path);
        Ok(())
    }
}

/// Backup manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupManifest {
    pub backup_date: chrono::DateTime<chrono::Utc>,
    pub source_dir: String,
    pub backup_dir: String,
    pub files_count: usize,
}

/// Model registry for tracking all security models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRegistry {
    pub models: Vec<RegisteredModel>,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Registered model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisteredModel {
    pub name: String,
    pub version: String,
    pub path: String,
    pub metadata_path: String,
    pub status: ModelStatus,
    pub registered_at: chrono::DateTime<chrono::Utc>,
}

/// Model status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelStatus {
    Active,
    Training,
    Deprecated,
    Failed,
}

impl ModelRegistry {
    /// Create new registry
    pub fn new() -> Self {
        Self {
            models: Vec::new(),
            last_updated: chrono::Utc::now(),
        }
    }
    
    /// Register a model
    pub fn register_model(&mut self, model: RegisteredModel) {
        // Remove existing model with same name
        self.models.retain(|m| m.name != model.name);
        
        // Add new model
        self.models.push(model);
        self.last_updated = chrono::Utc::now();
    }
    
    /// Save registry to disk
    pub fn save(&self, path: &Path) -> Result<(), ProcessingError> {
        let registry_json = serde_json::to_string_pretty(self)
            .map_err(|e| ProcessingError::SerializationError(e.to_string()))?;
        
        fs::write(path, registry_json)
            .map_err(|e| ProcessingError::SerializationError(e.to_string()))?;
        
        Ok(())
    }
    
    /// Load registry from disk
    pub fn load(path: &Path) -> Result<Self, ProcessingError> {
        let registry_json = fs::read_to_string(path)
            .map_err(|e| ProcessingError::DeserializationError(e.to_string()))?;
        
        let registry: ModelRegistry = serde_json::from_str(&registry_json)
            .map_err(|e| ProcessingError::DeserializationError(e.to_string()))?;
        
        Ok(registry)
    }
}