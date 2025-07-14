//! Document processing handlers

use axum::{extract::State, Json};
use std::sync::Arc;
use validator::Validate;

use crate::state::AppState;
use crate::models::{ProcessRequest, ProcessResponse, BatchRequest, BatchResponse, JobStatus};
use crate::error::ApiResult;
use crate::auth::Claims;

/// Process a single document
#[utoipa::path(
    post,
    path = "/api/v1/process",
    tag = "Processing",
    summary = "Process a document",
    description = "Process a single document and return results synchronously or asynchronously",
    request_body = ProcessRequest,
    responses(
        (status = 200, description = "Document processed successfully", body = ProcessResponse),
        (status = 400, description = "Invalid request", body = ErrorResponse),
        (status = 413, description = "Payload too large", body = ErrorResponse),
        (status = 422, description = "Validation error", body = ErrorResponse),
        (status = 500, description = "Processing error", body = ErrorResponse)
    ),
    security(
        ("bearer_auth" = [])
    )
)]
pub async fn process_document(
    State(state): State<Arc<AppState>>,
    claims: Claims,
    Json(request): Json<ProcessRequest>,
) -> ApiResult<Json<ProcessResponse>> {
    // Validate request
    request.validate()?;

    // Decode content from base64
    let content = base64::decode(&request.content)
        .map_err(|_| crate::error::ApiError::BadRequest {
            message: "Invalid base64 content".to_string(),
        })?;

    // Check file size
    if content.len() as u64 > state.config.max_file_size {
        return Err(crate::error::ApiError::PayloadTooLarge {
            message: format!(
                "File size {} bytes exceeds maximum of {} bytes",
                content.len(),
                state.config.max_file_size
            ),
        });
    }

    // Generate job ID
    let job_id = uuid::Uuid::new_v4().to_string();

    // Check cache first
    let cache_key = state.generate_cache_key(
        &format!("{:x}", md5::compute(&content)),
        &state.config.processing_config(),
    );

    if let Some(cached_result) = state.get_cached_result(&cache_key) {
        return Ok(Json(ProcessResponse {
            job_id,
            status: JobStatus::Completed,
            result: Some(crate::models::ProcessingResult {
                success: cached_result.success,
                content: cached_result.content,
                document_metadata: cached_result.metadata,
                outputs: std::collections::HashMap::new(), // TODO: convert outputs
                security_results: None, // TODO: convert security results
                warnings: cached_result.warnings,
                statistics: crate::models::ProcessingStatistics {
                    processing_time_ms: cached_result.processing_time_ms as u64,
                    neural_time_ms: None,
                    security_time_ms: None,
                    memory_usage_bytes: 0,
                    pages_processed: None,
                    character_count: Some(cached_result.content.len() as u64),
                },
            }),
            estimated_completion: None,
            submitted_at: chrono::Utc::now(),
            warnings: vec![],
        }));
    }

    if request.async_processing {
        // Queue for async processing
        let job = crate::jobs::ProcessingJob {
            id: job_id.clone(),
            user_id: claims.sub,
            content,
            filename: request.filename,
            content_type: request.content_type,
            options: request.options,
            submitted_at: chrono::Utc::now(),
        };

        state.job_queue.submit(job).await?;

        Ok(Json(ProcessResponse {
            job_id,
            status: JobStatus::Queued,
            result: None,
            estimated_completion: Some(chrono::Utc::now() + chrono::Duration::seconds(60)), // Rough estimate
            submitted_at: chrono::Utc::now(),
            warnings: vec![],
        }))
    } else {
        // Process synchronously
        let document = neural_doc_flow_core::Document::from_bytes(content, request.filename)?;
        let processing_config = convert_processing_options(request.options);
        let processor = neural_doc_flow_core::DocumentProcessor::new(processing_config)?;
        
        let result = processor.process(document).await?;

        // Cache the result
        state.cache_result(cache_key, result.clone());

        Ok(Json(ProcessResponse {
            job_id,
            status: JobStatus::Completed,
            result: Some(convert_processing_result(result)),
            estimated_completion: None,
            submitted_at: chrono::Utc::now(),
            warnings: vec![],
        }))
    }
}

/// Process multiple documents in batch
#[utoipa::path(
    post,
    path = "/api/v1/batch",
    tag = "Processing",
    summary = "Process documents in batch",
    description = "Process multiple documents in parallel",
    request_body = BatchRequest,
    responses(
        (status = 200, description = "Batch processing started", body = BatchResponse),
        (status = 400, description = "Invalid request", body = ErrorResponse),
        (status = 422, description = "Validation error", body = ErrorResponse)
    ),
    security(
        ("bearer_auth" = [])
    )
)]
pub async fn process_batch(
    State(state): State<Arc<AppState>>,
    claims: Claims,
    Json(request): Json<BatchRequest>,
) -> ApiResult<Json<BatchResponse>> {
    // Validate request
    request.validate()?;

    let batch_id = uuid::Uuid::new_v4().to_string();
    let mut document_jobs = std::collections::HashMap::new();

    // Create individual jobs for each document
    for doc in request.documents {
        let job_id = uuid::Uuid::new_v4().to_string();
        document_jobs.insert(doc.id.clone(), job_id.clone());

        // Decode content
        let content = base64::decode(&doc.content)
            .map_err(|_| crate::error::ApiError::BadRequest {
                message: format!("Invalid base64 content for document {}", doc.id),
            })?;

        // Create processing job
        let job = crate::jobs::ProcessingJob {
            id: job_id,
            user_id: claims.sub.clone(),
            content,
            filename: doc.filename,
            content_type: doc.content_type,
            options: doc.options.unwrap_or_else(|| request.options.clone()),
            submitted_at: chrono::Utc::now(),
        };

        state.job_queue.submit(job).await?;
    }

    Ok(Json(BatchResponse {
        batch_id,
        status: JobStatus::Queued,
        document_jobs,
        statistics: crate::models::BatchStatistics {
            total_documents: request.documents.len() as u32,
            completed_documents: 0,
            failed_documents: 0,
            pending_documents: request.documents.len() as u32,
            progress: 0,
            average_processing_time_ms: None,
        },
        submitted_at: chrono::Utc::now(),
        estimated_completion: Some(
            chrono::Utc::now() + chrono::Duration::seconds(request.documents.len() as i64 * 30)
        ),
    }))
}

/// Convert processing options to core config
fn convert_processing_options(options: crate::models::ProcessingOptions) -> neural_doc_flow_core::ProcessingConfig {
    let mut config = neural_doc_flow_core::ProcessingConfig::default();
    
    config.set_neural_enabled(options.neural_enhancement);
    config.set_security_level(options.security_level);
    config.set_timeout_ms(options.timeout_seconds * 1000);

    for format in options.output_formats {
        config.add_output_format(format).ok();
    }

    for (key, value) in options.custom_parameters {
        config.set_custom_option(key, value);
    }

    config
}

/// Convert processing result to API model
fn convert_processing_result(result: neural_doc_flow_processors::ProcessingResult) -> crate::models::ProcessingResult {
    crate::models::ProcessingResult {
        success: result.success,
        content: result.content,
        document_metadata: result.metadata,
        outputs: std::collections::HashMap::new(), // TODO: convert outputs
        security_results: None, // TODO: convert security results
        warnings: result.warnings,
        statistics: crate::models::ProcessingStatistics {
            processing_time_ms: result.processing_time_ms as u64,
            neural_time_ms: None,
            security_time_ms: None,
            memory_usage_bytes: 0, // TODO: track memory usage
            pages_processed: None,
            character_count: Some(result.content.len() as u64),
        },
    }
}