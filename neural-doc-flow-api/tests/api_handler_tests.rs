mod common;

use common::*;
use neural_doc_flow_api::models::{
    Document, DocumentUpload, ProcessingJob, ProcessingResult, ProcessingStatus,
};
use serde_json::json;

#[tokio::test]
async fn test_document_upload() {
    let (app, state) = create_test_app().await;
    
    // Authenticate
    let api_key = MockAuth::create_test_api_key(&state, "upload-test").await;

    let upload = DocumentUpload {
        filename: "test.pdf".to_string(),
        content_type: "application/pdf".to_string(),
        data: base64::encode("fake pdf content"),
        metadata: Some(json!({"author": "Test Author"})),
    };

    let response = TestRequest::post("/api/v1/documents")
        .api_key(&api_key.key)
        .json(&upload)
        .send(app)
        .await;

    response.assert_created();

    let doc: Document = response.json();
    assert_eq!(doc.filename, "test.pdf");
    assert_eq!(doc.content_type, "application/pdf");
    assert!(doc.metadata.is_some());
}

#[tokio::test]
async fn test_document_listing() {
    let (app, state) = create_test_app().await;
    
    // Authenticate
    let api_key = MockAuth::create_test_api_key(&state, "list-test").await;

    // Upload some documents
    for i in 0..3 {
        let upload = DocumentUpload {
            filename: format!("doc{}.pdf", i),
            content_type: "application/pdf".to_string(),
            data: base64::encode(format!("content {}", i)),
            metadata: None,
        };

        TestRequest::post("/api/v1/documents")
            .api_key(&api_key.key)
            .json(&upload)
            .send(app.clone())
            .await
            .assert_created();
    }

    // List documents
    let response = TestRequest::get("/api/v1/documents")
        .api_key(&api_key.key)
        .send(app)
        .await;

    response.assert_ok();

    let docs: Vec<Document> = response.json();
    assert!(docs.len() >= 3);
}

#[tokio::test]
async fn test_document_get_by_id() {
    let (app, state) = create_test_app().await;
    
    // Authenticate
    let api_key = MockAuth::create_test_api_key(&state, "get-test").await;

    // Upload a document
    let upload = DocumentUpload {
        filename: "get-test.pdf".to_string(),
        content_type: "application/pdf".to_string(),
        data: base64::encode("test content"),
        metadata: None,
    };

    let upload_response = TestRequest::post("/api/v1/documents")
        .api_key(&api_key.key)
        .json(&upload)
        .send(app.clone())
        .await;

    let doc: Document = upload_response.json();

    // Get the document by ID
    let response = TestRequest::get(&format!("/api/v1/documents/{}", doc.id))
        .api_key(&api_key.key)
        .send(app)
        .await;

    response.assert_ok();

    let retrieved: Document = response.json();
    assert_eq!(retrieved.id, doc.id);
    assert_eq!(retrieved.filename, doc.filename);
}

#[tokio::test]
async fn test_document_delete() {
    let (app, state) = create_test_app().await;
    
    // Authenticate
    let api_key = MockAuth::create_test_api_key(&state, "delete-test").await;

    // Upload a document
    let upload = DocumentUpload {
        filename: "delete-test.pdf".to_string(),
        content_type: "application/pdf".to_string(),
        data: base64::encode("to be deleted"),
        metadata: None,
    };

    let upload_response = TestRequest::post("/api/v1/documents")
        .api_key(&api_key.key)
        .json(&upload)
        .send(app.clone())
        .await;

    let doc: Document = upload_response.json();

    // Delete the document
    let response = TestRequest::delete(&format!("/api/v1/documents/{}", doc.id))
        .api_key(&api_key.key)
        .send(app.clone())
        .await;

    response.assert_ok();

    // Verify it's gone
    let response = TestRequest::get(&format!("/api/v1/documents/{}", doc.id))
        .api_key(&api_key.key)
        .send(app)
        .await;

    response.assert_not_found();
}

#[tokio::test]
async fn test_document_processing() {
    let (app, state) = create_test_app().await;
    
    // Authenticate
    let api_key = MockAuth::create_test_api_key(&state, "process-test").await;

    // Upload a document
    let upload = DocumentUpload {
        filename: "process-test.pdf".to_string(),
        content_type: "application/pdf".to_string(),
        data: base64::encode("content to process"),
        metadata: None,
    };

    let upload_response = TestRequest::post("/api/v1/documents")
        .api_key(&api_key.key)
        .json(&upload)
        .send(app.clone())
        .await;

    let doc: Document = upload_response.json();

    // Start processing
    let response = TestRequest::post(&format!("/api/v1/documents/{}/process", doc.id))
        .api_key(&api_key.key)
        .send(app)
        .await;

    response.assert_created();

    let job: ProcessingJob = response.json();
    assert_eq!(job.document_id, doc.id);
    assert_eq!(job.status, ProcessingStatus::Pending);
}

#[tokio::test]
async fn test_processing_status() {
    let (app, state) = create_test_app().await;
    
    // Authenticate
    let api_key = MockAuth::create_test_api_key(&state, "status-test").await;

    // Create a test document and job
    let doc = fixtures::test_document();
    sqlx::query!(
        r#"
        INSERT INTO documents (id, filename, content_type, size, hash, storage_path, created_at, updated_at)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
        "#,
        doc.id,
        doc.filename,
        doc.content_type,
        doc.size,
        doc.hash,
        doc.storage_path,
        doc.created_at,
        doc.updated_at
    )
    .execute(&state.db)
    .await
    .unwrap();

    let job = fixtures::test_processing_job();
    sqlx::query!(
        r#"
        INSERT INTO processing_jobs (id, document_id, status, created_at, processor_version)
        VALUES (?1, ?2, ?3, ?4, ?5)
        "#,
        job.id,
        doc.id,
        job.status as i32,
        job.created_at,
        job.processor_version
    )
    .execute(&state.db)
    .await
    .unwrap();

    // Check status
    let response = TestRequest::get(&format!("/api/v1/processing/{}", job.id))
        .api_key(&api_key.key)
        .send(app)
        .await;

    response.assert_ok();

    let retrieved: ProcessingJob = response.json();
    assert_eq!(retrieved.id, job.id);
}

#[tokio::test]
async fn test_processing_result() {
    let (app, state) = create_test_app().await;
    
    // Authenticate
    let api_key = MockAuth::create_test_api_key(&state, "result-test").await;

    // Create test data with completed processing
    let doc = fixtures::test_document();
    sqlx::query!(
        r#"
        INSERT INTO documents (id, filename, content_type, size, hash, storage_path, created_at, updated_at)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
        "#,
        doc.id,
        doc.filename,
        doc.content_type,
        doc.size,
        doc.hash,
        doc.storage_path,
        doc.created_at,
        doc.updated_at
    )
    .execute(&state.db)
    .await
    .unwrap();

    let mut job = fixtures::test_processing_job();
    job.document_id = doc.id.clone();
    job.status = ProcessingStatus::Completed;
    job.completed_at = Some(chrono::Utc::now());
    job.result = Some(json!({"extracted_text": "Test content"}));

    sqlx::query!(
        r#"
        INSERT INTO processing_jobs (id, document_id, status, created_at, completed_at, result, processor_version)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
        "#,
        job.id,
        job.document_id,
        job.status as i32,
        job.created_at,
        job.completed_at,
        job.result,
        job.processor_version
    )
    .execute(&state.db)
    .await
    .unwrap();

    // Get result
    let response = TestRequest::get(&format!("/api/v1/processing/{}/result", job.id))
        .api_key(&api_key.key)
        .send(app)
        .await;

    response.assert_ok();

    let result: ProcessingResult = response.json();
    assert_eq!(result.job_id, job.id);
    assert!(result.result.is_some());
}

#[tokio::test]
async fn test_bulk_document_upload() {
    let (app, state) = create_test_app().await;
    
    // Authenticate
    let api_key = MockAuth::create_test_api_key(&state, "bulk-test").await;

    let uploads = vec![
        DocumentUpload {
            filename: "bulk1.pdf".to_string(),
            content_type: "application/pdf".to_string(),
            data: base64::encode("content 1"),
            metadata: None,
        },
        DocumentUpload {
            filename: "bulk2.pdf".to_string(),
            content_type: "application/pdf".to_string(),
            data: base64::encode("content 2"),
            metadata: None,
        },
    ];

    let response = TestRequest::post("/api/v1/documents/bulk")
        .api_key(&api_key.key)
        .json(&uploads)
        .send(app)
        .await;

    response.assert_created();

    let docs: Vec<Document> = response.json();
    assert_eq!(docs.len(), 2);
}

#[tokio::test]
async fn test_document_search() {
    let (app, state) = create_test_app().await;
    
    // Authenticate
    let api_key = MockAuth::create_test_api_key(&state, "search-test").await;

    // Upload documents with metadata
    for i in 0..3 {
        let upload = DocumentUpload {
            filename: format!("search{}.pdf", i),
            content_type: "application/pdf".to_string(),
            data: base64::encode(format!("content {}", i)),
            metadata: Some(json!({
                "category": if i == 0 { "important" } else { "regular" }
            })),
        };

        TestRequest::post("/api/v1/documents")
            .api_key(&api_key.key)
            .json(&upload)
            .send(app.clone())
            .await
            .assert_created();
    }

    // Search for important documents
    let response = TestRequest::get("/api/v1/documents/search?q=important")
        .api_key(&api_key.key)
        .send(app)
        .await;

    response.assert_ok();

    let docs: Vec<Document> = response.json();
    assert!(docs.len() >= 1);
}

#[tokio::test]
async fn test_invalid_document_format() {
    let (app, state) = create_test_app().await;
    
    // Authenticate
    let api_key = MockAuth::create_test_api_key(&state, "invalid-test").await;

    let upload = DocumentUpload {
        filename: "test.exe".to_string(), // Invalid file type
        content_type: "application/x-executable".to_string(),
        data: base64::encode("invalid content"),
        metadata: None,
    };

    let response = TestRequest::post("/api/v1/documents")
        .api_key(&api_key.key)
        .json(&upload)
        .send(app)
        .await;

    response.assert_bad_request();
}

#[tokio::test]
async fn test_document_pagination() {
    let (app, state) = create_test_app().await;
    
    // Authenticate
    let api_key = MockAuth::create_test_api_key(&state, "pagination-test").await;

    // Upload many documents
    for i in 0..15 {
        let upload = DocumentUpload {
            filename: format!("page{}.pdf", i),
            content_type: "application/pdf".to_string(),
            data: base64::encode(format!("content {}", i)),
            metadata: None,
        };

        TestRequest::post("/api/v1/documents")
            .api_key(&api_key.key)
            .json(&upload)
            .send(app.clone())
            .await
            .assert_created();
    }

    // Get first page
    let response = TestRequest::get("/api/v1/documents?page=1&limit=10")
        .api_key(&api_key.key)
        .send(app.clone())
        .await;

    response.assert_ok();

    let docs: Vec<Document> = response.json();
    assert_eq!(docs.len(), 10);

    // Get second page
    let response = TestRequest::get("/api/v1/documents?page=2&limit=10")
        .api_key(&api_key.key)
        .send(app)
        .await;

    response.assert_ok();

    let docs: Vec<Document> = response.json();
    assert!(docs.len() >= 5);
}