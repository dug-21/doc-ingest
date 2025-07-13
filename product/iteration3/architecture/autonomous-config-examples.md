# Autonomous Configuration Examples

This document shows how domain-specific code is replaced with YAML configurations, demonstrating the system's flexibility across different document types.

## 📄 SEC Financial Extraction Configuration

```yaml
# sec-10k-extraction.yaml
name: "SEC Form 10-K Extraction"
version: "2.0.0"
description: "Extract financial data from SEC Form 10-K annual reports"

# Document identification patterns
document_patterns:
  - type: "form_10k"
    identifiers:
      - regex: "^UNITED STATES\s+SECURITIES AND EXCHANGE COMMISSION"
      - regex: "FORM\s+10-K"
      - contains_all:
          - "ANNUAL REPORT PURSUANT TO SECTION 13"
          - "Securities Exchange Act of 1934"
    confidence_threshold: 0.95

# Extraction goals define what to extract
extraction_goals:
  - name: "company_information"
    description: "Extract company name, CIK, fiscal year end"
    priority: "critical"
    location_hints:
      - section: "cover_page"
      - proximity_to: ["CENTRAL INDEX KEY", "Commission File Number"]
    output_format: "key_value_pairs"
    
  - name: "financial_statements"
    description: "Extract income statement, balance sheet, cash flow statement"
    priority: "critical"
    location_hints:
      - section_headers: 
          - "CONSOLIDATED STATEMENTS OF INCOME"
          - "CONSOLIDATED BALANCE SHEETS"
          - "CONSOLIDATED STATEMENTS OF CASH FLOWS"
      - table_indicators:
          - columns_contain: ["Revenue", "Net Income", "Total Assets"]
    output_format: "structured_table"
    extraction_strategy: "table_detection_with_headers"
    
  - name: "risk_factors"
    description: "Extract Item 1A Risk Factors"
    priority: "high"
    location_hints:
      - section_marker: "ITEM 1A.*RISK FACTORS"
      - end_marker: "ITEM 1B|ITEM 2"
    output_format: "structured_text"
    processing_hints:
      - preserve_bullet_points: true
      - extract_subsections: true
      
  - name: "md&a"
    description: "Management's Discussion and Analysis"
    priority: "high"
    location_hints:
      - section_marker: "ITEM 7.*MANAGEMENT.*DISCUSSION.*ANALYSIS"
      - contains: "Results of Operations"
    output_format: "structured_text"
    
  - name: "financial_notes"
    description: "Notes to Financial Statements"
    priority: "medium"
    location_hints:
      - section_marker: "NOTES TO.*FINANCIAL STATEMENTS"
      - numbered_sections: true
    output_format: "structured_sections"

# Validation rules ensure data quality
validation_rules:
  - name: "balance_sheet_equation"
    type: "mathematical"
    description: "Assets must equal liabilities plus equity"
    expression: |
      abs(balance_sheet.total_assets - 
          (balance_sheet.total_liabilities + balance_sheet.total_equity)) < 1000
    error_message: "Balance sheet doesn't balance"
    severity: "error"
    
  - name: "income_statement_consistency"
    type: "mathematical"
    description: "Net income calculation must be consistent"
    expression: |
      abs(income_statement.revenue - income_statement.expenses - 
          income_statement.net_income) < 1000
    severity: "warning"
    
  - name: "fiscal_year_consistency"
    type: "logical"
    description: "All statements must be for same fiscal year"
    expression: |
      income_statement.fiscal_year == balance_sheet.fiscal_year &&
      balance_sheet.fiscal_year == cash_flow.fiscal_year
    severity: "error"
    
  - name: "required_sections_present"
    type: "completeness"
    description: "All required sections must be found"
    required_sections:
      - "financial_statements"
      - "risk_factors"
      - "company_information"
    severity: "error"

# Output schemas define the structure of results
output_schemas:
  - name: "financial_table"
    format: "json"
    fields:
      - name: "fiscal_year"
        type: "integer"
        required: true
      - name: "period_end_date"
        type: "date"
        format: "YYYY-MM-DD"
        required: true
      - name: "rows"
        type: "array"
        items:
          - name: "label"
            type: "string"
          - name: "values"
            type: "array"
            items:
              type: "number"
              format: "decimal(15,2)"
              
  - name: "xbrl_mapping"
    format: "xml"
    namespace: "http://xbrl.sec.gov/dei/2021"
    mappings:
      - source: "company_information.name"
        target: "dei:EntityRegistrantName"
      - source: "company_information.cik"
        target: "dei:EntityCentralIndexKey"
      - source: "financial_statements.revenue"
        target: "us-gaap:Revenue"

# Model hints help with discovery
model_hints:
  - task: "table_extraction"
    preferred_models:
      - "microsoft/table-transformer-detection"
      - "google/detr-tables"
    minimum_confidence: 0.85
    
  - task: "financial_ner"
    preferred_models:
      - "nlpaueb/sec-bert-base"
      - "yiyanghkust/finbert-tone"
    
  - task: "section_classification"
    preferred_models:
      - "facebook/bart-large-mnli"
      - "microsoft/deberta-v3-base"

# Processing configuration
processing_config:
  max_pages: 500
  timeout_seconds: 300
  parallel_sections: true
  memory_limit_mb: 2048
  
# Learning configuration  
learning_config:
  save_feedback: true
  update_frequency: "after_each_batch"
  minimum_feedback_count: 10
  confidence_improvement_threshold: 0.02
```

## 🏥 Medical Records Configuration

```yaml
# medical-records-extraction.yaml
name: "Medical Records Extraction"
version: "1.0.0"
description: "Extract patient information from medical records"

document_patterns:
  - type: "patient_record"
    identifiers:
      - contains_any:
          - "PATIENT MEDICAL RECORD"
          - "CLINICAL SUMMARY"
          - "DISCHARGE SUMMARY"
      - contains_all:
          - "Patient Name"
          - "Date of Birth"
          - "Medical Record Number"
    confidence_threshold: 0.90

extraction_goals:
  - name: "patient_demographics"
    description: "Extract patient identifying information"
    priority: "critical"
    location_hints:
      - proximity_to: ["Patient Name:", "DOB:", "MRN:"]
      - section: "header"
    output_format: "structured_record"
    privacy_level: "phi"
    
  - name: "diagnoses"
    description: "Extract diagnosis codes and descriptions"
    priority: "critical"
    location_hints:
      - section_headers: ["DIAGNOSES", "DISCHARGE DIAGNOSES", "FINAL DIAGNOSIS"]
      - proximity_to: ["ICD-10:", "Primary Diagnosis:"]
    output_format: "code_list"
    
  - name: "medications"
    description: "Extract current medications"
    priority: "high"
    location_hints:
      - section_headers: ["MEDICATIONS", "CURRENT MEDICATIONS", "DISCHARGE MEDICATIONS"]
      - list_indicators: ["•", "-", "1.", "*"]
    output_format: "medication_list"
    
  - name: "lab_results"
    description: "Extract laboratory test results"
    priority: "medium"
    location_hints:
      - section_headers: ["LABORATORY RESULTS", "LAB VALUES"]
      - table_indicators: true
    output_format: "structured_table"

validation_rules:
  - name: "valid_mrn_format"
    type: "regex"
    expression: "^[0-9]{7,10}$"
    field: "patient_demographics.mrn"
    error_message: "Invalid MRN format"
    
  - name: "valid_icd10_codes"
    type: "code_validation"
    validator: "icd10_validator"
    field: "diagnoses.codes"
    
  - name: "phi_redaction"
    type: "privacy"
    description: "Ensure PHI is properly marked for redaction"
    check_fields: ["patient_demographics", "contact_information"]

output_schemas:
  - name: "patient_record"
    format: "fhir"
    version: "R4"
    resource_type: "Patient"
    
  - name: "diagnosis_record"
    format: "json"
    fields:
      - name: "code"
        type: "string"
        pattern: "^[A-Z][0-9]{2}(\\.[0-9]{1,4})?$"
      - name: "description"
        type: "string"
      - name: "date_diagnosed"
        type: "date"

model_hints:
  - task: "medical_ner"
    preferred_models:
      - "microsoft/BiomedNLP-PubMedBERT-base"
      - "emilyalsentzer/Bio_ClinicalBERT"
  
  - task: "phi_detection"
    preferred_models:
      - "obi/deid_roberta_i2b2"

processing_config:
  enable_phi_detection: true
  redact_phi_in_output: true
  audit_trail: true
```

## 📜 Legal Document Configuration

```yaml
# legal-contract-extraction.yaml
name: "Legal Contract Analysis"
version: "1.0.0"
description: "Extract key terms from legal contracts"

document_patterns:
  - type: "legal_contract"
    identifiers:
      - contains_any:
          - "AGREEMENT"
          - "CONTRACT"
          - "TERMS AND CONDITIONS"
      - structure_indicators:
          - numbered_sections: true
          - legal_language_score: 0.7

extraction_goals:
  - name: "parties"
    description: "Extract contracting parties"
    priority: "critical"
    location_hints:
      - section: "preamble"
      - patterns: ["between .* and .*", "WHEREAS .* \\(.*\\)"]
    
  - name: "effective_date"
    description: "Contract effective date"
    priority: "critical"
    location_hints:
      - patterns: ["Effective Date: .*", "dated as of .*", "commencing on .*"]
      
  - name: "term_duration"
    description: "Contract term and duration"
    priority: "high"
    location_hints:
      - section_headers: ["TERM", "DURATION", "PERIOD"]
      
  - name: "payment_terms"
    description: "Payment amounts and schedules"
    priority: "high"
    location_hints:
      - section_headers: ["PAYMENT", "COMPENSATION", "FEES"]
      - patterns: ["\\$[0-9,]+", "payment of .*"]

validation_rules:
  - name: "parties_identified"
    type: "completeness"
    description: "At least two parties must be identified"
    expression: "len(parties) >= 2"
    
  - name: "date_consistency"
    type: "temporal"
    description: "Effective date must be before or equal to expiration"
    expression: "effective_date <= expiration_date"

output_schemas:
  - name: "contract_summary"
    format: "json"
    fields:
      - name: "parties"
        type: "array"
        items:
          - name: "party_name"
            type: "string"
          - name: "party_type"
            type: "enum"
            values: ["individual", "corporation", "llc", "partnership"]
      - name: "key_dates"
        type: "object"
        properties:
          - effective_date: "date"
          - expiration_date: "date"
          - renewal_date: "date"
```

## 🏭 Manufacturing QC Reports

```yaml
# manufacturing-qc-extraction.yaml
name: "Manufacturing QC Report Analysis"
version: "1.0.0"

document_patterns:
  - type: "qc_report"
    identifiers:
      - contains_any:
          - "QUALITY CONTROL REPORT"
          - "INSPECTION REPORT"
          - "QC CERTIFICATE"
      - contains_all:
          - "Batch Number"
          - "Test Results"

extraction_goals:
  - name: "batch_information"
    priority: "critical"
    location_hints:
      - labels: ["Batch #:", "Lot Number:", "Production Date:"]
      
  - name: "test_results"
    priority: "critical"
    output_format: "measurement_table"
    location_hints:
      - table_headers: ["Parameter", "Result", "Specification", "Pass/Fail"]
      
  - name: "defects"
    priority: "high"
    location_hints:
      - section_headers: ["DEFECTS", "NON-CONFORMANCES", "DEVIATIONS"]

validation_rules:
  - name: "all_tests_within_spec"
    type: "range_check"
    description: "All measurements within specification limits"
    
  - name: "batch_number_format"
    type: "regex"
    expression: "^[A-Z]{2}[0-9]{6}$"

output_schemas:
  - name: "qc_result"
    format: "json"
    fields:
      - name: "batch_number"
        type: "string"
      - name: "overall_result"
        type: "enum"
        values: ["PASS", "FAIL", "CONDITIONAL"]
      - name: "test_results"
        type: "array"
```

## 🎓 Academic Transcript Configuration

```yaml
# academic-transcript-extraction.yaml
name: "Academic Transcript Extraction"
version: "1.0.0"

document_patterns:
  - type: "transcript"
    identifiers:
      - contains: "OFFICIAL TRANSCRIPT"
      - structure_indicators:
          - has_courses: true
          - has_grades: true
          - has_gpa: true

extraction_goals:
  - name: "student_info"
    priority: "critical"
    location_hints:
      - section: "header"
      - labels: ["Student Name:", "Student ID:", "Date of Birth:"]
      
  - name: "courses"
    priority: "critical"
    output_format: "course_list"
    location_hints:
      - table_columns: ["Course", "Title", "Credits", "Grade"]
      - section_markers: ["SEMESTER", "TERM", "YEAR"]
      
  - name: "gpa_summary"
    priority: "high"
    location_hints:
      - patterns: ["GPA: [0-9]\\.[0-9]+", "Cumulative GPA.*[0-9]\\.[0-9]+"]

validation_rules:
  - name: "valid_grades"
    type: "enum_check"
    allowed_values: ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D", "F", "W", "I", "P"]
    
  - name: "credit_calculation"
    type: "mathematical"
    description: "Total credits must sum correctly"

output_schemas:
  - name: "transcript"
    format: "json"
    includes_schema: "education_data_standard_v2"
```

## 🔧 Configuration Features Explained

### Model Discovery Hints
```yaml
model_hints:
  - task: "specific_extraction_task"
    preferred_models:
      - "model/name"  # Ranked by preference
    minimum_confidence: 0.85
    fallback_strategy: "use_general_model"
```

### Dynamic Location Hints
```yaml
location_hints:
  - section_headers: ["HEADER1", "HEADER2"]  # Look for these headers
  - proximity_to: ["keyword1", "keyword2"]    # Find near these terms
  - patterns: ["regex1", "regex2"]           # Match these patterns
  - table_indicators:                        # Identify tables by
      columns_contain: ["col1", "col2"]      # Column headers
      min_rows: 3                            # Minimum rows
  - section_marker: "ITEM.*"                 # Section start pattern
  - end_marker: "ITEM|SECTION"               # Section end pattern
```

### Validation Rules Types
```yaml
validation_rules:
  - type: "mathematical"      # Numeric calculations
  - type: "logical"          # Boolean conditions
  - type: "regex"            # Pattern matching
  - type: "completeness"     # Required fields check
  - type: "range_check"      # Value within bounds
  - type: "enum_check"       # Value in allowed list
  - type: "temporal"         # Date/time logic
  - type: "cross_reference"  # Inter-field validation
```

### Output Format Flexibility
```yaml
output_schemas:
  - format: "json"           # Standard JSON
  - format: "xml"            # With namespaces
  - format: "csv"            # Tabular data
  - format: "fhir"           # Healthcare standard
  - format: "xbrl"           # Financial standard
  - format: "custom"         # User-defined
    template: "path/to/template"
```

## 🚀 Adding New Document Types

To support a new document type, simply create a new YAML configuration:

1. **Identify** the document patterns
2. **Define** extraction goals
3. **Specify** validation rules
4. **Design** output schemas
5. **Add** model hints (optional)
6. **Configure** processing parameters

No code changes required! The autonomous system adapts automatically.

## 📊 Performance Tuning

```yaml
# Performance can be tuned per document type
processing_config:
  max_pages: 500              # Limit for large documents
  timeout_seconds: 300        # Processing timeout
  parallel_sections: true     # Enable parallelization
  memory_limit_mb: 2048       # Memory constraints
  
  # Model selection preferences
  prefer_speed_over_accuracy: false
  minimum_model_confidence: 0.85
  
  # Caching configuration
  enable_caching: true
  cache_ttl_hours: 24
  
  # Retry configuration
  max_retries: 3
  retry_backoff: "exponential"
```

## 🎯 Summary

These configurations demonstrate how the autonomous system:

1. **Eliminates domain-specific code** - Everything is configuration
2. **Supports any document type** - From financial to medical to legal
3. **Provides flexibility** - Easy to modify without coding
4. **Ensures quality** - Through comprehensive validation rules
5. **Optimizes automatically** - Based on document characteristics
6. **Learns and improves** - Through feedback configuration

The same autonomous engine processes all document types, with behavior entirely driven by these YAML configurations.