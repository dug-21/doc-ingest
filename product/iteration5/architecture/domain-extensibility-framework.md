# Domain Extensibility Framework

## Overview

The Domain Extensibility Framework (DEF) enables users to add new document domains without modifying core system code. Through declarative configuration, hot-reload capabilities, and plugin architecture, DEF supports unlimited domain expansion while maintaining system integrity and performance.

## Architecture

### Core Components

```
domain-extensibility/
├── core/
│   ├── domain-registry.ts      # Central domain registration
│   ├── config-loader.ts        # Dynamic configuration loading
│   ├── plugin-manager.ts       # Domain plugin lifecycle
│   └── validation-engine.ts    # Rule-based validation
├── models/
│   ├── model-registry.ts       # Neural model management
│   ├── model-loader.ts         # Dynamic model loading
│   └── model-interface.ts      # Standard model API
├── templates/
│   ├── template-engine.ts      # Template processing
│   ├── template-registry.ts    # Template management
│   └── inheritance.ts          # Template inheritance
└── config/
    ├── schema/                 # Configuration schemas
    ├── validators/             # Config validators
    └── hot-reload.ts          # Live configuration updates
```

## Configuration Schema

### Domain Definition (YAML)

```yaml
# domains/financial.yaml
domain:
  id: financial
  name: Financial Documents
  version: 1.0.0
  description: Configuration for financial document processing
  
  # Neural model configuration
  models:
    primary:
      type: transformer
      path: models/financial/fin-bert-v2
      config:
        max_length: 512
        batch_size: 32
    secondary:
      type: cnn
      path: models/financial/doc-cnn
      fallback: true
  
  # Document types and patterns
  document_types:
    - id: annual_report
      patterns:
        - "annual report"
        - "10-K"
        - "yearly financial statement"
      confidence_threshold: 0.85
      
    - id: quarterly_report
      patterns:
        - "quarterly report"
        - "10-Q"
        - "Q[1-4] \\d{4}"
      confidence_threshold: 0.80
      
    - id: balance_sheet
      patterns:
        - "balance sheet"
        - "statement of financial position"
      confidence_threshold: 0.90
  
  # Extraction rules
  extraction:
    entities:
      - name: revenue
        patterns:
          - regex: "(?:total )?revenue[s]?:?\\s*\\$?([\\d,]+(?:\\.\\d+)?[MBK]?)"
          - nlp: "MONEY after {revenue, sales, income}"
        post_process:
          - normalize_currency
          - convert_to_millions
          
      - name: company_name
        patterns:
          - nlp: "ORG in first paragraph"
          - regex: "^([A-Z][\\w\\s&,]+(?:Inc|Corp|LLC|Ltd))\\.?"
        validation:
          - is_valid_company_name
          
      - name: fiscal_year
        patterns:
          - regex: "fiscal year (?:ended|ending) \\w+ \\d+,? (\\d{4})"
          - nlp: "DATE with year pattern"
        validation:
          - is_valid_year
          - is_not_future_year
    
    tables:
      financial_metrics:
        headers:
          - ["metric", "value", "change"]
          - ["item", "current year", "prior year"]
        extraction_strategy: "columnar"
        value_normalization: true
  
  # Validation rules
  validation:
    rules:
      - id: revenue_positive
        field: revenue
        condition: "value > 0"
        severity: error
        message: "Revenue must be positive"
        
      - id: fiscal_year_range
        field: fiscal_year
        condition: "value >= 1900 && value <= current_year()"
        severity: error
        message: "Invalid fiscal year"
        
      - id: balance_sheet_balance
        type: cross_field
        condition: "assets == liabilities + equity"
        tolerance: 0.01
        severity: warning
        message: "Balance sheet doesn't balance"
    
    quality_checks:
      - completeness:
          required_fields: [company_name, fiscal_year, revenue]
          threshold: 0.95
      - consistency:
          cross_validate: true
          historical_comparison: true
  
  # Output templates
  output:
    default_template: financial_standard
    templates:
      - id: financial_standard
        inherit: base_template
        format: json
        structure:
          metadata:
            domain: "{{ domain.name }}"
            document_type: "{{ document.type }}"
            confidence: "{{ extraction.confidence }}"
          company:
            name: "{{ entities.company_name }}"
            fiscal_year: "{{ entities.fiscal_year }}"
          financials:
            revenue: "{{ entities.revenue | currency }}"
            tables: "{{ tables.financial_metrics | format_table }}"
            
      - id: financial_summary
        format: markdown
        template: |
          # {{ entities.company_name }} Financial Summary
          
          **Fiscal Year:** {{ entities.fiscal_year }}
          **Revenue:** {{ entities.revenue | currency }}
          
          {{ tables.financial_metrics | markdown_table }}
```

### Domain Plugin Interface

```typescript
// Domain Plugin Interface
export interface DomainPlugin {
  // Metadata
  id: string;
  version: string;
  domain: DomainConfig;
  
  // Lifecycle hooks
  onLoad(): Promise<void>;
  onUnload(): Promise<void>;
  onConfigUpdate(newConfig: DomainConfig): Promise<void>;
  
  // Processing hooks
  preProcess(document: Document): Promise<Document>;
  postProcess(results: ExtractionResults): Promise<ExtractionResults>;
  
  // Custom extractors
  customExtractors?: {
    [key: string]: CustomExtractor;
  };
  
  // Custom validators
  customValidators?: {
    [key: string]: ValidationFunction;
  };
  
  // Custom formatters
  customFormatters?: {
    [key: string]: FormatterFunction;
  };
}

// Custom Extractor Interface
export interface CustomExtractor {
  name: string;
  extract(text: string, context: ExtractionContext): Promise<any>;
  confidence(result: any): number;
}

// Validation Function
export type ValidationFunction = (
  value: any, 
  context: ValidationContext
) => ValidationResult;

// Formatter Function
export type FormatterFunction = (
  value: any,
  options?: FormatterOptions
) => string;
```

## Example Domain Configurations

### Legal Domain

```yaml
# domains/legal.yaml
domain:
  id: legal
  name: Legal Documents
  version: 1.0.0
  
  models:
    primary:
      type: transformer
      path: models/legal/legal-bert
      config:
        max_length: 1024
        enable_ner: true
  
  document_types:
    - id: contract
      patterns:
        - "agreement"
        - "contract"
        - "terms and conditions"
      sub_types:
        - employment
        - service
        - nda
        - purchase
        
    - id: court_filing
      patterns:
        - "plaintiff v[s]?\\. defendant"
        - "case no\\.?"
        - "docket"
  
  extraction:
    entities:
      - name: parties
        type: list
        patterns:
          - nlp: "PERSON or ORG near {party, plaintiff, defendant}"
          - regex: "between ([^,]+) (?:and|&) ([^,]+)"
        
      - name: effective_date
        patterns:
          - regex: "effective (?:as of |date[:]? )([^,\\.]+)"
          - nlp: "DATE after {effective, commence}"
          
      - name: jurisdiction
        patterns:
          - nlp: "GPE near {law, jurisdiction, governed}"
          - regex: "laws of ([^,\\.]+)"
    
    clauses:
      - type: termination
        keywords: [terminate, termination, end, expire]
        extract_full_clause: true
        
      - type: liability
        keywords: [liability, liable, damages, indemnify]
        extract_full_clause: true
  
  validation:
    rules:
      - id: parties_min_two
        field: parties
        condition: "length(value) >= 2"
        severity: error
        
      - id: valid_date_format
        field: effective_date
        condition: "is_valid_date(value)"
        severity: error
  
  output:
    templates:
      - id: legal_summary
        format: json
        structure:
          document_type: "{{ document.type }}"
          parties: "{{ entities.parties | join(', ') }}"
          effective_date: "{{ entities.effective_date | date_format }}"
          key_clauses: "{{ clauses | summarize }}"
```

### Medical Domain

```yaml
# domains/medical.yaml
domain:
  id: medical
  name: Medical Documents
  version: 1.0.0
  
  # Compliance and privacy
  compliance:
    standards: [HIPAA, GDPR]
    pii_handling: redact
    audit_logging: true
  
  models:
    primary:
      type: transformer
      path: models/medical/bio-bert
      config:
        enable_medical_ner: true
        
    specialist:
      radiology:
        path: models/medical/rad-bert
      pathology:
        path: models/medical/path-bert
  
  document_types:
    - id: clinical_note
      patterns:
        - "chief complaint"
        - "history of present illness"
        - "assessment and plan"
        
    - id: lab_report
      patterns:
        - "laboratory results"
        - "test results"
        - "reference range"
        
    - id: radiology_report
      patterns:
        - "findings"
        - "impression"
        - "technique"
  
  extraction:
    entities:
      - name: patient_id
        patterns:
          - regex: "(?:MRN|patient id)[:]?\\s*([\\d-]+)"
        pii: true
        
      - name: diagnoses
        type: list
        patterns:
          - nlp: "MEDICAL_CONDITION"
          - icd10_lookup: true
          
      - name: medications
        type: list
        patterns:
          - nlp: "MEDICATION with dosage"
          - drug_database_lookup: true
          
      - name: vitals
        type: structured
        components:
          blood_pressure:
            pattern: "(\\d+)/(\\d+)\\s*mm\\s*Hg"
          heart_rate:
            pattern: "(?:HR|heart rate)[:]?\\s*(\\d+)"
          temperature:
            pattern: "(?:temp|temperature)[:]?\\s*([\\d\\.]+)\\s*°?[CF]"
  
  validation:
    medical_specific:
      - validate_icd10_codes: true
      - validate_drug_interactions: true
      - validate_dosage_ranges: true
      - validate_vital_ranges: true
  
  output:
    privacy_mode: true
    templates:
      - id: clinical_summary
        format: hl7-fhir
        redact_pii: true
```

### Scientific Papers Domain

```yaml
# domains/scientific.yaml
domain:
  id: scientific
  name: Scientific Papers
  version: 1.0.0
  
  models:
    primary:
      type: transformer
      path: models/scientific/scibert
      config:
        enable_citation_parsing: true
        enable_formula_extraction: true
  
  document_types:
    - id: research_paper
      patterns:
        - "abstract"
        - "introduction"
        - "methodology"
        - "results"
        - "conclusion"
        
    - id: review_article
      patterns:
        - "systematic review"
        - "meta-analysis"
        - "literature review"
  
  extraction:
    metadata:
      - name: title
        location: first_page
        patterns:
          - css: "h1.paper-title"
          - nlp: "longest text in first 10%"
          
      - name: authors
        type: list
        patterns:
          - nlp: "PERSON near {author, by}"
          - regex: "^([^\\d]+?)(?:\\d|\\*|†)"
          
      - name: doi
        patterns:
          - regex: "(?:doi[:]?\\s*|https://doi\\.org/)([\\d\\.]+/[\\w\\.]+)"
          
      - name: publication_date
        patterns:
          - nlp: "DATE in header"
          - regex: "(?:published|received)[:]?\\s*([^,;]+)"
    
    content:
      - name: abstract
        section: abstract
        max_length: 500
        
      - name: keywords
        type: list
        patterns:
          - after: "keywords:"
          - nlp: "key phrases"
          
      - name: citations
        type: list
        format: bibtex
        patterns:
          - regex: "\\[\\d+\\]"
          - inline_citation_parser: true
          
      - name: figures
        type: list
        extract:
          - caption
          - reference_number
          - page_location
          
      - name: formulas
        type: list
        latex: true
        patterns:
          - between: ["\\begin{equation}", "\\end{equation}"]
          - regex: "\\$\\$(.+?)\\$\\$"
  
  validation:
    academic_checks:
      - citation_format: true
      - doi_validity: true
      - author_orcid: optional
      - formula_syntax: true
  
  output:
    templates:
      - id: paper_metadata
        format: json
        structure:
          title: "{{ metadata.title }}"
          authors: "{{ metadata.authors }}"
          doi: "{{ metadata.doi }}"
          abstract: "{{ content.abstract }}"
          keywords: "{{ content.keywords }}"
          citation_count: "{{ content.citations | length }}"
          
      - id: bibtex_entry
        format: bibtex
        template: |
          @article{{ '{' }}{{ metadata.doi | slugify }},
            title = {{ '{' }}{{ metadata.title }}{{ '}' }},
            author = {{ '{' }}{{ metadata.authors | bibtex_authors }}{{ '}' }},
            year = {{ '{' }}{{ metadata.publication_date | year }}{{ '}' }},
            doi = {{ '{' }}{{ metadata.doi }}{{ '}' }}
          {{ '}' }}
```

## Implementation Guide

### 1. Domain Registration

```typescript
// Domain registration example
import { DomainRegistry } from '@docai/domain-extensibility';

const registry = new DomainRegistry({
  configPath: './domains',
  hotReload: true,
  validation: 'strict'
});

// Auto-discover domains
await registry.discover();

// Manual registration
await registry.register({
  configFile: './domains/custom.yaml',
  plugin: './plugins/custom-domain.js'
});

// Domain lifecycle events
registry.on('domain:loaded', (domain) => {
  console.log(`Domain ${domain.id} loaded`);
});

registry.on('domain:updated', (domain) => {
  console.log(`Domain ${domain.id} configuration updated`);
});
```

### 2. Model Management

```typescript
// Model registry usage
import { ModelRegistry } from '@docai/domain-extensibility';

const modelRegistry = new ModelRegistry({
  modelsPath: './models',
  cacheSize: '4GB',
  gpuEnabled: true
});

// Register domain-specific model
await modelRegistry.register('financial', {
  primary: {
    type: 'transformer',
    path: './models/financial/fin-bert-v2',
    preload: true
  }
});

// Dynamic model loading
const model = await modelRegistry.load('financial', 'primary');
const results = await model.process(document);
```

### 3. Template Engine

```typescript
// Template usage
import { TemplateEngine } from '@docai/domain-extensibility';

const engine = new TemplateEngine({
  templatesPath: './templates',
  enableInheritance: true
});

// Register custom filters
engine.registerFilter('currency', (value) => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD'
  }).format(value);
});

// Render output
const output = await engine.render('financial_summary', {
  entities: extractedEntities,
  tables: extractedTables,
  metadata: documentMetadata
});
```

### 4. Validation Engine

```typescript
// Validation setup
import { ValidationEngine } from '@docai/domain-extensibility';

const validator = new ValidationEngine();

// Register custom validators
validator.register('is_valid_company_name', (value) => {
  const companyPattern = /^[A-Z][\w\s&,]+(?:Inc|Corp|LLC|Ltd)\.?$/;
  return {
    valid: companyPattern.test(value),
    message: 'Invalid company name format'
  };
});

// Validate extraction results
const validationResults = await validator.validate(
  extractionResults,
  domainConfig.validation.rules
);
```

## Hot Reload Configuration

```typescript
// Hot reload setup
import { ConfigLoader } from '@docai/domain-extensibility';

const loader = new ConfigLoader({
  watch: true,
  debounce: 1000
});

loader.on('config:changed', async (domain, changes) => {
  console.log(`Configuration changed for ${domain}:`, changes);
  
  // Reload affected components
  if (changes.includes('models')) {
    await modelRegistry.reload(domain);
  }
  
  if (changes.includes('templates')) {
    await templateEngine.reload(domain);
  }
});
```

## Plugin Development

```javascript
// Example domain plugin
export default class CustomDomainPlugin {
  constructor(config) {
    this.config = config;
    this.customExtractors = {
      customField: new CustomFieldExtractor()
    };
  }
  
  async onLoad() {
    // Initialize plugin resources
    await this.loadCustomModels();
    await this.registerCustomValidators();
  }
  
  async preProcess(document) {
    // Pre-processing logic
    return this.normalizeDocument(document);
  }
  
  async postProcess(results) {
    // Post-processing logic
    return this.enrichResults(results);
  }
}
```

## Best Practices

### 1. Configuration Management
- Use version control for domain configurations
- Implement configuration validation before deployment
- Maintain backward compatibility for configuration changes
- Document all custom patterns and rules

### 2. Performance Optimization
- Lazy-load domain configurations and models
- Implement model caching strategies
- Use appropriate batch sizes for processing
- Monitor memory usage for large domains

### 3. Security Considerations
- Validate all user-provided configurations
- Implement sandboxing for custom plugins
- Audit access to sensitive domain data
- Encrypt stored models and configurations

### 4. Testing Strategy
- Create test suites for each domain
- Validate extraction accuracy metrics
- Test configuration hot-reload scenarios
- Benchmark performance across domains

## Monitoring and Analytics

```yaml
# Domain analytics configuration
analytics:
  metrics:
    - extraction_accuracy
    - processing_time
    - validation_pass_rate
    - model_confidence_distribution
    
  reporting:
    dashboard: true
    export_format: [json, csv]
    schedule: daily
    
  alerts:
    - metric: extraction_accuracy
      threshold: 0.85
      condition: below
      
    - metric: processing_time
      threshold: 5000  # ms
      condition: above
```

## Migration Guide

### From Hard-coded Domains

1. Extract domain logic into configuration files
2. Convert hard-coded patterns to YAML/TOML format
3. Refactor custom code into plugins
4. Update model loading to use registry
5. Migrate templates to template engine format

### Version Upgrades

```yaml
# Version migration configuration
migration:
  from_version: 0.9.0
  to_version: 1.0.0
  
  steps:
    - backup_current_config
    - validate_new_schema
    - migrate_patterns
    - update_model_paths
    - test_extraction
    - deploy_new_version
```

## Conclusion

The Domain Extensibility Framework provides a robust, scalable solution for adding new document domains without code changes. Through declarative configuration, plugin architecture, and hot-reload capabilities, organizations can rapidly adapt to new document types while maintaining system stability and performance.