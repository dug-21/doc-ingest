"""
Autonomous Document Processing Agent Implementation
Demonstrates domain elimination through self-organizing pipelines
"""

import yaml
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from transformers import pipeline, AutoModel, AutoTokenizer
import torch
from huggingface_hub import HfApi, ModelFilter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExtractionRequirement:
    """Represents a single field extraction requirement from YAML"""
    name: str
    field_type: str
    required: bool
    extraction_method: str
    source: str
    validation_rules: List[Dict]
    confidence_threshold: float = 0.85


@dataclass
class ModelCandidate:
    """Represents a candidate model for a specific task"""
    model_id: str
    task_type: str
    architecture: str
    performance_score: float
    capabilities: List[str]
    resource_requirements: Dict


class DocumentSchema:
    """Parses and manages YAML schema definitions"""
    
    def __init__(self, schema_path: str):
        with open(schema_path, 'r') as f:
            self.schema = yaml.safe_load(f)
        
        self.document_profile = self.schema['document_profile']
        self.output_spec = self.schema['output']
        self.quality_spec = self.schema.get('quality', {})
        self.model_hints = self.schema.get('model_hints', {})
        
        # Parse extraction requirements
        self.requirements = self._parse_requirements()
    
    def _parse_requirements(self) -> List[ExtractionRequirement]:
        """Convert YAML fields to extraction requirements"""
        requirements = []
        
        for field in self.output_spec['fields']:
            req = ExtractionRequirement(
                name=field['name'],
                field_type=field['type'],
                required=field['required'],
                extraction_method=field['extraction']['method'],
                source=field['extraction'].get('source', 'document'),
                validation_rules=field['extraction'].get('validation', []),
                confidence_threshold=self.quality_spec.get('min_confidence', 0.85)
            )
            requirements.append(req)
        
        return requirements
    
    def get_required_capabilities(self) -> Dict[str, List[str]]:
        """Determine what capabilities are needed based on schema"""
        capabilities = {
            'text_extraction': [],
            'structure_detection': [],
            'table_extraction': [],
            'entity_recognition': [],
            'visual_analysis': []
        }
        
        # Analyze extraction methods
        for req in self.requirements:
            if req.extraction_method == 'direct':
                capabilities['text_extraction'].append('high_quality')
            elif req.extraction_method == 'table':
                capabilities['table_extraction'].append(req.source)
            elif req.extraction_method == 'ner':
                capabilities['entity_recognition'].append(req.field_type)
            elif req.extraction_method == 'semantic':
                capabilities['text_extraction'].append('semantic_search')
            elif req.extraction_method == 'visual':
                capabilities['visual_analysis'].append(req.field_type)
        
        # Check structure requirements
        if self.document_profile['structure'].get('hierarchical'):
            capabilities['structure_detection'].append('hierarchical')
        
        if self.document_profile['structure'].get('tables', {}).get('expected'):
            capabilities['table_extraction'].append('detection')
        
        return {k: v for k, v in capabilities.items() if v}


class ModelDiscoveryAgent:
    """Agent responsible for finding appropriate models for tasks"""
    
    def __init__(self):
        self.hf_api = HfApi()
        self.model_cache = {}
        self.performance_history = {}
    
    def discover_models(self, capabilities: Dict[str, List[str]]) -> Dict[str, ModelCandidate]:
        """Discover models that match required capabilities"""
        discovered_models = {}
        
        for capability_type, requirements in capabilities.items():
            logger.info(f"Discovering models for {capability_type}: {requirements}")
            
            # Search Hugging Face Hub
            candidates = self._search_models(capability_type, requirements)
            
            # Evaluate candidates
            best_model = self._select_best_model(candidates, requirements)
            
            if best_model:
                discovered_models[capability_type] = best_model
                logger.info(f"Selected {best_model.model_id} for {capability_type}")
        
        return discovered_models
    
    def _search_models(self, task_type: str, requirements: List[str]) -> List[ModelCandidate]:
        """Search Hugging Face Hub for suitable models"""
        candidates = []
        
        # Define search mappings
        task_mappings = {
            'text_extraction': ['token-classification', 'text-generation'],
            'structure_detection': ['image-segmentation', 'object-detection'],
            'table_extraction': ['table-question-answering', 'object-detection'],
            'entity_recognition': ['token-classification', 'ner'],
            'visual_analysis': ['image-classification', 'object-detection']
        }
        
        # Search for models
        for hf_task in task_mappings.get(task_type, []):
            try:
                models = self.hf_api.list_models(
                    filter=ModelFilter(task=hf_task),
                    sort="downloads",
                    direction=-1,
                    limit=10
                )
                
                for model in models:
                    # Create candidate
                    candidate = ModelCandidate(
                        model_id=model.modelId,
                        task_type=task_type,
                        architecture=model.config.get('model_type', 'unknown'),
                        performance_score=self._estimate_performance(model),
                        capabilities=self._extract_capabilities(model),
                        resource_requirements=self._estimate_resources(model)
                    )
                    candidates.append(candidate)
            
            except Exception as e:
                logger.warning(f"Error searching models for {hf_task}: {e}")
        
        return candidates
    
    def _estimate_performance(self, model_info) -> float:
        """Estimate model performance based on metadata"""
        score = 0.5  # Base score
        
        # Popularity boost
        if model_info.downloads > 10000:
            score += 0.2
        elif model_info.downloads > 1000:
            score += 0.1
        
        # Recent update boost
        if hasattr(model_info, 'lastModified'):
            # Boost for recently updated models
            score += 0.1
        
        # Fine-tuned model boost
        if 'finetuned' in model_info.modelId.lower():
            score += 0.15
        
        return min(score, 1.0)
    
    def _extract_capabilities(self, model_info) -> List[str]:
        """Extract capabilities from model metadata"""
        capabilities = []
        
        # Check tags
        if hasattr(model_info, 'tags'):
            capabilities.extend(model_info.tags)
        
        # Infer from model name
        model_name = model_info.modelId.lower()
        if 'layout' in model_name:
            capabilities.append('layout_understanding')
        if 'table' in model_name:
            capabilities.append('table_extraction')
        if 'ocr' in model_name:
            capabilities.append('ocr')
        
        return capabilities
    
    def _estimate_resources(self, model_info) -> Dict:
        """Estimate resource requirements"""
        # Simple estimation based on model metadata
        if hasattr(model_info, 'safetensors'):
            model_size = sum(file.size for file in model_info.safetensors.files) / 1e9
        else:
            model_size = 1.0  # Default 1GB
        
        return {
            'memory_gb': model_size * 2,  # Rough estimate
            'requires_gpu': model_size > 2.0
        }
    
    def _select_best_model(self, candidates: List[ModelCandidate], 
                          requirements: List[str]) -> Optional[ModelCandidate]:
        """Select best model based on requirements and performance"""
        if not candidates:
            return None
        
        # Score candidates
        scored_candidates = []
        for candidate in candidates:
            score = candidate.performance_score
            
            # Boost for matching requirements
            for req in requirements:
                if req in candidate.capabilities:
                    score += 0.1
            
            # Check resource constraints
            if candidate.resource_requirements['requires_gpu'] and not torch.cuda.is_available():
                score -= 0.3
            
            scored_candidates.append((score, candidate))
        
        # Return best candidate
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return scored_candidates[0][1] if scored_candidates else None


class PipelineBuilder:
    """Builds dynamic processing pipelines based on requirements and models"""
    
    def __init__(self, schema: DocumentSchema, models: Dict[str, ModelCandidate]):
        self.schema = schema
        self.models = models
        self.pipeline_stages = []
    
    def build_pipeline(self) -> 'DocumentPipeline':
        """Construct processing pipeline"""
        logger.info("Building processing pipeline...")
        
        # Create stages for each requirement
        for requirement in self.schema.requirements:
            stage = self._create_extraction_stage(requirement)
            if stage:
                self.pipeline_stages.append(stage)
        
        # Add validation stages
        validation_stage = self._create_validation_stage()
        self.pipeline_stages.append(validation_stage)
        
        # Create pipeline
        pipeline = DocumentPipeline(self.pipeline_stages)
        
        logger.info(f"Pipeline built with {len(self.pipeline_stages)} stages")
        return pipeline
    
    def _create_extraction_stage(self, requirement: ExtractionRequirement) -> 'PipelineStage':
        """Create extraction stage for a requirement"""
        method_handlers = {
            'direct': DirectExtractionStage,
            'table': TableExtractionStage,
            'ner': NERExtractionStage,
            'semantic': SemanticExtractionStage,
            'pattern': PatternExtractionStage,
            'visual': VisualExtractionStage
        }
        
        stage_class = method_handlers.get(requirement.extraction_method)
        if not stage_class:
            logger.warning(f"Unknown extraction method: {requirement.extraction_method}")
            return None
        
        # Find appropriate model
        model_candidate = self._find_model_for_requirement(requirement)
        
        return stage_class(requirement, model_candidate)
    
    def _find_model_for_requirement(self, requirement: ExtractionRequirement) -> Optional[ModelCandidate]:
        """Find the best model for a specific requirement"""
        # Map extraction methods to capability types
        method_to_capability = {
            'direct': 'text_extraction',
            'table': 'table_extraction',
            'ner': 'entity_recognition',
            'semantic': 'text_extraction',
            'pattern': 'text_extraction',
            'visual': 'visual_analysis'
        }
        
        capability_type = method_to_capability.get(requirement.extraction_method)
        return self.models.get(capability_type)
    
    def _create_validation_stage(self) -> 'ValidationStage':
        """Create validation stage based on quality requirements"""
        return ValidationStage(self.schema.quality_spec)


class PipelineStage(ABC):
    """Abstract base class for pipeline stages"""
    
    def __init__(self, requirement: ExtractionRequirement, model: Optional[ModelCandidate]):
        self.requirement = requirement
        self.model = model
        self.model_instance = None
    
    @abstractmethod
    def process(self, document_data: Dict) -> Dict:
        """Process document data and extract required information"""
        pass
    
    def load_model(self):
        """Load the actual model"""
        if self.model and not self.model_instance:
            logger.info(f"Loading model {self.model.model_id}")
            # Load model based on type
            # This is simplified - real implementation would handle different model types
            self.model_instance = pipeline(
                self.model.task_type,
                model=self.model.model_id,
                device=0 if torch.cuda.is_available() else -1
            )


class DirectExtractionStage(PipelineStage):
    """Direct text extraction from document sections"""
    
    def process(self, document_data: Dict) -> Dict:
        source = self.requirement.source
        
        # Extract from specified source
        if source in document_data.get('sections', {}):
            text = document_data['sections'][source]['text']
            return {
                self.requirement.name: text,
                f"{self.requirement.name}_confidence": 0.95
            }
        
        return {
            self.requirement.name: None,
            f"{self.requirement.name}_confidence": 0.0
        }


class TableExtractionStage(PipelineStage):
    """Table extraction using specialized models"""
    
    def process(self, document_data: Dict) -> Dict:
        if not self.model_instance:
            self.load_model()
        
        tables = document_data.get('tables', [])
        source_table = None
        
        # Find the specified table
        for table in tables:
            if table.get('name') == self.requirement.source:
                source_table = table
                break
        
        if source_table:
            # Extract data based on requirement
            # This is simplified - real implementation would use the model
            return {
                self.requirement.name: source_table.get('data', []),
                f"{self.requirement.name}_confidence": 0.9
            }
        
        return {
            self.requirement.name: None,
            f"{self.requirement.name}_confidence": 0.0
        }


class NERExtractionStage(PipelineStage):
    """Named entity recognition extraction"""
    
    def process(self, document_data: Dict) -> Dict:
        if not self.model_instance:
            self.load_model()
        
        source_text = document_data.get('sections', {}).get(self.requirement.source, {}).get('text', '')
        
        if source_text and self.model_instance:
            # Run NER
            entities = self.model_instance(source_text)
            
            # Filter by entity type if specified
            filtered_entities = [
                ent for ent in entities 
                if self.requirement.field_type == 'any' or ent['entity_group'] == self.requirement.field_type
            ]
            
            return {
                self.requirement.name: filtered_entities,
                f"{self.requirement.name}_confidence": np.mean([ent['score'] for ent in filtered_entities]) if filtered_entities else 0.0
            }
        
        return {
            self.requirement.name: [],
            f"{self.requirement.name}_confidence": 0.0
        }


class SemanticExtractionStage(PipelineStage):
    """Semantic search and question answering extraction"""
    
    def process(self, document_data: Dict) -> Dict:
        if not self.model_instance:
            # Load QA model
            self.model_instance = pipeline(
                "question-answering",
                model="deepset/roberta-base-squad2"
            )
        
        source_text = document_data.get('sections', {}).get(self.requirement.source, {}).get('text', '')
        
        if source_text and 'query' in self.requirement.extraction_method:
            query = self.requirement.extraction_method['query']
            
            # Run QA
            result = self.model_instance(
                question=query,
                context=source_text
            )
            
            return {
                self.requirement.name: result['answer'],
                f"{self.requirement.name}_confidence": result['score']
            }
        
        return {
            self.requirement.name: None,
            f"{self.requirement.name}_confidence": 0.0
        }


class PatternExtractionStage(PipelineStage):
    """Pattern-based extraction using regex or NLP patterns"""
    
    def process(self, document_data: Dict) -> Dict:
        import re
        
        source_text = document_data.get('sections', {}).get(self.requirement.source, {}).get('text', '')
        patterns = self.requirement.extraction_method.get('patterns', [])
        
        for pattern in patterns:
            match = re.search(pattern, source_text)
            if match:
                return {
                    self.requirement.name: match.group(1) if match.groups() else match.group(0),
                    f"{self.requirement.name}_confidence": 0.9
                }
        
        return {
            self.requirement.name: None,
            f"{self.requirement.name}_confidence": 0.0
        }


class VisualExtractionStage(PipelineStage):
    """Visual element extraction from document images"""
    
    def process(self, document_data: Dict) -> Dict:
        # This would use visual models for extraction
        # Simplified for demonstration
        visual_elements = document_data.get('visual_elements', [])
        
        relevant_elements = [
            elem for elem in visual_elements
            if elem.get('type') == self.requirement.field_type
        ]
        
        if relevant_elements:
            return {
                self.requirement.name: relevant_elements,
                f"{self.requirement.name}_confidence": 0.85
            }
        
        return {
            self.requirement.name: [],
            f"{self.requirement.name}_confidence": 0.0
        }


class ValidationStage(PipelineStage):
    """Validates extraction results against quality requirements"""
    
    def __init__(self, quality_spec: Dict):
        self.quality_spec = quality_spec
        self.validation_rules = quality_spec.get('validation_rules', [])
        self.min_confidence = quality_spec.get('min_confidence', 0.85)
    
    def process(self, document_data: Dict) -> Dict:
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check confidence thresholds
        for key, value in document_data.items():
            if key.endswith('_confidence'):
                field_name = key.replace('_confidence', '')
                if value < self.min_confidence:
                    validation_results['warnings'].append(
                        f"Low confidence for {field_name}: {value:.2f}"
                    )
        
        # Apply validation rules
        for rule in self.validation_rules:
            rule_type = rule.get('type')
            
            if rule_type == 'completeness':
                # Check required fields
                threshold = rule.get('threshold', 0.9)
                filled_fields = sum(1 for k, v in document_data.items() 
                                  if not k.endswith('_confidence') and v is not None)
                total_fields = len([k for k in document_data.keys() 
                                  if not k.endswith('_confidence')])
                
                if filled_fields / total_fields < threshold:
                    validation_results['errors'].append(
                        f"Completeness below threshold: {filled_fields}/{total_fields}"
                    )
                    validation_results['valid'] = False
            
            elif rule_type == 'consistency':
                # Check field consistency
                fields_to_check = rule.get('cross_check_fields', [])
                # Implement consistency checks
                pass
        
        document_data['validation'] = validation_results
        return document_data


class DocumentPipeline:
    """Executes the processing pipeline"""
    
    def __init__(self, stages: List[PipelineStage]):
        self.stages = stages
    
    def process(self, document_path: str) -> Dict:
        """Process a document through all pipeline stages"""
        logger.info(f"Processing document: {document_path}")
        
        # Initial document analysis (simplified)
        document_data = self._analyze_document(document_path)
        
        # Run through each stage
        for stage in self.stages:
            logger.info(f"Running stage: {stage.__class__.__name__}")
            document_data = stage.process(document_data)
        
        return document_data
    
    def _analyze_document(self, document_path: str) -> Dict:
        """Initial document analysis to extract basic structure"""
        # This would use document analysis models
        # Simplified for demonstration
        return {
            'path': document_path,
            'sections': {
                'header': {'text': 'Document Header Content'},
                'body': {'text': 'Main document content...'},
                'footer': {'text': 'Footer content'}
            },
            'tables': [
                {
                    'name': 'financial_data',
                    'data': [['Q1', '100'], ['Q2', '150']]
                }
            ],
            'visual_elements': [
                {
                    'type': 'chart',
                    'description': 'Quarterly revenue chart'
                }
            ]
        }


class AutonomousDocumentAgent:
    """Main autonomous agent that orchestrates the entire process"""
    
    def __init__(self):
        self.discovery_agent = ModelDiscoveryAgent()
        self.schema_cache = {}
        self.pipeline_cache = {}
    
    def process_document(self, document_path: str, schema_path: str) -> Dict:
        """Process a document using the specified schema"""
        
        # Load schema
        schema = DocumentSchema(schema_path)
        
        # Determine required capabilities
        capabilities = schema.get_required_capabilities()
        logger.info(f"Required capabilities: {capabilities}")
        
        # Discover appropriate models
        models = self.discovery_agent.discover_models(capabilities)
        logger.info(f"Discovered {len(models)} models")
        
        # Build processing pipeline
        builder = PipelineBuilder(schema, models)
        pipeline = builder.build_pipeline()
        
        # Process document
        results = pipeline.process(document_path)
        
        # Format output according to schema
        formatted_output = self._format_output(results, schema)
        
        return formatted_output
    
    def _format_output(self, results: Dict, schema: DocumentSchema) -> Dict:
        """Format results according to schema output specification"""
        output_format = schema.output_spec.get('format', 'json')
        
        # Extract only requested fields
        output = {}
        for field in schema.output_spec['fields']:
            field_name = field['name']
            if field_name in results:
                output[field_name] = results[field_name]
        
        # Add metadata
        output['_metadata'] = {
            'document': results.get('path'),
            'schema': schema.document_profile['name'],
            'validation': results.get('validation', {}),
            'extraction_confidence': {
                k.replace('_confidence', ''): v 
                for k, v in results.items() 
                if k.endswith('_confidence')
            }
        }
        
        return output


# Example usage
if __name__ == "__main__":
    # Create autonomous agent
    agent = AutonomousDocumentAgent()
    
    # Process different document types with appropriate schemas
    
    # Example 1: SEC 10-K
    results = agent.process_document(
        document_path="sample_10k.pdf",
        schema_path="schemas/sec_10k_schema.yaml"
    )
    print("10-K Results:", json.dumps(results, indent=2))
    
    # Example 2: Legal Contract
    results = agent.process_document(
        document_path="sample_contract.pdf",
        schema_path="schemas/legal_contract_schema.yaml"
    )
    print("Contract Results:", json.dumps(results, indent=2))
    
    # Example 3: Medical Record
    results = agent.process_document(
        document_path="sample_medical.pdf",
        schema_path="schemas/medical_record_schema.yaml"
    )
    print("Medical Results:", json.dumps(results, indent=2))