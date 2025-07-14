"""
Neural Document Flow - Python Bindings

A powerful document processing library with AI-powered extraction, 
security analysis, and plugin support.

Example usage:
    import neuraldocflow

    # Create processor with high security
    processor = neuraldocflow.Processor(
        security_level="high",
        enable_neural=True,
        plugins=["docx", "tables", "images"]
    )

    # Process a document
    result = processor.process_document("document.pdf", output_format="json")

    # Access results
    print(result.text)
    print(result.tables)
    print(result.security_analysis)
"""

# Import the compiled Rust module
from ._neuraldocflow import *

# Re-export key classes and functions for convenience
__all__ = [
    # Main processor
    "Processor",
    "create_processor",
    
    # Results and types
    "ProcessingResult",
    "SecurityAnalysis",
    "ThreatLevel",
    "SecurityAction",
    "BehavioralRisk",
    
    # Plugin management
    "PluginManager",
    
    # Errors
    "NeuralDocFlowError",
    
    # Utility functions
    "get_supported_formats",
    "get_version",
    
    # Constants
    "__version__",
    "API_VERSION",
]

# Version information
__version__ = "0.1.0"
__author__ = "DAA Neural Coordinator"
__email__ = "daa@neural.ai"
__license__ = "MIT"

# Module docstring
__doc__ = """
Neural Document Flow Python Bindings

This module provides Python bindings for the neural document processing framework.
It offers high-performance document processing with AI-powered features, security
analysis, and extensible plugin architecture.

Key Features:
- Multi-format document processing (PDF, DOCX, images, etc.)
- AI-powered text extraction and enhancement
- Real-time security analysis and threat detection
- Plugin system for custom processors
- High-performance Rust core with Python convenience

Quick Start:
    >>> import neuraldocflow
    >>> processor = neuraldocflow.create_processor()
    >>> result = processor.process_document("document.pdf")
    >>> print(result.text)
"""