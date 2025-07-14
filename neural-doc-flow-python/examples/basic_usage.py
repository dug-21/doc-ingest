#!/usr/bin/env python3
"""
Basic usage example for Neural Document Flow Python bindings.

This example demonstrates the core functionality of the library
including document processing, security analysis, and plugin usage.
"""

import neuraldocflow
import json
import sys
from pathlib import Path


def main():
    print("Neural Document Flow - Basic Usage Example")
    print("=" * 50)
    
    # Get version information
    version_info = neuraldocflow.get_version()
    print(f"Library version: {version_info['neuraldocflow']}")
    print(f"Python bindings: {version_info['python_bindings']}")
    print(f"API version: {version_info['api_version']}")
    print()
    
    # Get supported formats
    formats = neuraldocflow.get_supported_formats()
    print("Supported formats:")
    for category, mime_types in formats.items():
        print(f"  {category}: {', '.join(mime_types)}")
    print()
    
    # Create processor with different security levels
    processors = {
        "basic": neuraldocflow.create_processor(
            security_level="basic",
            enable_neural=False,
            plugins=["text"]
        ),
        "standard": neuraldocflow.create_processor(
            security_level="standard",
            enable_neural=True,
            plugins=["docx", "tables"]
        ),
        "high": neuraldocflow.create_processor(
            security_level="high",
            enable_neural=True,
            plugins=["docx", "tables", "images"]
        )
    }
    
    # Display processor configurations
    print("Processor configurations:")
    for name, processor in processors.items():
        config = processor.config
        print(f"  {name.capitalize()}:")
        print(f"    Security: {config.get('security_enabled', False)}")
        print(f"    Neural: {config.get('neural_enabled', False)}")
        print(f"    Plugins: {config.get('plugins_enabled', False)}")
    print()
    
    # Example: Process a text document
    sample_text = """
    This is a sample document for testing the Neural Document Flow library.
    
    The document contains:
    - Multiple paragraphs
    - Some potentially suspicious content: <script>alert('test')</script>
    - Regular text content
    
    This library provides comprehensive document processing with security analysis.
    """
    
    print("Processing sample text document...")
    
    # Use the high-security processor
    processor = processors["high"]
    
    try:
        # Process the text as bytes
        text_bytes = sample_text.encode('utf-8')
        result = processor.process_bytes(text_bytes, "text/plain", "json")
        
        print(f"Processing successful: {result.success}")
        print(f"Source: {result.source}")
        print(f"Format: {result.format}")
        print()
        
        # Display extracted text
        if result.text:
            print("Extracted text:")
            print(f"  Length: {len(result.text)} characters")
            print(f"  Preview: {result.text[:100]}...")
        print()
        
        # Display security analysis
        if result.security_analysis:
            analysis = result.security_analysis
            print("Security Analysis:")
            print(f"  Threat level: {analysis.threat_level}")
            print(f"  Malware probability: {analysis.malware_probability:.2%}")
            print(f"  Anomaly score: {analysis.anomaly_score:.2f}")
            print(f"  Recommended action: {analysis.recommended_action}")
            print(f"  Is safe: {analysis.is_safe}")
            print(f"  Requires attention: {analysis.requires_attention}")
            
            if analysis.threat_categories:
                print(f"  Threat categories: {', '.join(analysis.threat_categories)}")
            
            if analysis.behavioral_risks:
                print("  Behavioral risks:")
                for risk in analysis.behavioral_risks:
                    print(f"    - {risk.risk_type}: {risk.description} (severity: {risk.severity:.2f})")
        print()
        
        # Display metadata
        print("Metadata:")
        for key, value in result.metadata.items():
            print(f"  {key}: {value}")
        print()
        
        # Display processing statistics
        print("Processing Statistics:")
        stats = result.stats
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print()
        
        # Convert to different formats
        print("Output formats:")
        
        # JSON format
        json_output = result.to_string("json")
        print(f"JSON length: {len(json_output)} characters")
        
        # Markdown format
        markdown_output = result.to_string("markdown")
        print(f"Markdown length: {len(markdown_output)} characters")
        
        # HTML format
        html_output = result.to_string("html")
        print(f"HTML length: {len(html_output)} characters")
        
        # XML format
        xml_output = result.to_string("xml")
        print(f"XML length: {len(xml_output)} characters")
        print()
        
        # Display summary
        summary = result.summary
        print("Result Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
    except neuraldocflow.NeuralDocFlowError as e:
        print(f"Processing failed: {e}")
        print(f"Error type: {e.error_type}")
        if e.error_code:
            print(f"Error code: {e.error_code}")
    
    print("\n" + "=" * 50)
    print("Example completed successfully!")


def demonstrate_plugin_manager():
    """Demonstrate plugin manager functionality."""
    print("\nPlugin Manager Example")
    print("-" * 30)
    
    try:
        # Create plugin manager
        plugin_manager = neuraldocflow.PluginManager(
            enable_hot_reload=True,
            enable_sandboxing=True,
            max_plugins=10
        )
        
        # Load built-in plugins
        loaded_count = plugin_manager.load_builtin_plugins()
        print(f"Loaded {loaded_count} built-in plugins")
        
        # Get available plugins
        plugins = plugin_manager.get_available_plugins()
        print("\nAvailable plugins:")
        for name, info in plugins.items():
            print(f"  {name}:")
            print(f"    Description: {info.get('description', 'N/A')}")
            print(f"    Version: {info.get('version', 'N/A')}")
            print(f"    Type: {info.get('type', 'N/A')}")
            print(f"    Enabled: {info.get('enabled', False)}")
            if 'supported_formats' in info:
                formats = info['supported_formats']
                if isinstance(formats, list):
                    print(f"    Formats: {', '.join(formats[:3])}{'...' if len(formats) > 3 else ''}")
        
        # Test format-specific plugin lookup
        print("\nPlugins for specific formats:")
        test_formats = [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
            "image/jpeg"
        ]
        
        for mime_type in test_formats:
            supporting_plugins = plugin_manager.get_plugins_for_format(mime_type)
            print(f"  {mime_type}: {', '.join(supporting_plugins) if supporting_plugins else 'None'}")
        
        # Display plugin manager configuration
        print("\nPlugin Manager Configuration:")
        config = plugin_manager.config
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        # Get usage statistics
        print("\nPlugin Usage Statistics:")
        stats = plugin_manager.get_usage_stats()
        for plugin_name, plugin_stats in stats.items():
            print(f"  {plugin_name}:")
            for stat_name, stat_value in plugin_stats.items():
                print(f"    {stat_name}: {stat_value}")
    
    except neuraldocflow.NeuralDocFlowError as e:
        print(f"Plugin manager error: {e}")


def demonstrate_batch_processing():
    """Demonstrate batch processing capabilities."""
    print("\nBatch Processing Example")
    print("-" * 30)
    
    # Create sample documents in memory
    sample_docs = [
        ("doc1.txt", "This is the first sample document.", "text/plain"),
        ("doc2.txt", "This is the second sample document with more content.", "text/plain"),
        ("doc3.txt", "This is the third document. It contains suspicious content: <script>alert('xss')</script>", "text/plain"),
    ]
    
    try:
        processor = neuraldocflow.create_processor(
            security_level="high",
            enable_neural=True,
            plugins=["text"]
        )
        
        print(f"Processing {len(sample_docs)} documents in batch...")
        
        # Note: For real batch processing, you would use file paths
        # Here we'll process each document individually for demonstration
        results = []
        for filename, content, mime_type in sample_docs:
            try:
                content_bytes = content.encode('utf-8')
                result = processor.process_bytes(content_bytes, mime_type, "json")
                results.append(result)
                
                print(f"\n{filename}:")
                print(f"  Success: {result.success}")
                print(f"  Text length: {len(result.text) if result.text else 0}")
                
                if result.security_analysis:
                    analysis = result.security_analysis
                    print(f"  Threat level: {analysis.threat_level}")
                    print(f"  Safe: {analysis.is_safe}")
                    if analysis.threat_categories:
                        print(f"  Threats: {', '.join(analysis.threat_categories)}")
                
            except neuraldocflow.NeuralDocFlowError as e:
                print(f"  Error processing {filename}: {e}")
                # Create error result
                error_result = neuraldocflow.ProcessingResult()
                error_result.source = filename
                error_result.error = str(e)
                results.append(error_result)
        
        # Summary
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        print(f"\nBatch processing summary:")
        print(f"  Total documents: {len(results)}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Success rate: {successful/len(results)*100:.1f}%")
    
    except neuraldocflow.NeuralDocFlowError as e:
        print(f"Batch processing error: {e}")


if __name__ == "__main__":
    try:
        main()
        demonstrate_plugin_manager()
        demonstrate_batch_processing()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)