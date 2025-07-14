"""
Basic tests for Neural Document Flow Python bindings.
"""

import pytest
import tempfile
import os
from pathlib import Path

# Import the module - this will test if the bindings load correctly
try:
    import neuraldocflow
    BINDINGS_AVAILABLE = True
except ImportError as e:
    BINDINGS_AVAILABLE = False
    IMPORT_ERROR = str(e)


@pytest.mark.skipif(not BINDINGS_AVAILABLE, reason=f"Bindings not available: {IMPORT_ERROR if not BINDINGS_AVAILABLE else ''}")
class TestBasicFunctionality:
    """Test basic functionality of the Python bindings."""
    
    def test_version_info(self):
        """Test version information retrieval."""
        version_info = neuraldocflow.get_version()
        
        assert isinstance(version_info, dict)
        assert "neuraldocflow" in version_info
        assert "python_bindings" in version_info
        assert "api_version" in version_info
        assert "rust_core" in version_info
    
    def test_supported_formats(self):
        """Test supported formats retrieval."""
        formats = neuraldocflow.get_supported_formats()
        
        assert isinstance(formats, dict)
        assert "text" in formats
        assert "pdf" in formats
        assert "office" in formats
        assert "images" in formats
        
        # Check format content
        assert "text/plain" in formats["text"]
        assert "application/pdf" in formats["pdf"]
    
    def test_processor_creation(self):
        """Test processor creation with different configurations."""
        # Basic processor
        processor = neuraldocflow.create_processor()
        assert processor is not None
        
        # Custom configuration
        processor = neuraldocflow.create_processor(
            security_level="high",
            enable_neural=True,
            plugins=["text"]
        )
        assert processor is not None
        
        # Check configuration
        config = processor.config
        assert isinstance(config, dict)
        assert "security_enabled" in config
        assert "neural_enabled" in config
    
    def test_processor_invalid_security_level(self):
        """Test processor creation with invalid security level."""
        with pytest.raises(neuraldocflow.NeuralDocFlowError):
            neuraldocflow.create_processor(security_level="invalid")
    
    def test_text_processing(self):
        """Test basic text processing."""
        processor = neuraldocflow.create_processor(
            security_level="basic",
            enable_neural=False,
            plugins=["text"]
        )
        
        # Sample text
        sample_text = "This is a test document for the neural document flow library."
        text_bytes = sample_text.encode('utf-8')
        
        # Process the text
        result = processor.process_bytes(text_bytes, "text/plain", "json")
        
        assert result is not None
        assert result.success
        assert result.text is not None
        assert sample_text in result.text
        assert result.source == "unknown"
        assert result.format == "json"
    
    def test_security_analysis(self):
        """Test security analysis functionality."""
        processor = neuraldocflow.create_processor(
            security_level="high",
            enable_neural=False,
            plugins=["text"]
        )
        
        # Safe content
        safe_text = "This is a safe document with normal content."
        safe_bytes = safe_text.encode('utf-8')
        
        result = processor.process_bytes(safe_bytes, "text/plain", "json")
        assert result.success
        
        if result.security_analysis:
            analysis = result.security_analysis
            assert hasattr(analysis, 'threat_level')
            assert hasattr(analysis, 'malware_probability')
            assert hasattr(analysis, 'is_safe')
            assert hasattr(analysis, 'recommended_action')
        
        # Suspicious content
        suspicious_text = "This document contains <script>alert('xss')</script> content."
        suspicious_bytes = suspicious_text.encode('utf-8')
        
        result = processor.process_bytes(suspicious_bytes, "text/plain", "json")
        assert result.success
        
        if result.security_analysis:
            analysis = result.security_analysis
            # Should detect the script content
            assert analysis.malware_probability > 0.0 or len(analysis.threat_categories) > 0
    
    def test_result_formats(self):
        """Test different output formats."""
        processor = neuraldocflow.create_processor(
            security_level="basic",
            enable_neural=False,
            plugins=["text"]
        )
        
        sample_text = "Sample document content for format testing."
        text_bytes = sample_text.encode('utf-8')
        
        # Test different formats
        formats = ["json", "markdown", "html", "xml"]
        
        for fmt in formats:
            result = processor.process_bytes(text_bytes, "text/plain", fmt)
            assert result.success
            assert result.format == fmt
            
            # Test format conversion
            formatted_output = result.to_string(fmt)
            assert isinstance(formatted_output, str)
            assert len(formatted_output) > 0
    
    def test_processor_properties(self):
        """Test processor properties and methods."""
        processor = neuraldocflow.create_processor(
            security_level="standard",
            enable_neural=True,
            plugins=["text"]
        )
        
        # Test properties
        assert isinstance(processor.neural_enabled, bool)
        assert processor.neural_enabled == True
        
        security_status = processor.security_status
        assert isinstance(security_status, dict)
        assert "enabled" in security_status
        
        # Test plugin information
        plugins = processor.get_available_plugins()
        assert isinstance(plugins, dict)
        assert len(plugins) > 0
        
        # Test neural enable/disable
        processor.set_neural_enabled(False)
        assert processor.neural_enabled == False
        
        processor.set_neural_enabled(True)
        assert processor.neural_enabled == True


@pytest.mark.skipif(not BINDINGS_AVAILABLE, reason=f"Bindings not available: {IMPORT_ERROR if not BINDINGS_AVAILABLE else ''}")
class TestPluginManager:
    """Test plugin manager functionality."""
    
    def test_plugin_manager_creation(self):
        """Test plugin manager creation."""
        plugin_manager = neuraldocflow.PluginManager()
        assert plugin_manager is not None
        
        # Test with custom configuration
        plugin_manager = neuraldocflow.PluginManager(
            enable_hot_reload=True,
            enable_sandboxing=True,
            max_plugins=10
        )
        assert plugin_manager is not None
    
    def test_builtin_plugins(self):
        """Test built-in plugin loading."""
        plugin_manager = neuraldocflow.PluginManager()
        
        # Load built-in plugins
        loaded_count = plugin_manager.load_builtin_plugins()
        assert loaded_count > 0
        
        # Get available plugins
        plugins = plugin_manager.get_available_plugins()
        assert isinstance(plugins, dict)
        assert len(plugins) > 0
        
        # Check for expected built-in plugins
        expected_plugins = ["text", "docx", "tables", "images", "pdf"]
        for plugin_name in expected_plugins:
            assert plugin_name in plugins
            
            plugin_info = plugins[plugin_name]
            assert "description" in plugin_info
            assert "type" in plugin_info
            assert plugin_info["type"] == "builtin"
    
    def test_plugin_format_support(self):
        """Test plugin format support queries."""
        plugin_manager = neuraldocflow.PluginManager()
        plugin_manager.load_builtin_plugins()
        
        # Test format-specific plugin lookup
        text_plugins = plugin_manager.get_plugins_for_format("text/plain")
        assert isinstance(text_plugins, list)
        assert "text" in text_plugins
        
        pdf_plugins = plugin_manager.get_plugins_for_format("application/pdf")
        assert isinstance(pdf_plugins, list)
        assert "pdf" in pdf_plugins
    
    def test_plugin_enable_disable(self):
        """Test plugin enable/disable functionality."""
        plugin_manager = neuraldocflow.PluginManager()
        plugin_manager.load_builtin_plugins()
        
        # Test enabling existing plugin
        assert plugin_manager.enable_plugin("text") == True
        assert plugin_manager.is_plugin_enabled("text") == True
        
        # Test disabling plugin
        assert plugin_manager.disable_plugin("text") == True
        
        # Test non-existent plugin
        with pytest.raises(neuraldocflow.NeuralDocFlowError):
            plugin_manager.enable_plugin("nonexistent")
    
    def test_plugin_info(self):
        """Test plugin information retrieval."""
        plugin_manager = neuraldocflow.PluginManager()
        plugin_manager.load_builtin_plugins()
        
        # Get info for existing plugin
        info = plugin_manager.get_plugin_info("text")
        assert isinstance(info, dict)
        assert "name" in info
        assert "description" in info
        assert "type" in info
        
        # Test non-existent plugin
        with pytest.raises(neuraldocflow.NeuralDocFlowError):
            plugin_manager.get_plugin_info("nonexistent")


@pytest.mark.skipif(not BINDINGS_AVAILABLE, reason=f"Bindings not available: {IMPORT_ERROR if not BINDINGS_AVAILABLE else ''}")
class TestErrorHandling:
    """Test error handling and exception types."""
    
    def test_error_types(self):
        """Test different error types."""
        # Test that error types exist
        assert hasattr(neuraldocflow, 'NeuralDocFlowError')
        
        # Test creating errors
        error = neuraldocflow.NeuralDocFlowError(
            "Test error message",
            "TestError",
            1234
        )
        assert error.message == "Test error message"
        assert error.error_type == "TestError"
        assert error.error_code == 1234
    
    def test_processing_errors(self):
        """Test processing error scenarios."""
        processor = neuraldocflow.create_processor()
        
        # Test invalid MIME type (should not necessarily fail, but test error handling)
        try:
            result = processor.process_bytes(b"test", "invalid/mime-type", "json")
            # Processing might succeed even with unknown MIME type
            assert result is not None
        except neuraldocflow.NeuralDocFlowError:
            # Or it might fail, which is also acceptable
            pass
    
    def test_file_not_found(self):
        """Test file not found error."""
        processor = neuraldocflow.create_processor()
        
        with pytest.raises(neuraldocflow.NeuralDocFlowError):
            processor.process_document("/nonexistent/file.txt")


@pytest.mark.skipif(not BINDINGS_AVAILABLE, reason=f"Bindings not available: {IMPORT_ERROR if not BINDINGS_AVAILABLE else ''}")
class TestFileProcessing:
    """Test file-based processing."""
    
    def test_temporary_file_processing(self):
        """Test processing with temporary files."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            test_content = "This is test content for file processing."
            f.write(test_content)
            temp_path = f.name
        
        try:
            processor = neuraldocflow.create_processor(
                security_level="basic",
                enable_neural=False,
                plugins=["text"]
            )
            
            # Process the file
            result = processor.process_document(temp_path, "json")
            
            assert result.success
            assert result.text is not None
            assert test_content in result.text
            assert result.source != "unknown"  # Should use file name
            
        finally:
            # Clean up
            os.unlink(temp_path)


def test_import_structure():
    """Test that the module structure is correct."""
    if not BINDINGS_AVAILABLE:
        pytest.skip(f"Bindings not available: {IMPORT_ERROR}")
    
    # Test main module imports
    assert hasattr(neuraldocflow, 'Processor')
    assert hasattr(neuraldocflow, 'create_processor')
    assert hasattr(neuraldocflow, 'ProcessingResult')
    assert hasattr(neuraldocflow, 'SecurityAnalysis')
    assert hasattr(neuraldocflow, 'PluginManager')
    assert hasattr(neuraldocflow, 'NeuralDocFlowError')
    assert hasattr(neuraldocflow, 'get_supported_formats')
    assert hasattr(neuraldocflow, 'get_version')
    
    # Test version constants
    assert hasattr(neuraldocflow, '__version__')
    assert hasattr(neuraldocflow, 'API_VERSION')


def test_async_import():
    """Test async module import."""
    if not BINDINGS_AVAILABLE:
        pytest.skip(f"Bindings not available: {IMPORT_ERROR}")
    
    try:
        from neuraldocflow.async_processor import AsyncProcessor
        assert AsyncProcessor is not None
    except ImportError as e:
        pytest.fail(f"Failed to import async processor: {e}")


if __name__ == "__main__":
    # Run basic smoke test
    if BINDINGS_AVAILABLE:
        print("✓ Neural Document Flow Python bindings loaded successfully")
        
        version = neuraldocflow.get_version()
        print(f"✓ Version: {version}")
        
        formats = neuraldocflow.get_supported_formats()
        print(f"✓ Supported formats: {list(formats.keys())}")
        
        processor = neuraldocflow.create_processor()
        print(f"✓ Processor created: {processor}")
        
        print("✓ Basic smoke test passed")
    else:
        print(f"✗ Failed to import bindings: {IMPORT_ERROR}")
        exit(1)