"""
Async wrapper for the neural document flow processor.

Provides async/await support for document processing operations.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Any
from pathlib import Path

from . import Processor, ProcessingResult


class AsyncProcessor:
    """
    Async wrapper for the Processor class.
    
    Provides async/await interface for document processing while
    maintaining compatibility with the synchronous Rust implementation.
    
    Example:
        async def process_documents():
            processor = AsyncProcessor(
                security_level="high",
                enable_neural=True,
                plugins=["docx", "tables"]
            )
            
            result = await processor.process_document("document.pdf")
            print(result.text)
            
            # Process multiple documents concurrently
            results = await processor.process_batch([
                "doc1.pdf", "doc2.docx", "doc3.txt"
            ])
    """
    
    def __init__(
        self,
        security_level: str = "standard",
        enable_neural: bool = True,
        plugins: Optional[List[str]] = None,
        max_workers: int = 4
    ):
        """
        Initialize the async processor.
        
        Args:
            security_level: Security level ("disabled", "basic", "standard", "high")
            enable_neural: Enable neural processing features
            plugins: List of plugin names to enable
            max_workers: Maximum number of worker threads for concurrent processing
        """
        self._processor = Processor(security_level, enable_neural, plugins)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def process_document(
        self,
        file_path: str,
        output_format: str = "json"
    ) -> ProcessingResult:
        """
        Asynchronously process a document from file path.
        
        Args:
            file_path: Path to the document file
            output_format: Output format ("json", "markdown", "html", "xml")
            
        Returns:
            ProcessingResult containing extracted content and metadata
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._processor.process_document,
            file_path,
            output_format
        )
    
    async def process_bytes(
        self,
        data: bytes,
        mime_type: str,
        output_format: str = "json"
    ) -> ProcessingResult:
        """
        Asynchronously process a document from byte data.
        
        Args:
            data: Document data as bytes
            mime_type: MIME type of the document
            output_format: Output format ("json", "markdown", "html", "xml")
            
        Returns:
            ProcessingResult containing extracted content and metadata
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._processor.process_bytes,
            data,
            mime_type,
            output_format
        )
    
    async def process_batch(
        self,
        file_paths: List[str],
        output_format: str = "json",
        max_concurrent: int = 4
    ) -> List[ProcessingResult]:
        """
        Process multiple documents concurrently.
        
        Args:
            file_paths: List of file paths to process
            output_format: Output format for all documents
            max_concurrent: Maximum number of concurrent operations
            
        Returns:
            List of ProcessingResult objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(file_path: str) -> ProcessingResult:
            async with semaphore:
                return await self.process_document(file_path, output_format)
        
        tasks = [process_with_semaphore(path) for path in file_paths]
        return await asyncio.gather(*tasks, return_exceptions=False)
    
    async def process_directory(
        self,
        directory: str,
        pattern: str = "*",
        recursive: bool = True,
        output_format: str = "json",
        max_concurrent: int = 4
    ) -> Dict[str, ProcessingResult]:
        """
        Process all files in a directory matching a pattern.
        
        Args:
            directory: Directory path to process
            pattern: File pattern to match (e.g., "*.pdf", "*.docx")
            recursive: Whether to search subdirectories
            output_format: Output format for all documents
            max_concurrent: Maximum number of concurrent operations
            
        Returns:
            Dictionary mapping file paths to ProcessingResult objects
        """
        directory_path = Path(directory)
        
        if recursive:
            files = list(directory_path.rglob(pattern))
        else:
            files = list(directory_path.glob(pattern))
        
        file_paths = [str(f) for f in files if f.is_file()]
        results = await self.process_batch(file_paths, output_format, max_concurrent)
        
        return dict(zip(file_paths, results))
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get processor configuration."""
        return self._processor.config
    
    @property
    def neural_enabled(self) -> bool:
        """Check if neural processing is enabled."""
        return self._processor.neural_enabled
    
    @property
    def security_status(self) -> Dict[str, Any]:
        """Get security configuration."""
        return self._processor.security_status
    
    def get_available_plugins(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available plugins."""
        return self._processor.get_available_plugins()
    
    def set_neural_enabled(self, enabled: bool):
        """Enable or disable neural processing."""
        self._processor.set_neural_enabled(enabled)
    
    async def close(self):
        """Close the async processor and cleanup resources."""
        self._executor.shutdown(wait=True)
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Convenience function for quick async processing
async def process_document_async(
    file_path: str,
    security_level: str = "standard",
    enable_neural: bool = True,
    plugins: Optional[List[str]] = None,
    output_format: str = "json"
) -> ProcessingResult:
    """
    Convenience function for processing a single document asynchronously.
    
    Args:
        file_path: Path to the document file
        security_level: Security level to use
        enable_neural: Enable neural processing
        plugins: List of plugins to enable
        output_format: Output format
        
    Returns:
        ProcessingResult containing extracted content and metadata
        
    Example:
        result = await process_document_async("document.pdf", security_level="high")
        print(result.text)
    """
    async with AsyncProcessor(security_level, enable_neural, plugins) as processor:
        return await processor.process_document(file_path, output_format)


# Convenience function for batch processing
async def process_batch_async(
    file_paths: List[str],
    security_level: str = "standard",
    enable_neural: bool = True,
    plugins: Optional[List[str]] = None,
    output_format: str = "json",
    max_concurrent: int = 4
) -> List[ProcessingResult]:
    """
    Convenience function for processing multiple documents asynchronously.
    
    Args:
        file_paths: List of file paths to process
        security_level: Security level to use
        enable_neural: Enable neural processing
        plugins: List of plugins to enable
        output_format: Output format
        max_concurrent: Maximum concurrent operations
        
    Returns:
        List of ProcessingResult objects
        
    Example:
        results = await process_batch_async([
            "doc1.pdf", "doc2.docx", "doc3.txt"
        ], security_level="high")
    """
    async with AsyncProcessor(security_level, enable_neural, plugins) as processor:
        return await processor.process_batch(file_paths, output_format, max_concurrent)