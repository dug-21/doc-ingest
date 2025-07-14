#!/usr/bin/env python3
"""
Async processing example for Neural Document Flow Python bindings.

This example demonstrates asynchronous document processing capabilities
including concurrent batch processing and directory scanning.
"""

import asyncio
import time
import tempfile
import os
from pathlib import Path
import neuraldocflow
from neuraldocflow.async_processor import AsyncProcessor, process_document_async, process_batch_async


async def main():
    print("Neural Document Flow - Async Processing Example")
    print("=" * 55)
    
    # Create sample documents for testing
    await create_sample_documents()
    
    # Demonstrate async single document processing
    await demo_async_single_processing()
    
    # Demonstrate async batch processing
    await demo_async_batch_processing()
    
    # Demonstrate async directory processing
    await demo_async_directory_processing()
    
    # Demonstrate concurrent processing performance
    await demo_performance_comparison()
    
    print("\n" + "=" * 55)
    print("Async processing example completed!")


async def create_sample_documents():
    """Create sample documents for testing."""
    print("Creating sample documents...")
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp(prefix="neuraldocflow_"))
    
    sample_docs = {
        "safe_document.txt": """
        This is a safe document with normal content.
        
        It contains:
        - Regular paragraphs
        - Normal text
        - No suspicious content
        
        This document should be processed without any security warnings.
        """,
        
        "suspicious_document.txt": """
        This document contains some suspicious content.
        
        Suspicious elements:
        - <script>alert('XSS attempt')</script>
        - eval("malicious code")
        - javascript:void(0)
        
        The security scanner should detect these patterns.
        """,
        
        "large_document.txt": """
        This is a larger document for performance testing.
        
        """ + "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 100 + """
        
        This document is intentionally large to test processing performance
        and memory usage during batch operations.
        """,
        
        "structured_document.txt": """
        Document Title: Sample Report
        
        Section 1: Introduction
        This section introduces the document.
        
        Section 2: Data Analysis
        Here we present some data:
        
        Table: Sample Data
        Name    | Value  | Status
        Item A  | 100    | Active
        Item B  | 200    | Inactive
        Item C  | 150    | Active
        
        Section 3: Conclusion
        This document demonstrates structured content extraction.
        """,
        
        "multilingual_document.txt": """
        English: This document contains multiple languages.
        Español: Este documento contiene múltiples idiomas.
        Français: Ce document contient plusieurs langues.
        Deutsch: Dieses Dokument enthält mehrere Sprachen.
        
        The neural processor should handle multilingual content appropriately.
        """
    }
    
    # Write sample documents
    global SAMPLE_DIR
    SAMPLE_DIR = temp_dir
    
    for filename, content in sample_docs.items():
        file_path = temp_dir / filename
        file_path.write_text(content.strip(), encoding='utf-8')
    
    print(f"Created {len(sample_docs)} sample documents in {temp_dir}")
    print()


async def demo_async_single_processing():
    """Demonstrate async single document processing."""
    print("Async Single Document Processing")
    print("-" * 40)
    
    # Create async processor
    async with AsyncProcessor(
        security_level="high",
        enable_neural=True,
        plugins=["text"]
    ) as processor:
        
        # Process a safe document
        safe_doc = SAMPLE_DIR / "safe_document.txt"
        print(f"Processing {safe_doc.name}...")
        
        start_time = time.time()
        result = await processor.process_document(str(safe_doc), "json")
        processing_time = time.time() - start_time
        
        print(f"  Processing time: {processing_time:.3f} seconds")
        print(f"  Success: {result.success}")
        print(f"  Text length: {len(result.text) if result.text else 0} characters")
        
        if result.security_analysis:
            analysis = result.security_analysis
            print(f"  Threat level: {analysis.threat_level}")
            print(f"  Is safe: {analysis.is_safe}")
        
        # Process a suspicious document
        print()
        suspicious_doc = SAMPLE_DIR / "suspicious_document.txt"
        print(f"Processing {suspicious_doc.name}...")
        
        start_time = time.time()
        result = await processor.process_document(str(suspicious_doc), "json")
        processing_time = time.time() - start_time
        
        print(f"  Processing time: {processing_time:.3f} seconds")
        print(f"  Success: {result.success}")
        print(f"  Text length: {len(result.text) if result.text else 0} characters")
        
        if result.security_analysis:
            analysis = result.security_analysis
            print(f"  Threat level: {analysis.threat_level}")
            print(f"  Is safe: {analysis.is_safe}")
            print(f"  Requires attention: {analysis.requires_attention}")
            
            if analysis.threat_categories:
                print(f"  Threat categories: {', '.join(analysis.threat_categories)}")
            
            if analysis.behavioral_risks:
                print(f"  Behavioral risks: {len(analysis.behavioral_risks)}")
                for risk in analysis.behavioral_risks[:3]:  # Show first 3
                    print(f"    - {risk.risk_type}: {risk.severity:.2f}")
    
    print()


async def demo_async_batch_processing():
    """Demonstrate async batch processing."""
    print("Async Batch Processing")
    print("-" * 30)
    
    # Get all sample documents
    sample_files = list(SAMPLE_DIR.glob("*.txt"))
    file_paths = [str(f) for f in sample_files]
    
    print(f"Processing {len(file_paths)} documents in batch...")
    
    # Create async processor
    async with AsyncProcessor(
        security_level="standard",
        enable_neural=True,
        plugins=["text"],
        max_workers=4
    ) as processor:
        
        start_time = time.time()
        results = await processor.process_batch(
            file_paths,
            output_format="json",
            max_concurrent=3
        )
        total_time = time.time() - start_time
        
        print(f"Batch processing completed in {total_time:.3f} seconds")
        print(f"Average time per document: {total_time/len(results):.3f} seconds")
        print()
        
        # Analyze results
        successful = 0
        total_text_length = 0
        security_issues = 0
        
        for i, result in enumerate(results):
            filename = Path(file_paths[i]).name
            print(f"  {filename}:")
            print(f"    Success: {result.success}")
            
            if result.success:
                successful += 1
                text_length = len(result.text) if result.text else 0
                total_text_length += text_length
                print(f"    Text length: {text_length} characters")
                
                if result.security_analysis:
                    analysis = result.security_analysis
                    print(f"    Threat level: {analysis.threat_level}")
                    if not analysis.is_safe:
                        security_issues += 1
                        print(f"    ⚠️  Security concerns detected")
            else:
                print(f"    Error: {result.error}")
        
        print(f"\nBatch Summary:")
        print(f"  Total documents: {len(results)}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {len(results) - successful}")
        print(f"  Security issues: {security_issues}")
        print(f"  Total text extracted: {total_text_length} characters")
        print(f"  Success rate: {successful/len(results)*100:.1f}%")
    
    print()


async def demo_async_directory_processing():
    """Demonstrate async directory processing."""
    print("Async Directory Processing")
    print("-" * 35)
    
    async with AsyncProcessor(
        security_level="high",
        enable_neural=True,
        plugins=["text"]
    ) as processor:
        
        print(f"Processing all .txt files in {SAMPLE_DIR}...")
        
        start_time = time.time()
        results = await processor.process_directory(
            str(SAMPLE_DIR),
            pattern="*.txt",
            recursive=False,
            output_format="json",
            max_concurrent=3
        )
        total_time = time.time() - start_time
        
        print(f"Directory processing completed in {total_time:.3f} seconds")
        print(f"Found and processed {len(results)} files")
        print()
        
        # Show results summary
        for file_path, result in results.items():
            filename = Path(file_path).name
            if result.success:
                status = "✅"
                details = f"({len(result.text) if result.text else 0} chars)"
                
                if result.security_analysis and not result.security_analysis.is_safe:
                    status = "⚠️ "
                    details += f" - {result.security_analysis.threat_level}"
            else:
                status = "❌"
                details = f"Error: {result.error}"
            
            print(f"  {status} {filename} {details}")
    
    print()


async def demo_performance_comparison():
    """Demonstrate performance comparison between sync and async processing."""
    print("Performance Comparison: Sync vs Async")
    print("-" * 45)
    
    # Get sample files
    sample_files = list(SAMPLE_DIR.glob("*.txt"))
    file_paths = [str(f) for f in sample_files]
    
    # Test synchronous processing
    print("Testing synchronous processing...")
    sync_processor = neuraldocflow.create_processor(
        security_level="standard",
        enable_neural=True,
        plugins=["text"]
    )
    
    start_time = time.time()
    sync_results = []
    for file_path in file_paths:
        try:
            result = sync_processor.process_document(file_path, "json")
            sync_results.append(result)
        except Exception as e:
            print(f"Sync error for {file_path}: {e}")
    
    sync_time = time.time() - start_time
    print(f"Synchronous processing time: {sync_time:.3f} seconds")
    
    # Test asynchronous processing
    print("Testing asynchronous processing...")
    
    start_time = time.time()
    async_results = await process_batch_async(
        file_paths,
        security_level="standard",
        enable_neural=True,
        plugins=["text"],
        output_format="json",
        max_concurrent=3
    )
    async_time = time.time() - start_time
    print(f"Asynchronous processing time: {async_time:.3f} seconds")
    
    # Compare results
    speedup = sync_time / async_time if async_time > 0 else 0
    print(f"\nPerformance comparison:")
    print(f"  Synchronous: {sync_time:.3f}s")
    print(f"  Asynchronous: {async_time:.3f}s")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Async advantage: {((sync_time - async_time) / sync_time * 100):.1f}% faster")
    
    # Verify results consistency
    sync_successful = sum(1 for r in sync_results if r.success)
    async_successful = sum(1 for r in async_results if r.success)
    
    print(f"\nResult consistency:")
    print(f"  Sync successful: {sync_successful}/{len(sync_results)}")
    print(f"  Async successful: {async_successful}/{len(async_results)}")
    print(f"  Results match: {'✅' if sync_successful == async_successful else '❌'}")


async def demo_convenience_functions():
    """Demonstrate convenience functions."""
    print("Convenience Functions Demo")
    print("-" * 30)
    
    # Single document processing
    sample_file = SAMPLE_DIR / "safe_document.txt"
    
    print("Using process_document_async convenience function...")
    result = await process_document_async(
        str(sample_file),
        security_level="high",
        enable_neural=True,
        plugins=["text"],
        output_format="markdown"
    )
    
    print(f"  Success: {result.success}")
    print(f"  Format: {result.format}")
    print(f"  Text preview: {result.text[:100] if result.text else 'None'}...")
    
    # Batch processing
    file_paths = [str(f) for f in SAMPLE_DIR.glob("*.txt")][:3]  # First 3 files
    
    print(f"\nUsing process_batch_async convenience function...")
    results = await process_batch_async(
        file_paths,
        security_level="standard",
        enable_neural=False,
        plugins=["text"],
        output_format="json",
        max_concurrent=2
    )
    
    print(f"  Processed {len(results)} documents")
    successful = sum(1 for r in results if r.success)
    print(f"  Success rate: {successful}/{len(results)}")


# Cleanup function
async def cleanup():
    """Clean up temporary files."""
    import shutil
    if 'SAMPLE_DIR' in globals():
        try:
            shutil.rmtree(SAMPLE_DIR)
            print(f"Cleaned up temporary directory: {SAMPLE_DIR}")
        except Exception as e:
            print(f"Cleanup warning: {e}")


if __name__ == "__main__":
    try:
        # Run the async main function
        asyncio.run(main())
        
        # Also demonstrate convenience functions
        print("\n" + "-" * 55)
        asyncio.run(demo_convenience_functions())
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        try:
            asyncio.run(cleanup())
        except:
            pass