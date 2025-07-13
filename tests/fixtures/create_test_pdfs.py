#!/usr/bin/env python3
"""Create test PDF files for the neural document flow test suite."""

import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.lib.units import inch
from PyPDF2 import PdfWriter, PdfReader
import io

def create_simple_pdf(filename, title, content):
    """Create a simple single-page PDF with text."""
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    # Add metadata
    c.setTitle(title)
    c.setAuthor("Test Suite")
    c.setSubject("Test Document")
    c.setKeywords(["test", "example", "neural"])
    
    # Add title
    c.setFont("Helvetica-Bold", 24)
    c.drawString(100, height - 100, title)
    
    # Add content
    c.setFont("Helvetica", 12)
    y = height - 150
    for line in content.split('\n'):
        c.drawString(100, y, line)
        y -= 20
    
    # Add page number
    c.setFont("Helvetica", 10)
    c.drawString(width/2 - 20, 50, "Page 1")
    
    c.save()

def create_multipage_pdf(filename, title, pages=5):
    """Create a multi-page PDF document."""
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    # Add metadata
    c.setTitle(title)
    c.setAuthor("Test Suite")
    
    for page_num in range(1, pages + 1):
        # Title
        c.setFont("Helvetica-Bold", 24)
        c.drawString(100, height - 100, f"{title} - Page {page_num}")
        
        # Content
        c.setFont("Helvetica", 12)
        y = height - 150
        content = f"""This is page {page_num} of {pages}.
        
Lorem ipsum dolor sit amet, consectetur adipiscing elit.
Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.

Section {page_num}.1: Introduction
This section introduces the content for page {page_num}.

Section {page_num}.2: Details
Here are more details about the content on this page.
- Item 1: Description of first item
- Item 2: Description of second item
- Item 3: Description of third item

Section {page_num}.3: Conclusion
This concludes the content for page {page_num}."""
        
        for line in content.split('\n'):
            c.drawString(100, y, line.strip())
            y -= 15
            if y < 100:
                break
        
        # Page number
        c.setFont("Helvetica", 10)
        c.drawString(width/2 - 20, 50, f"Page {page_num}")
        
        if page_num < pages:
            c.showPage()
    
    c.save()

def create_table_pdf(filename, title):
    """Create a PDF with tables."""
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    # Title
    c.setFont("Helvetica-Bold", 24)
    c.drawString(100, height - 100, title)
    
    # Table header
    c.setFont("Helvetica-Bold", 12)
    y = height - 200
    c.drawString(100, y, "ID")
    c.drawString(200, y, "Name")
    c.drawString(350, y, "Value")
    c.drawString(450, y, "Status")
    
    # Draw table lines
    c.line(90, y - 5, 550, y - 5)
    
    # Table data
    c.setFont("Helvetica", 11)
    data = [
        ("001", "Document A", "100.50", "Completed"),
        ("002", "Document B", "250.75", "In Progress"),
        ("003", "Document C", "175.25", "Pending"),
        ("004", "Document D", "300.00", "Completed"),
        ("005", "Document E", "425.50", "Failed"),
    ]
    
    y -= 20
    for row in data:
        c.drawString(100, y, row[0])
        c.drawString(200, y, row[1])
        c.drawString(350, y, row[2])
        c.drawString(450, y, row[3])
        y -= 20
    
    # Summary
    c.setFont("Helvetica-Bold", 12)
    y -= 30
    c.drawString(100, y, "Summary:")
    c.setFont("Helvetica", 11)
    y -= 20
    c.drawString(100, y, "Total Documents: 5")
    y -= 15
    c.drawString(100, y, "Total Value: 1,252.00")
    
    c.save()

def create_complex_pdf(filename, title):
    """Create a complex PDF with various elements."""
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    # Page 1: Title and introduction
    c.setFont("Helvetica-Bold", 32)
    c.drawString(width/2 - 150, height/2, title)
    
    c.setFont("Helvetica", 14)
    c.drawString(width/2 - 100, height/2 - 50, "A Complex Test Document")
    
    c.setFont("Helvetica", 10)
    c.drawString(width/2 - 50, 100, "Created: January 2024")
    
    c.showPage()
    
    # Page 2: Text with formatting
    c.setFont("Helvetica-Bold", 18)
    c.drawString(100, height - 100, "Chapter 1: Introduction")
    
    c.setFont("Helvetica", 12)
    text = """This document demonstrates various PDF features including:
    • Multiple pages with different content types
    • Text formatting and styles
    • Lists and bullet points
    • Tables and structured data
    • Headers and footers
    • Metadata and properties"""
    
    y = height - 150
    for line in text.split('\n'):
        c.drawString(100, y, line)
        y -= 20
    
    # Add some shapes
    c.setStrokeColor(colors.blue)
    c.setFillColor(colors.lightblue)
    c.rect(100, y - 50, 200, 40, fill=1)
    
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(110, y - 35, "Important Note")
    
    c.showPage()
    
    # Page 3: Data visualization (simple chart)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(100, height - 100, "Chapter 2: Data Analysis")
    
    # Draw a simple bar chart
    c.setFont("Helvetica", 10)
    y = height - 200
    bar_data = [("Q1", 75), ("Q2", 85), ("Q3", 95), ("Q4", 110)]
    
    for i, (label, value) in enumerate(bar_data):
        x = 150 + i * 100
        bar_height = value * 2
        
        # Draw bar
        c.setFillColor(colors.green)
        c.rect(x, y - bar_height, 50, bar_height, fill=1)
        
        # Draw label
        c.setFillColor(colors.black)
        c.drawString(x + 15, y - bar_height - 20, label)
        c.drawString(x + 10, y - bar_height + 5, str(value))
    
    c.save()

def create_corrupted_pdf(filename):
    """Create a corrupted PDF file for error testing."""
    with open(filename, 'wb') as f:
        # Write partial PDF header
        f.write(b"%PDF-1.4\n")
        # Write some random bytes
        f.write(b"This is not a valid PDF content\n")
        f.write(b"\x00\x01\x02\x03\x04\x05")
        # Don't write proper PDF structure

def create_empty_pdf(filename):
    """Create an empty PDF with minimal content."""
    c = canvas.Canvas(filename, pagesize=letter)
    # Just save without adding any content
    c.save()

def create_special_chars_pdf(filename):
    """Create a PDF with special characters and unicode."""
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    c.setTitle("Special Characters Test")
    
    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(100, height - 100, "Special Characters Test Document")
    
    # Special characters
    c.setFont("Helvetica", 12)
    y = height - 150
    
    special_content = [
        "ASCII Special: < > & \" ' @ # $ % ^ * ( ) { } [ ]",
        "Math Symbols: ± × ÷ ≤ ≥ ≠ ∞ √ ∑ ∏ ∫",
        "Currency: $ € £ ¥ ₹ ₽ ¢",
        "Arrows: ← → ↑ ↓ ↔ ⇐ ⇒ ⇑ ⇓ ⇔",
        "Quotes: "Hello" 'World' „Test" «French» »Spanish«",
        "Diacritics: café naïve résumé façade",
        "Greek: α β γ δ ε ζ η θ ι κ λ μ",
        "Bullets: • ◦ ▪ ▫ ■ □ ● ○",
    ]
    
    for line in special_content:
        try:
            c.drawString(100, y, line)
        except:
            # If character can't be rendered, show placeholder
            c.drawString(100, y, "[Special characters line]")
        y -= 20
    
    c.save()

def main():
    """Create all test PDF files."""
    fixtures_dir = "pdfs"
    os.makedirs(fixtures_dir, exist_ok=True)
    
    # Create various test PDFs
    print("Creating test PDFs...")
    
    # Simple PDFs
    create_simple_pdf(
        os.path.join(fixtures_dir, "simple.pdf"),
        "Simple Test Document",
        "This is a simple test document.\nIt contains basic text content.\nUsed for testing basic PDF extraction."
    )
    
    # Multi-page PDF
    create_multipage_pdf(
        os.path.join(fixtures_dir, "multipage.pdf"),
        "Multi-Page Test Document",
        pages=10
    )
    
    # Table PDF
    create_table_pdf(
        os.path.join(fixtures_dir, "table.pdf"),
        "Table Test Document"
    )
    
    # Complex PDF
    create_complex_pdf(
        os.path.join(fixtures_dir, "complex.pdf"),
        "Complex Test Document"
    )
    
    # Empty PDF
    create_empty_pdf(os.path.join(fixtures_dir, "empty.pdf"))
    
    # Special characters PDF
    create_special_chars_pdf(os.path.join(fixtures_dir, "special_chars.pdf"))
    
    # Corrupted PDF
    create_corrupted_pdf(os.path.join(fixtures_dir, "corrupted.pdf"))
    
    print("Test PDFs created successfully!")

if __name__ == "__main__":
    main()