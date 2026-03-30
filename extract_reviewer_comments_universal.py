#!/usr/bin/env python3
"""
Universal script to extract reviewer comments from any downloads directory.
Usage: python extract_reviewer_comments_universal.py [directory_name]
"""

import os
import sys
import logging
from pathlib import Path
import fitz  # PyMuPDF

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_reviewer_comments_from_pdf(pdf_path):
    """Extract reviewer comments section from PDF."""
    try:
        doc = fitz.open(str(pdf_path))
        full_text = ""
        
        # Extract all text
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            full_text += page.get_text() + "\n"
        
        doc.close()
        
        # Find reviewer comments section
        full_text_lower = full_text.lower()
        
        # Look for start patterns
        start_patterns = [
            "reviewers' comments:",
            "reviewers comments:",
            "reviewer comments:",
            "reviewer #1:",
            "reviewer #2:",
        ]
        
        start_pos = -1
        for pattern in start_patterns:
            pos = full_text_lower.find(pattern)
            if pos != -1:
                start_pos = pos
                break
        
        if start_pos == -1:
            # Use full text if no pattern found
            return full_text
        
        # Look for end patterns
        end_patterns = [
            "author rebuttal to initial comments",
            "authors rebuttal to initial comments",
            "author response",
            "rebuttal"
        ]
        
        end_pos = -1
        for pattern in end_patterns:
            pos = full_text_lower.find(pattern, start_pos)
            if pos != -1:
                end_pos = pos
                break
        
        if end_pos == -1:
            extracted_text = full_text[start_pos:]
        else:
            extracted_text = full_text[start_pos:end_pos]
        
        return extracted_text.strip()
        
    except Exception as e:
        logger.error(f"Error extracting from {pdf_path}: {e}")
        return ""

def extract_from_txt(txt_path):
    """Extract full content from text file."""
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(txt_path, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading {txt_path}: {e}")
            return ""
    except Exception as e:
        logger.error(f"Error reading {txt_path}: {e}")
        return ""

def process_directory(downloads_dir):
    """Process a downloads directory."""
    if not downloads_dir.exists():
        print(f"Directory not found: {downloads_dir}")
        return 0, 0
    
    processed = 0
    successful = 0
    
    print(f"Processing directory: {downloads_dir}")
    
    # Process each DOI directory
    for doi_dir in downloads_dir.iterdir():
        if doi_dir.is_dir():
            print(f"  Processing: {doi_dir.name}")
            
            # Look for peer_review files
            peer_review_pdf = doi_dir / "peer_review.pdf"
            peer_review_txt = doi_dir / "peer_review.txt"
            
            extracted_text = ""
            source_file = None
            
            if peer_review_pdf.exists():
                extracted_text = extract_reviewer_comments_from_pdf(peer_review_pdf)
                source_file = "peer_review.pdf"
            elif peer_review_txt.exists():
                extracted_text = extract_from_txt(peer_review_txt)
                source_file = "peer_review.txt"
            
            if extracted_text and source_file:
                # Save extracted text
                output_path = doi_dir / "reviewer_comments_extracted.txt"
                try:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(f"# Extracted from {source_file}\n")
                        f.write(f"# Directory: {doi_dir.name}\n")
                        f.write(f"# Characters: {len(extracted_text)}\n\n")
                        f.write(extracted_text)
                    
                    print(f"    ✓ Saved {len(extracted_text)} characters to reviewer_comments_extracted.txt")
                    successful += 1
                    
                except Exception as e:
                    print(f"    ✗ Error saving: {e}")
            else:
                print(f"    ✗ No peer_review file found or extraction failed")
            
            processed += 1
    
    return processed, successful

def main():
    """Extract reviewer comments from downloads directories."""
    
    # Get directory from command line argument or use defaults
    if len(sys.argv) > 1:
        directories = [sys.argv[1]]
    else:
        # Check for common download directories
        directories = ["synbio_downloads", "bioeng_downloads"]
    
    total_processed = 0
    total_successful = 0
    
    for dir_name in directories:
        downloads_dir = Path(dir_name)
        if downloads_dir.exists():
            processed, successful = process_directory(downloads_dir)
            total_processed += processed
            total_successful += successful
            print()
    
    if total_processed == 0:
        print("No valid downloads directories found.")
        print("Available directories:")
        for item in Path(".").iterdir():
            if item.is_dir() and "download" in item.name.lower():
                print(f"  {item.name}")
        return
    
    print(f"Overall Summary:")
    print(f"Directories processed: {total_processed}")
    print(f"Successful extractions: {total_successful}")
    print(f"Files saved as 'reviewer_comments_extracted.txt' in each DOI folder")

if __name__ == "__main__":
    main()
