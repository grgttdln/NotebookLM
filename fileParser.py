"""
File Parser Module
Handles document parsing for PDF, DOCX, and TXT files
"""

from pathlib import Path
from typing import List, Dict, Optional
import pypdf
import docx2txt
import tempfile
import os


class FileParser:
    """Parse documents and extract text content"""
    
    def __init__(self):
        self.supported_extensions = {'.pdf', '.docx', '.doc', '.txt', '.md'}
    
    def parse_file(self, file_path: str, file_name: str) -> Dict[str, any]:
        """
        Parse a file and extract text content
        
        Args:
            file_path: Path to the uploaded file
            file_name: Original filename
            
        Returns:
            Dictionary with 'text', 'metadata', and 'file_name'
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {extension}")
        
        text = ""
        metadata = {
            'file_name': file_name,
            'file_type': extension,
            'file_path': file_path
        }
        
        try:
            if extension == '.pdf':
                text = self._parse_pdf(file_path)
                metadata['page_count'] = self._get_pdf_page_count(file_path)
            elif extension in ['.docx', '.doc']:
                text = self._parse_docx(file_path)
            elif extension in ['.txt', '.md']:
                text = self._parse_text(file_path)
            
            if not text.strip():
                raise ValueError(f"No text content extracted from {file_name}")
            
            metadata['text_length'] = len(text)
            metadata['word_count'] = len(text.split())
            
            return {
                'text': text,
                'metadata': metadata
            }
        except Exception as e:
            raise ValueError(f"Error parsing {file_name}: {str(e)}")
    
    def _parse_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        text_parts = []
        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return '\n\n'.join(text_parts)
    
    def _get_pdf_page_count(self, file_path: str) -> int:
        """Get page count from PDF"""
        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            return len(pdf_reader.pages)
    
    def _parse_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        return docx2txt.process(file_path)
    
    def _parse_text(self, file_path: str) -> str:
        """Extract text from TXT/MD file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

