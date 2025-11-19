"""
File Parser Module
Handles document parsing for PDF, DOCX, and TXT files with automatic text normalization
"""

from pathlib import Path
from typing import List, Dict, Optional
import pypdf
import docx2txt
import tempfile
import os
import re


class FileParser:
    """Parse documents and extract text content with automatic text cleaning"""
    
    def __init__(self):
        self.supported_extensions = {'.pdf', '.docx', '.doc', '.txt', '.md'}
    
    def clean_extracted_text(self, text: str) -> str:
        """
        Clean extracted text to fix PDF extraction artifacts.
     
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text with normalized spacing and fixed word breaks
        """
        if not text:
            return ""
        
        # Step 1: Fix hyphenated line breaks first (before other processing)
        text = self.fix_hyphen_linebreaks(text)
        
        # Step 2: Fix character-level spacing patterns
        text = self.fix_character_spacing(text)
        
        # Step 3: Fix words split with newlines (non-hyphenated)
        text = self._fix_newline_word_breaks(text)
        
        # Step 4: Normalize whitespace
        text = self.normalize_whitespace(text)
        
        return text.strip()
    
    def fix_character_spacing(self, text: str) -> str:
        """
        Fix character-level spacing patterns in words:
        - "D ev el op me nt" → "Development"
        - "S t r o n g" → "Strong"
        - "meanin gful" → "meaningful"
        
        Args:
            text: Text with potential character spacing issues
            
        Returns:
            Text with character spacing fixed
        """
        if not text:
            return ""
        
        def fix_spaced_chars(match):
            chars = match.group(0)
            # Remove spaces between single characters
            fixed = chars.replace(' ', '')
            # Only fix if it forms a reasonable word (3+ chars, mostly letters)
            if len(fixed) >= 3 and fixed.isalnum() and fixed.isalpha():
                return fixed
            return chars
        
        text = re.sub(r'\b([a-zA-Z0-9] [a-zA-Z0-9]{2,}(?: [a-zA-Z0-9])*)\b', fix_spaced_chars, text)
        
        def fix_char_sequence(match):
            seq = match.group(0)
            # Remove all spaces
            fixed = seq.replace(' ', '')
            # Only fix if:
            # 1. Result is 3+ characters
            # 2. Mostly alphabetic (at least 70%)
            # 3. Forms a valid word pattern (starts with letter, contains letters/numbers)
            if len(fixed) >= 3:
                alpha_count = sum(1 for c in fixed if c.isalpha())
                if alpha_count / len(fixed) >= 0.7 and fixed[0].isalpha():
                    return fixed
            return seq
        
        text = re.sub(r'\b([a-zA-Z0-9](?: [a-zA-Z0-9]){2,})\b', fix_char_sequence, text)
        
        def fix_internal_spacing(match):
            part1 = match.group(1)
            part2 = match.group(2)
            combined = part1 + part2
            
            if (part1.islower() and part2.islower() and 
                3 <= len(combined) <= 30 and 
                combined.isalpha()):
                return combined
            return match.group(0)
        
        text = re.sub(r'\b([a-z]{2,}) ([a-z]{2,})\b', fix_internal_spacing, text)
        
        return text
    
    def fix_hyphen_linebreaks(self, text: str) -> str:
        """
        Fix hyphenated line breaks     
        
        Args:
            text: Text with potential hyphenated line breaks
            
        Returns:
            Text with hyphenated line breaks fixed
        """
        if not text:
            return ""
        
        def fix_hyphen_break(match):
            before = match.group(1)
            after = match.group(2)
            return before + after
        
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', fix_hyphen_break, text)
        
        text = re.sub(r'(\w+)-\s*\r\n\s*(\w+)', fix_hyphen_break, text)
        text = re.sub(r'(\w+)-\s*\r\s*(\w+)', fix_hyphen_break, text)
        
        return text
    
    def _fix_newline_word_breaks(self, text: str) -> str:
        """
        Fix words split with newlines (non-hyphenated)
        
        Args:
            text: Text with potential newline word breaks
            
        Returns:
            Text with newline word breaks fixed
        """
        if not text:
            return ""
        
        def fix_newline_break(match):
            before = match.group(1)
            after = match.group(2)
            combined = before + after
            
            if (before.islower() and after.islower() and 
                len(combined) >= 3 and len(combined) <= 30 and
                combined.isalpha()):
                return combined
            return match.group(0)
        
        text = re.sub(r'\b([a-z]{2,})\s*\n\s*([a-z]{2,})\b', fix_newline_break, text)
        
        text = re.sub(r'\b([a-z]{2,})\s*\r\n\s*([a-z]{2,})\b', fix_newline_break, text)
        text = re.sub(r'\b([a-z]{2,})\s*\r\s*([a-z]{2,})\b', fix_newline_break, text)
        
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace
        
        Args:
            text: Text with potentially messy whitespace
            
        Returns:
            Text with normalized whitespace
        """
        if not text:
            return ""
        
        text = text.replace('\t', ' ')
        
        text = re.sub(r'[ \t]+', ' ', text)
        
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        text = re.sub(r' +\n', '\n', text)
        text = re.sub(r'\n +', '\n', text)
        
        text = re.sub(r'\n\n +', '\n\n', text)
        text = re.sub(r' +\n\n', '\n\n', text)
        
        lines = text.split('\n')
        lines = [line.rstrip() for line in lines]
        text = '\n'.join(lines)
        
        text = re.sub(r' {2,}', ' ', text)
        
        return text
    
    def parse_file(self, file_path: str, file_name: str) -> Dict[str, any]:
        """
        Parse a file and extract text content with automatic cleaning
        
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
            
            # Clean extracted text to fix PDF extraction artifacts
            text = self.clean_extracted_text(text)
            
            if not text.strip():
                raise ValueError(f"No text content after cleaning from {file_name}")
            
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
