"""
Chunker Module
Handles text chunking for RAG pipeline
"""

from typing import List, Dict
import re


class Chunker:
    """Split text into chunks for embedding and retrieval"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize chunker
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Split text into chunks
        
        Args:
            text: Text content to chunk
            metadata: Metadata dictionary to attach to chunks
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text.strip():
            return []
        
        # Split by paragraphs first for better semantic boundaries
        paragraphs = self._split_into_paragraphs(text)
        
        chunks = []
        current_chunk = ""
        chunk_start_idx = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(para) + 1 > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_index'] = len(chunks)
                chunk_metadata['chunk_start'] = chunk_start_idx
                chunk_metadata['chunk_end'] = chunk_start_idx + len(current_chunk)
                
                chunks.append({
                    'text': current_chunk.strip(),
                    'metadata': chunk_metadata
                })
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                current_chunk = overlap_text + para
                chunk_start_idx = chunk_metadata['chunk_end'] - len(overlap_text)
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                    chunk_start_idx = text.find(para)
        
        # Add final chunk
        if current_chunk.strip():
            chunk_metadata = metadata.copy()
            chunk_metadata['chunk_index'] = len(chunks)
            chunk_metadata['chunk_start'] = chunk_start_idx
            chunk_metadata['chunk_end'] = chunk_start_idx + len(current_chunk)
            
            chunks.append({
                'text': current_chunk.strip(),
                'metadata': chunk_metadata
            })
        
        # If text is shorter than chunk_size, ensure we still create at least one chunk
        if not chunks and text.strip():
            chunk_metadata = metadata.copy()
            chunk_metadata['chunk_index'] = 0
            chunk_metadata['chunk_start'] = 0
            chunk_metadata['chunk_end'] = len(text)
            chunks.append({
                'text': text.strip(),
                'metadata': chunk_metadata
            })
        
        return chunks
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        # Split by double newlines or single newline followed by capital letter
        paragraphs = re.split(r'\n\s*\n', text)
        # Further split very long paragraphs
        result = []
        for para in paragraphs:
            if len(para) > self.chunk_size:
                # Split long paragraphs by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current = ""
                for sentence in sentences:
                    if len(current) + len(sentence) > self.chunk_size and current:
                        result.append(current)
                        current = sentence
                    else:
                        current += (" " if current else "") + sentence
                if current:
                    result.append(current)
            else:
                result.append(para)
        return result
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get overlap text from the end of a chunk"""
        if len(text) <= overlap_size:
            return text
        # Try to overlap at sentence boundaries
        overlap_text = text[-overlap_size:]
        # Find first sentence boundary in overlap
        sentence_match = re.search(r'[.!?]\s+', overlap_text)
        if sentence_match:
            return overlap_text[sentence_match.end():]
        return overlap_text

