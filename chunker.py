"""
Chunker Module
Handles text chunking for RAG pipeline with sentence-aware chunking and text cleaning
"""

from typing import List, Dict
import re


class Chunker:
    """Split text into chunks for embedding and retrieval with sentence-aware chunking"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize chunker
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def clean_text(self, text: str) -> str:
        """
        Clean text before chunking:
        - Remove excessive newlines
        - Fix broken words split across lines
        - Remove hyphenated line breaks
        - Fix spacing issues inside words
        - Normalize multiple spaces into one
        - Preserve paragraph breaks
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text with preserved paragraph structure
        """
        if not text:
            return ""
        
        # Step 1: Remove hyphenated line breaks (e.g., "inter-\naction" → "interaction")
        # Match hyphen followed by newline(s) and optional whitespace
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        # Step 2: Fix broken words split across lines (no hyphen)
        # Match word character, newline(s), then word character (likely same word)
        # Be careful not to break legitimate paragraph breaks
        # Only fix if it looks like a broken word (short gap, lowercase letters)
        text = re.sub(r'(\w)\s*\n\s*(\w)', lambda m: m.group(1) + m.group(2) if len(m.group(1) + m.group(2)) < 20 else m.group(1) + ' ' + m.group(2), text)
        
        # Step 3: Normalize newlines - preserve paragraph breaks (double newlines)
        # Replace multiple newlines (3+) with double newline (paragraph break)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Step 4: Fix spacing issues inside words (e.g., "meanin gful" → "meaningful")
        # Match space between two lowercase letters (likely broken word)
        text = re.sub(r'([a-z])\s+([a-z])', lambda m: m.group(1) + m.group(2) if len(m.group(1) + m.group(2)) < 20 else m.group(1) + ' ' + m.group(2), text)
        
        # Step 5: Normalize multiple spaces into single space (but preserve paragraph breaks)
        # Replace multiple spaces/tabs with single space, but keep newlines
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Step 6: Clean up newlines - remove single newlines that aren't paragraph breaks
        # Replace single newline with space (preserving double newlines for paragraphs)
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        
        # Step 7: Normalize paragraph breaks to consistent double newline
        text = re.sub(r'\n{2,}', '\n\n', text)
        
        # Step 8: Remove leading/trailing whitespace from each line
        lines = text.split('\n')
        lines = [line.strip() for line in lines]
        text = '\n'.join(lines)
        
        # Step 9: Final cleanup - remove excessive spaces around paragraph breaks
        text = re.sub(r' \n\n ', '\n\n', text)
        text = re.sub(r'\n\n ', '\n\n', text)
        text = re.sub(r' \n\n', '\n\n', text)
        
        # Step 10: Remove any remaining excessive whitespace
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex pattern
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        if not text.strip():
            return []
        
        # Split by sentence endings: . ! ? followed by whitespace
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Split text into chunks using sentence-aware chunking
        
        Args:
            text: Text content to chunk
            metadata: Metadata dictionary to attach to chunks
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or not text.strip():
            return []
        
        # Step 1: Clean the text
        cleaned_text = self.clean_text(text)
        
        if not cleaned_text.strip():
            return []
        
        # Step 2: Split into paragraphs (soft boundaries)
        paragraphs = self._split_into_paragraphs(cleaned_text)
        
        # Step 3: Split paragraphs into sentences
        all_sentences = []
        paragraph_boundaries = []  # Track which sentences belong to which paragraph
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            sentences = self.split_into_sentences(para)
            if sentences:
                start_idx = len(all_sentences)
                all_sentences.extend(sentences)
                end_idx = len(all_sentences)
                paragraph_boundaries.append((start_idx, end_idx))
        
        if not all_sentences:
            # Fallback: if no sentences found, treat entire text as one chunk
            chunk_metadata = metadata.copy()
            chunk_metadata['chunk_index'] = 0
            chunk_metadata['chunk_start'] = 0
            chunk_metadata['chunk_end'] = len(cleaned_text)
            return [{
                'text': cleaned_text,
                'metadata': chunk_metadata
            }]
        
        # Step 4: Build chunks by adding sentences until chunk_size is reached
        chunks = []
        current_chunk_sentences = []
        current_chunk_length = 0
        chunk_start_idx = 0
        
        i = 0
        while i < len(all_sentences):
            sentence = all_sentences[i]
            sentence_length = len(sentence) + 1  # +1 for space separator
            
            # Check if adding this sentence would exceed chunk size
            if current_chunk_length + sentence_length > self.chunk_size and current_chunk_sentences:
                # Save current chunk
                chunk_text = ' '.join(current_chunk_sentences)
                chunk_end_idx = self._find_chunk_end_in_text(cleaned_text, chunk_start_idx, chunk_text)
                
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_index'] = len(chunks)
                chunk_metadata['chunk_start'] = chunk_start_idx
                chunk_metadata['chunk_end'] = chunk_end_idx
                
                chunks.append({
                    'text': chunk_text,
                    'metadata': chunk_metadata
                })
                
                # Calculate overlap: go back sentences until we have enough overlap
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk_sentences, 
                    self.chunk_overlap
                )
                
                # Start new chunk with overlap
                current_chunk_sentences = overlap_sentences.copy()
                current_chunk_length = sum(len(s) + 1 for s in overlap_sentences) - 1
                
                # Find start position of overlap in cleaned text
                overlap_text = ' '.join(overlap_sentences)
                chunk_start_idx = cleaned_text.find(overlap_text, chunk_start_idx)
                if chunk_start_idx == -1:
                    chunk_start_idx = chunk_end_idx - len(overlap_text)
                
                # Continue with current sentence (don't increment i)
                continue
            
            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            current_chunk_length += sentence_length
            
            # If this is the first sentence of a chunk, find its start position
            if len(current_chunk_sentences) == 1:
                chunk_start_idx = cleaned_text.find(sentence)
                if chunk_start_idx == -1:
                    chunk_start_idx = 0
            
            i += 1
        
        # Add final chunk
        if current_chunk_sentences:
            chunk_text = ' '.join(current_chunk_sentences)
            chunk_end_idx = self._find_chunk_end_in_text(cleaned_text, chunk_start_idx, chunk_text)
            
            chunk_metadata = metadata.copy()
            chunk_metadata['chunk_index'] = len(chunks)
            chunk_metadata['chunk_start'] = chunk_start_idx
            chunk_metadata['chunk_end'] = chunk_end_idx
            
            chunks.append({
                'text': chunk_text,
                'metadata': chunk_metadata
            })
        
        # Ensure at least one chunk is returned
        if not chunks and cleaned_text.strip():
            chunk_metadata = metadata.copy()
            chunk_metadata['chunk_index'] = 0
            chunk_metadata['chunk_start'] = 0
            chunk_metadata['chunk_end'] = len(cleaned_text)
            chunks.append({
                'text': cleaned_text,
                'metadata': chunk_metadata
            })
        
        return chunks
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs using double newlines as boundaries
        
        Args:
            text: Text to split
            
        Returns:
            List of paragraphs
        """
        # Split by double newlines (paragraph breaks)
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Filter out empty paragraphs and strip whitespace
        paragraphs = [para.strip() for para in paragraphs if para.strip()]
        
        return paragraphs
    
    def _get_overlap_sentences(self, sentences: List[str], overlap_size: int) -> List[str]:
        """
        Get overlap sentences from the end of a chunk
        
        Args:
            sentences: List of sentences in current chunk
            overlap_size: Desired overlap size in characters
            
        Returns:
            List of sentences for overlap
        """
        if not sentences:
            return []
        
        # Build overlap from the end, adding sentences until we reach overlap_size
        overlap_sentences = []
        current_length = 0
        
        for sentence in reversed(sentences):
            sentence_length = len(sentence) + (1 if overlap_sentences else 0)  # +1 for space if not first
            if current_length + sentence_length <= overlap_size or not overlap_sentences:
                overlap_sentences.insert(0, sentence)
                current_length += sentence_length
            else:
                break
        
        return overlap_sentences
    
    def _find_chunk_end_in_text(self, text: str, start_idx: int, chunk_text: str) -> int:
        """
        Find the end index of a chunk in the cleaned text
        
        Args:
            text: Cleaned text
            start_idx: Start index of chunk
            chunk_text: Chunk text to locate
            
        Returns:
            End index in cleaned text
        """
        # Try to find exact match first
        idx = text.find(chunk_text, start_idx)
        if idx != -1:
            return idx + len(chunk_text)
        
        # Fallback: approximate based on length
        return min(start_idx + len(chunk_text), len(text))
