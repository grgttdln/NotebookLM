"""
Retriever Module
Handles vector storage and retrieval using Hugging Face embeddings
"""

from typing import List, Dict, Optional
import numpy as np
import json
import os
from pathlib import Path


class Retriever:
    """Vector store and retriever using Hugging Face embeddings"""
    
    def __init__(self, storage_path: str = "./vectorstore/hf_store.json"):
        """
        Initialize retriever
        
        Args:
            storage_path: Path to store vector data
        """
        self.storage_path = storage_path
        self.vectors: List[List[float]] = []
        self.chunks: List[Dict] = []
        self.document_store: Dict[str, Dict] = {}  # Store full document texts
        
        # Load existing data if available
        self._load_storage()
    
    def add_document(self, document_id: str, document_text: str, metadata: Dict):
        """
        Store document text for later retrieval
        
        Args:
            document_id: Unique document identifier
            document_text: Full document text
            metadata: Document metadata
        """
        self.document_store[document_id] = {
            'text': document_text,
            'metadata': metadata
        }
    
    def add_chunks(self, chunks: List[Dict], embeddings: List[List[float]]):
        """
        Add chunks with their embeddings to the vector store
        
        Args:
            chunks: List of chunk dictionaries
            embeddings: List of embedding vectors
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        for chunk, embedding in zip(chunks, embeddings):
            self.chunks.append(chunk)
            self.vectors.append(embedding)
        
        # Save to disk
        self._save_storage()
    
    def retrieve(self, query_embedding: List[float], k: int = 4, document_ids: Optional[List[str]] = None) -> List[Dict]:
        """
        Retrieve top-k most similar chunks
        
        Args:
            query_embedding: Query embedding vector
            k: Number of chunks to retrieve
            document_ids: Optional list of document IDs to filter by. If None, searches all documents.
            
        Returns:
            List of chunk dictionaries with similarity scores
        """
        if not self.vectors:
            return []
        
        # Validate that vectors and chunks are in sync, auto-repair if needed
        if len(self.vectors) != len(self.chunks):
            print(f"Warning: Vector store mismatch detected (vectors: {len(self.vectors)}, chunks: {len(self.chunks)}). Attempting repair...")
            self.repair_vector_store()
            if len(self.vectors) != len(self.chunks):
                raise ValueError(
                    f"Vector store is corrupted: vectors ({len(self.vectors)}) and chunks ({len(self.chunks)}) "
                    f"have different lengths after repair attempt. Please re-upload your documents."
                )
        
        # Calculate cosine similarities
        query_vec = np.array(query_embedding)
        similarities = []
        
        for i, vec in enumerate(self.vectors):
            # Bounds check
            if i >= len(self.chunks):
                continue
            
            # Filter by document_ids if provided
            if document_ids is not None:
                chunk_metadata = self.chunks[i].get('metadata', {})
                chunk_doc_id = chunk_metadata.get('document_id')
                if chunk_doc_id not in document_ids:
                    continue
            
            vec_array = np.array(vec)
            dot_product = np.dot(query_vec, vec_array)
            norm_query = np.linalg.norm(query_vec)
            norm_vec = np.linalg.norm(vec_array)
            
            if norm_query == 0 or norm_vec == 0:
                similarity = 0.0
            else:
                similarity = float(dot_product / (norm_query * norm_vec))
            
            similarities.append((i, similarity))
        
        # If no similarities found (e.g., all filtered out), return empty
        if not similarities:
            return []
        
        # Sort by similarity and get top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:k]
        
        # Return chunks with similarity scores
        results = []
        for idx, similarity_score in top_k:
            # Additional bounds check
            if idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx].copy()
            chunk['similarity'] = similarity_score
            results.append(chunk)
        
        return results
    
    def get_document(self, document_id: str) -> Optional[Dict]:
        """
        Get full document by ID
        
        Args:
            document_id: Document identifier
            
        Returns:
            Document dictionary or None
        """
        return self.document_store.get(document_id)
    
    def get_all_documents(self) -> List[Dict]:
        """
        Get all stored documents
        
        Returns:
            List of document dictionaries
        """
        return [
            {'id': doc_id, **doc_data}
            for doc_id, doc_data in self.document_store.items()
        ]
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and all its associated chunks
        
        Args:
            document_id: Document identifier to delete
            
        Returns:
            True if document was deleted, False if not found
        """
        if document_id not in self.document_store:
            return False
        
        # Remove document from store
        del self.document_store[document_id]
        
        # Remove all chunks associated with this document
        # We need to track indices to remove in reverse order
        indices_to_remove = []
        for i, chunk in enumerate(self.chunks):
            if chunk.get('metadata', {}).get('document_id') == document_id:
                indices_to_remove.append(i)
        
        # Remove chunks and vectors in reverse order to maintain indices
        for idx in reversed(indices_to_remove):
            self.chunks.pop(idx)
            self.vectors.pop(idx)
        
        # Save to disk
        self._save_storage()
        
        return True
    
    def _save_storage(self):
        """Save vector store to disk"""
        try:
            storage_dir = Path(self.storage_path).parent
            storage_dir.mkdir(parents=True, exist_ok=True)
            
            # Save chunks and document store (vectors are too large for JSON)
            data = {
                'chunks': self.chunks,
                'document_store': self.document_store
            }
            
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Save vectors separately as numpy array
            vectors_path = self.storage_path.replace('.json', '_vectors.npy')
            if self.vectors:
                np.save(vectors_path, np.array(self.vectors))
        except Exception as e:
            print(f"Warning: Could not save storage: {e}")
    
    def repair_vector_store(self):
        """
        Repair vector store by removing orphaned vectors or chunks.
        Keeps only pairs that exist in both arrays.
        """
        vectors_len = len(self.vectors)
        chunks_len = len(self.chunks)
        
        if vectors_len == chunks_len:
            print("Vector store is already in sync.")
            return  # Already in sync
        
        print(f"Repairing vector store: {vectors_len} vectors, {chunks_len} chunks")
        
        try:
            # Keep only the minimum length to ensure they match
            min_len = min(vectors_len, chunks_len)
            
            # Trim both arrays to the same length
            self.vectors = self.vectors[:min_len].copy() if hasattr(self.vectors, 'copy') else self.vectors[:min_len]
            self.chunks = self.chunks[:min_len].copy() if hasattr(self.chunks, 'copy') else self.chunks[:min_len]
            
            # Verify repair worked
            if len(self.vectors) != len(self.chunks):
                raise ValueError(f"Repair failed: vectors ({len(self.vectors)}) and chunks ({len(self.chunks)}) still mismatched after trim")
            
            # Rebuild document_store to only include documents that still have chunks
            valid_doc_ids = set()
            for chunk in self.chunks:
                doc_id = chunk.get('metadata', {}).get('document_id')
                if doc_id:
                    valid_doc_ids.add(doc_id)
            
            # Remove documents that no longer have chunks
            doc_ids_to_remove = [
                doc_id for doc_id in self.document_store.keys()
                if doc_id not in valid_doc_ids
            ]
            for doc_id in doc_ids_to_remove:
                del self.document_store[doc_id]
            
            # Save repaired store
            self._save_storage()
            print(f"Repair complete: {min_len} vector-chunk pairs retained, {len(doc_ids_to_remove)} orphaned documents removed")
            
            # Final verification
            if len(self.vectors) != len(self.chunks):
                raise ValueError(f"Repair verification failed: vectors ({len(self.vectors)}) and chunks ({len(self.chunks)}) still mismatched")
                
        except Exception as e:
            print(f"Error during repair: {e}")
            raise
    
    def _load_storage(self):
        """Load vector store from disk"""
        try:
            if not os.path.exists(self.storage_path):
                return
            
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.chunks = data.get('chunks', [])
            self.document_store = data.get('document_store', {})
            
            # Load vectors
            vectors_path = self.storage_path.replace('.json', '_vectors.npy')
            if os.path.exists(vectors_path):
                vectors_array = np.load(vectors_path)
                self.vectors = vectors_array.tolist()
            
            # Check for corruption and repair if needed
            if len(self.vectors) != len(self.chunks):
                print(f"Warning: Detected mismatch on load (vectors: {len(self.vectors)}, chunks: {len(self.chunks)}). Repairing...")
                self.repair_vector_store()
        except Exception as e:
            print(f"Warning: Could not load storage: {e}")
            self.vectors = []
            self.chunks = []
            self.document_store = {}

