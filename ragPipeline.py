"""
RAG Pipeline Module
Coordinates the end-to-end RAG process
"""

from typing import List, Dict, Optional
from fileParser import FileParser
from chunker import Chunker
from groqClient import HuggingFaceClient
from retriever import Retriever
import uuid


class RAGPipeline:
    """Main RAG pipeline coordinator"""
    
    def __init__(
        self,
        huggingface_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        storage_path: str = "./vectorstore/hf_store.json"
    ):
        """
        Initialize RAG pipeline
        
        Args:
            huggingface_api_key: Hugging Face API token (for embeddings)
            groq_api_key: Groq API token (for LLM)
            chunk_size: Chunk size for text splitting
            chunk_overlap: Chunk overlap
            storage_path: Path for vector storage
        """
        self.file_parser = FileParser()
        self.chunker = Chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.hf_client = HuggingFaceClient(
            huggingface_api_key=huggingface_api_key,
            groq_api_key=groq_api_key
        )
        self.retriever = Retriever(storage_path=storage_path)
    
    def process_document(self, file_path: str, file_name: str) -> Dict:
        """
        Process a document: parse, chunk, embed, and store
        
        Args:
            file_path: Path to uploaded file
            file_name: Original filename
            
        Returns:
            Dictionary with document ID and processing info
        """
        # Parse document
        parsed = self.file_parser.parse_file(file_path, file_name)
        document_text = parsed['text']
        metadata = parsed['metadata']
        
        # Generate document ID
        document_id = str(uuid.uuid4())
        metadata['document_id'] = document_id
        
        # Store full document
        self.retriever.add_document(document_id, document_text, metadata)
        
        # Chunk text
        chunks = self.chunker.chunk_text(document_text, metadata)
        
        if not chunks:
            raise ValueError("No chunks created from document")
        
        # Generate embeddings for chunks
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = self.hf_client.get_embeddings(chunk_texts)
        
        # Add document_id to each chunk's metadata
        for chunk in chunks:
            chunk['metadata']['document_id'] = document_id
        
        # Store chunks and embeddings
        self.retriever.add_chunks(chunks, embeddings)
        
        return {
            'document_id': document_id,
            'file_name': file_name,
            'chunk_count': len(chunks),
            'metadata': metadata
        }
    
    def query(self, question: str, k: int = 4, document_ids: Optional[List[str]] = None) -> Dict:
        """
        Query the RAG system
        
        Args:
            question: User question
            k: Number of chunks to retrieve
            document_ids: Optional list of document IDs to filter by. If None, searches all documents.
            
        Returns:
            Dictionary with answer, citations, and retrieved chunks
        """
        if not question or not question.strip():
            return {
                'answer': "Please provide a question.",
                'citations': [],
                'chunks': []
            }
        
        try:
            # Embed question
            question_embedding = self.hf_client.get_embedding(question)
            
            if not question_embedding:
                raise ValueError("Failed to generate embedding for question")
            
            # Retrieve relevant chunks
            retrieved_chunks = self.retriever.retrieve(question_embedding, k=k, document_ids=document_ids)
            
            if not retrieved_chunks:
                # Check if documents exist but were filtered out
                all_docs = self.retriever.get_all_documents()
                if all_docs:
                    if document_ids:
                        return {
                            'answer': "No relevant content found in the selected documents. Try selecting different documents or rephrasing your question.",
                            'citations': [],
                            'chunks': []
                        }
                    else:
                        return {
                            'answer': "No relevant content found in the documents. Try rephrasing your question.",
                            'citations': [],
                            'chunks': []
                        }
                else:
                    return {
                        'answer': "I don't have any documents loaded yet. Please upload a document first.",
                        'citations': [],
                        'chunks': []
                    }
            
            # Format context from retrieved chunks
            context_parts = []
            citations = []
            
            for i, chunk in enumerate(retrieved_chunks, 1):
                if not chunk or 'text' not in chunk:
                    continue
                    
                chunk_text = chunk.get('text', '')
                chunk_metadata = chunk.get('metadata', {})
                document_id = chunk_metadata.get('document_id', 'unknown')
                file_name = chunk_metadata.get('file_name', 'Unknown')
                chunk_index = chunk_metadata.get('chunk_index', 0)
                
                context_parts.append(f"[{i}] {chunk_text}")
                
                citations.append({
                    'citation_id': i,
                    'document_id': document_id,
                    'file_name': file_name,
                    'chunk_index': chunk_index,
                    'chunk_text': chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
                    'similarity': chunk.get('similarity', 0.0),
                    'metadata': chunk_metadata
                })
            
            if not context_parts:
                return {
                    'answer': "No valid content found in retrieved chunks.",
                    'citations': [],
                    'chunks': []
                }
            
            context = "\n\n".join(context_parts)
            
            # Generate answer using LLM
            answer = self.hf_client.generate_response(question, context)
            
            return {
                'answer': answer,
                'citations': citations,
                'chunks': retrieved_chunks
            }
        except Exception as e:
            # Re-raise with more context
            raise ValueError(f"Error in RAG query: {str(e)}") from e
    
    def get_document(self, document_id: str) -> Optional[Dict]:
        """Get full document by ID"""
        return self.retriever.get_document(document_id)
    
    def get_all_documents(self) -> List[Dict]:
        """Get all stored documents"""
        return self.retriever.get_all_documents()
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its chunks"""
        return self.retriever.delete_document(document_id)

