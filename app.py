"""
FastAPI Backend for RAG Web App
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import os
import uuid
from pathlib import Path

from ragPipeline import RAGPipeline

app = FastAPI(title="NotebookLM RAG App")

# Serve static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG pipeline
rag_pipeline = RAGPipeline(
    huggingface_api_key=os.getenv("HUGGINGFACE_API_KEY"),
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Store uploaded files temporarily
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


class ChatMessage(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    document_ids: Optional[List[str]] = None


class ChatResponse(BaseModel):
    answer: str
    citations: List[dict]
    conversation_id: str


@app.get("/")
async def root():
    """Serve the main HTML page"""
    static_file = static_dir / "index.html"
    if static_file.exists():
        return FileResponse(str(static_file))
    return {"message": "NotebookLM RAG API"}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and process a document
    
    Returns document ID and processing status
    """
    # Validate file type
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ['.pdf', '.docx', '.doc', '.txt', '.md']:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Supported: PDF, DOCX, TXT, MD"
        )
    
    # Save uploaded file temporarily for processing
    temp_file_path = UPLOAD_DIR / f"{uuid.uuid4()}{file_ext}"
    
    try:
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process document (this generates a document_id)
        result = rag_pipeline.process_document(str(temp_file_path), file.filename)
        document_id = result['document_id']
        
        # Save file permanently with document ID as filename
        saved_file_path = UPLOAD_DIR / f"{document_id}{file_ext}"
        temp_file_path.rename(saved_file_path)
        
        # Update metadata to include file path
        rag_pipeline.retriever.document_store[document_id]['metadata']['saved_file_path'] = str(saved_file_path)
        rag_pipeline.retriever.document_store[document_id]['metadata']['file_extension'] = file_ext
        rag_pipeline.retriever._save_storage()  # Save to disk
        
        return {
            "success": True,
            "document_id": document_id,
            "file_name": result['file_name'],
            "chunk_count": result['chunk_count'],
            "metadata": result['metadata']
        }
    except Exception as e:
        # Clean up temp file on error
        if temp_file_path.exists():
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """
    Chat endpoint - ask questions and get answers with citations
    """
    try:
        result = rag_pipeline.query(message.question, k=4, document_ids=message.document_ids)
        
        # Generate conversation ID if not provided
        conversation_id = message.conversation_id or str(uuid.uuid4())
        
        return ChatResponse(
            answer=result['answer'],
            citations=result['citations'],
            conversation_id=conversation_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/api/documents")
async def get_documents():
    """Get list of all uploaded documents"""
    try:
        documents = rag_pipeline.get_all_documents()
        return {
            "documents": [
                {
                    "id": doc['id'],
                    "file_name": doc['metadata'].get('file_name', 'Unknown'),
                    "file_type": doc['metadata'].get('file_type', ''),
                    "text_length": doc['metadata'].get('text_length', 0),
                    "word_count": doc['metadata'].get('word_count', 0)
                }
                for doc in documents
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")


@app.get("/api/documents/{document_id}")
async def get_document(document_id: str):
    """Get full document content by ID"""
    try:
        document = rag_pipeline.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        metadata = document['metadata'].copy()
        # Get file extension from metadata or infer from file_name
        file_extension = metadata.get('file_extension', '')
        if not file_extension and metadata.get('file_name'):
            file_name = metadata['file_name']
            file_extension = Path(file_name).suffix.lower()
        
        # Check if file exists - try saved_file_path first, then try standard pattern
        saved_file_path = metadata.get('saved_file_path')
        file_exists = False
        
        if saved_file_path and Path(saved_file_path).exists():
            file_exists = True
        elif file_extension:
            # Try standard pattern: uploads/{document_id}{extension}
            potential_path = UPLOAD_DIR / f"{document_id}{file_extension}"
            if potential_path.exists():
                file_exists = True
                # Update metadata for future use
                metadata['saved_file_path'] = str(potential_path)
                metadata['file_extension'] = file_extension
                rag_pipeline.retriever.document_store[document_id]['metadata'] = metadata
                rag_pipeline.retriever._save_storage()
        
        file_url = f"/api/documents/{document_id}/file" if file_exists else None
        
        return {
            "document_id": document_id,
            "text": document['text'],
            "metadata": metadata,
            "file_url": file_url,
            "file_type": file_extension
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving document: {str(e)}")


@app.get("/api/documents/{document_id}/file")
async def get_document_file(document_id: str):
    """Serve the original document file"""
    try:
        document = rag_pipeline.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        metadata = document['metadata']
        saved_file_path = metadata.get('saved_file_path')
        
        # If saved_file_path not in metadata, try standard pattern
        file_path = None
        if saved_file_path and Path(saved_file_path).exists():
            file_path = Path(saved_file_path)
        else:
            # Try to find file by document_id and extension
            file_extension = metadata.get('file_extension', '')
            if not file_extension and metadata.get('file_name'):
                file_extension = Path(metadata['file_name']).suffix.lower()
            
            if file_extension:
                potential_path = UPLOAD_DIR / f"{document_id}{file_extension}"
                if potential_path.exists():
                    file_path = potential_path
        
        if not file_path or not file_path.exists():
            raise HTTPException(status_code=404, detail="Original file not found")
        
        file_name = metadata.get('file_name', f"document{file_path.suffix}")
        
        # Determine media type
        file_ext = file_path.suffix.lower()
        if file_ext == '.pdf':
            media_type = "application/pdf"
        elif file_ext in ['.docx', '.doc']:
            media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document" if file_ext == '.docx' else "application/msword"
        else:
            media_type = "application/octet-stream"
        
        # For inline display, use Response with proper headers instead of FileResponse with filename
        with open(file_path, "rb") as f:
            content = f.read()
        
        headers = {
            "Content-Disposition": f"inline; filename=\"{file_name}\""
        }
        
        return Response(
            content=content,
            media_type=media_type,
            headers=headers
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving file: {str(e)}")


@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and all its chunks"""
    try:
        # Get document metadata before deleting to find saved file
        document = rag_pipeline.get_document(document_id)
        saved_file_path = None
        if document:
            saved_file_path = document['metadata'].get('saved_file_path')
        
        success = rag_pipeline.delete_document(document_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete the saved file if it exists
        if saved_file_path and Path(saved_file_path).exists():
            try:
                os.remove(saved_file_path)
            except Exception as e:
                print(f"Warning: Could not delete file {saved_file_path}: {e}")
        
        return {"success": True, "message": "Document deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")


@app.post("/api/repair")
async def repair_vector_store():
    """Repair corrupted vector store"""
    try:
        rag_pipeline.retriever.repair_vector_store()
        return {
            "success": True,
            "message": "Vector store repaired successfully",
            "vectors_count": len(rag_pipeline.retriever.vectors),
            "chunks_count": len(rag_pipeline.retriever.chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error repairing vector store: {str(e)}")


@app.get("/api/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)

