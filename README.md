# NotebookLM RAG Web App

A RAG (Retrieval-Augmented Generation) web application similar to Google NotebookLM, built with FastAPI and using Hugging Face Inference API for both embeddings and LLM responses.

## Features

- **Three-Pane Layout:**

  - **Left Pane:** List of uploaded documents
  - **Center Pane:** Chat/conversation interface with follow-up questions
  - **Right Pane:** Document content viewer with citation linking

- **Document Upload:**

  - Upload PDF, DOCX, and TXT files directly from the browser
  - Real-time upload and indexing progress indicators
  - Automatic text extraction and chunking

- **RAG Pipeline:**
  - Uses Hugging Face Inference API for embeddings (free API)
  - Uses Hugging Face Inference API for LLM responses (free API)
  - Local vector storage (no external vector DBs like Pinecone, Weaviate, Chroma, FAISS)
  - Semantic search and retrieval
  - Citations linking to source document chunks

**Note:** Uses Hugging Face's free Inference API for both embeddings and LLM responses.

## Architecture

The application follows a clean, modular structure:

- **fileParser.py** - Handles document parsing (PDF, DOCX, TXT)
- **chunker.py** - Splits text into manageable chunks
- **groqClient.py** - All Hugging Face API interactions (embeddings & LLM)
- **retriever.py** - Vector storage and retrieval
- **ragPipeline.py** - Coordinates the end-to-end RAG process
- **app.py** - FastAPI backend with REST endpoints
- **static/** - Frontend HTML, CSS, and JavaScript

## Installation

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Set up Hugging Face API token:**

Get a free API token from [Hugging Face Settings](https://huggingface.co/settings/tokens)

```bash
export HUGGINGFACE_API_KEY=your_huggingface_token_here
```

Or create a `.env` file:

```
HUGGINGFACE_API_KEY=your_huggingface_token_here
```

## Usage

1. **Start the server:**

```bash
python app.py
```

Or using uvicorn directly:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

2. **Open in browser:**

Navigate to `http://localhost:8000`

3. **Upload documents:**

- Click the "+ Upload" button in the left pane
- Select PDF, DOCX, or TXT files
- Wait for upload and indexing to complete

4. **Ask questions:**

- Type your question in the chat input
- The bot will retrieve relevant chunks and generate an answer
- Click on citations to view the source chunks in the right pane

## API Endpoints

- `GET /` - Serve the main web interface
- `POST /api/upload` - Upload and process a document
- `POST /api/chat` - Send a chat message and get response
- `GET /api/documents` - Get list of all uploaded documents
- `GET /api/documents/{document_id}` - Get full document content
- `GET /api/health` - Health check endpoint

## Configuration

You can modify the following in `ragPipeline.py`:

- `chunk_size` - Size of text chunks (default: 1000 characters)
- `chunk_overlap` - Overlap between chunks (default: 200 characters)
- `storage_path` - Path for vector storage

In `groqClient.py`:

- `embedding_model` - Hugging Face embedding model name (default: "sentence-transformers/all-MiniLM-L6-v2")
- `llm_model` - Hugging Face LLM model name (default: "mistralai/Mistral-7B-Instruct-v0.2")

## Requirements

- Python 3.8+
- Hugging Face API token (free at https://huggingface.co/settings/tokens)
- See `requirements.txt` for Python dependencies

## Notes

- Documents are stored locally in `./vectorstore/`
- Uploaded files are temporarily stored in `./uploads/` during processing
- The vector store persists between sessions
- Hugging Face API has rate limits on free tier - suitable for development/testing

## Troubleshooting

**Error: HUGGINGFACE_API_KEY not set**

- Make sure you've set the API token as an environment variable or in a `.env` file
- Get a free token at: https://huggingface.co/settings/tokens
- Note: Free tier has rate limits, suitable for development/testing

**Upload fails**

- Check file format (PDF, DOCX, TXT, MD only)
- Ensure file is not corrupted or password-protected
- Check server logs for detailed error messages

## License

This project is provided as-is for educational and development purposes.
