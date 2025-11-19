#!/bin/bash

# Run the NotebookLM RAG Web App

echo "Starting NotebookLM RAG Web App..."
echo "Make sure HUGGINGFACE_API_KEY is set in your environment or .env file"
echo ""

# Check if virtual environment exists and activate it
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Run the app
python app.py

