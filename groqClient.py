"""
Hugging Face Client Module
Handles all Hugging Face API interactions for embeddings and Groq API for LLM
"""

import os
import time
from typing import List, Optional
from dotenv import load_dotenv

# Try to use huggingface_hub InferenceClient (recommended)
try:
    from huggingface_hub import InferenceClient
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    import requests

# Try to use Groq SDK
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Try to import numpy for type checking
try:
    import numpy as np
    NP_AVAILABLE = True
except ImportError:
    NP_AVAILABLE = False
    np = None

load_dotenv()


class HuggingFaceClient:
    """Client for Hugging Face API - handles both embeddings and LLM calls."""
    
    def __init__(
        self, 
        huggingface_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "nvidia/nemotron-nano-12b-v2-vl:free"
    ):
        """
        Initialize Hugging Face client for embeddings and Groq/OpenRouter client for LLM
        
        Args:
            huggingface_api_key: Hugging Face API token (if None, reads from HUGGINGFACE_API_KEY env var)
            groq_api_key: Groq API token (if None, reads from GROQ_API_KEY env var)
            openrouter_api_key: OpenRouter API token (if None, reads from OPENROUTER_API_KEY env var)
            embedding_model: Hugging Face embedding model name (default: sentence-transformers/all-MiniLM-L6-v2)
            llm_model: LLM model name (default: llama-3.1-8b-instant)
        """
        # Initialize Hugging Face Inference API for embeddings
        self.huggingface_api_key = huggingface_api_key or os.getenv("HUGGINGFACE_API_KEY")
        if not self.huggingface_api_key:
            raise ValueError(
                "HUGGINGFACE_API_KEY not set. Get a free API token at: https://huggingface.co/settings/tokens\n"
                "Then set it: export HUGGINGFACE_API_KEY=your_token_here"
            )
        
        # Initialize LLM Client (OpenRouter or Groq)
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        
        self.embedding_model_name = embedding_model
        self.llm_model = llm_model
        
        self.use_openrouter = False
        
        if self.openrouter_api_key:
            self.use_openrouter = True
            print(f"Using OpenRouter API for LLM: {llm_model}")
            if self.llm_model == "llama-3.1-8b-instant" or self.llm_model == "meta-llama/llama-3.1-8b-instruct:free": 
                 self.llm_model = "nvidia/llama-3.1-nemotron-70b-instruct:free" 
                 print(f"Switched default model to OpenRouter equivalent: {self.llm_model}")
        
        elif self.groq_api_key:
            # Initialize Groq client
            if GROQ_AVAILABLE:
                self.groq_client = Groq(api_key=self.groq_api_key)
                print(f"Using Groq API for LLM: {llm_model}")
            else:
                raise ValueError(
                    "Groq package not installed. Install it with: pip install groq\n"
                )
        else:
            raise ValueError(
                "No LLM API key set. Please set either GROQ_API_KEY or OPENROUTER_API_KEY.\n"
            )
        
        # Use huggingface_hub InferenceClient for embeddings if available
        if HF_HUB_AVAILABLE:
            try:
                # Initialize embedding client
                self.embedding_client = InferenceClient(
                    model=embedding_model,
                    token=self.huggingface_api_key
                )
                self.use_hf_client = True
                print(f"Using Hugging Face InferenceClient for embeddings: {embedding_model}")
            except Exception as e:
                print(f"Warning: Could not initialize InferenceClient, falling back to direct API: {e}")
                self.use_hf_client = False
                self._setup_direct_api()
        else:
            self.use_hf_client = False
            self._setup_direct_api()
        
        # Always set up direct API URLs as fallback for embeddings (even if using InferenceClient)
        if not hasattr(self, 'embedding_api_url'):
            self._setup_direct_api()
    
    def _setup_direct_api(self):
        """Setup direct API calls as fallback for embeddings"""
        # Use new router endpoint format for embeddings only
        self.embedding_api_url = f"https://router.huggingface.co/hf-inference/{self.embedding_model_name}"
        self.api_headers = {"Authorization": f"Bearer {self.huggingface_api_key}"}
        print(f"Using Hugging Face direct API for embeddings: {self.embedding_model_name}")
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts using Hugging Face Inference API
        
        Uses Hugging Face's free Inference API for embeddings.
        Optimized to batch requests when possible for better performance.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (guaranteed to match length of texts)
        """
        if not texts:
            return []
        
        try:
            if self.use_hf_client:
                # Use huggingface_hub InferenceClient (recommended)
                # Try batch processing first (faster for multiple texts)
                try:
                    # Attempt batch processing - send all texts at once
                    batch_embeddings = self.embedding_client.feature_extraction(texts)
                    
                    # Handle batch response format
                    # The API might return different formats, so we need to handle them carefully
                    processed_embeddings = self._process_batch_embeddings(batch_embeddings, len(texts))
                    
                    # Validate we got the right number of embeddings
                    if len(processed_embeddings) != len(texts):
                        # Count mismatch - fall back to individual requests
                        raise ValueError(f"Batch embedding count mismatch: expected {len(texts)}, got {len(processed_embeddings)}")
                    
                    return processed_embeddings
                        
                except (AttributeError, TypeError, ValueError) as e:
                    # Batch processing failed, fall back to individual requests
                    # This handles cases where the API doesn't support batching or returns wrong format
                    print(f"Batch embedding failed ({e}), falling back to individual requests")
                    return self._get_embeddings_individual(texts)
            else:
                # Fallback to direct API calls
                return self._get_embeddings_direct_api(texts)
        except Exception as e:
            raise ValueError(f"Error generating embeddings: {str(e)}")
    
    def _get_embeddings_individual(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings by making individual requests (fallback method)
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            try:
                embedding = self.embedding_client.feature_extraction(text)
                processed = self._process_embedding_response(embedding)
                embeddings.append(processed)
            except Exception as e:
                # If individual requests fail, try direct API
                print(f"Individual embedding request failed, trying direct API: {e}")
                return self._get_embeddings_direct_api(texts)
        
        # Validate count before returning
        if len(embeddings) != len(texts):
            raise ValueError(f"Embedding count mismatch: expected {len(texts)}, got {len(embeddings)}")
        
        return embeddings
    
    def _process_batch_embeddings(self, batch_response, expected_count: int) -> List[List[float]]:
        """
        Process batch embedding response from API
        
        Args:
            batch_response: Raw batch response (can be various formats)
            expected_count: Expected number of embeddings
            
        Returns:
            List of embedding vectors
        """
        processed_embeddings = []
        
        # Handle different batch response formats
        if isinstance(batch_response, list):
            if len(batch_response) == 0:
                raise ValueError("Empty batch response received")
            
            # Check if it's a list of embeddings (proper batch)
            first_item = batch_response[0]
            is_numpy_array = NP_AVAILABLE and isinstance(first_item, np.ndarray)
            is_list_of_lists = isinstance(first_item, (list, tuple)) and len(first_item) > 0
            
            if is_numpy_array or is_list_of_lists:
                # This is a proper batch response - list of embeddings
                for embedding in batch_response:
                    processed_emb = self._process_embedding_response(embedding)
                    processed_embeddings.append(processed_emb)
            else:
                # Might be a single embedding or different format
                # Try processing the whole thing as a single response
                processed = self._process_embedding_response(batch_response)
                # If it's a nested list, it might contain multiple embeddings
                if isinstance(processed, list) and len(processed) > 0:
                    if isinstance(processed[0], (list, tuple)):
                        # Nested structure - each element is an embedding
                        processed_embeddings = processed
                    else:
                        # Single embedding
                        processed_embeddings = [processed]
                else:
                    processed_embeddings = [processed]
        else:
            # Single embedding returned (shouldn't happen with batch)
            processed = self._process_embedding_response(batch_response)
            processed_embeddings = [processed]
        
        return processed_embeddings
    
    def _process_embedding_response(self, embedding, max_depth: int = 10) -> List[float]:
        """
        Process embedding response from API into a flat list of floats
        
        Args:
            embedding: Raw embedding response (can be various formats)
            max_depth: Maximum recursion depth to prevent infinite loops
            
        Returns:
            List of floats representing the embedding vector
        """
        if max_depth <= 0:
            raise ValueError("Maximum recursion depth exceeded while processing embedding response")
        
        # Handle numpy arrays first
        if NP_AVAILABLE and isinstance(embedding, np.ndarray):
            # Flatten if needed
            if embedding.ndim > 1:
                embedding = embedding.flatten()
            result = embedding.tolist()
            # Ensure it's flat (no nested lists)
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], (list, tuple)):
                return self._process_embedding_response(result, max_depth - 1)
            return result
        
        # Convert to list if it's a tuple
        if isinstance(embedding, tuple):
            embedding = list(embedding)
        
        # Handle lists
        if isinstance(embedding, list):
            # If empty, return empty list
            if len(embedding) == 0:
                return []
            
            # Check if first element is a list/tuple (nested structure)
            first_elem = embedding[0]
            
            # If first element is a number (int/float), this is already a flat list
            if isinstance(first_elem, (int, float)):
                # Already flat - just ensure all elements are floats
                return [float(x) for x in embedding]
            
            # If first element is a list/tuple/numpy array, we have nested structure
            if isinstance(first_elem, (list, tuple)) or (NP_AVAILABLE and isinstance(first_elem, np.ndarray)):
                # If it's a single nested element, unwrap it
                if len(embedding) == 1:
                    # Process the single nested element with reduced depth
                    return self._process_embedding_response(first_elem, max_depth - 1)
                else:
                    # Multiple nested elements - this shouldn't happen for a single embedding
                    # But if it does, take the first one
                    return self._process_embedding_response(first_elem, max_depth - 1)
        
        # Handle other types - try to convert to list
        if hasattr(embedding, 'tolist'):
            result = embedding.tolist()
            # If result is still nested, process it with reduced depth
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], (list, tuple)):
                return self._process_embedding_response(result, max_depth - 1)
            return result
        
        # If we get here, try to convert to float directly
        try:
            return [float(embedding)]
        except (ValueError, TypeError):
            raise ValueError(f"Unable to process embedding response: {type(embedding)}")
    
    def _get_embeddings_direct_api(self, texts: List[str]) -> List[List[float]]:
        """
        Fallback method using direct API calls
        Optimized to batch requests when possible
        """
        import requests
        
        # Try batch request first (faster for multiple texts)
        api_url = f"https://router.huggingface.co/hf-inference/{self.embedding_model_name}"
        
        try:
            # Attempt batch request - send all texts at once
            response = requests.post(
                api_url,
                headers=self.api_headers,
                json={"inputs": texts},  # Send list of texts for batch processing
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                # Handle batch response - should be a list of embeddings
                if isinstance(result, list):
                    # Process batch response
                    processed_batch = self._process_batch_embeddings(result, len(texts))
                    if len(processed_batch) == len(texts):
                        return processed_batch
                    # If count doesn't match, fall through to individual requests
                # If response format is unexpected, fall through to individual requests
            elif response.status_code == 503:
                # Model is loading, wait a bit and retry
                time.sleep(5)
                response = requests.post(
                    api_url,
                    headers=self.api_headers,
                    json={"inputs": texts},
                    timeout=30
                )
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list):
                        processed_batch = self._process_batch_embeddings(result, len(texts))
                        if len(processed_batch) == len(texts):
                            return processed_batch
        except (requests.RequestException, ValueError, KeyError):
            # Batch request failed or unsupported, fall back to individual requests
            pass
        
        # Fallback to individual requests if batch fails
        embeddings = []
        for text in texts:
            response = requests.post(
                api_url,
                headers=self.api_headers,
                json={"inputs": text},
                timeout=30
            )
            
            if response.status_code != 200:
                error_msg = response.text
                if response.status_code == 503:
                    # Model is loading, wait a bit
                    time.sleep(5)
                    response = requests.post(
                        api_url,
                        headers=self.api_headers,
                        json={"inputs": text},
                        timeout=30
                    )
                    if response.status_code != 200:
                        raise ValueError(f"Hugging Face API error: {response.status_code} - {response.text}")
                else:
                    raise ValueError(f"Hugging Face API error: {response.status_code} - {error_msg}")
            
            result = response.json()
            # Handle different response formats
            if isinstance(result, list):
                embedding = result[0] if result else []
            elif isinstance(result, dict) and "embedding" in result:
                embedding = result["embedding"]
            else:
                embedding = result
            
            embeddings.append(embedding)
        
        return embeddings
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector
        """
        return self.get_embeddings([text])[0]
    
    def generate_response(
        self,
        question: str,
        context: str,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> str:
        """
        Generate LLM response using Groq API.

        Args:
            question: User question.
            context: Retrieved context from documents.
            model: LLM model name (defaults to self.llm_model).
            temperature: Temperature for generation.

        Returns:
            Generated response text.
        """
    
        model = model or self.llm_model

        prompt = fprompt = f"""
                                You are an expert AI assistant answering questions using ONLY the context provided.

                                --------------------
                                CONTEXT:
                                {context}
                                --------------------

                                GUIDELINES:
                                - Use the context as the single source of truth.
                                - If the context does not contain the answer, say: 
                                "The document does not provide enough information to answer this."
                                - Do NOT hallucinate or invent facts.
                                - When multiple sources disagree, note the discrepancy.
                                - Provide a clear, concise, well-structured answer.
                                - Include a short summary at the end.
                                - Maintain factual accuracy and stay grounded in the retrieved chunks.

                                QUESTION:
                                {question}

                                Now provide the best possible answer.
                            """


        try:
            if self.use_openrouter:
                import requests
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.openrouter_api_key}",
                        "HTTP-Referer": "http://localhost:8000", # Optional, for including your app on openrouter.ai rankings
                        "X-Title": "NotebookLM RAG", # Optional
                    },
                    json={
                        "model": model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a helpful assistant that answers questions based on provided document context. Answer directly and concisely. Do not list page numbers or sources unless specifically asked. If the context doesn't contain relevant information, simply say you don't have that information in the provided context."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "temperature": max(temperature, 0.1),
                        "max_tokens": 1000,
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get('choices') and len(result['choices']) > 0:
                        return result['choices'][0]['message']['content'].strip()
                    else:
                        raise ValueError(f"Invalid response from OpenRouter: {result}")
                else:
                    raise ValueError(f"OpenRouter API error: {response.status_code} - {response.text}")

            else:
                # Groq API call
                response = self.groq_client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that answers questions based on provided document context. Answer directly and concisely. Do not list page numbers or sources unless specifically asked. If the context doesn't contain relevant information, simply say you don't have that information in the provided context."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=max(temperature, 0.1),  # Minimum 0.1 for more natural responses
                    max_tokens=1000,
                )

                # Extract response
                if response.choices and len(response.choices) > 0:
                    return response.choices[0].message.content.strip()

            raise ValueError("No response generated from API.")

        except Exception as e:
            raise ValueError(f"Error generating response from LLM API: {e}")
