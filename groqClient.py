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

load_dotenv()


class HuggingFaceClient:
    """Client for Hugging Face API - handles both embeddings and LLM calls."""
    
    def __init__(
        self, 
        huggingface_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "llama-3.1-8b-instant"
    ):
        """
        Initialize Hugging Face client for embeddings and Groq client for LLM
        
        Args:
            huggingface_api_key: Hugging Face API token (if None, reads from HUGGINGFACE_API_KEY env var)
            groq_api_key: Groq API token (if None, reads from GROQ_API_KEY env var)
            embedding_model: Hugging Face embedding model name (default: sentence-transformers/all-MiniLM-L6-v2)
            llm_model: Groq LLM model name (default: llama-3.1-8b-instant)
        """
        # Initialize Hugging Face Inference API for embeddings
        self.huggingface_api_key = huggingface_api_key or os.getenv("HUGGINGFACE_API_KEY")
        if not self.huggingface_api_key:
            raise ValueError(
                "HUGGINGFACE_API_KEY not set. Get a free API token at: https://huggingface.co/settings/tokens\n"
                "Then set it: export HUGGINGFACE_API_KEY=your_token_here"
            )
        
        # Initialize Groq API for LLM
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY not set. Get an API key from Groq.\n"
                "Then set it: export GROQ_API_KEY=your_token_here"
            )
        
        self.embedding_model_name = embedding_model
        self.llm_model = llm_model
        
        # Initialize Groq client
        if GROQ_AVAILABLE:
            self.groq_client = Groq(api_key=self.groq_api_key)
            print(f"Using Groq API for LLM: {llm_model}")
        else:
            raise ValueError(
                "Groq package not installed. Install it with: pip install groq\n"
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
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            if self.use_hf_client:
                # Use huggingface_hub InferenceClient (recommended)
                embeddings = []
                for text in texts:
                    try:
                        # Use feature_extraction method for embedding models
                        embedding = self.embedding_client.feature_extraction(text)
                        
                        # Handle different response formats
                        if isinstance(embedding, list):
                            # If it's a nested list, flatten or take first element
                            if len(embedding) > 0 and isinstance(embedding[0], (list, tuple)):
                                # Nested list - take first element or flatten
                                embedding = embedding[0] if len(embedding) == 1 else embedding
                            else:
                                # Already flat list
                                embedding = embedding
                        
                        # Convert numpy arrays to lists
                        if hasattr(embedding, 'tolist'):
                            embedding = embedding.tolist()
                        elif isinstance(embedding, (list, tuple)) and len(embedding) > 0:
                            # Convert nested numpy arrays
                            embedding = [e.tolist() if hasattr(e, 'tolist') else e for e in embedding]
                        
                        embeddings.append(embedding)
                    except AttributeError:
                        # If feature_extraction doesn't exist, fall back to direct API
                        return self._get_embeddings_direct_api(texts)
                    except Exception as e:
                        # Try direct API as fallback
                        try:
                            return self._get_embeddings_direct_api(texts)
                        except Exception as e2:
                            raise ValueError(f"Hugging Face API error: {str(e)} (fallback also failed: {str(e2)})")
                return embeddings
            else:
                # Fallback to direct API calls
                return self._get_embeddings_direct_api(texts)
        except Exception as e:
            raise ValueError(f"Error generating embeddings: {str(e)}")
    
    def _get_embeddings_direct_api(self, texts: List[str]) -> List[List[float]]:
        """Fallback method using direct API calls"""
        import requests
        embeddings = []
        for text in texts:
            # Use new router endpoint format for embeddings
            api_url = f"https://router.huggingface.co/hf-inference/{self.embedding_model_name}"
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
                    api_url = f"https://router.huggingface.co/hf-inference/{self.embedding_model_name}"
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
        temperature: float = 0.0
    ) -> str:
        """
        Generate LLM response using Groq API
        
        Args:
            question: User question
            context: Retrieved context from documents
            model: LLM model name (defaults to self.llm_model)
            temperature: Temperature for generation
            
        Returns:
            Generated response text
        """
        model = model or self.llm_model
        
        prompt = f"""Use the following context to answer the question. If you don't know the answer based on the context, say that you don't know.

Context:
{context}

Question: {question}

Answer:"""
        
        try:
            # Use Groq API
            response = self.groq_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on the provided context."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_tokens=1000
            )
            
            # Extract the response text
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content.strip()
            else:
                raise ValueError("No response generated from Groq API")
                
        except Exception as e:
            raise ValueError(f"Error generating response from Groq API: {str(e)}")

