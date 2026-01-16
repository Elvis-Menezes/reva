"""
Configuration Module for Intent Ingestion Pipeline

This module handles:
- Environment variable loading
- Configuration validation
- Default values for optional settings

Environment Variables Required:
- QDRANT_URL: URL of your Qdrant instance (e.g., https://xxx.qdrant.io)
- QDRANT_API_KEY: API key for Qdrant authentication

Optional Environment Variables:
- QDRANT_COLLECTION_NAME: Name of the collection (default: "intent_knowledge_base")
- EMBEDDING_MODEL: HuggingFace model name (default: "all-MiniLM-L6-v2")
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
# Looks in current directory and parent directories
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)


class ConfigurationError(Exception):
    """Raised when required configuration is missing."""
    pass


class Config:
    """
    Central configuration class for the ingestion pipeline.
    
    All secrets and configurable values are loaded from environment variables.
    This enables:
    - Security: No hardcoded secrets
    - Flexibility: Different configs for dev/staging/prod
    - Portability: Easy deployment across environments
    """
    
    # =========================================================================
    # QDRANT CONFIGURATION
    # =========================================================================
    
    # Required: Qdrant server URL
    # Example: https://your-cluster-url.qdrant.io or http://localhost:6333
    QDRANT_URL: str = os.getenv("QDRANT_URL")
    
    # Required: Qdrant API key for authentication
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY")
    
    # Optional: Collection name (default provided)
    QDRANT_COLLECTION_NAME: str = os.getenv(
        "QDRANT_COLLECTION_NAME", 
        "intent_knowledge_base"
    )
    
    # =========================================================================
    # EMBEDDING MODEL CONFIGURATION
    # =========================================================================
    
    # Optional: Embedding model from HuggingFace
    # Default: all-MiniLM-L6-v2 (384 dimensions, fast, good for intent matching)
    # 
    # Alternative models optimized for semantic similarity:
    # - "sentence-transformers/all-mpnet-base-v2" (768 dim, more accurate)
    # - "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" (multilingual)
    # - "BAAI/bge-small-en-v1.5" (384 dim, strong performance)
    # - "intfloat/e5-small-v2" (384 dim, requires "query: " prefix)
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # =========================================================================
    # OPENAI CONFIGURATION (Optional - for future use)
    # =========================================================================
    
    # If you want to use OpenAI embeddings instead of sentence-transformers
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    @classmethod
    def validate(cls) -> None:
        """
        Validate that all required configuration is present.
        
        Raises:
            ConfigurationError: If required variables are missing
        """
        missing = []
        
        if not cls.QDRANT_URL:
            missing.append("QDRANT_URL")
        
        if not cls.QDRANT_API_KEY:
            missing.append("QDRANT_API_KEY")
        
        if missing:
            raise ConfigurationError(
                f"Missing required environment variables: {', '.join(missing)}\n"
                f"Please add them to your .env file:\n"
                f"  QDRANT_URL=https://your-cluster.qdrant.io\n"
                f"  QDRANT_API_KEY=your-api-key"
            )
    
    @classmethod
    def show(cls) -> dict:
        """
        Return a dictionary of configuration (with secrets masked).
        
        Useful for debugging and logging.
        """
        return {
            "QDRANT_URL": cls.QDRANT_URL,
            "QDRANT_API_KEY": f"{cls.QDRANT_API_KEY[:8]}..." if cls.QDRANT_API_KEY else None,
            "QDRANT_COLLECTION_NAME": cls.QDRANT_COLLECTION_NAME,
            "EMBEDDING_MODEL": cls.EMBEDDING_MODEL,
            "OPENAI_API_KEY": f"{cls.OPENAI_API_KEY[:8]}..." if cls.OPENAI_API_KEY else None,
        }


# Validate on import (optional - can be disabled for testing)
# Uncomment the line below to enforce validation at import time
# Config.validate()
