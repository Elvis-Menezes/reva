"""
Intent-Based Knowledge Base Ingestion Pipeline for Qdrant

This module provides a production-ready pipeline for:
1. Loading structured intent-based knowledge bases
2. Constructing semantic embedding text from intent groups
3. Generating normalized embeddings using sentence transformers
4. Storing vectors in Qdrant with proper metadata

Architecture Principles:
- Each intent object = one atomic vector (no token-based splitting)
- Clean separation of concerns (loading, chunking, embedding, storage)
- Framework-agnostic design (works with LangChain, Parlant, custom agents)
- Environment-based configuration for secrets
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class IntentChunk:
    """
    Represents a single intent unit from the knowledge base.
    
    Each IntentChunk is treated as ONE atomic vector - never split.
    This preserves:
    - Intent phrasing (questions/paraphrases)
    - Authoritative resolution (answer)
    - Journey context (topic/type)
    """
    chunk_id: str
    topic: str
    questions: list[str]
    answer: str
    metadata: dict[str, Any]
    
    @property
    def intent_count(self) -> int:
        """Number of intent variations (questions) in this chunk."""
        return len(self.questions)
    
    def to_embedding_text(self) -> str:
        """
        Construct the semantic embedding string for this intent.
        
        Format (MUST follow exactly):
        Category: <topic>
        
        User intents:
        - <question 1>
        - <question 2>
        ...
        
        Resolved answer:
        <answer text>
        
        Why this format:
        - Category provides domain context for the embedding
        - User intents capture all paraphrases (critical for intent matching)
        - Resolved answer anchors the semantic meaning
        - This holistic representation enables accurate cosine similarity search
        """
        intent_lines = "\n".join(f"- {q}" for q in self.questions)
        
        embedding_text = f"""Category: {self.topic}

User intents:
{intent_lines}

Resolved answer:
{self.answer}"""
        
        return embedding_text
    
    def to_payload(self) -> dict[str, Any]:
        """
        Generate the Qdrant payload for this chunk.
        
        Payload includes the answer and questions for RAG retrieval.
        This enables the LLM to access the full context when responding.
        """
        return {
            "chunk_id": self.chunk_id,
            "topic": self.topic,
            "questions": self.questions,  # Store questions for context
            "answer": self.answer,  # Store answer for RAG retrieval
            "intent_count": self.intent_count,
            "type": "intent_qa",
            "source": "knowledge_base",
            # Store additional metadata if present
            **{k: v for k, v in self.metadata.items() if v}
        }


# ============================================================================
# LOADING MODULE
# ============================================================================

class KnowledgeBaseLoader:
    """
    Handles loading and parsing of structured knowledge base files.
    
    Responsibilities:
    - File I/O and validation
    - JSON parsing
    - Conversion to IntentChunk objects
    """
    
    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        
    def load(self) -> list[IntentChunk]:
        """
        Load knowledge base from JSON file.
        
        Returns:
            List of IntentChunk objects, each representing one atomic intent unit.
            
        Raises:
            FileNotFoundError: If knowledge base file doesn't exist
            json.JSONDecodeError: If file contains invalid JSON
            ValueError: If JSON structure is invalid
        """
        logger.info(f"Loading knowledge base from: {self.file_path}")
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"Knowledge base not found: {self.file_path}")
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("Knowledge base must be a JSON array of intent objects")
        
        chunks = []
        for idx, item in enumerate(data):
            try:
                chunk = self._parse_intent_item(item, idx)
                chunks.append(chunk)
            except (KeyError, TypeError) as e:
                logger.warning(f"Skipping invalid item at index {idx}: {e}")
                continue
        
        logger.info(f"Loaded {len(chunks)} intent chunks from knowledge base")
        return chunks
    
    def _parse_intent_item(self, item: dict, index: int) -> IntentChunk:
        """
        Parse a single JSON object into an IntentChunk.
        
        Validates required fields and provides sensible defaults.
        """
        # Required fields
        if "chunk_id" not in item:
            raise KeyError(f"Missing 'chunk_id' at index {index}")
        if "questions" not in item or not item["questions"]:
            raise KeyError(f"Missing or empty 'questions' at index {index}")
        if "answer" not in item:
            raise KeyError(f"Missing 'answer' at index {index}")
        
        return IntentChunk(
            chunk_id=str(item["chunk_id"]),
            topic=str(item.get("topic", "General")),
            questions=list(item["questions"]),
            answer=str(item["answer"]),
            metadata=dict(item.get("metadata", {}))
        )


# ============================================================================
# EMBEDDING MODULE
# ============================================================================

class IntentEmbedder:
    """
    Generates normalized embeddings for intent chunks.
    
    Design decisions:
    - Uses sentence-transformers for high-quality semantic embeddings
    - Normalizes vectors for optimal cosine similarity performance
    - Batch processing for efficiency
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize the embedding model.
        
        Args:
            model_name: HuggingFace model identifier. Defaults to a model
                       optimized for semantic similarity and intent matching.
        """
        self.model_name = model_name or Config.EMBEDDING_MODEL
        logger.info(f"Loading embedding model: {self.model_name}")
        
        self.model = SentenceTransformer(self.model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Embedding model loaded. Vector dimension: {self.dimension}")
    
    def embed_chunks(
        self, 
        chunks: list[IntentChunk], 
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of intent chunks.
        
        Args:
            chunks: List of IntentChunk objects
            batch_size: Number of chunks to embed at once
            show_progress: Whether to display progress bar
            
        Returns:
            Normalized embedding matrix of shape (n_chunks, embedding_dim)
        """
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        # Construct embedding texts using the exact format specified
        texts = [chunk.to_embedding_text() for chunk in chunks]
        
        # Generate embeddings with batch processing
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # Critical for cosine similarity
        )
        
        # Verify normalization (should be ~1.0 for all vectors)
        norms = np.linalg.norm(embeddings, axis=1)
        logger.debug(f"Embedding norms - min: {norms.min():.4f}, max: {norms.max():.4f}")
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query string.
        
        Used for retrieval queries at inference time.
        """
        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding


# ============================================================================
# STORAGE MODULE
# ============================================================================

class QdrantStorage:
    """
    Handles all interactions with the Qdrant vector database.
    
    Design principles:
    - Collection management (create/verify)
    - Batch upsert for efficiency
    - Proper error handling and logging
    - Framework-agnostic (can be queried by LangChain, Parlant, etc.)
    """
    
    def __init__(
        self,
        url: str = None,
        api_key: str = None,
        collection_name: str = None
    ):
        """
        Initialize Qdrant client connection.
        
        Args:
            url: Qdrant server URL (defaults to Config)
            api_key: API key for authentication (defaults to Config)
            collection_name: Name of the vector collection (defaults to Config)
        """
        self.url = url or Config.QDRANT_URL
        self.api_key = api_key or Config.QDRANT_API_KEY
        self.collection_name = collection_name or Config.QDRANT_COLLECTION_NAME
        
        logger.info(f"Connecting to Qdrant at: {self.url}")
        
        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
            timeout=60
        )
        
        # Verify connection
        try:
            self.client.get_collections()
            logger.info("Successfully connected to Qdrant")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    def ensure_collection(self, vector_size: int, recreate: bool = False) -> None:
        """
        Create or verify the vector collection exists with correct configuration.
        
        Args:
            vector_size: Dimension of the embedding vectors
            recreate: If True, delete and recreate collection (WARNING: data loss)
        """
        collection_exists = self._collection_exists()
        
        if recreate and collection_exists:
            logger.warning(f"Recreating collection: {self.collection_name}")
            self.client.delete_collection(self.collection_name)
            collection_exists = False
        
        if not collection_exists:
            logger.info(f"Creating collection: {self.collection_name} with dimension {vector_size}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE  # Optimized for normalized embeddings
                )
            )
            logger.info(f"Collection '{self.collection_name}' created successfully")
        else:
            # Verify existing collection has correct vector size
            collection_info = self.client.get_collection(self.collection_name)
            existing_size = collection_info.config.params.vectors.size
            
            if existing_size != vector_size:
                raise ValueError(
                    f"Collection vector size mismatch. "
                    f"Expected: {vector_size}, Found: {existing_size}. "
                    f"Use recreate=True to fix this."
                )
            
            logger.info(f"Using existing collection: {self.collection_name}")
    
    def _collection_exists(self) -> bool:
        """Check if the collection already exists."""
        collections = self.client.get_collections().collections
        return any(c.name == self.collection_name for c in collections)
    
    def upsert_chunks(
        self,
        chunks: list[IntentChunk],
        embeddings: np.ndarray,
        batch_size: int = 100
    ) -> int:
        """
        Insert or update intent chunks with their embeddings.
        
        Args:
            chunks: List of IntentChunk objects
            embeddings: Corresponding embedding matrix
            batch_size: Number of points to upsert at once
            
        Returns:
            Number of points successfully upserted
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks and embeddings count mismatch: "
                f"{len(chunks)} chunks, {len(embeddings)} embeddings"
            )
        
        logger.info(f"Upserting {len(chunks)} points to collection '{self.collection_name}'")
        
        # Prepare points for upsert
        points = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point = PointStruct(
                id=idx,  # Using index as ID; could also hash chunk_id
                vector=embedding.tolist(),
                payload=chunk.to_payload()
            )
            points.append(point)
        
        # Batch upsert for efficiency
        total_upserted = 0
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=True
            )
            total_upserted += len(batch)
            logger.debug(f"Upserted batch {i // batch_size + 1}: {len(batch)} points")
        
        logger.info(f"Successfully upserted {total_upserted} points")
        return total_upserted
    
    def get_collection_info(self) -> dict:
        """Get information about the collection."""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "vectors_count": info.points_count,
            "points_count": info.points_count,
            "status": info.status
        }


# ============================================================================
# PIPELINE ORCHESTRATOR
# ============================================================================

class IntentIngestionPipeline:
    """
    Orchestrates the complete ingestion pipeline.
    
    This is the main entry point that coordinates:
    1. Loading the knowledge base
    2. Constructing embedding text for each intent
    3. Generating normalized embeddings
    4. Storing vectors in Qdrant
    """
    
    def __init__(
        self,
        knowledge_base_path: str | Path,
        embedding_model: str = None,
        qdrant_url: str = None,
        qdrant_api_key: str = None,
        collection_name: str = None
    ):
        """
        Initialize the pipeline with all required components.
        
        Args:
            knowledge_base_path: Path to the JSON knowledge base file
            embedding_model: Optional custom embedding model name
            qdrant_url: Optional Qdrant URL (defaults to env var)
            qdrant_api_key: Optional Qdrant API key (defaults to env var)
            collection_name: Optional collection name (defaults to env var)
        """
        self.loader = KnowledgeBaseLoader(knowledge_base_path)
        self.embedder = IntentEmbedder(embedding_model)
        self.storage = QdrantStorage(
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=collection_name
        )
    
    def run(self, recreate_collection: bool = False) -> dict:
        """
        Execute the complete ingestion pipeline.
        
        Args:
            recreate_collection: If True, delete and recreate the collection
                               WARNING: This will delete existing data!
        
        Returns:
            Summary statistics of the ingestion run
        """
        logger.info("=" * 60)
        logger.info("Starting Intent Ingestion Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Load knowledge base
        chunks = self.loader.load()
        
        # Step 2: Generate embeddings
        embeddings = self.embedder.embed_chunks(chunks)
        
        # Step 3: Ensure collection exists with correct configuration
        self.storage.ensure_collection(
            vector_size=self.embedder.dimension,
            recreate=recreate_collection
        )
        
        # Step 4: Store vectors
        upserted_count = self.storage.upsert_chunks(chunks, embeddings)
        
        # Summary
        collection_info = self.storage.get_collection_info()
        
        summary = {
            "chunks_loaded": len(chunks),
            "embeddings_generated": len(embeddings),
            "points_upserted": upserted_count,
            "collection_name": collection_info["name"],
            "total_vectors": collection_info["vectors_count"],
            "embedding_dimension": self.embedder.dimension,
            "embedding_model": self.embedder.model_name
        }
        
        logger.info("=" * 60)
        logger.info("Ingestion Pipeline Complete")
        logger.info(f"Summary: {json.dumps(summary, indent=2)}")
        logger.info("=" * 60)
        
        return summary


# ============================================================================
# QUERY/RETRIEVAL MODULE (Bonus)
# ============================================================================

class IntentRetriever:
    """
    Retrieves relevant intent chunks based on user queries.
    
    This class demonstrates how the indexed vectors can be queried.
    Compatible with any framework (LangChain, Parlant, custom agents).
    """
    
    def __init__(
        self,
        embedding_model: str = None,
        qdrant_url: str = None,
        qdrant_api_key: str = None,
        collection_name: str = None
    ):
        self.embedder = IntentEmbedder(embedding_model)
        self.storage = QdrantStorage(
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=collection_name
        )
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> list[dict]:
        """
        Search for relevant intents based on a query.
        
        Args:
            query: User's natural language query
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0-1 for cosine)
            
        Returns:
            List of matching results with scores and payloads
        """
        logger.info(f"Searching for: '{query}'")
        
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)
        
        # Search in Qdrant
        results = self.storage.client.query_points(
            collection_name=self.storage.collection_name,
            query=query_embedding.tolist(),
            limit=top_k,
            score_threshold=score_threshold if score_threshold > 0 else None
        ).points
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "score": result.score,
                "chunk_id": result.payload.get("chunk_id"),
                "topic": result.payload.get("topic"),
                "intent_count": result.payload.get("intent_count"),
                "type": result.payload.get("type"),
                "source": result.payload.get("source")
            })
        
        logger.info(f"Found {len(formatted_results)} results")
        return formatted_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main entry point for running the ingestion pipeline.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Intent-Based Knowledge Base Ingestion Pipeline"
    )
    parser.add_argument(
        "--knowledge-base",
        type=str,
        default="heyo.txt",
        help="Path to the knowledge base JSON file"
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate the collection (WARNING: deletes existing data)"
    )
    parser.add_argument(
        "--test-query",
        type=str,
        help="Optional: Run a test query after ingestion"
    )
    
    args = parser.parse_args()
    
    # Run ingestion pipeline
    pipeline = IntentIngestionPipeline(
        knowledge_base_path=args.knowledge_base
    )
    summary = pipeline.run(recreate_collection=args.recreate)
    
    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    
    # Optional: Run test query
    if args.test_query:
        print("\n" + "=" * 60)
        print(f"TEST QUERY: '{args.test_query}'")
        print("=" * 60)
        
        retriever = IntentRetriever()
        results = retriever.search(args.test_query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"\n--- Result {i} (Score: {result['score']:.4f}) ---")
            print(f"Chunk ID: {result['chunk_id']}")
            print(f"Topic: {result['topic']}")
            print(f"Intent Count: {result['intent_count']}")


if __name__ == "__main__":
    main()
