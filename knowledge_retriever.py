"""
Knowledge Retriever for Parlant Agent

In Parlant, RAG is implemented via Retrievers, NOT Tools.
Retrievers run in parallel with guideline matching, reducing latency.
They provide grounding context that the agent "should know" vs. data to "load".

This module implements a proper Parlant retriever that:
1. Queries Qdrant for relevant knowledge
2. Returns structured RetrieverResult (not formatted markdown)
3. Lets the agent synthesize answers from retrieved context
"""

import logging
import time
from typing import Optional

import parlant.sdk as p
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from config import Config

# Enable DEBUG level for this module
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class KnowledgeRetriever:
    """
    Singleton retriever that queries Qdrant for intent-based knowledge.
    
    Uses SentenceTransformers for embedding and cosine similarity search.
    Returns raw semantic data - formatting is the agent's responsibility.
    """
    
    _instance: Optional["KnowledgeRetriever"] = None
    
    def __new__(cls) -> "KnowledgeRetriever":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        if self._initialized:
            logger.debug("[RETRIEVER] Already initialized, skipping")
            return
        
        total_start = time.time()
        logger.info("=" * 60)
        logger.info("[RETRIEVER] Starting KnowledgeRetriever initialization...")
        logger.info("=" * 60)
        
        # Step 1: Load SentenceTransformer model
        logger.info(f"[RETRIEVER] Step 1/3: Loading embedding model: {Config.EMBEDDING_MODEL}")
        model_start = time.time()
        try:
            self.model = SentenceTransformer(Config.EMBEDDING_MODEL)
            logger.info(f"[RETRIEVER] ✓ Model loaded in {time.time() - model_start:.2f}s")
        except Exception as e:
            logger.error(f"[RETRIEVER] ✗ Model loading FAILED after {time.time() - model_start:.2f}s: {e}")
            raise
        
        # Step 2: Connect to Qdrant
        logger.info(f"[RETRIEVER] Step 2/3: Connecting to Qdrant at {Config.QDRANT_URL}")
        qdrant_start = time.time()
        try:
            self.client = QdrantClient(
                url=Config.QDRANT_URL,
                api_key=Config.QDRANT_API_KEY,
                timeout=60
            )
            logger.info(f"[RETRIEVER] ✓ Qdrant client created in {time.time() - qdrant_start:.2f}s")
        except Exception as e:
            logger.error(f"[RETRIEVER] ✗ Qdrant connection FAILED after {time.time() - qdrant_start:.2f}s: {e}")
            raise
        
        # Step 3: Set collection name
        logger.info(f"[RETRIEVER] Step 3/3: Setting collection name: {Config.QDRANT_COLLECTION_NAME}")
        self.collection_name = Config.QDRANT_COLLECTION_NAME
        self._initialized = True
        
        logger.info("=" * 60)
        logger.info(f"[RETRIEVER] ✓ TOTAL initialization time: {time.time() - total_start:.2f}s")
        logger.info("=" * 60)
    
    def search(self, query: str, top_k: int = 5, min_score: float = 0.3) -> list[dict]:
        """
        Search Qdrant for relevant knowledge entries.
        
        Args:
            query: Natural language query from conversation context
            top_k: Maximum results to return
            min_score: Minimum cosine similarity threshold
            
        Returns:
            List of knowledge entries with topic, answer, and confidence score
        """
        search_start = time.time()
        logger.debug(f"[SEARCH] Starting search: '{query[:50]}...' (top_k={top_k}, min_score={min_score})")
        
        # Generate query embedding using same model as ingestion
        embed_start = time.time()
        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True
        ).tolist()
        logger.debug(f"[SEARCH] Embedding generated in {time.time() - embed_start:.3f}s")
        
        # Query Qdrant
        qdrant_start = time.time()
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True
        ).points
        logger.debug(f"[SEARCH] Qdrant query completed in {time.time() - qdrant_start:.3f}s")
        
        # Filter by minimum score and extract relevant fields
        # Return structured data only - no formatting here
        filtered_results = []
        for r in results:
            if r.score >= min_score:
                filtered_results.append({
                    "topic": r.payload.get("topic", "General"),
                    "answer": r.payload.get("answer", ""),
                    "questions": r.payload.get("questions", []),
                    "confidence": round(r.score, 3)
                })
        
        logger.debug(f"[SEARCH] Found {len(filtered_results)} results in {time.time() - search_start:.3f}s total")
        return filtered_results


# Global singleton instance
_retriever = KnowledgeRetriever()


async def heyo_knowledge_retriever(context: p.RetrieverContext) -> p.RetrieverResult:
    """
    Parlant retriever that grounds the agent with Heyo product knowledge.
    
    This retriever is called automatically on every message.
    It extracts the user's query and fetches relevant knowledge from Qdrant.
    
    Key Parlant patterns:
    - Retriever returns RetrieverResult with structured data
    - Data becomes part of the agent's "known context" for this response
    - Agent uses this context to formulate grounded answers
    - No tool calls needed - this runs in parallel with guideline matching
    
    Args:
        context: Parlant's RetrieverContext with conversation state
        
    Returns:
        RetrieverResult with knowledge entries or None if no relevant data
    """
    retriever_start = time.time()
    logger.info("[PARLANT-RETRIEVER] >>> heyo_knowledge_retriever CALLED <<<")
    
    # Get the last customer message as the query
    last_message = context.interaction.last_customer_message
    if not last_message:
        logger.info("[PARLANT-RETRIEVER] No last message, returning None")
        return p.RetrieverResult(data=None)
    
    query = last_message.content
    logger.info(f"[PARLANT-RETRIEVER] Query: '{query[:100]}...'")
    
    # Search knowledge base
    logger.info("[PARLANT-RETRIEVER] Calling _retriever.search()...")
    results = _retriever.search(query, top_k=5, min_score=0.3)
    
    if not results:
        # No relevant knowledge found - agent should acknowledge this
        logger.info(f"[PARLANT-RETRIEVER] No results found, completed in {time.time() - retriever_start:.3f}s")
        return p.RetrieverResult(
            data={
                "status": "no_results",
                "message": "No relevant information found in knowledge base for this query."
            }
        )
    
    # Return structured knowledge data
    # The agent will synthesize this into a natural response
    logger.info(f"[PARLANT-RETRIEVER] Returning {len(results)} entries, completed in {time.time() - retriever_start:.3f}s")
    return p.RetrieverResult(
        data={
            "status": "found",
            "knowledge_entries": results,
            "query": query
        }
    )
