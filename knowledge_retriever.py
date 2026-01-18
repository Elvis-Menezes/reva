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

import asyncio
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
    
    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> list[dict]:
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

# Confidence bands for intent-level similarity
HIGH_CONFIDENCE_THRESHOLD = 0.78
MEDIUM_CONFIDENCE_THRESHOLD = 0.6
LOW_CONFIDENCE_THRESHOLD = 0.45

# Timeout guard to avoid blocking response generation
RETRIEVER_TIMEOUT_SECONDS = 3.0


def _confidence_band(score: float) -> str:
    if score >= HIGH_CONFIDENCE_THRESHOLD:
        return "high"
    if score >= MEDIUM_CONFIDENCE_THRESHOLD:
        return "medium"
    if score >= LOW_CONFIDENCE_THRESHOLD:
        return "low"
    return "none"


def _extract_trace_id(context: p.RetrieverContext) -> str | None:
    interaction = getattr(context, "interaction", None)
    for obj in (context, interaction):
        if not obj:
            continue
        for attr in ("trace_id", "traceId", "traceID"):
            value = getattr(obj, attr, None)
            if value:
                return value
    return None


def _extract_journey_info(context: p.RetrieverContext) -> tuple[str | None, str | None]:
    interaction = getattr(context, "interaction", None)
    journey_id = None
    journey_state = None
    for obj in (interaction, context):
        if not obj:
            continue
        journey_id = journey_id or getattr(obj, "journey_id", None) or getattr(obj, "journeyId", None)
        journey_state = journey_state or getattr(obj, "journey_state", None) or getattr(
            obj, "journeyState", None
        )
        journey = getattr(obj, "journey", None)
        if journey:
            journey_id = journey_id or getattr(journey, "id", None) or getattr(journey, "name", None)
            journey_state = journey_state or getattr(journey, "state", None)
    return journey_id, journey_state


_journey_state_cache: dict[str, str] = {}


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
    trace_id = _extract_trace_id(context)
    journey_id, journey_state = _extract_journey_info(context)
    if trace_id and journey_id and journey_state:
        cache_key = f"{trace_id}:{journey_id}"
        previous_state = _journey_state_cache.get(cache_key)
        if previous_state != journey_state:
            _journey_state_cache[cache_key] = journey_state
            logger.info(
                "[PARLANT-JOURNEY] Transition trace_id=%s journey=%s state=%s",
                trace_id,
                journey_id,
                journey_state,
            )
    logger.info(
        "[PARLANT-RETRIEVER] >>> heyo_knowledge_retriever CALLED <<< trace_id=%s",
        trace_id,
    )
    
    # Get the last customer message as the query
    last_message = context.interaction.last_customer_message
    if not last_message:
        logger.info("[PARLANT-RETRIEVER] No last message, returning None")
        return p.RetrieverResult(data=None)
    
    query = last_message.content
    logger.info(f"[PARLANT-RETRIEVER] Query: '{query[:100]}...'")
    
    # Search knowledge base in a worker thread to avoid blocking the event loop
    logger.info("[PARLANT-RETRIEVER] Calling _retriever.search()...")
    try:
        results = await asyncio.wait_for(
            asyncio.to_thread(_retriever.search, query, 5, 0.0),
            timeout=RETRIEVER_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "[PARLANT-RETRIEVER] Timeout after %.2fs; returning no_results",
            RETRIEVER_TIMEOUT_SECONDS,
        )
        return p.RetrieverResult(
            data={
                "status": "timeout",
                "confidence_band": "none",
                "knowledge_entries": [],
                "query": query,
                "trace_id": trace_id,
            }
        )
    except Exception as exc:
        logger.exception("[PARLANT-RETRIEVER] Retrieval failed: %s", exc)
        return p.RetrieverResult(
            data={
                "status": "error",
                "confidence_band": "none",
                "knowledge_entries": [],
                "query": query,
                "trace_id": trace_id,
                "error": str(exc),
            }
        )
    
    if not results:
        # No relevant knowledge found - agent should acknowledge this
        logger.info(
            "[PARLANT-RETRIEVER] No results found, completed in %.3fs",
            time.time() - retriever_start,
        )
        return p.RetrieverResult(
            data={
                "status": "no_results",
                "confidence_band": "none",
                "knowledge_entries": [],
                "query": query,
                "trace_id": trace_id,
            }
        )

    top_score = max(r["confidence"] for r in results)
    band = _confidence_band(top_score)
    if band in {"none", "low"}:
        logger.info(
            "[PARLANT-RETRIEVER] Low confidence (%.3f), returning no_results",
            top_score,
        )
        return p.RetrieverResult(
            data={
                "status": "low_confidence",
                "confidence_band": band,
                "knowledge_entries": [],
                "query": query,
                "top_score": top_score,
                "trace_id": trace_id,
            }
        )

    # Only surface entries that meet the low-confidence threshold
    results = [r for r in results if r["confidence"] >= LOW_CONFIDENCE_THRESHOLD]
    
    # Return structured knowledge data
    # The agent will synthesize this into a natural response
    logger.info(
        "[PARLANT-RETRIEVER] Returning %d entries (%s), completed in %.3fs",
        len(results),
        band,
        time.time() - retriever_start,
    )
    return p.RetrieverResult(
        data={
            "status": "found",
            "confidence_band": band,
            "knowledge_entries": results,
            "query": query,
            "top_score": top_score,
            "trace_id": trace_id,
        }
    )
