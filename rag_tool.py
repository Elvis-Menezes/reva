"""
RAG Tool for Parlant Agent

This module provides a retrieval-augmented generation (RAG) tool
that integrates with Parlant to search the Qdrant knowledge base
and return relevant answers for user queries.
"""

import asyncio
import logging
import time

import parlant.sdk as p
from parlant.sdk import ToolContext, ToolResult
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from config import Config

logger = logging.getLogger(__name__)


class RAGRetriever:
    """Enhanced retriever that returns full answers for Parlant."""
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to avoid reloading the model multiple times."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        logger.info("Initializing RAG Retriever...")
        logger.info(f"Loading embedding model: {Config.EMBEDDING_MODEL}")
        
        self.model = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.client = QdrantClient(
            url=Config.QDRANT_URL,
            api_key=Config.QDRANT_API_KEY,
            timeout=60
        )
        self.collection_name = Config.QDRANT_COLLECTION_NAME
        self._initialized = True
        
        logger.info(f"RAG Retriever initialized. Collection: {self.collection_name}")
    
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Search the knowledge base for relevant intents.
        
        Args:
            query: User's natural language query
            top_k: Number of results to return
            
        Returns:
            List of matching results with scores, topics, questions, and answers
        """
        logger.info(f"Searching for: '{query}' (top_k={top_k})")
        
        # Generate query embedding
        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True
        ).tolist()
        
        # Search Qdrant
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True
        ).points
        
        formatted_results = [
            {
                "score": round(r.score, 3),
                "topic": r.payload.get("topic", "Unknown"),
                "chunk_id": r.payload.get("chunk_id", ""),
                "questions": r.payload.get("questions", []),
                "answer": r.payload.get("answer", "No answer available"),
            }
            for r in results
        ]
        
        logger.info(f"Found {len(formatted_results)} results")
        return formatted_results


# Initialize retriever as singleton
_retriever = RAGRetriever()


HIGH_CONFIDENCE_THRESHOLD = 0.78
MEDIUM_CONFIDENCE_THRESHOLD = 0.6
LOW_CONFIDENCE_THRESHOLD = 0.45
TOOL_TIMEOUT_SECONDS = 3.0


def _confidence_band(score: float) -> str:
    if score >= HIGH_CONFIDENCE_THRESHOLD:
        return "high"
    if score >= MEDIUM_CONFIDENCE_THRESHOLD:
        return "medium"
    if score >= LOW_CONFIDENCE_THRESHOLD:
        return "low"
    return "none"


def _extract_trace_id(context: ToolContext) -> str | None:
    interaction = getattr(context, "interaction", None)
    for obj in (context, interaction):
        if not obj:
            continue
        for attr in ("trace_id", "traceId", "traceID"):
            value = getattr(obj, attr, None)
            if value:
                return value
    return None


@p.tool
async def search_knowledge_base(
    context: ToolContext,
    query: str,
    top_k: int = 5
) -> ToolResult:
    """
    Search the Heyo knowledge base to find answers to user questions.
    
    Use this tool whenever the user asks about:
    - Heyo features, pricing, plans, or setup
    - Technical support questions
    - How-to guides and instructions
    - Billing, refunds, or account issues
    - Any factual information about the product
    
    Args:
        context: Parlant tool context (automatically provided)
        query: The user's question or search query
        top_k: Number of results to retrieve (default: 5)
        
    Returns:
        ToolResult with structured knowledge base results and confidence metadata
    """
    start = time.time()
    trace_id = _extract_trace_id(context)
    logger.info("[TOOL] search_knowledge_base called trace_id=%s", trace_id)

    try:
        results = await asyncio.wait_for(
            asyncio.to_thread(_retriever.search, query, top_k),
            timeout=TOOL_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        return ToolResult(
            data={
                "status": "timeout",
                "confidence_band": "none",
                "results": [],
                "query": query,
                "trace_id": trace_id,
            }
        )
    except Exception as exc:
        logger.exception("[TOOL] search_knowledge_base failed: %s", exc)
        return ToolResult(
            data={
                "status": "error",
                "confidence_band": "none",
                "results": [],
                "query": query,
                "trace_id": trace_id,
                "error": str(exc),
            }
        )

    if not results:
        return ToolResult(
            data={
                "status": "no_results",
                "confidence_band": "none",
                "results": [],
                "query": query,
                "trace_id": trace_id,
            }
        )

    top_score = max(r["score"] for r in results)
    band = _confidence_band(top_score)
    filtered = [r for r in results if r["score"] >= LOW_CONFIDENCE_THRESHOLD]
    if band in {"none", "low"}:
        filtered = []

    return ToolResult(
        data={
            "status": "found" if filtered else "low_confidence",
            "confidence_band": band,
            "results": filtered,
            "query": query,
            "top_score": top_score,
            "trace_id": trace_id,
            "timing_ms": round((time.time() - start) * 1000),
        }
    )