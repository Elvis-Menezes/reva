"""
RAG Tool for Parlant Agent

This module provides a retrieval-augmented generation (RAG) tool
that integrates with Parlant to search the Qdrant knowledge base
and return relevant answers for user queries.
"""

import logging
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
                "score": r.score,
                "topic": r.payload.get("topic", "Unknown"),
                "chunk_id": r.payload.get("chunk_id", ""),
                "questions": r.payload.get("questions", []),
                "answer": r.payload.get("answer", "No answer available")
            }
            for r in results
        ]
        
        logger.info(f"Found {len(formatted_results)} results")
        return formatted_results


# Initialize retriever as singleton
_retriever = RAGRetriever()


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
        ToolResult with formatted knowledge base results
    """
    results = _retriever.search(query, top_k=top_k)
    
    if not results:
        return ToolResult(data="No relevant information found in the knowledge base.")
    
    # Format results for the LLM
    output_parts = []
    
    for i, result in enumerate(results, 1):
        score = result["score"]
        topic = result["topic"]
        answer = result["answer"]
        questions = result["questions"]
        
        # Only include results with reasonable relevance
        if score < 0.3:
            continue
            
        # Format each result
        questions_str = ", ".join(questions[:3]) if questions else "N/A"
        output_parts.append(
            f"**Result {i}** (Relevance: {score:.2f})\n"
            f"- Topic: {topic}\n"
            f"- Related queries: {questions_str}\n"
            f"- Answer: {answer}"
        )
    
    if not output_parts:
        return ToolResult(data="No sufficiently relevant results found. The query may be outside the knowledge base scope.")
    
    return ToolResult(data="\n\n---\n\n".join(output_parts))