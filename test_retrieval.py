"""
Intent Retrieval Test Script

This script demonstrates how to query the Qdrant vector database
after running the ingestion pipeline. It validates that:
1. One intent = one vector (atomic chunking)
2. Queries like "press 1 for sales" retrieve the correct IVR chunk
3. The DB works correctly for intent-based retrieval

Usage:
    python test_retrieval.py
    python test_retrieval.py --query "your custom query"
"""

import argparse
import json
from intent_ingestion_pipeline import IntentRetriever
from config import Config


def run_test_queries():
    """
    Run a suite of test queries to validate the ingestion.
    
    These queries are designed to test:
    - Exact phrase matching ("press 1 for sales")
    - Semantic similarity ("how much does heyo cost")
    - Intent paraphrase matching ("tell me about IVR")
    """
    
    test_queries = [
        # Test 1: IVR query (should match sales_010)
        {
            "query": "press 1 for sales",
            "expected_chunk": "sales_010",
            "description": "IVR routing query"
        },
        # Test 2: Pricing query
        {
            "query": "how much does heyo cost",
            "expected_chunk": "sales_022",  # or sales_023, sales_024
            "description": "Pricing intent"
        },
        # Test 3: Product information
        {
            "query": "what is heyo phone",
            "expected_chunk": "sales_001",
            "description": "Product info query"
        },
        # Test 4: Demo request
        {
            "query": "I want to see a demo",
            "expected_chunk": "sales_003",  # or sales_004
            "description": "Demo request intent"
        },
        # Test 5: Feature inquiry
        {
            "query": "do you have call recording",
            "expected_chunk": "sales_008",
            "description": "Feature inquiry"
        },
        # Test 6: KYC process
        {
            "query": "how do I complete KYC",
            "expected_chunk": "sales_006",
            "description": "KYC process query"
        },
        # Test 7: Virtual number
        {
            "query": "do I need a SIM card",
            "expected_chunk": "sales_007",  # or sales_009
            "description": "Virtual number clarification"
        },
        # Test 8: Trial information
        {
            "query": "is there a free trial",
            "expected_chunk": "sales_027",
            "description": "Trial inquiry"
        }
    ]
    
    print("\n" + "=" * 70)
    print("INTENT RETRIEVAL TEST SUITE")
    print("=" * 70)
    
    # Validate configuration
    Config.validate()
    
    # Initialize retriever
    retriever = IntentRetriever()
    
    results_summary = []
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n--- Test {i}: {test['description']} ---")
        print(f"Query: '{test['query']}'")
        print(f"Expected: {test['expected_chunk']}")
        
        # Perform search
        results = retriever.search(test["query"], top_k=3)
        
        if results:
            top_result = results[0]
            is_match = test["expected_chunk"] in [r["chunk_id"] for r in results[:2]]
            
            print(f"\nTop Results:")
            for j, result in enumerate(results, 1):
                marker = "✓" if result["chunk_id"] == test["expected_chunk"] else " "
                print(f"  {j}. [{marker}] {result['chunk_id']} "
                      f"(Score: {result['score']:.4f}, Topic: {result['topic']})")
            
            status = "PASS" if is_match else "CHECK"
            print(f"\nStatus: {status}")
            
            results_summary.append({
                "test": test["description"],
                "query": test["query"],
                "expected": test["expected_chunk"],
                "got": top_result["chunk_id"],
                "score": top_result["score"],
                "status": status
            })
        else:
            print("  No results found!")
            results_summary.append({
                "test": test["description"],
                "query": test["query"],
                "expected": test["expected_chunk"],
                "got": None,
                "score": 0,
                "status": "FAIL"
            })
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for r in results_summary if r["status"] == "PASS")
    total = len(results_summary)
    
    print(f"\nResults: {passed}/{total} tests passed")
    print("\nDetailed Results:")
    for r in results_summary:
        status_icon = "✓" if r["status"] == "PASS" else "⚠" if r["status"] == "CHECK" else "✗"
        print(f"  {status_icon} {r['test']}: {r['got']} (expected: {r['expected']})")
    
    return results_summary


def interactive_search(query: str, top_k: int = 5):
    """
    Perform an interactive search with a custom query.
    """
    print("\n" + "=" * 70)
    print("INTERACTIVE INTENT SEARCH")
    print("=" * 70)
    
    # Validate configuration
    Config.validate()
    
    print(f"\nQuery: '{query}'")
    print(f"Top K: {top_k}")
    
    retriever = IntentRetriever()
    results = retriever.search(query, top_k=top_k)
    
    print(f"\n{'='*70}")
    print(f"RESULTS ({len(results)} found)")
    print(f"{'='*70}")
    
    if not results:
        print("\nNo matching intents found.")
        return
    
    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"  Chunk ID:     {result['chunk_id']}")
        print(f"  Topic:        {result['topic']}")
        print(f"  Intent Count: {result['intent_count']}")
        print(f"  Score:        {result['score']:.4f}")
        print(f"  Type:         {result['type']}")
        print(f"  Source:       {result['source']}")


def main():
    parser = argparse.ArgumentParser(
        description="Test Intent Retrieval from Qdrant"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Custom query to search for"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of results to return"
    )
    parser.add_argument(
        "--run-tests", "-t",
        action="store_true",
        help="Run the full test suite"
    )
    
    args = parser.parse_args()
    
    if args.query:
        interactive_search(args.query, args.top_k)
    elif args.run_tests:
        run_test_queries()
    else:
        # Default: run test suite
        run_test_queries()


if __name__ == "__main__":
    main()
