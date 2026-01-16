# Reva - Heyo Support Agent

Reva is an intelligent support agent for the **Heyo** virtual phone platform, built using the [Parlant SDK](https://github.com/parlant-io/parlant). 

This agent demonstrates the **Retriever-based RAG (Retrieval-Augmented Generation)** pattern, which allows the agent to ground its responses in a knowledge base without requiring the LLM to call external tools manually.

## Features

- **Knowledge Retrieval**: Automatically fetches relevant context from a Qdrant vector database for every user message.
- **Behavioral Guidelines**: strict rules for personality, escalation protocols, and "don't guess" policies.
- **Glossary Terms**: Domain-specific vocabulary definitions (e.g., IVR, WABA) for consistent understanding.
- **Traceability**: Extensive debug logging for monitoring performance and decision-making.

## Prerequisites

- Python 3.12+
- OpenAI API Key
- Qdrant Vector Database (Cloud or Local instance)

## Installation

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Create a `.env` file in the root directory with the following variables:

```env
# NLP Service Configuration
OPENAI_API_KEY=sk-...
NLP_SERVICE="openai"

# Vector Database Configuration
QDRANT_URL=https://your-cluster.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_key
QDRANT_COLLECTION="Reva"
```

## Usage

To start the agent:

```bash
python reva.py
```

The server will initialize, cache guidelines (which may take a moment on first run), and then listen for connections.

## Project Structure

- **`reva.py`**: The main entry point. Initializes the Parlant server, creates the agent, defines behavioral guidelines, and attaches the knowledge retriever.
- **`knowledge_retriever.py`**: Contains the logic for querying the Vector DB (Qdrant) and formatting results for the Parlant agent.
- **`config.py`**: Configuration management using environment variables.
- **`intent_ingestion_pipeline.py`**: (Optional) Script to ingest documents from `heyo.txt` into the Qdrant vector store.
- **`heyo.txt`**: Source knowledge base for the agent.

## Troubleshooting

If you encounter slow startup times during "Caching entity embeddings":
- This phase involves calls to the OpenAI API for each guideline and term.
- Check `NLP_SERVICE="openai"` is set correctly.
- Ensure your API key has sufficient rate limits.
# reva
