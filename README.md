# Reva: Production-Grade Conversational AI Agent

A production-ready chatbot application built on the **Parlant SDK** featuring a hierarchical agent architecture, guidelines-driven behavior, structured conversation journeys, and retrieval-augmented generation (RAG) with confidence-based response strategies.

---

## Table of Contents

- [High-Level System Overview](#high-level-system-overview)
- [Architecture](#architecture)
  - [Agent Hierarchy](#agent-hierarchy)
  - [Component Interaction](#component-interaction)
- [Core Components](#core-components)
  - [Agent Definition (reva.py)](#agent-definition-revapy)
  - [Knowledge Retriever (knowledge_retriever.py)](#knowledge-retriever-knowledge_retrieverpy)
  - [Journey & Tool System (journeys_tools.py)](#journey--tool-system-journeys_toolspy)
  - [Bill Processor (bill_processor.py)](#bill-processor-bill_processorpy)
  - [Bill Upload Service (bill_upload_service.py)](#bill-upload-service-bill_upload_servicepy)
  - [Intent Ingestion Pipeline (intent_ingestion_pipeline.py)](#intent-ingestion-pipeline-intent_ingestion_pipelinepy)
  - [Configuration (config.py)](#configuration-configpy)
- [Guidelines System](#guidelines-system)
- [Journey Definitions](#journey-definitions)
  - [Onboarding Journey](#onboarding-journey)
  - [Support Journey](#support-journey)
- [Tool Specifications](#tool-specifications)
- [RAG Architecture](#rag-architecture)
  - [Confidence Bands](#confidence-bands)
  - [Retriever vs Tool Pattern](#retriever-vs-tool-pattern)
- [Data Flow](#data-flow)
- [Configuration & Environment](#configuration--environment)
- [Testing](#testing)
- [API Reference](#api-reference)

---

## High-Level System Overview

Reva is a customer support agent for "Heyo" that combines:

| Capability | Implementation |
|------------|----------------|
| **Natural Language Understanding** | OpenAI NLP service via Parlant SDK |
| **Knowledge Grounding** | Qdrant vector database + SentenceTransformers |
| **Behavioral Guardrails** | 7 declarative guidelines with condition/action pairs |
| **Structured Workflows** | State-machine journeys with tool transitions |
| **Document Processing** | OpenAI Vision API for bill/invoice analysis |
| **Human Escalation** | Deterministic handoff workflow |

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           REVA AGENT SYSTEM                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   User Message ──▶ [Parlant Server] ──▶ [Agent: Reva]                  │
│                           │                   │                         │
│                           ▼                   ▼                         │
│              ┌─────────────────────┐  ┌─────────────────┐              │
│              │  heyo_knowledge_    │  │   Guidelines    │              │
│              │     retriever       │  │   (7 rules)     │              │
│              │   (runs parallel)   │  │                 │              │
│              └─────────────────────┘  └─────────────────┘              │
│                           │                   │                         │
│                           ▼                   ▼                         │
│              ┌─────────────────────────────────────────┐               │
│              │         Journey State Machine           │               │
│              │  ┌─────────────┐   ┌─────────────────┐ │               │
│              │  │ Onboarding  │   │    Support      │ │               │
│              │  │   Journey   │   │    Journey      │ │               │
│              │  └─────────────┘   └─────────────────┘ │               │
│              └─────────────────────────────────────────┘               │
│                           │                                             │
│                           ▼                                             │
│              ┌─────────────────────────────────────────┐               │
│              │              Tool Invocations           │               │
│              │  • verify_id_tool                       │               │
│              │  • check_refund_eligibility             │               │
│              │  • process_bill_image                   │               │
│              │  • initiate_human_handoff               │               │
│              └─────────────────────────────────────────┘               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Architecture

### Agent Hierarchy

The system uses a **single parent agent** pattern where one primary agent handles all interactions, with behavior modulated by:

1. **Guidelines** - Declarative condition→action rules evaluated on every turn
2. **Retrievers** - Context providers that run in parallel with guideline matching
3. **Journeys** - State machines that guide multi-turn conversations
4. **Tools** - Executable functions for external actions

```python
# Agent initialization pattern (reva.py)
async with p.Server(nlp_service=p.NLPServices.openai) as server:
    agent = await server.create_agent(
        name="Reva",
        description="A calm and professional support agent for Heyo..."
    )
    
    # Attach retriever (parallel context loading)
    await agent.attach_retriever(heyo_knowledge_retriever, id="heyo_knowledge")
    
    # Create guidelines (behavioral rules)
    await agent.create_guideline(condition="...", action="...")
    
    # Create journeys (structured workflows)
    await create_onboarding_journey(agent)
    await create_support_journey(agent)
```

### Component Interaction

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        REQUEST PROCESSING FLOW                           │
└──────────────────────────────────────────────────────────────────────────┘

                         ┌─────────────────┐
                         │  User Message   │
                         └────────┬────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         PARALLEL EXECUTION                               │
│  ┌─────────────────────────┐       ┌─────────────────────────────────┐ │
│  │   Retriever Pipeline    │       │    Guideline Matching Engine    │ │
│  │                         │       │                                 │ │
│  │  1. Extract last msg    │       │  1. Evaluate all 7 conditions  │ │
│  │  2. Generate embedding  │       │  2. Collect matching actions   │ │
│  │  3. Query Qdrant        │       │  3. Merge into response plan   │ │
│  │  4. Filter by score     │       │                                 │ │
│  │  5. Return confidence   │       │                                 │ │
│  │     band + entries      │       │                                 │ │
│  └─────────────────────────┘       └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   Journey State Check   │
                    │                         │
                    │  Is user in a journey?  │
                    │  What transitions are   │
                    │  available?             │
                    └───────────┬─────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
     ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
     │  Chat State │   │  Tool State │   │   End/Exit  │
     │  (respond)  │   │ (invoke fn) │   │             │
     └─────────────┘   └─────────────┘   └─────────────┘
```

---

## Core Components

### Agent Definition (`reva.py`)

The main entry point that bootstraps the Parlant server and configures the agent.

**Responsibilities:**
- Initialize Parlant server with OpenAI NLP service
- Create the Reva agent with description
- Attach the knowledge retriever
- Define all behavioral guidelines
- Register conversation journeys

**Key Pattern:** Async context manager ensures proper cleanup of server resources.

```python
async def main():
    async with p.Server(nlp_service=p.NLPServices.openai) as server:
        agent = await server.create_agent(name="Reva", description="...")
        await agent.attach_retriever(heyo_knowledge_retriever, id="heyo_knowledge")
        # ... guidelines and journeys
```

---

### Knowledge Retriever (`knowledge_retriever.py`)

Implements the Parlant retriever pattern for grounding agent responses with domain knowledge.

**Key Design Decisions:**

| Decision | Rationale |
|----------|-----------|
| Singleton pattern | Avoid reloading SentenceTransformer model on each request |
| Async with timeout | 3-second timeout prevents blocking response generation |
| Confidence bands | High/medium/low/none determines response strategy |
| Structured RetrieverResult | Agent synthesizes from data, not pre-formatted text |

**Confidence Band Thresholds:**

```python
HIGH_CONFIDENCE_THRESHOLD = 0.78    # Answer directly and authoritatively
MEDIUM_CONFIDENCE_THRESHOLD = 0.6   # Cautious answer + clarifying question
LOW_CONFIDENCE_THRESHOLD = 0.45     # Acknowledge limitation, offer escalation
# Below 0.45 = "none"               # No relevant knowledge found
```

**Retriever Function Signature:**

```python
async def heyo_knowledge_retriever(context: p.RetrieverContext) -> p.RetrieverResult:
    """
    Parlant retriever that grounds the agent with Heyo product knowledge.
    
    Key Parlant patterns:
    - Retriever returns RetrieverResult with structured data
    - Data becomes part of the agent's "known context" for this response
    - Agent uses this context to formulate grounded answers
    - No tool calls needed - runs in parallel with guideline matching
    """
```

**Return Structure:**

```python
{
    "status": "found" | "low_confidence" | "no_results" | "timeout" | "error",
    "confidence_band": "high" | "medium" | "low" | "none",
    "knowledge_entries": [
        {
            "topic": "Pricing",
            "answer": "Heyo starts at $29/month...",
            "questions": ["How much does Heyo cost?", "What's the pricing?"],
            "confidence": 0.85
        }
    ],
    "query": "original user query",
    "top_score": 0.85,
    "trace_id": "abc123"
}
```

---

### Journey & Tool System (`journeys_tools.py`)

Defines structured conversation workflows and the tools that power them.

#### Tools Overview

| Tool | Type | Purpose |
|------|------|---------|
| `verify_id_tool` | Standard | KYC document verification (deterministic stub) |
| `initiate_human_handoff` | Standard | Queue escalation to human agent |
| `check_refund_eligibility` | Consequential | Business logic for refund decisions |
| `process_bill_image` | Consequential | Vision-based bill analysis |

**Tool Result Pattern:**

All tools return structured `p.ToolResult` objects with:

```python
{
    "status": "verified" | "eligible" | "queued" | "error",
    "side_effect": "none" | "handoff_requested",
    "retryable": True | False,  # For error cases
    # ... tool-specific data
}
```

**Consequential Tool Marking:**

```python
@p.tool(consequential=True)  # Requires explicit user confirmation
async def check_refund_eligibility(...) -> p.ToolResult:
    """Marked consequential because refund decisions have business impact."""
```

---

### Bill Processor (`bill_processor.py`)

Handles document analysis using OpenAI Vision API with deterministic business rules.

**Architecture:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                      BILL PROCESSING PIPELINE                        │
└─────────────────────────────────────────────────────────────────────┘

   File ID ──▶ get_bill_path() ──▶ Image exists?
                                        │
                                  ┌─────┴─────┐
                                  ▼           ▼
                               [Yes]        [No] ──▶ FileNotFoundError
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   OpenAI Vision API     │
                    │   (gpt-4o, high detail) │
                    │                         │
                    │   Extraction Prompt:    │
                    │   - invoice_number      │
                    │   - date (YYYY-MM-DD)   │
                    │   - total_amount        │
                    │   - currency            │
                    │   - vendor_name         │
                    │   - payment_status      │
                    │   - line_items          │
                    │   - confidence          │
                    └───────────┬─────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │  BillData (dataclass)   │
                    └───────────┬─────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │  assess_refund_         │
                    │  eligibility()          │
                    │                         │
                    │  Deterministic Rules:   │
                    │  • total_amount exists  │
                    │  • date within 30 days  │
                    │  • status != "unpaid"   │
                    │  • confidence check     │
                    └───────────┬─────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │  RefundAssessment       │
                    │  - eligible: bool       │
                    │  - reason: str          │
                    │  - requires_review      │
                    │  - max_refund_amount    │
                    └─────────────────────────┘
```

**Refund Eligibility Rules (Deterministic):**

| Rule | Condition | Result |
|------|-----------|--------|
| Missing amount | `total_amount is None` | Ineligible, requires review |
| Missing date | `date is None` | Ineligible, requires review |
| Outside window | `days_since > 30` | Ineligible |
| Future date | `days_since < 0` | Ineligible, requires review |
| Unpaid bill | `payment_status == "unpaid"` | Ineligible |
| Low confidence | `extraction_confidence == "low"` | Eligible, requires review |
| All checks pass | Default | Eligible |

---

### Bill Upload Service (`bill_upload_service.py`)

FastAPI-based microservice for handling file uploads.

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upload-bill` | Upload bill image, returns `file_id` |
| `GET` | `/bill/{file_id}` | Get bill metadata |
| `DELETE` | `/bill/{file_id}` | Delete uploaded bill |
| `GET` | `/health` | Health check |

**Upload Response:**

```json
{
    "file_id": "3b71a1dc684c",
    "filename": "receipt.png",
    "stored_path": "3b71a1dc684c.png",
    "size_bytes": 245120,
    "uploaded_at": "2024-01-15T10:30:00Z",
    "message": "Upload successful. Use file_id '3b71a1dc684c' in chat to process this bill."
}
```

**Validation Rules:**
- Allowed extensions: `.jpg`, `.jpeg`, `.png`, `.pdf`
- Maximum file size: 10 MB
- Content-type validation

---

### Intent Ingestion Pipeline (`intent_ingestion_pipeline.py`)

Production-ready ETL pipeline for populating the Qdrant knowledge base.

**Pipeline Stages:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INTENT INGESTION PIPELINE                         │
└─────────────────────────────────────────────────────────────────────┘

   ┌─────────────────┐
   │  JSON Knowledge │
   │  Base File      │
   └────────┬────────┘
            │
            ▼
   ┌─────────────────────────────────────────┐
   │  Stage 1: KnowledgeBaseLoader           │
   │                                         │
   │  • Load JSON array of intent objects    │
   │  • Validate required fields             │
   │  • Convert to IntentChunk dataclass     │
   └───────────────────┬─────────────────────┘
                       │
                       ▼
   ┌─────────────────────────────────────────┐
   │  Stage 2: IntentEmbedder                │
   │                                         │
   │  • Construct semantic embedding text    │
   │  • Batch encode with SentenceTransformer│
   │  • Normalize vectors for cosine sim     │
   └───────────────────┬─────────────────────┘
                       │
                       ▼
   ┌─────────────────────────────────────────┐
   │  Stage 3: QdrantStorage                 │
   │                                         │
   │  • Ensure collection exists             │
   │  • Batch upsert points                  │
   │  • Store payload with questions/answers │
   └─────────────────────────────────────────┘
```

**IntentChunk Structure:**

```python
@dataclass
class IntentChunk:
    chunk_id: str           # Unique identifier (e.g., "sales_010")
    topic: str              # Category (e.g., "Pricing", "IVR")
    questions: list[str]    # Intent paraphrases
    answer: str             # Authoritative resolution
    metadata: dict          # Additional context
```

**Embedding Text Format:**

```
Category: {topic}

User intents:
- {question_1}
- {question_2}
- ...

Resolved answer:
{answer}
```

**Key Design Principle:** One intent = one vector. Never split intents across multiple vectors.

**CLI Usage:**

```bash
# Basic ingestion
python intent_ingestion_pipeline.py --knowledge-base heyo.txt

# Recreate collection (WARNING: deletes existing data)
python intent_ingestion_pipeline.py --knowledge-base heyo.txt --recreate

# Ingest and run test query
python intent_ingestion_pipeline.py --knowledge-base heyo.txt --test-query "how much does heyo cost"
```

---

### Configuration (`config.py`)

Centralized environment-based configuration management.

**Required Environment Variables:**

| Variable | Description | Example |
|----------|-------------|---------|
| `QDRANT_URL` | Qdrant server URL | `https://xxx.qdrant.io` |
| `QDRANT_API_KEY` | Qdrant API key | `your-api-key` |
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |

**Optional Environment Variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_COLLECTION_NAME` | `intent_knowledge_base` | Vector collection name |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace model |

**Configuration Validation:**

```python
from config import Config

# Validate on startup
Config.validate()  # Raises ConfigurationError if missing required vars

# Debug configuration (secrets masked)
print(Config.show())
# {
#     "QDRANT_URL": "https://xxx.qdrant.io",
#     "QDRANT_API_KEY": "abc12345...",
#     "QDRANT_COLLECTION_NAME": "intent_knowledge_base",
#     "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
#     "OPENAI_API_KEY": "sk-abc1..."
# }
```

---

## Guidelines System

Guidelines are declarative condition→action rules that modulate agent behavior on every turn. They run in parallel with retriever execution.

| # | Condition | Action |
|---|-----------|--------|
| 1 | Retriever returns **high confidence** knowledge | Answer directly and authoritatively using ONLY retrieved knowledge. Do not add facts not present. |
| 2 | Retriever returns **medium confidence** knowledge | Provide cautious answer. Ask clarifying question to confirm intent. |
| 3 | Retriever returns **low confidence** or no results | Acknowledge lack of reliable information. Offer escalation. |
| 4 | User asks for link, URL, or web address | Only provide URLs from retrieved knowledge. Never invent URLs. |
| 5 | User expresses frustration or mentions repeated issues | Acknowledge with empathy first. Then help. Offer escalation proactively. |
| 6 | User asks for account change or external action | Never claim completion without tool confirmation. Explain what's needed. |
| 7 | Tool result is present in context | Summarize tool result clearly. Ask what user wants next. |

**Guideline Definition Pattern:**

```python
await agent.create_guideline(
    condition="the retriever returns high confidence knowledge for the question",
    action=(
        "Answer directly and authoritatively using ONLY the retrieved knowledge entries. "
        "Do not add facts that are not present in the retrieved knowledge."
    )
)
```

---

## Journey Definitions

Journeys are state machines that guide multi-turn conversations through structured workflows.

### Onboarding Journey

**Trigger Conditions:** Customer needs to set up account or register for WABA

```
                    ┌─────────────────────┐
                    │    Initial State    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ "Welcome! KYC or    │
                    │  WABA registration?"│
                    └──────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
              ▼                                 ▼
┌─────────────────────────┐       ┌─────────────────────────┐
│ User chooses KYC        │       │ User chooses WABA       │
│                         │       │                         │
│ "Please share ID        │       │ "What phone number      │
│  details..."            │       │  for WhatsApp?"         │
└────────────┬────────────┘       └────────────┬────────────┘
             │                                  │
             ▼                                  ▼
┌─────────────────────────┐       ┌─────────────────────────┐
│ User provides document  │       │ User provides number    │
│                         │       │                         │
│ [TOOL: verify_id_tool]  │       │ "I will use that        │
│                         │       │  number for WABA..."    │
└────────────┬────────────┘       └─────────────────────────┘
             │
             ▼
┌─────────────────────────┐
│ "Thanks! Verification   │
│  is complete."          │
└─────────────────────────┘
```

### Support Journey

**Trigger Conditions:** User has a problem, is stuck, or requests a refund

```
                    ┌─────────────────────┐
                    │    Initial State    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ "Sorry you're       │
                    │  running into this. │
                    │  Share account      │
                    │  details?"          │
                    └──────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
              ▼                                 ▼
┌─────────────────────────┐       ┌─────────────────────────┐
│ User provides details   │       │ User asks for refund    │
│                         │       │                         │
│ "I'll look into this    │       │ [TOOL: check_refund_    │
│  and propose solution." │       │  eligibility]           │
└────────────┬────────────┘       └────────────┬────────────┘
             │                                  │
             ▼                                  ▼
┌─────────────────────────┐       ┌─────────────────────────┐
│ First solution failed   │       │ "Upload bill image      │
│                         │       │  if you have one..."    │
│ [TOOL: initiate_human_  │       └────────────┬────────────┘
│  handoff]               │                    │
└────────────┬────────────┘                    ▼
             │                   ┌─────────────────────────┐
             ▼                   │ User provides bill URL  │
┌─────────────────────────┐      │                         │
│ "I've escalated this    │      │ [TOOL: process_bill_    │
│  to a specialist..."    │      │  image]                 │
└─────────────────────────┘      └────────────┬────────────┘
                                              │
                                              ▼
                                 ┌─────────────────────────┐
                                 │ "If eligible, confirm.  │
                                 │  Otherwise, explain."   │
                                 └─────────────────────────┘
```

**Journey Definition Pattern:**

```python
async def create_support_journey(agent: p.Agent) -> None:
    support = await agent.create_journey(
        title="Support & Retention",
        description="Handles technical issues, billing inquiries, and human handoffs.",
        conditions=["The user has a problem, is stuck, or requests a refund"],
    )

    t1 = await support.initial_state.transition_to(
        chat_state="I'm sorry you're running into this. Can you share your account details?"
    )

    t2_refund = await t1.target.transition_to(
        tool_state=check_refund_eligibility,
        condition="The user is asking for a refund",
        tool_instruction="Check the refund eligibility for this customer's order.",
    )
```

---

## Tool Specifications

### `verify_id_tool`

```python
@p.tool
async def verify_id_tool(
    context: p.ToolContext,
    document_id: str | None = None,
    document_type: str | None = None,
    country: str | None = None,
) -> p.ToolResult
```

**Purpose:** KYC document verification (deterministic stub)

**Inputs:**
- `document_id` (required): Document identifier
- `document_type` (required): Type of document (e.g., "passport", "license")
- `country` (optional): Country of issuance

**Outputs:**
```python
# Success
{"status": "verified", "verification_id": "abc123...", "document_type": "...", "country": "...", "side_effect": "none"}

# Invalid input
{"status": "invalid_input", "missing_fields": ["document_id"], "retryable": True}
```

---

### `initiate_human_handoff`

```python
@p.tool
async def initiate_human_handoff(
    context: p.ToolContext,
    reason: str,
    urgency: str = "normal",
) -> p.ToolResult
```

**Purpose:** Queue escalation to human agent

**Inputs:**
- `reason` (required): Why escalation is needed
- `urgency` (optional): `"normal"` | `"high"` | `"critical"`

**Outputs:**
```python
{"status": "queued", "handoff_id": "abc123...", "urgency": "normal", "requires_human": True, "side_effect": "handoff_requested"}
```

---

### `check_refund_eligibility`

```python
@p.tool(consequential=True)
async def check_refund_eligibility(
    context: p.ToolContext,
    account_id: str | None = None,
    order_id: str | None = None,
    days_since_purchase: int | None = None,
    reason: str | None = None,
) -> p.ToolResult
```

**Purpose:** Determine refund eligibility based on business rules

**Business Rules:**
- Refund window: 7 days
- Blocked reasons: "fraud", "abuse"

**Outputs:**
```python
# Eligible
{"status": "eligible", "eligible": True, "refund_window_days": 7, "days_since_purchase": 3, "blocked_reason": False, "side_effect": "none"}

# Ineligible
{"status": "ineligible", "eligible": False, "refund_window_days": 7, "days_since_purchase": 10, "blocked_reason": False, "side_effect": "none"}
```

---

### `process_bill_image`

```python
@p.tool(consequential=True)
async def process_bill_image(
    context: p.ToolContext,
    bill_image_url: str,
) -> p.ToolResult
```

**Purpose:** Vision-based bill analysis for refund verification

**Inputs:**
- `bill_image_url` (required): The `file_id` from upload service or direct URL

**Outputs:**
```python
{
    "status": "processed",
    "file_id": "3b71a1dc684c",
    "bill": {
        "invoice_number": "INV-2024-001",
        "date": "2024-01-10",
        "days_since_issued": 5,
        "total_amount": 29.99,
        "currency": "USD",
        "vendor_name": "Heyo Inc",
        "payment_status": "paid",
        "line_items_count": 3,
        "extraction_confidence": "high"
    },
    "refund": {
        "eligible": True,
        "reason": "Bill verified successfully (5 days old, within 30-day refund window)",
        "requires_review": False,
        "max_refund_amount": 29.99,
        "refund_window_days": 30
    },
    "side_effect": "none"
}
```

---

## RAG Architecture

### Confidence Bands

The RAG system uses confidence bands to determine response strategy:

```
Score Range          Band        Agent Behavior
─────────────────────────────────────────────────────────────
≥ 0.78               high        Answer directly, authoritatively
0.60 - 0.77          medium      Cautious answer + clarifying question
0.45 - 0.59          low         Acknowledge limitation, offer escalation
< 0.45               none        No relevant knowledge found
```

### Retriever vs Tool Pattern

Parlant distinguishes between **Retrievers** and **Tools**:

| Aspect | Retriever | Tool |
|--------|-----------|------|
| **Execution** | Parallel with guidelines | Sequential, on-demand |
| **Purpose** | Context the agent "should know" | Actions to "perform" |
| **Timing** | Every message | When explicitly invoked |
| **Return type** | `RetrieverResult` | `ToolResult` |
| **Side effects** | None (read-only) | May have side effects |

**Why use Retriever for RAG:**
- Reduces latency by running parallel to guideline matching
- Knowledge is always available without explicit tool calls
- Agent can naturally incorporate grounded facts

---

## Data Flow

### Complete Request Lifecycle

```
1. User sends message
         │
         ▼
2. Parlant Server receives request
         │
         ├──────────────────────────────┐
         │                              │
         ▼                              ▼
3a. heyo_knowledge_retriever()    3b. Guideline evaluation
    • Extract last message            • Match all 7 conditions
    • Generate embedding              • Collect applicable actions
    • Query Qdrant (3s timeout)
    • Calculate confidence band
         │                              │
         └──────────────┬───────────────┘
                        │
                        ▼
4. Agent synthesizes response
   • Retriever data provides grounding
   • Guidelines shape behavior
   • Journey state determines structure
         │
         ▼
5. Check journey transitions
   • Evaluate transition conditions
   • Determine next state (chat/tool)
         │
         ├─────────────────────────────┐
         │                             │
         ▼                             ▼
6a. Chat state                   6b. Tool state
    • Generate response              • Invoke tool function
    • Apply guidelines               • Process ToolResult
                                     • Summarize per guideline 7
         │                             │
         └──────────────┬──────────────┘
                        │
                        ▼
7. Return response to user
```

---

## Configuration & Environment

### Environment File Template (`.env`)

```bash
# Required: Qdrant Vector Database
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key

# Required: OpenAI (for NLP service and Vision API)
OPENAI_API_KEY=sk-your-openai-api-key

# Optional: Customization
QDRANT_COLLECTION_NAME=intent_knowledge_base
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Supported Embedding Models

| Model | Dimensions | Notes |
|-------|------------|-------|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Default, fast, good for intent matching |
| `sentence-transformers/all-mpnet-base-v2` | 768 | More accurate, slower |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | Multilingual support |
| `BAAI/bge-small-en-v1.5` | 384 | Strong benchmark performance |

---

## Testing

### Retrieval Test Suite (`test_retrieval.py`)

Validates that the ingested knowledge base returns correct results:

```bash
# Run full test suite
python test_retrieval.py --run-tests

# Interactive single query
python test_retrieval.py --query "how much does heyo cost" --top-k 5
```

**Test Cases:**
- IVR routing queries
- Pricing intent matching
- Product information retrieval
- Demo request detection
- Feature inquiry handling
- KYC process questions
- Virtual number clarification
- Trial information lookup

---

## API Reference

### Bill Upload Service

**Base URL:** `http://localhost:8801`

#### Upload Bill

```http
POST /upload-bill
Content-Type: multipart/form-data

file: <binary>
```

**Response:**
```json
{
    "file_id": "3b71a1dc684c",
    "filename": "receipt.png",
    "stored_path": "3b71a1dc684c.png",
    "size_bytes": 245120,
    "uploaded_at": "2024-01-15T10:30:00.000Z",
    "message": "Upload successful. Use file_id '3b71a1dc684c' in chat to process this bill."
}
```

#### Get Bill Info

```http
GET /bill/{file_id}
```

**Response:**
```json
{
    "file_id": "3b71a1dc684c",
    "filename": "3b71a1dc684c.png",
    "size_bytes": 245120,
    "exists": true
}
```

#### Delete Bill

```http
DELETE /bill/{file_id}
```

**Response:**
```json
{
    "file_id": "3b71a1dc684c",
    "deleted": true
}
```

#### Health Check

```http
GET /health
```

**Response:**
```json
{
    "status": "healthy",
    "service": "bill-upload"
}
```

---

## Dependencies

### Core Backend

| Package | Version | Purpose |
|---------|---------|---------|
| `parlant-sdk` | - | Agent framework |
| `openai` | ≥1.0.0 | NLP service, Vision API |
| `qdrant-client` | ≥1.9.0 | Vector database |
| `sentence-transformers` | ≥2.2.0 | Embedding generation |
| `torch` | ≥2.0.0 | ML runtime |
| `transformers` | ≥4.35.0 | Model architecture |

### API & Services

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | ≥0.109.0 | Bill upload service |
| `uvicorn` | ≥0.27.0 | ASGI server |
| `python-multipart` | ≥0.0.6 | File upload handling |
| `httpx` | ≥0.26.0 | Async HTTP client |

### Utilities

| Package | Version | Purpose |
|---------|---------|---------|
| `python-dotenv` | ≥1.0.0 | Environment management |
| `Pillow` | ≥10.0.0 | Image processing |
| `numpy` | ≥1.24.0 | Numerical operations |

---

## Running the System

### 1. Start Bill Upload Service

```bash
uvicorn bill_upload_service:app --port 8801
```

### 2. Run Intent Ingestion (first time or updates)

```bash
python intent_ingestion_pipeline.py --knowledge-base heyo.txt
```

### 3. Start Agent Server

```bash
python reva.py
```

---

## License

[Specify your license here]
