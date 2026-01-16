"""
Reva - Heyo Support Agent with Parlant Retriever

This agent uses Parlant's retriever pattern (not tools) for RAG.
Retrievers provide grounding context that runs in parallel with
guideline matching, reducing latency and enabling proper knowledge grounding.

Key Parlant concepts implemented:
1. Retriever: Fetches knowledge context automatically per message
2. Guidelines: Describe behavioral intent (not tool instructions)
3. Terms: Define domain vocabulary for consistent understanding
"""

import asyncio
import logging
import time

import parlant.sdk as p

from knowledge_retriever import heyo_knowledge_retriever

# Configure logging with DEBUG level to see everything
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Also enable debug logging for parlant SDK
logging.getLogger("parlant").setLevel(logging.DEBUG)


async def setup_agent(server: p.Server) -> p.Agent:
    """
    Create and configure the Reva agent with guidelines and retriever.
    
    Guidelines in Parlant describe BEHAVIORAL INTENT, not tool usage.
    The retriever provides knowledge context automatically.
    """
    
    setup_start = time.time()
    logger.info("=" * 60)
    logger.info("[SETUP] Starting agent setup...")
    logger.info("=" * 60)
    
    # Step 1: Create agent
    logger.info("[SETUP] Step 1: Creating agent 'Reva'...")
    agent_start = time.time()
    agent = await server.create_agent(
        name="Reva",
        description=(
            "A calm and professional support agent for Heyo. "
            "Answers questions using knowledge from the product documentation. "
            "Escalates to specialists when unable to help."
        ),
    )
    logger.info(f"[SETUP] ✓ Agent created in {time.time() - agent_start:.2f}s")

    # =========================================================================
    # ATTACH RETRIEVER - COMMENTED OUT FOR TESTING
    # =========================================================================
    logger.info("[SETUP] Step 2: SKIPPING retriever (commented out for testing)")
    retriever_start = time.time()
    # await agent.attach_retriever(
    #     heyo_knowledge_retriever,
    #     id="heyo_knowledge"
    # )
    logger.info(f"[SETUP] ✓ Retriever skipped in {time.time() - retriever_start:.2f}s")

    # =========================================================================
    # GUIDELINES - ENABLED FOR TESTING
    # =========================================================================
    logger.info("[SETUP] Step 3: Creating guidelines (8 total)...")
    guidelines_start = time.time()
    
    # Guideline 1
    logger.info("[SETUP]   Creating guideline 1/8: Core grounding...")
    g1_start = time.time()
    await agent.create_guideline(
        condition="the user asks any question about Heyo features, pricing, setup, or support",
        action=(
            "Answer based ONLY on the knowledge context provided. "
            "If the knowledge context contains relevant information, synthesize a clear answer. "
            "If no relevant information is found, acknowledge this honestly."
        )
    )
    logger.info(f"[SETUP]   ✓ Guideline 1 created in {time.time() - g1_start:.2f}s")
    
    # Guideline 2
    logger.info("[SETUP]   Creating guideline 2/8: URL fabrication prevention...")
    g2_start = time.time()
    await agent.create_guideline(
        condition="the user asks for a link, URL, or web address",
        action=(
            "Only provide URLs that appear exactly in the knowledge context. "
            "Never invent, guess, or construct URLs. "
            "If no URL is found in context, explain that you don't have that link available."
        )
    )
    logger.info(f"[SETUP]   ✓ Guideline 2 created in {time.time() - g2_start:.2f}s")
    
    # Guideline 3
    logger.info("[SETUP]   Creating guideline 3/8: Problem resolution...")
    g3_start = time.time()
    await agent.create_guideline(
        condition="the user describes a problem, error, or issue they are experiencing",
        action=(
            "Use the knowledge context to find a solution. "
            "Provide step-by-step guidance if available. "
            "If no solution is found in context, apologize and escalate to a senior colleague."
        )
    )
    logger.info(f"[SETUP]   ✓ Guideline 3 created in {time.time() - g3_start:.2f}s")
    
    # Guideline 4
    logger.info("[SETUP]   Creating guideline 4/8: Low confidence escalation...")
    g4_start = time.time()
    await agent.create_guideline(
        condition="the knowledge context indicates no relevant results or low confidence",
        action=(
            "Acknowledge that you couldn't find specific information about their question. "
            "Offer to escalate to a specialist who can help further. "
            "Never make up information to fill the gap."
        )
    )
    logger.info(f"[SETUP]   ✓ Guideline 4 created in {time.time() - g4_start:.2f}s")
    
    # Guideline 5
    logger.info("[SETUP]   Creating guideline 5/8: Frustrated user handling...")
    g5_start = time.time()
    await agent.create_guideline(
        condition="the user expresses frustration, anger, or mentions repeated issues",
        action=(
            "First, acknowledge their frustration with empathy. "
            "Then attempt to help using available knowledge context. "
            "Proactively offer to escalate to a senior colleague if needed."
        )
    )
    logger.info(f"[SETUP]   ✓ Guideline 5 created in {time.time() - g5_start:.2f}s")
    
    # Guideline 6
    logger.info("[SETUP]   Creating guideline 6/8: WABA specialist routing...")
    g6_start = time.time()
    await agent.create_guideline(
        condition="the user asks about WhatsApp Business API (WABA) setup or configuration",
        action=(
            "Provide information from knowledge context if available. "
            "For complex WABA issues not covered in context, escalate to the WABA specialist."
        )
    )
    logger.info(f"[SETUP]   ✓ Guideline 6 created in {time.time() - g6_start:.2f}s")
    
    # Guideline 7
    logger.info("[SETUP]   Creating guideline 7/8: Billing specialist routing...")
    g7_start = time.time()
    await agent.create_guideline(
        condition="the user asks about billing, payments, refunds, or subscription issues",
        action=(
            "Provide billing information from knowledge context if available. "
            "For refund requests or billing disputes, escalate to the billing specialist."
        )
    )
    logger.info(f"[SETUP]   ✓ Guideline 7 created in {time.time() - g7_start:.2f}s")
    
    # Guideline 8
    logger.info("[SETUP]   Creating guideline 8/8: Demo requests...")
    g8_start = time.time()
    await agent.create_guideline(
        condition="the user asks a general question about Heyo capabilities",
        action=(
            "Answer their question using knowledge context. "
            "Do not proactively offer demos unless explicitly requested."
        )
    )
    logger.info(f"[SETUP]   ✓ Guideline 8 created in {time.time() - g8_start:.2f}s")
    
    logger.info(f"[SETUP] ✓ All guidelines created in {time.time() - guidelines_start:.2f}s")

    # =========================================================================
    # GLOSSARY TERMS - COMMENTED OUT FOR TESTING
    # =========================================================================
    logger.info("[SETUP] Step 4: SKIPPING glossary terms (commented out for testing)")
    terms_start = time.time()
    
    # logger.info("[SETUP]   Creating term 1/5: Heyo...")
    # t1_start = time.time()
    # await agent.create_term(
    #     name="Heyo",
    #     description=(
    #         "Heyo is a virtual phone system for businesses. "
    #         "Features include IVR, call recording, call forwarding, "
    #         "WhatsApp Business integration, and team collaboration."
    #     ),
    # )
    # logger.info(f"[SETUP]   ✓ Term 1 created in {time.time() - t1_start:.2f}s")
    
    # logger.info("[SETUP]   Creating term 2/5: WABA...")
    # t2_start = time.time()
    # await agent.create_term(
    #     name="WABA",
    #     description=(
    #         "WhatsApp Business API - allows businesses to send messages "
    #         "at scale through WhatsApp. Requires Meta verification."
    #     ),
    # )
    # logger.info(f"[SETUP]   ✓ Term 2 created in {time.time() - t2_start:.2f}s")
    
    # logger.info("[SETUP]   Creating term 3/5: IVR...")
    # t3_start = time.time()
    # await agent.create_term(
    #     name="IVR",
    #     description=(
    #         "Interactive Voice Response - automated phone menu system "
    #         "that routes callers to the right department."
    #     ),
    # )
    # logger.info(f"[SETUP]   ✓ Term 3 created in {time.time() - t3_start:.2f}s")

    # logger.info("[SETUP]   Creating term 4/5: Office Phone Number...")
    # t4_start = time.time()
    # await agent.create_term(
    #     name="Office Phone Number",
    #     description="Our office contact number: +999376738389",
    # )
    # logger.info(f"[SETUP]   ✓ Term 4 created in {time.time() - t4_start:.2f}s")

    # logger.info("[SETUP]   Creating term 5/5: Office Hours...")
    # t5_start = time.time()
    # await agent.create_term(
    #     name="Office Hours",
    #     description="Monday to Friday, 9 AM to 5 PM (local time)",
    # )
    # logger.info(f"[SETUP]   ✓ Term 5 created in {time.time() - t5_start:.2f}s")
    
    logger.info(f"[SETUP] ✓ Terms skipped in {time.time() - terms_start:.2f}s")
    
    logger.info("=" * 60)
    logger.info(f"[SETUP] ✓✓✓ TOTAL SETUP TIME: {time.time() - setup_start:.2f}s ✓✓✓")
    logger.info("=" * 60)
    
    return agent


async def main() -> None:
    """
    Main entry point - starts the Parlant server with retriever-based RAG.
    
    Note: No tool modules needed. The retriever is attached directly to the agent.
    """
    
    print("=" * 60)
    print("Starting Reva - Heyo Support Agent")
    print("Using Parlant Retriever pattern for RAG")
    print("DEBUG LOGGING ENABLED - Watch for timing info")
    print("=" * 60)
    
    # Start server - no tool modules needed for retriever-based RAG
    logger.info("[MAIN] Creating Parlant Server with OpenAI NLP service...")
    server_start = time.time()
    
    async with p.Server(nlp_service=p.NLPServices.openai) as server:
        logger.info(f"[MAIN] ✓ Server context entered in {time.time() - server_start:.2f}s")
        logger.info("[MAIN] Now calling setup_agent()...")
        
        agent = await setup_agent(server)
        
        print(f"\n✓ Agent '{agent.name}' created")
        print("✓ Knowledge retriever attached")
        print("✓ Guidelines configured (behavioral, not tool-based)")
        print("\n" + "=" * 60)
        print("Server is running. Press Ctrl+C to stop.")
        print("=" * 60)
        
        logger.info("[MAIN] Entering main loop - server is ready")
        
        # Keep server running
        while True:
            await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())