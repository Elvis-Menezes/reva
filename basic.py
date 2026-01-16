import asyncio
import parlant.sdk as p
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

async def main():
    # Initialize the server with the OpenAI service
    async with p.Server(nlp_service=p.NLPServices.openai) as server:
        # Create your agent as usual
        agent = await server.create_agent(
            name="Reva",
            description=(
            "A calm and professional support agent for Heyo. "
            "Answers questions using knowledge from the product documentation. "
            "Escalates to specialists when unable to help."
        ),
        )
        
        await agent.create_guideline(
            condition="the customer is frustrated",
            action="respond with empathy and offer to escalate",
            criticality=p.Criticality.HIGH
)
        await agent.create_guideline(
        condition="the user asks any question about Heyo features, pricing, setup, or support",
        action=(
            "Answer based ONLY on the knowledge context provided. "
            "If the knowledge context contains relevant information, synthesize a clear answer. "
            "If no relevant information is found, acknowledge this honestly."
        )
    )    
        
if __name__ == "__main__":
    asyncio.run(main())