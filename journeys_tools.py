import hashlib
import logging

import parlant.sdk as p

from bill_processor import process_bill_for_refund

logger = logging.getLogger(__name__)


def _stable_id(*parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return digest[:12]


@p.tool
async def verify_id_tool(
    context: p.ToolContext,
    document_id: str | None = None,
    document_type: str | None = None,
    country: str | None = None,
) -> p.ToolResult:
    """
    Deterministic ID verification stub.
    Returns structured metadata for explainability and next-step guidance.
    """
    if not document_id or not document_type:
        return p.ToolResult(
            data={
                "status": "invalid_input",
                "missing_fields": [
                    name
                    for name, value in {
                        "document_id": document_id,
                        "document_type": document_type,
                    }.items()
                    if not value
                ],
                "retryable": True,
            }
        )

    verification_id = _stable_id(document_id, document_type, country or "unknown")
    return p.ToolResult(
        data={
            "status": "verified",
            "verification_id": verification_id,
            "document_type": document_type,
            "country": country or "unknown",
            "side_effect": "none",
        }
    )


@p.tool
async def initiate_human_handoff(
    context: p.ToolContext,
    reason: str,
    urgency: str = "normal",
) -> p.ToolResult:
    """
    Deterministic escalation stub. Marks the request for human follow-up.
    """
    handoff_id = _stable_id(reason, urgency)
    return p.ToolResult(
        data={
            "status": "queued",
            "handoff_id": handoff_id,
            "urgency": urgency,
            "requires_human": True,
            "side_effect": "handoff_requested",
        }
    )


@p.tool(consequential=True)
async def check_refund_eligibility(
    context: p.ToolContext,
    account_id: str | None = None,
    order_id: str | None = None,
    days_since_purchase: int | None = None,
    reason: str | None = None,
) -> p.ToolResult:
    """
    Checks refund eligibility based on account and order metadata.
    Marked as consequential because refund decisions have business impact.
    """
    missing = [
        name
        for name, value in {
            "account_id": account_id,
            "order_id": order_id,
            "days_since_purchase": days_since_purchase,
        }.items()
        if value is None
    ]
    if missing:
        return p.ToolResult(
            data={
                "status": "invalid_input",
                "missing_fields": missing,
                "retryable": True,
            }
        )

    in_window = days_since_purchase <= 7
    blocked_reason = (reason or "").lower() in {"fraud", "abuse"}
    eligible = in_window and not blocked_reason
    return p.ToolResult(
        data={
            "status": "eligible" if eligible else "ineligible",
            "eligible": eligible,
            "refund_window_days": 7,
            "days_since_purchase": days_since_purchase,
            "blocked_reason": blocked_reason,
            "side_effect": "none",
        }
    )


@p.tool(consequential=True)
async def process_bill_image(
    context: p.ToolContext,
    bill_image_url: str,
) -> p.ToolResult:
    """
    Uses OpenAI Vision to analyze the bill and extract transaction details
    for refund verification. Marked as consequential because it affects
    refund decisions.
    
    Args:
        context: Parlant tool context (automatically provided)
        bill_image_url: The file_id or URL of the uploaded bill image
    
    Returns:
        ToolResult with bill data and refund eligibility assessment
    """
    logger.info(f"[TOOL] process_bill_image called with bill_image_url={bill_image_url}")

    if not bill_image_url or not bill_image_url.strip():
        return p.ToolResult(
            data={
                "status": "invalid_input",
                "error": "missing_bill_url",
                "message": "Please provide the file_id or URL of your bill.",
                "retryable": True,
            },
            metadata={"vision_agent_trace": "invalid_input"}
        )

    # Process the bill (runs OpenAI Vision extraction + eligibility check)
    result = process_bill_for_refund(bill_image_url.strip())

    logger.info(f"[TOOL] process_bill_image result: status={result.get('status')}")
    return p.ToolResult(
        data=result,
        metadata={"vision_agent_trace": "success" if result.get("status") == "processed" else "failed"}
    )


async def create_onboarding_journey(agent: p.Agent) -> None:
    onboarding = await agent.create_journey(
        title="Onboarding & Setup",
        description="Guides post-signup users through KYC, staff setup, and WABA registration.",
        conditions=["The customer needs to set up their account or register for WABA"],
    )

    t1 = await onboarding.initial_state.transition_to(
        chat_state=(
            "Welcome! Would you like to start with KYC verification or WABA registration?"
        )
    )

    t2_kyc = await t1.target.transition_to(
        chat_state="Please share your official ID details to proceed with KYC.",
        condition="The user chooses KYC verification",
    )

    t3_verify = await t2_kyc.target.transition_to(
        tool_state=verify_id_tool,
        condition="The user provides an identity document",
        tool_instruction="Verify the identity document provided by the customer.",
    )

    await t3_verify.target.transition_to(
        chat_state=(
            "Thanks! Your verification is complete. "
            
        )
    )

    t2_waba = await t1.target.transition_to(
        chat_state="What professional phone number should we use for WhatsApp?",
        condition="The user chooses WABA registration",
    )

    await t2_waba.target.transition_to(
        chat_state=(
            "Great. I will use that number for WABA registration and share next steps."
        )
    )


async def create_support_journey(agent: p.Agent) -> None:
    support = await agent.create_journey(
        title="Support & Retention",
        description="Handles technical issues, billing inquiries, and human handoffs.",
        conditions=["The user has a problem, is stuck, or requests a refund"],
    )

    t1 = await support.initial_state.transition_to(
        chat_state="I’m sorry you’re running into this. Can you share your account details?"
    )

    t2_solve = await t1.target.transition_to(
        chat_state=(
            "I’ll look into this and propose one solution. "
            "If it doesn’t help, I can escalate."
        ),
        condition="The user provides their account details",
    )

    t3_handoff = await t2_solve.target.transition_to(
        tool_state=initiate_human_handoff,
        condition="The first solution failed to solve the user's problem",
        tool_instruction="Escalate to a human agent because automated solutions did not work.",
    )

    await t3_handoff.target.transition_to(
        chat_state=(
            "I’ve escalated this to a specialist. "
            "Would you like to add any more details while we connect you?"
        )
    )

    t2_refund = await t1.target.transition_to(
        tool_state=check_refund_eligibility,
        condition="The user is asking for a refund",
        tool_instruction="Check the refund eligibility for this customer's order.",
    )

    t3_refund_chat = await t2_refund.target.transition_to(
        chat_state=(
            "I checked the refund eligibility. "
            "If you have a bill or invoice image, please upload it and share the file_id. "
            "Otherwise, let me know how you'd like to proceed."
        )
    )

    # Bill image processing path
    t4_bill = await t3_refund_chat.target.transition_to(
        tool_state=process_bill_image,
        condition="The user provides a bill URL or image link",
        tool_instruction="Verify the bill provided by the customer using vision analysis.",
    )

    await t4_bill.target.transition_to(
        chat_state="If the bill is eligible, confirm the refund. Otherwise, explain why.",
        condition="The vision tool has completed the analysis",
    )
