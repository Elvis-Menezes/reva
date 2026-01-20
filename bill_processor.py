"""
Bill Processor Module

Handles bill/invoice image processing using OpenAI Vision API.
Extracts structured data from bill images for refund eligibility assessment.

This module is deterministic in its processing logic - the extraction
happens via OpenAI Vision, but the eligibility rules are fixed.
"""

import base64
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Any

from openai import OpenAI

from bill_upload_service import get_bill_path

logger = logging.getLogger(__name__)

# Initialize OpenAI client (uses OPENAI_API_KEY env var)
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Lazy initialization of OpenAI client."""
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


@dataclass
class BillData:
    """Structured data extracted from a bill image."""

    invoice_number: str | None
    date: str | None
    total_amount: float | None
    currency: str | None
    vendor_name: str | None
    payment_status: str | None
    line_items: list[dict[str, Any]]
    raw_text: str | None
    extraction_confidence: str  # "high", "medium", "low"


@dataclass
class RefundAssessment:
    """Refund eligibility assessment based on bill data."""

    eligible: bool
    reason: str
    bill_data: BillData
    requires_review: bool
    max_refund_amount: float | None


# Extraction prompt for OpenAI Vision
EXTRACTION_PROMPT = """Analyze this bill/invoice image and extract the following information.
Return ONLY a valid JSON object with these fields:

{
  "invoice_number": "string or null if not found",
  "date": "string in YYYY-MM-DD format or null",
  "total_amount": "number or null",
  "currency": "3-letter code like USD, INR, EUR or null",
  "vendor_name": "string or null",
  "payment_status": "paid, unpaid, pending, or null if unclear",
  "line_items": [{"description": "string", "amount": number}] or [],
  "raw_text": "brief summary of visible text",
  "confidence": "high, medium, or low based on image clarity"
}

If a field cannot be determined, use null.
Do not include any text outside the JSON object."""


def _encode_image_to_base64(file_path: Path) -> str:
    """Read and encode image file to base64."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _get_media_type(file_path: Path) -> str:
    """Determine media type from file extension."""
    ext = file_path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".pdf": "application/pdf",
    }
    return media_types.get(ext, "image/jpeg")


def extract_bill_data(file_id: str) -> BillData:
    """
    Extract structured data from a bill image using OpenAI Vision.

    Args:
        file_id: The unique identifier of the uploaded bill

    Returns:
        BillData with extracted information

    Raises:
        FileNotFoundError: If bill with file_id doesn't exist
        ValueError: If extraction fails
    """
    # Get file path
    file_path = get_bill_path(file_id)
    if file_path is None:
        raise FileNotFoundError(f"Bill with file_id '{file_id}' not found")

    logger.info(f"[BILL-PROCESSOR] Processing bill: {file_id}")

    # Encode image
    image_data = _encode_image_to_base64(file_path)
    media_type = _get_media_type(file_path)

    # Call OpenAI Vision API
    client = _get_client()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": EXTRACTION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_data}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
        max_tokens=1000,
    )

    # Parse response
    raw_content = response.choices[0].message.content or "{}"
    logger.debug(f"[BILL-PROCESSOR] Raw extraction: {raw_content}")

    try:
        # Strip markdown code fences if present
        content = raw_content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            content = content.rsplit("```", 1)[0]

        data = json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"[BILL-PROCESSOR] JSON parse error: {e}")
        # Return minimal data on parse failure
        return BillData(
            invoice_number=None,
            date=None,
            total_amount=None,
            currency=None,
            vendor_name=None,
            payment_status=None,
            line_items=[],
            raw_text=raw_content[:500],
            extraction_confidence="low",
        )

    return BillData(
        invoice_number=data.get("invoice_number"),
        date=data.get("date"),
        total_amount=data.get("total_amount"),
        currency=data.get("currency"),
        vendor_name=data.get("vendor_name"),
        payment_status=data.get("payment_status"),
        line_items=data.get("line_items", []),
        raw_text=data.get("raw_text"),
        extraction_confidence=data.get("confidence", "medium"),
    )


def _parse_bill_date(date_str: str | None) -> date | None:
    """
    Parse bill date from various formats.
    
    OpenAI Vision should return YYYY-MM-DD, but we handle common variations.
    """
    if not date_str:
        return None
    
    # Common date formats to try
    formats = [
        "%Y-%m-%d",      # 2022-04-01 (ISO format - expected from Vision)
        "%d/%m/%Y",      # 01/04/2022
        "%m/%d/%Y",      # 04/01/2022
        "%d-%m-%Y",      # 01-04-2022
        "%d %B %Y",      # 01 April 2022
        "%d %b %Y",      # 01 Apr 2022
        "%B %d, %Y",     # April 01, 2022
        "%b %d, %Y",     # Apr 01, 2022
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue
    
    logger.warning(f"[BILL-PROCESSOR] Could not parse date: {date_str}")
    return None


def _calculate_days_since(bill_date: date) -> int:
    """Calculate number of days between bill date and today."""
    today = date.today()
    delta = today - bill_date
    return delta.days


# Refund window in days
REFUND_WINDOW_DAYS = 30


def assess_refund_eligibility(bill_data: BillData) -> RefundAssessment:
    """
    Assess refund eligibility based on extracted bill data.

    This is deterministic business logic - no LLM calls here.

    Rules:
    - Must have total_amount to be eligible
    - Bill date must be within 30 days (REFUND_WINDOW_DAYS)
    - Payment status must not be "unpaid"
    - Low confidence extractions require manual review
    - Max refund is the total_amount from the bill
    """
    # Check for required data
    if bill_data.total_amount is None:
        return RefundAssessment(
            eligible=False,
            reason="Could not extract total amount from bill image",
            bill_data=bill_data,
            requires_review=True,
            max_refund_amount=None,
        )

    # Check bill date - MUST be within refund window
    bill_date = _parse_bill_date(bill_data.date)
    if bill_date is None:
        return RefundAssessment(
            eligible=False,
            reason="Could not determine bill date - please provide a clearer image",
            bill_data=bill_data,
            requires_review=True,
            max_refund_amount=None,
        )
    
    days_since = _calculate_days_since(bill_date)
    if days_since > REFUND_WINDOW_DAYS:
        return RefundAssessment(
            eligible=False,
            reason=f"Bill is {days_since} days old. Refunds are only available within {REFUND_WINDOW_DAYS} days of purchase.",
            bill_data=bill_data,
            requires_review=False,
            max_refund_amount=None,
        )
    
    if days_since < 0:
        return RefundAssessment(
            eligible=False,
            reason=f"Bill date ({bill_data.date}) is in the future - please verify the bill",
            bill_data=bill_data,
            requires_review=True,
            max_refund_amount=None,
        )

    # Check payment status
    if bill_data.payment_status == "unpaid":
        return RefundAssessment(
            eligible=False,
            reason="Bill shows as unpaid - cannot refund unpaid bills",
            bill_data=bill_data,
            requires_review=False,
            max_refund_amount=None,
        )

    # Low confidence requires human review
    if bill_data.extraction_confidence == "low":
        return RefundAssessment(
            eligible=True,
            reason=f"Bill from {days_since} days ago extracted with low confidence - requires manual review",
            bill_data=bill_data,
            requires_review=True,
            max_refund_amount=bill_data.total_amount,
        )

    # Standard eligible case
    return RefundAssessment(
        eligible=True,
        reason=f"Bill verified successfully ({days_since} days old, within {REFUND_WINDOW_DAYS}-day refund window)",
        bill_data=bill_data,
        requires_review=False,
        max_refund_amount=bill_data.total_amount,
    )


def process_bill_for_refund(file_id: str) -> dict[str, Any]:
    """
    Main entry point: process a bill image and assess refund eligibility.

    Args:
        file_id: The unique identifier of the uploaded bill

    Returns:
        Dictionary with bill data and refund assessment suitable for ToolResult
    """
    try:
        # Extract bill data
        bill_data = extract_bill_data(file_id)

        # Assess eligibility
        assessment = assess_refund_eligibility(bill_data)

        # Calculate days since bill for display
        bill_date = _parse_bill_date(bill_data.date)
        days_since = _calculate_days_since(bill_date) if bill_date else None

        return {
            "status": "processed",
            "file_id": file_id,
            "bill": {
                "invoice_number": bill_data.invoice_number,
                "date": bill_data.date,
                "days_since_issued": days_since,
                "total_amount": bill_data.total_amount,
                "currency": bill_data.currency,
                "vendor_name": bill_data.vendor_name,
                "payment_status": bill_data.payment_status,
                "line_items_count": len(bill_data.line_items),
                "extraction_confidence": bill_data.extraction_confidence,
            },
            "refund": {
                "eligible": assessment.eligible,
                "reason": assessment.reason,
                "requires_review": assessment.requires_review,
                "max_refund_amount": assessment.max_refund_amount,
                "refund_window_days": REFUND_WINDOW_DAYS,
            },
            "side_effect": "none",
        }

    except FileNotFoundError:
        return {
            "status": "error",
            "error": "file_not_found",
            "message": f"No bill found with file_id '{file_id}'. Please upload a bill first.",
            "retryable": True,
        }
    except Exception as e:
        logger.exception(f"[BILL-PROCESSOR] Error processing bill {file_id}: {e}")
        return {
            "status": "error",
            "error": "processing_failed",
            "message": str(e),
            "retryable": True,
        }
