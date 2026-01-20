"""
Bill Upload Service

FastAPI service for handling bill/invoice image uploads.
Validates file types, stores temporarily, and returns a file_id
that can be referenced in the Parlant chat.

Run with: uvicorn bill_upload_service:app --port 8801
"""

import os
import uuid
import shutil
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Storage directory for uploaded bills
UPLOAD_DIR = Path(__file__).parent / "uploads" / "bills"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Allowed file types
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}
MAX_FILE_SIZE_MB = 10

app = FastAPI(
    title="Bill Upload Service",
    description="Handles bill/invoice image uploads for refund processing",
    version="1.0.0",
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _validate_file(file: UploadFile) -> None:
    """Validate file type and size."""
    # Check extension
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Check content type
    content_type = file.content_type or ""
    valid_types = {"image/jpeg", "image/png", "application/pdf"}
    if content_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type '{content_type}'. Upload JPEG, PNG, or PDF.",
        )


@app.post("/upload-bill")
async def upload_bill(file: UploadFile = File(...)):
    """
    Upload a bill/invoice image for processing.

    Returns:
        file_id: Unique identifier to reference this file in chat
        filename: Original filename
        uploaded_at: Timestamp of upload
    """
    _validate_file(file)

    # Generate unique file_id
    file_id = uuid.uuid4().hex[:12]
    ext = Path(file.filename or "unknown").suffix.lower()
    stored_filename = f"{file_id}{ext}"
    file_path = UPLOAD_DIR / stored_filename

    # Check file size by reading content
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.1f}MB). Maximum: {MAX_FILE_SIZE_MB}MB",
        )

    # Save file
    with open(file_path, "wb") as f:
        f.write(content)

    return {
        "file_id": file_id,
        "filename": file.filename,
        "stored_path": str(stored_filename),
        "size_bytes": len(content),
        "uploaded_at": datetime.utcnow().isoformat(),
        "message": f"Upload successful. Use file_id '{file_id}' in chat to process this bill.",
    }


@app.get("/bill/{file_id}")
async def get_bill_info(file_id: str):
    """
    Get information about an uploaded bill.
    """
    # Find file with this ID
    matches = list(UPLOAD_DIR.glob(f"{file_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail=f"Bill '{file_id}' not found")

    file_path = matches[0]
    stat = file_path.stat()

    return {
        "file_id": file_id,
        "filename": file_path.name,
        "size_bytes": stat.st_size,
        "exists": True,
    }


@app.delete("/bill/{file_id}")
async def delete_bill(file_id: str):
    """
    Delete an uploaded bill after processing.
    """
    matches = list(UPLOAD_DIR.glob(f"{file_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail=f"Bill '{file_id}' not found")

    file_path = matches[0]
    file_path.unlink()

    return {"file_id": file_id, "deleted": True}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "bill-upload"}


def get_bill_path(file_id: str) -> Path | None:
    """
    Utility function to get the path of an uploaded bill.
    Used by the bill processor.
    """
    matches = list(UPLOAD_DIR.glob(f"{file_id}.*"))
    return matches[0] if matches else None


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8801)
