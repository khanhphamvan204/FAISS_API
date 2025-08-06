# app/routes/health.py
from fastapi import APIRouter, HTTPException
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/", response_model=dict)
async def health_check():
    """Kiểm tra trạng thái API."""
    logger.info("Health check requested")
    return {"status": "ok"}