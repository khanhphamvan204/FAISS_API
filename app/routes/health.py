# app/routes/health.py
from fastapi import APIRouter, HTTPException, Depends
import logging
from app.services.auth_service import verify_token

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/", response_model=dict)
async def health_check(current_user: dict = Depends(verify_token)):
    """Kiểm tra trạng thái API."""
    logger.info("Health check requested")
    return {"status": "ok"}