# app/routes/documents.py
from fastapi import APIRouter, HTTPException
import logging
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from app.config import Config
import os
import json

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/types", response_model=dict)
async def get_file_types():
    """Lấy danh sách các loại file được hỗ trợ."""
    return {
        "file_types": [
            {"value": "public", "label": "Thông báo chung (Public)", "description": "Tài liệu công khai cho tất cả người dùng"},
            {"value": "student", "label": "Sinh viên (Student)", "description": "Tài liệu dành cho sinh viên"},
            {"value": "teacher", "label": "Giảng viên (Teacher)", "description": "Tài liệu dành cho giảng viên"},
            {"value": "admin", "label": "Quản trị viên (Admin)", "description": "Tài liệu dành cho quản trị viên"}
        ]
    }

@router.get("/list", response_model=dict)
async def list_documents(file_type: str = None, limit: int = 100, skip: int = 0):
    """Lấy danh sách tài liệu."""
    try:
        documents = []
        
        # Thử lấy từ MongoDB trước
        try:
            client = MongoClient(Config.DATABASE_URL)
            db = client["faiss_db"]
            collection = db["metadata"]
            
            filter_dict = {"file_type": file_type} if file_type else {}
            cursor = collection.find(filter_dict).skip(skip).limit(limit).sort("createdAt", -1)
            documents = list(cursor)
            client.close()
            
            if documents:
                logger.info(f"Retrieved {len(documents)} documents from MongoDB")
                return {"documents": documents, "total": len(documents), "source": "mongodb"}
        except PyMongoError as e:
            logger.error(f"Failed to retrieve documents from MongoDB: {str(e)}")
        
        # Fallback: Lấy từ metadata.json
        base_path = Config.DATA_PATH
        metadata_paths = [
            f"{base_path}/{Config.FILE_TYPE_PATHS[role]['vector_folder']}/metadata.json"
            for role in Config.FILE_TYPE_PATHS
        ]
        
        for metadata_file in metadata_paths:
            try:
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata_list = json.load(f)
                    
                    if file_type:
                        metadata_list = [item for item in metadata_list if item.get('file_type') == file_type]
                    
                    documents.extend(metadata_list)
            except Exception as e:
                logger.error(f"Error reading {metadata_file}: {str(e)}")
        
        documents.sort(key=lambda x: x.get('createdAt', ''), reverse=True)
        total = len(documents)
        documents = documents[skip:skip + limit]
        
        logger.info(f"Retrieved {len(documents)} documents from JSON files")
        return {"documents": documents, "total": total, "source": "json", "showing": len(documents)}
    
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")