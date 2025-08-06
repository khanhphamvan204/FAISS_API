# app/services/metadata_service.py
import os
import json
import logging
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from fastapi import HTTPException
from app.config import Config
from app.services.file_service import get_file_paths

logger = logging.getLogger(__name__)

def save_metadata(metadata):
    try:
        _, vector_db_path = get_file_paths(metadata.file_type, metadata.filename)
        metadata_file = os.path.join(vector_db_path, "metadata.json")
        
        client = MongoClient(Config.DATABASE_URL)
        db = client["faiss_db"]
        collection = db["metadata"]
        
        metadata_dict = metadata.dict(by_alias=True)
        collection.insert_one(metadata_dict)
        logger.info(f"Successfully saved metadata to MongoDB for _id: {metadata.id}")
        client.close()
    except PyMongoError as e:
        logger.error(f"Failed to save metadata to MongoDB: {str(e)}")
        existing_metadata = []
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                existing_metadata = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logger.info(f"No existing metadata.json found at {metadata_file}, creating new")
        
        metadata_dict = metadata.dict(by_alias=True)
        existing_metadata.append(metadata_dict)
        
        os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(existing_metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"Fallback: Successfully saved metadata to {metadata_file}")

def delete_metadata(doc_id: str) -> bool:
    success = False
    
    try:
        client = MongoClient(Config.DATABASE_URL)
        db = client["faiss_db"]
        collection = db["metadata"]
        
        result = collection.delete_one({"_id": doc_id})
        if result.deleted_count > 0:
            logger.info(f"Successfully deleted metadata from MongoDB for _id: {doc_id}")
            success = True
        client.close()
    except PyMongoError as e:
        logger.error(f"Failed to delete metadata from MongoDB: {str(e)}")
    
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
                
                original_length = len(metadata_list)
                metadata_list = [item for item in metadata_list if item.get('_id') != doc_id]
                
                if len(metadata_list) < original_length:
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        json.dump(metadata_list, f, ensure_ascii=False, indent=2)
                    logger.info(f"Successfully deleted metadata from {metadata_file}")
                    success = True
        except Exception as e:
            logger.error(f"Error deleting from {metadata_file}: {str(e)}")
    
    return success

def find_document_info(doc_id: str) -> dict:
    try:
        client = MongoClient(Config.DATABASE_URL)
        db = client["faiss_db"]
        collection = db["metadata"]
        
        doc_info = collection.find_one({"_id": doc_id})
        client.close()
        
        if doc_info:
            return doc_info
    except PyMongoError as e:
        logger.error(f"Failed to find document in MongoDB: {str(e)}")
    
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
                
                for item in metadata_list:
                    if item.get('_id') == doc_id:
                        return item
        except Exception as e:
            logger.error(f"Error reading {metadata_file}: {str(e)}")
    
    return None