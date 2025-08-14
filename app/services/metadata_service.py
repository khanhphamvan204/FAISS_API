# app/services/metadata_service.py
import os
import json
import logging
from contextlib import contextmanager
from pymongo import MongoClient
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError
from fastapi import HTTPException
from app.config import Config
from app.services.file_service import get_file_paths

logger = logging.getLogger(__name__)

# Connection manager singleton
class MongoManager:
    _instance = None
    _client = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_client(self):
        if self._client is None:
            try:
                # Log the connection URL for debugging (without password)
                connection_url = Config.DATABASE_URL
                if '@' in connection_url:
                    # Hide password in logs
                    masked_url = connection_url.split('@')[0].split('//')[0] + '//***:***@' + connection_url.split('@')[1]
                    logger.info(f"Attempting MongoDB connection to: {masked_url}")
                else:
                    logger.info(f"Attempting MongoDB connection to: {connection_url}")
                
                self._client = MongoClient(
                    Config.DATABASE_URL,
                    maxPoolSize=10,
                    minPoolSize=2,
                    maxIdleTimeMS=30000,
                    waitQueueTimeoutMS=5000,
                    serverSelectionTimeoutMS=10000,  # Longer timeout for external connection
                    connectTimeoutMS=10000,           # Longer timeout for external connection
                    socketTimeoutMS=30000,
                    retryWrites=True,
                    retryReads=True,
                    # Additional options for external MongoDB
                    heartbeatFrequencyMS=30000,
                )
                # Test connection with timeout
                self._client.admin.command('ping')
                logger.info("MongoDB connection established successfully")
                
                # Log server info for debugging
                server_info = self._client.server_info()
                logger.info(f"Connected to MongoDB version: {server_info.get('version', 'unknown')}")
                
            except ServerSelectionTimeoutError as e:
                logger.warning(f"MongoDB server not available: {e}")
                logger.info("Will fallback to JSON file storage for this session")
                self._client = None
                # Don't raise exception, allow fallback
                return None
            except Exception as e:
                logger.error(f"Failed to establish MongoDB connection: {e}")
                self._client = None
                # Don't raise exception, allow fallback
                return None
        return self._client
    
    def is_connected(self):
        """Check if MongoDB is connected and available"""
        try:
            if self._client is None:
                return False
            self._client.admin.command('ping')
            return True
        except:
            return False
    
    def close(self):
        if self._client:
            self._client.close()
            self._client = None
            logger.info("MongoDB connection closed")

# Global instance
mongo_manager = MongoManager()

@contextmanager
def get_mongo_connection():
    """Context manager for MongoDB connections"""
    client = None
    try:
        client = mongo_manager.get_client()
        if client is None:
            # MongoDB not available, will use fallback
            yield None
            return
        yield client
    except Exception as e:
        logger.error(f"MongoDB operation failed: {e}")
        yield None  # Return None instead of raising exception

def save_metadata(metadata):
    """Save metadata to MongoDB with proper error handling"""
    mongodb_success = False
    
    # Try MongoDB first
    try:
        with get_mongo_connection() as client:
            if client is not None:
                db = client["faiss_db"]
                collection = db["metadata"]
                
                metadata_dict = metadata.dict(by_alias=True)
                result = collection.insert_one(metadata_dict)
                
                if result.inserted_id:
                    logger.info(f"Successfully saved metadata to MongoDB for _id: {metadata.id}")
                    mongodb_success = True
                else:
                    raise PyMongoError("Insert operation failed")
    except Exception as e:
        logger.warning(f"MongoDB save failed: {str(e)}")
    
    # Always also save to JSON file as backup/fallback
    try:
        _, vector_db_path = get_file_paths(metadata.file_type, metadata.filename)
        metadata_file = os.path.join(vector_db_path, "metadata.json")
        
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
        
        if mongodb_success:
            logger.info(f"Successfully saved metadata to both MongoDB and JSON file: {metadata_file}")
        else:
            logger.info(f"Fallback: Successfully saved metadata to JSON file: {metadata_file}")
        return True
        
    except Exception as file_error:
        logger.error(f"Failed to save to JSON file: {file_error}")
        return mongodb_success  # Return True if MongoDB worked, False if both failed

def delete_metadata(doc_id: str) -> bool:
    """Delete metadata with proper connection handling"""
    mongodb_success = False
    json_success = False
    
    # Try MongoDB first
    try:
        with get_mongo_connection() as client:
            if client is not None:
                db = client["faiss_db"]
                collection = db["metadata"]
                
                result = collection.delete_one({"_id": doc_id})
                if result.deleted_count > 0:
                    logger.info(f"Successfully deleted metadata from MongoDB for _id: {doc_id}")
                    mongodb_success = True
    except Exception as e:
        logger.warning(f"MongoDB delete failed: {str(e)}")
    
    # Also try JSON files (for backup/fallback)
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
                    json_success = True
        except Exception as e:
            logger.error(f"Error deleting from {metadata_file}: {str(e)}")
    
    return mongodb_success or json_success

def find_document_info(doc_id: str) -> dict:
    """Find document info with proper connection handling"""
    # Try MongoDB first
    try:
        with get_mongo_connection() as client:
            if client is not None:
                db = client["faiss_db"]
                collection = db["metadata"]
                
                doc_info = collection.find_one({"_id": doc_id})
                
                if doc_info:
                    logger.info(f"Found document in MongoDB: {doc_id}")
                    return doc_info
    except Exception as e:
        logger.warning(f"MongoDB search failed: {str(e)}")
    
    # Fallback to JSON files
    logger.info("Searching in JSON files as fallback")
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
                        logger.info(f"Found document in JSON file: {metadata_file}")
                        return item
        except Exception as e:
            logger.error(f"Error reading {metadata_file}: {str(e)}")
    
    logger.warning(f"Document not found: {doc_id}")
    return None

def get_connection_status():
    """Get current MongoDB connection status"""
    status = {
        "mongodb_connected": mongo_manager.is_connected(),
        "connection_url": Config.DATABASE_URL.split('@')[1] if '@' in Config.DATABASE_URL else Config.DATABASE_URL
    }
    return status

# Optional: Add this to your FastAPI app startup
def initialize_mongo():
    """Initialize MongoDB connection at startup"""
    try:
        client = mongo_manager.get_client()
        if client:
            logger.info("MongoDB initialized successfully")
        else:
            logger.warning("MongoDB not available - using JSON fallback mode")
    except Exception as e:
        logger.error(f"Failed to initialize MongoDB: {e}")

# Optional: Add this to your FastAPI app shutdown
def close_mongo():
    """Close MongoDB connection at shutdown"""
    mongo_manager.close()