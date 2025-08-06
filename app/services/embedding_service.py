# app/services/embedding_service.py
import os
import logging
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from app.config import Config
from app.services.file_service import get_file_paths
from app.services.metadata_service import find_document_info
from .document_loader import load_new_documents

logger = logging.getLogger(__name__)

def add_to_embedding(file_path: str, metadata):
    documents = load_new_documents(file_path, metadata)
    if not documents:
        logger.warning(f"No documents loaded from {file_path}")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    if not chunks:
        logger.warning(f"No chunks created from {file_path}")
        return
    
    _, vector_db_path = get_file_paths(metadata.file_type, metadata.filename)
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    index_exists = os.path.exists(f"{vector_db_path}/index.faiss") and os.path.exists(f"{vector_db_path}/index.pkl")
    db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True) if index_exists else FAISS.from_documents(chunks, embedding_model)
    
    db.add_documents(chunks)
    os.makedirs(vector_db_path, exist_ok=True)
    db.save_local(vector_db_path)
    logger.info(f"Successfully saved FAISS index to {vector_db_path}")

def delete_from_faiss_index(vector_db_path: str, doc_id: str) -> bool:
    try:
        index_path = f"{vector_db_path}/index.faiss"
        pkl_path = f"{vector_db_path}/index.pkl"
        
        if not (os.path.exists(index_path) and os.path.exists(pkl_path)):
            logger.warning(f"No FAISS index found at {vector_db_path}")
            return True
        
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
        
        docstore = db.docstore
        index_to_docstore_id = db.index_to_docstore_id
        ids_to_delete = [docstore_id for index, docstore_id in index_to_docstore_id.items() 
                        if docstore.search(docstore_id).metadata.get('_id') == doc_id]
        
        if ids_to_delete:
            db.delete(ids=ids_to_delete)
            db.save_local(vector_db_path)
            logger.info(f"Deleted {len(ids_to_delete)} documents with _id: {doc_id}")
        else:
            logger.warning(f"No documents found with _id: {doc_id}")
        
        return True
    except Exception as e:
        logger.error(f"Error deleting from FAISS index: {str(e)}")
        return False

def update_document_metadata_in_vector_store(doc_id: str, old_metadata: dict, new_metadata) -> bool:
    try:
        old_file_type = old_metadata.get('file_type')
        old_filename = old_metadata.get('filename')
        
        _, old_vector_db_path = get_file_paths(old_file_type, old_filename)
        
        if not (os.path.exists(f"{old_vector_db_path}/index.faiss") and os.path.exists(f"{old_vector_db_path}/index.pkl")):
            logger.warning(f"Vector database not found at {old_vector_db_path}")
            return False
        
        success = delete_from_faiss_index(old_vector_db_path, doc_id)
        if not success:
            return False
        
        file_path = new_metadata.url
        if os.path.exists(file_path):
            add_to_embedding(file_path, new_metadata)
            return True
        else:
            logger.error(f"File not found for re-embedding: {file_path}")
            return False
    except Exception as e:
        logger.error(f"Error updating metadata in vector store: {str(e)}")
        return False

def update_metadata_only(doc_id: str, new_metadata) -> bool:
    try:
        _, vector_db_path = get_file_paths(new_metadata.file_type, new_metadata.filename)
        
        if not (os.path.exists(f"{vector_db_path}/index.faiss") and os.path.exists(f"{vector_db_path}/index.pkl")):
            logger.warning(f"Vector database not found at {vector_db_path}")
            return False
        
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
        
        docstore = db.docstore
        index_to_docstore_id = db.index_to_docstore_id
        updated_count = 0
        new_metadata_dict = new_metadata.dict(by_alias=True)
        
        for index, docstore_id in index_to_docstore_id.items():
            doc = docstore.search(docstore_id)
            if doc and doc.metadata.get('_id') == doc_id:
                doc.metadata = new_metadata_dict
                docstore.add({docstore_id: doc})
                updated_count += 1
        
        if updated_count > 0:
            db.save_local(vector_db_path)
            logger.info(f"Updated metadata for {updated_count} chunks of document: {doc_id}")
            return True
        else:
            logger.warning(f"No chunks found for document: {doc_id}")
            return False
    except Exception as e:
        logger.error(f"Error updating metadata only: {str(e)}")
        return False

def smart_metadata_update(doc_id: str, old_metadata: dict, new_metadata, force_re_embed: bool = False) -> bool:
    try:
        file_type_changed = old_metadata.get('file_type') != new_metadata.file_type
        filename_changed = old_metadata.get('filename') != new_metadata.filename
        
        if file_type_changed or filename_changed or force_re_embed:
            return update_document_metadata_in_vector_store(doc_id, old_metadata, new_metadata)
        else:
            success = update_metadata_only(doc_id, new_metadata)
            if not success:
                return update_document_metadata_in_vector_store(doc_id, old_metadata, new_metadata)
            return success
    except Exception as e:
        logger.error(f"Error in smart metadata update: {str(e)}")
        return False