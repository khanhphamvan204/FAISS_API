# app/services/embedding_service.py
import os
import logging
import gc
from typing import List
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from app.config import Config
from app.services.file_service import get_file_paths
from app.services.metadata_service import find_document_info
from .document_loader import load_new_documents
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class SentenceTransformerEmbeddings:
    """
    Wrapper để tương thích với LangChain FAISS và SemanticChunker
    """
    _model = None
    
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        if SentenceTransformerEmbeddings._model is None:
            SentenceTransformerEmbeddings._model = SentenceTransformer(model_name)
        self.model = SentenceTransformerEmbeddings._model
    
    def __call__(self, text: str) -> List[float]:
        """Make the class callable - required by FAISS"""
        return self.embed_query(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        try:
            # Process in small batches to avoid memory issues
            batch_size = 8
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                embeddings = self.model.encode(
                    batch,
                    convert_to_tensor=False,
                    show_progress_bar=False,
                    batch_size=4
                )
                all_embeddings.extend(embeddings.tolist())
                
                # Clear memory after each batch
                gc.collect()
            
            logger.info(f"Successfully embedded {len(texts)} documents")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error in embed_documents: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Embed single query"""
        try:
            embedding = self.model.encode([text], convert_to_tensor=False)
            return embedding[0].tolist()
        except Exception as e:
            logger.error(f"Error in embed_query: {e}")
            raise

def get_embedding_model():
    """Get consistent embedding model"""
    return SentenceTransformerEmbeddings()

def get_text_splitter(use_semantic: bool = True):
    """Get text splitter - semantic or traditional"""
    try:
        if use_semantic:
            logger.info("Using SemanticChunker for text splitting")
            embedding_model = get_embedding_model()
            return SemanticChunker(
                embeddings=embedding_model,
                breakpoint_threshold_type="percentile",  # hoặc "standard_deviation"
                breakpoint_threshold_amount=95,  # top 5% differences become breakpoints
                number_of_chunks=None,  # không giới hạn số chunk
                buffer_size=1,  # số câu buffer xung quanh breakpoint
            )
        else:
            logger.info("Using RecursiveCharacterTextSplitter for text splitting")
            return RecursiveCharacterTextSplitter(
                chunk_size=2000, 
                chunk_overlap=200
            )
    except Exception as e:
        logger.warning(f"Failed to create SemanticChunker: {e}")
        logger.info("Falling back to RecursiveCharacterTextSplitter")
        return RecursiveCharacterTextSplitter(
            chunk_size=2000, 
            chunk_overlap=200
        )

def add_to_embedding(file_path: str, metadata, use_semantic_chunking: bool = True):
    """Add documents to vector database"""
    try:
        logger.info(f"Starting embedding process for: {file_path}")
        
        # Load documents
        documents = load_new_documents(file_path, metadata)
        if not documents:
            logger.warning(f"No documents loaded from {file_path}")
            return False

        # Split into chunks with semantic or traditional splitter
        text_splitter = get_text_splitter(use_semantic=use_semantic_chunking)
        
        try:
            chunks = text_splitter.split_documents(documents)
            logger.info(f"Successfully created {len(chunks)} chunks using {'semantic' if use_semantic_chunking else 'traditional'} chunking")
        except Exception as e:
            if use_semantic_chunking:
                logger.warning(f"Semantic chunking failed: {e}. Falling back to traditional chunking")
                text_splitter = get_text_splitter(use_semantic=False)
                chunks = text_splitter.split_documents(documents)
                logger.info(f"Created {len(chunks)} chunks using fallback traditional chunking")
            else:
                raise e
        
        if not chunks:
            logger.warning(f"No chunks created from {file_path}")
            return False
        
        # Get paths and embedding model
        _, vector_db_path = get_file_paths(metadata.file_type, metadata.filename)
        embedding_model = get_embedding_model()
        
        # Ensure directory exists
        os.makedirs(vector_db_path, exist_ok=True)
        
        # Check if index exists
        index_exists = (
            os.path.exists(f"{vector_db_path}/index.faiss") and 
            os.path.exists(f"{vector_db_path}/index.pkl")
        )
        
        if index_exists:
            logger.info("Loading existing FAISS index")
            try:
                db = FAISS.load_local(
                    vector_db_path, 
                    embedding_model, 
                    allow_dangerous_deserialization=True
                )
                # Add new chunks to existing database
                db.add_documents(chunks)
                logger.info(f"Added {len(chunks)} chunks to existing database")
            except Exception as e:
                logger.error(f"Failed to load existing index: {e}")
                logger.info("Creating new FAISS index")
                db = FAISS.from_documents(chunks, embedding_model)
        else:
            logger.info("Creating new FAISS index")
            db = FAISS.from_documents(chunks, embedding_model)
        
        # Save the database
        try:
            db.save_local(vector_db_path)
            logger.info(f"Successfully saved FAISS index to {vector_db_path}")
            
            # Verify files were created
            if os.path.exists(f"{vector_db_path}/index.faiss"):
                faiss_size = os.path.getsize(f"{vector_db_path}/index.faiss")
                logger.info(f"FAISS index file size: {faiss_size} bytes")
            
            # Clear memory
            gc.collect()
            return True
            
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Error in add_to_embedding: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def delete_from_faiss_index(vector_db_path: str, doc_id: str) -> bool:
    """Delete documents from FAISS index"""
    try:
        index_path = f"{vector_db_path}/index.faiss"
        pkl_path = f"{vector_db_path}/index.pkl"
        
        if not (os.path.exists(index_path) and os.path.exists(pkl_path)):
            logger.warning(f"No FAISS index found at {vector_db_path}")
            return True
        
        # Use consistent embedding model
        embedding_model = get_embedding_model()
        db = FAISS.load_local(
            vector_db_path, 
            embedding_model, 
            allow_dangerous_deserialization=True
        )
        
        # Find documents to delete
        docstore = db.docstore
        index_to_docstore_id = db.index_to_docstore_id
        ids_to_delete = []
        
        for index, docstore_id in index_to_docstore_id.items():
            doc = docstore.search(docstore_id)
            if doc and doc.metadata.get('_id') == doc_id:
                ids_to_delete.append(docstore_id)
        
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

def update_document_metadata_in_vector_store(doc_id: str, old_metadata: dict, new_metadata, use_semantic_chunking: bool = True) -> bool:
    """Update document by re-embedding"""
    try:
        old_file_type = old_metadata.get('file_type')
        old_filename = old_metadata.get('filename')
        
        _, old_vector_db_path = get_file_paths(old_file_type, old_filename)
        
        # Delete old document
        success = delete_from_faiss_index(old_vector_db_path, doc_id)
        if not success:
            return False
        
        # Re-embed with new metadata
        file_path = new_metadata.url
        if os.path.exists(file_path):
            return add_to_embedding(file_path, new_metadata, use_semantic_chunking)
        else:
            logger.error(f"File not found for re-embedding: {file_path}")
            return False
            
    except Exception as e:
        logger.error(f"Error updating metadata in vector store: {str(e)}")
        return False

def update_metadata_only(doc_id: str, new_metadata) -> bool:
    """Update only metadata without re-embedding"""
    try:
        _, vector_db_path = get_file_paths(new_metadata.file_type, new_metadata.filename)
        
        if not (os.path.exists(f"{vector_db_path}/index.faiss") and 
                os.path.exists(f"{vector_db_path}/index.pkl")):
            logger.warning(f"Vector database not found at {vector_db_path}")
            return False
        
        # Use consistent embedding model
        embedding_model = get_embedding_model()
        db = FAISS.load_local(
            vector_db_path, 
            embedding_model, 
            allow_dangerous_deserialization=True
        )
        
        # Update metadata
        docstore = db.docstore
        index_to_docstore_id = db.index_to_docstore_id
        updated_count = 0
        new_metadata_dict = new_metadata.dict(by_alias=True)
        
        for index, docstore_id in index_to_docstore_id.items():
            doc = docstore.search(docstore_id)
            if doc and doc.metadata.get('_id') == doc_id:
                doc.metadata.update(new_metadata_dict)
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

def smart_metadata_update(doc_id: str, old_metadata: dict, new_metadata, force_re_embed: bool = False, use_semantic_chunking: bool = True) -> bool:
    """Smart metadata update with fallback logic"""
    try:
        file_type_changed = old_metadata.get('file_type') != new_metadata.file_type
        filename_changed = old_metadata.get('filename') != new_metadata.filename
        
        if file_type_changed or filename_changed or force_re_embed:
            return update_document_metadata_in_vector_store(doc_id, old_metadata, new_metadata, use_semantic_chunking)
        else:
            success = update_metadata_only(doc_id, new_metadata)
            if not success:
                logger.info("Metadata-only update failed, attempting re-embedding")
                return update_document_metadata_in_vector_store(doc_id, old_metadata, new_metadata, use_semantic_chunking)
            return success
            
    except Exception as e:
        logger.error(f"Error in smart metadata update: {str(e)}")
        return False