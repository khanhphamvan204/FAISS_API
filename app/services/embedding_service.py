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

# Lựa chọn 1: HuggingFace Embeddings (nhẹ nhất)
from langchain_community.embeddings import HuggingFaceEmbeddings

# Lựa chọn 2: OpenAI Embeddings (cần API key)
# from langchain_community.embeddings import OpenAIEmbeddings

# Lựa chọn 3: Ollama Embeddings (local, nhẹ)
# from langchain_community.embeddings import OllamaEmbeddings

logger = logging.getLogger(__name__)

def get_embedding_model():
    """Get consistent embedding model - multiple options"""
    
    # Option 1: HuggingFace Embeddings (nhẹ nhất, không cần GPU)
    return HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Option 2: OpenAI Embeddings (nếu có API key)
    # return OpenAIEmbeddings(
    #     model="text-embedding-3-small",  # Nhẹ và rẻ
    #     openai_api_key=os.getenv("OPENAI_API_KEY")
    # )
    
    # Option 3: Ollama Embeddings (local, rất nhẹ)
    # return OllamaEmbeddings(
    #     model="nomic-embed-text:latest",
    #     base_url="http://localhost:11434"
    # )
    
    # Option 4: Fake Embeddings (để test, không dùng production)
    # from langchain_community.embeddings import FakeEmbeddings
    # return FakeEmbeddings(size=384)

def semantic_sliding_window_chunking(documents: List[Document], embedding_model, window_overlap: float = 0.2) -> List[Document]:
    """
    Sliding Window với tỷ lệ overlap dựa trên semantic boundaries
    
    Args:
        documents: List các Document cần chia
        embedding_model: Model embedding
        window_overlap: Tỷ lệ overlap (0.0-1.0)
    
    Returns:
        List các Document chunks có overlap theo tỷ lệ
    """
    logger.info(f"Applying semantic sliding window chunking with {window_overlap*100}% overlap")
    
    all_chunks = []
    
    for doc in documents:
        try:
            # Tạo semantic chunker
            semantic_chunker = SemanticChunker(
                embeddings=embedding_model,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=95,
                sentence_split_regex=r'(?<=[.?!…:])\s+(?=[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ\"\'\(\[0-9])',
                buffer_size=2
            )
            
            # Chia văn bản thành semantic chunks
            text_chunks = semantic_chunker.split_text(doc.page_content)
            logger.info(f"Created {len(text_chunks)} semantic chunks for document")
            
            # Áp dụng sliding window với overlap
            sliding_chunks = []
            
            for i in range(len(text_chunks)):
                if i == 0:
                    # Chunk đầu tiên không có overlap
                    sliding_chunks.append(text_chunks[i])
                else:
                    # Tính toán overlap size dựa trên tỷ lệ
                    prev_chunk_words = text_chunks[i-1].split()
                    overlap_words_count = int(len(prev_chunk_words) * window_overlap)
                    
                    if overlap_words_count > 0:
                        # Lấy overlap theo từ thay vì ký tự để tự nhiên hơn
                        overlap_text = ' '.join(prev_chunk_words[-overlap_words_count:])
                        new_chunk_text = overlap_text + " " + text_chunks[i]
                    else:
                        new_chunk_text = text_chunks[i]
                    
                    sliding_chunks.append(new_chunk_text)
            
            # Tạo Document objects từ sliding chunks
            for j, chunk_text in enumerate(sliding_chunks):
                chunk_metadata = doc.metadata.copy()
                chunk_metadata['chunk_index'] = j
                chunk_metadata['total_chunks'] = len(sliding_chunks)
                chunk_metadata['has_overlap'] = j > 0
                chunk_metadata['overlap_ratio'] = window_overlap if j > 0 else 0.0
                
                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata=chunk_metadata
                )
                all_chunks.append(chunk_doc)
                
        except Exception as e:
            logger.warning(f"Semantic sliding window failed for document: {e}")
            # Fallback to traditional chunking
            fallback_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000, 
                chunk_overlap=int(2000 * window_overlap)  # Áp dụng overlap ratio
            )
            fallback_chunks = fallback_splitter.split_documents([doc])
            
            # Thêm metadata về fallback
            for chunk in fallback_chunks:
                chunk.metadata['chunking_method'] = 'fallback_traditional'
                chunk.metadata['overlap_ratio'] = window_overlap
            
            all_chunks.extend(fallback_chunks)
    
    logger.info(f"Total chunks created with sliding window: {len(all_chunks)}")
    return all_chunks

def get_text_splitter(use_semantic: bool = True, window_overlap: float = 0.2):
    """Get text splitter - semantic with sliding window or traditional"""
    try:
        if use_semantic:
            logger.info(f"Using SemanticChunker with sliding window (overlap: {window_overlap*100}%)")
            embedding_model = get_embedding_model()
            
            # Tạo wrapper function để sử dụng semantic sliding window
            class SemanticSlidingWindowSplitter:
                def __init__(self, embedding_model, window_overlap):
                    self.embedding_model = embedding_model
                    self.window_overlap = window_overlap
                
                def split_documents(self, documents):
                    return semantic_sliding_window_chunking(
                        documents, 
                        self.embedding_model, 
                        self.window_overlap
                    )
            
            return SemanticSlidingWindowSplitter(embedding_model, window_overlap)
            
        else:
            logger.info(f"Using RecursiveCharacterTextSplitter with {window_overlap*100}% overlap")
            return RecursiveCharacterTextSplitter(
                chunk_size=2000, 
                chunk_overlap=int(2000 * window_overlap)
            )
    except Exception as e:
        logger.warning(f"Failed to create SemanticSlidingWindow: {e}")
        logger.info("Falling back to RecursiveCharacterTextSplitter")
        return RecursiveCharacterTextSplitter(
            chunk_size=2000, 
            chunk_overlap=int(2000 * window_overlap)
        )

def add_to_embedding(file_path: str, metadata, use_semantic_chunking: bool = True, window_overlap: float = 0.2):
    """Add documents to vector database with sliding window overlap"""
    try:
        logger.info(f"Starting embedding process for: {file_path}")
        
        # Load documents
        documents = load_new_documents(file_path, metadata)
        if not documents:
            logger.warning(f"No documents loaded from {file_path}")
            return False

        # Split into chunks with semantic sliding window or traditional splitter
        text_splitter = get_text_splitter(use_semantic=use_semantic_chunking, window_overlap=window_overlap)
        
        try:
            chunks = text_splitter.split_documents(documents)
            chunking_method = "semantic sliding window" if use_semantic_chunking else "traditional"
            logger.info(f"Successfully created {len(chunks)} chunks using {chunking_method} with {window_overlap*100}% overlap")
        except Exception as e:
            if use_semantic_chunking:
                logger.warning(f"Semantic sliding window chunking failed: {e}. Falling back to traditional chunking")
                text_splitter = get_text_splitter(use_semantic=False, window_overlap=window_overlap)
                chunks = text_splitter.split_documents(documents)
                logger.info(f"Created {len(chunks)} chunks using fallback traditional chunking with {window_overlap*100}% overlap")
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
            
            # Log chunking statistics
            overlap_count = sum(1 for chunk in chunks if chunk.metadata.get('has_overlap', False))
            logger.info(f"Chunking statistics: {overlap_count}/{len(chunks)} chunks have overlap")
            
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

def update_document_metadata_in_vector_store(doc_id: str, old_metadata: dict, new_metadata, use_semantic_chunking: bool = True, window_overlap: float = 0.2) -> bool:
    """Update document by re-embedding with sliding window overlap"""
    try:
        old_file_type = old_metadata.get('file_type')
        old_filename = old_metadata.get('filename')
        
        _, old_vector_db_path = get_file_paths(old_file_type, old_filename)
        
        # Delete old document
        success = delete_from_faiss_index(old_vector_db_path, doc_id)
        if not success:
            return False
        
        # Re-embed with new metadata and sliding window overlap
        file_path = new_metadata.url
        if os.path.exists(file_path):
            return add_to_embedding(file_path, new_metadata, use_semantic_chunking, window_overlap)
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

def smart_metadata_update(doc_id: str, old_metadata: dict, new_metadata, force_re_embed: bool = False, use_semantic_chunking: bool = True, window_overlap: float = 0.2) -> bool:
    """Smart metadata update with fallback logic and sliding window overlap"""
    try:
        file_type_changed = old_metadata.get('file_type') != new_metadata.file_type
        filename_changed = old_metadata.get('filename') != new_metadata.filename
        
        if file_type_changed or filename_changed or force_re_embed:
            return update_document_metadata_in_vector_store(doc_id, old_metadata, new_metadata, use_semantic_chunking, window_overlap)
        else:
            success = update_metadata_only(doc_id, new_metadata)
            if not success:
                logger.info("Metadata-only update failed, attempting re-embedding")
                return update_document_metadata_in_vector_store(doc_id, old_metadata, new_metadata, use_semantic_chunking, window_overlap)
            return success
            
    except Exception as e:
        logger.error(f"Error in smart metadata update: {str(e)}")
        return False