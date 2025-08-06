# app/services/document_loader.py
import os
import logging
import pdfplumber
import pandas as pd
from langchain.docstore.document import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredExcelLoader
)
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw
import fitz

logger = logging.getLogger(__name__)

# Khởi tạo OCR - Global variables
ocr = None
ocr_method = None

def init_ocr():
    """Khởi tạo OCR và xác định phương thức hoạt động"""
    global ocr, ocr_method
    
    if ocr is not None:
        return True
    
    try:
        logger.info("Đang khởi tạo PaddleOCR...")
        ocr = PaddleOCR(use_angle_cls=True, lang='vi')
        
        # Test để xác định phương thức nào hoạt động
        test_img = "test_temp.png"
        
        img = Image.new('RGB', (200, 50), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "TEST", fill='black')
        img.save(test_img)
        
        # Thử các phương thức
        try:
            result = ocr(test_img)
            ocr_method = "direct"
        except:
            try:
                result = ocr.ocr(test_img)
                ocr_method = "ocr_method"
            except:
                try:
                    result = ocr.predict(test_img)
                    ocr_method = "predict"
                except:
                    ocr_method = None
        
        # Cleanup test file
        if os.path.exists(test_img):
            os.remove(test_img)
            
        if ocr_method:
            logger.info(f"PaddleOCR initialized successfully with method: {ocr_method}")
        else:
            logger.error("Failed to determine OCR method")
            
        return ocr_method is not None
        
    except Exception as e:
        logger.error(f"Failed to initialize PaddleOCR: {str(e)}")
        return False

def extract_text_with_paddle(pdf_path: str, page_num: int) -> str:
    """Trích xuất văn bản từ trang PDF scan bằng PaddleOCR"""
    global ocr, ocr_method
    
    if not init_ocr():
        logger.error("Cannot initialize OCR")
        return ""
    
    temp_img_path = None
    try:
        pdf_document = fitz.open(pdf_path)
        page = pdf_document[page_num - 1]
        
        # Render page thành hình ảnh với độ phân giải cao
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        
        temp_img_path = f"temp_page_{os.getpid()}_{page_num}.png"
        pix.save(temp_img_path)
        pdf_document.close()
        
        # Gọi OCR bằng phương thức đã xác định
        result = None
        if ocr_method == "direct":
            result = ocr(temp_img_path)
        elif ocr_method == "ocr_method":
            result = ocr.ocr(temp_img_path)
        elif ocr_method == "predict":
            result = ocr.predict(temp_img_path)
        else:
            return ""
        
        # Xử lý kết quả
        text = ""
        if result and isinstance(result, list) and len(result) > 0:
            # Xử lý cấu trúc thông thường [[bbox, (text, confidence)], ...]
            if isinstance(result[0], list):
                for line in result[0]:
                    if isinstance(line, list) and len(line) >= 2:
                        if isinstance(line[1], tuple) and len(line[1]) >= 1:
                            text += str(line[1][0]) + "\n"
                        elif isinstance(line[1], str):
                            text += line[1] + "\n"
            
            # Fallback: extract text từ bất kỳ cấu trúc nào
            else:
                def extract_text_recursive(obj):
                    texts = []
                    if isinstance(obj, str):
                        texts.append(obj)
                    elif isinstance(obj, (list, tuple)):
                        for item in obj:
                            texts.extend(extract_text_recursive(item))
                    elif isinstance(obj, dict):
                        for value in obj.values():
                            texts.extend(extract_text_recursive(value))
                    return texts
                
                all_texts = extract_text_recursive(result)
                meaningful_texts = [t for t in all_texts if isinstance(t, str) and len(t.strip()) > 2 and not t.strip().isdigit()]
                text = "\n".join(meaningful_texts)
        
        logger.info(f"OCR extracted text length: {len(text)} for page {page_num}")
        return text.strip()
        
    except Exception as e:
        logger.error(f"OCR Error for {pdf_path} page {page_num}: {str(e)}")
        return ""
    finally:
        if temp_img_path and os.path.exists(temp_img_path):
            try:
                os.remove(temp_img_path)
            except:
                pass


def process_pdf(file_path: str) -> tuple[list, list]:
    """Xử lý file PDF, trích xuất bảng và văn bản với OCR fallback."""
    tables, texts = [], []
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Processing PDF: {file_path}")
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Trích xuất bảng
                extracted_tables = page.extract_tables()
                if extracted_tables:
                    for table in extracted_tables:
                        if table and len(table) > 1:
                            try:
                                headers = table[0] if table[0] else [f"Col_{i}" for i in range(len(table[1]))]
                                data = table[1:]
                                df = pd.DataFrame(data, columns=headers)
                                tables.append(df)
                            except Exception as e:
                                logger.warning(f"Error processing table on page {page_num + 1}: {str(e)}")
                                continue
                
                # Trích xuất văn bản
                text = page.extract_text()
                if text and text.strip():
                    texts.append(text)
                    logger.info(f"Extracted text using pdfplumber for page {page_num + 1}")
                else:
                    # Fallback to OCR if no text found
                    logger.info(f"No text found with pdfplumber, trying OCR for page {page_num + 1}")
                    ocr_text = extract_text_with_paddle(file_path, page_num + 1)
                    if ocr_text and ocr_text.strip():
                        texts.append(ocr_text)
                        logger.info(f"Successfully extracted text using OCR for page {page_num + 1}")
                    else:
                        logger.warning(f"No text extracted for page {page_num + 1}")
        
        logger.info(f"Extracted {len(tables)} tables and {len(texts)} text segments from {file_path}")
    except Exception as e:
        logger.error(f"PDF Processing Error for {file_path}: {str(e)}")
    
    return tables, texts

def load_new_documents(file_path: str, metadata) -> list:
    documents = []
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return documents

    extension = file_path.lower().split('.')[-1]
    supported_extensions = {
        'pdf': PyPDFLoader,
        'txt': TextLoader,
        'docx': Docx2txtLoader,
        'csv': CSVLoader,
        'xlsx': UnstructuredExcelLoader,
        'xls': UnstructuredExcelLoader
    }

    if extension in supported_extensions:
        try:
            logger.info(f"Loading document: {file_path} with extension {extension}")
            if extension == 'pdf':
                tables, texts = process_pdf(file_path)
                metadata_dict = metadata.dict(by_alias=True)
                for table in tables:
                    table_text = table.to_csv(index=False)
                    documents.append(Document(page_content=table_text, metadata=metadata_dict))
                for text in texts:
                    documents.append(Document(page_content=text, metadata=metadata_dict))
            else:
                loader = supported_extensions[extension](file_path)
                loaded_docs = loader.load()
                metadata_dict = metadata.dict(by_alias=True)
                for doc in loaded_docs:
                    documents.append(Document(page_content=doc.page_content, metadata=metadata_dict))
            logger.info(f"Loaded {len(documents)} documents from {file_path}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
    
    return documents