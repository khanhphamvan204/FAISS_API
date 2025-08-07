# # app/services/document_loader.py
# import os
# import logging
# import cv2
# import numpy as np
# import pytesseract
# from pdf2image import convert_from_path
# from PIL import Image
# from langchain.docstore.document import Document
# from langchain_community.document_loaders import (
#     TextLoader,
#     Docx2txtLoader,
#     CSVLoader,
#     UnstructuredExcelLoader
# )

# logger = logging.getLogger(__name__)

# # Cấu hình đường dẫn Tesseract
# base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# pytesseract.pytesseract.tesseract_cmd = os.path.join(base_dir, "app", "models", "Tesseract-OCR", "tesseract.exe")
# tessdata_dir = os.path.join(base_dir, "app", "models", "Tesseract-OCR", "tessdata")

# # Global variables for OCR
# ocr_initialized = False

# def init_tesseract():
#     """Khởi tạo và kiểm tra Tesseract OCR"""
#     global ocr_initialized
    
#     if ocr_initialized:
#         return True
    
#     try:
#         logger.info("Initializing Tesseract OCR...")
        
#         # Kiểm tra xem tesseract executable có tồn tại không
#         if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
#             logger.error(f"Tesseract executable not found at: {pytesseract.pytesseract.tesseract_cmd}")
#             return False
        
#         # Kiểm tra tessdata directory
#         if not os.path.exists(tessdata_dir):
#             logger.error(f"Tessdata directory not found at: {tessdata_dir}")
#             return False
        
#         # Kiểm tra các file ngôn ngữ có sẵn
#         available_langs = []
#         for lang_file in ['vie.traineddata', 'eng.traineddata', 'osd.traineddata']:
#             lang_path = os.path.join(tessdata_dir, lang_file)
#             if os.path.exists(lang_path):
#                 available_langs.append(lang_file.replace('.traineddata', ''))
        
#         if not available_langs:
#             logger.error(f"No language data files found in: {tessdata_dir}")
#             return False
        
#         logger.info(f"Available languages: {available_langs}")
        
#         # Set environment variable cho tessdata
#         tessdata_path = tessdata_dir.replace('\\', '/')
#         os.environ['TESSDATA_PREFIX'] = tessdata_path
        
#         # Test với tiếng Việt hoặc tiếng Anh
#         test_lang = 'vie' if 'vie' in available_langs else 'eng'
#         if test_lang not in available_langs:
#             logger.error("Neither Vietnamese nor English language files found")
#             return False
        
#         # Test OCR
#         test_img = Image.new('RGB', (200, 50), color='white')
#         from PIL import ImageDraw
#         draw = ImageDraw.Draw(test_img)
#         draw.text((10, 10), "TEST", fill='black')
        
#         config = f'--tessdata-dir {tessdata_path} -l {test_lang}'
#         test_result = pytesseract.image_to_string(test_img, config=config)
        
#         ocr_initialized = True
#         logger.info(f"Tesseract OCR initialized successfully with {test_lang}")
#         return True
        
#     except Exception as e:
#         logger.error(f"Failed to initialize Tesseract OCR: {str(e)}")
#         return False

# def preprocess_image_for_ocr(image):
#     """
#     Tiền xử lý hình ảnh để cải thiện OCR
#     """
#     try:
#         # Convert PIL Image to numpy array
#         img_array = np.array(image)
        
#         # Convert RGB to BGR for OpenCV
#         if len(img_array.shape) == 3:
#             img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
#             gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
#         else:
#             gray = img_array
        
#         # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#         enhanced = clahe.apply(gray)
        
#         # Denoise
#         denoised = cv2.medianBlur(enhanced, 3)
        
#         # Threshold
#         _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
#         return Image.fromarray(thresh)
        
#     except Exception as e:
#         logger.warning(f"Image preprocessing failed: {str(e)}, using original image")
#         return image

# def extract_text_with_tesseract(image, page_num=1):
#     """Trích xuất văn bản từ hình ảnh bằng Tesseract OCR"""
#     if not init_tesseract():
#         logger.error("Cannot initialize Tesseract OCR")
#         return ""
    
#     try:
#         # Preprocess image
#         processed_img = preprocess_image_for_ocr(image)
        
#         # Prefer Vietnamese, fallback to English
#         selected_lang = 'vie'
#         if not os.path.exists(os.path.join(tessdata_dir, f"{selected_lang}.traineddata")):
#             selected_lang = 'eng'
#             if not os.path.exists(os.path.join(tessdata_dir, f"{selected_lang}.traineddata")):
#                 logger.error("Neither Vietnamese nor English language data found")
#                 return ""
        
#         # Use path with forward slashes for consistency
#         tessdata_path = tessdata_dir.replace('\\', '/')
        
#         # Try different PSM modes for better results
#         psm_modes = [6, 4, 3, 1]  # Different page segmentation modes
#         best_text = ""
#         best_length = 0
        
#         for psm in psm_modes:
#             try:
#                 config = f'--tessdata-dir {tessdata_path} -l {selected_lang} --psm {psm} -c preserve_interword_spaces=1'
#                 text = pytesseract.image_to_string(processed_img, config=config)
                
#                 # Choose the result with most content
#                 if len(text.strip()) > best_length:
#                     best_text = text
#                     best_length = len(text.strip())
                    
#             except Exception as e:
#                 logger.warning(f"OCR failed with PSM {psm}: {str(e)}")
#                 continue
        
#         logger.info(f"OCR extracted {best_length} characters from page {page_num} using {selected_lang}")
#         return best_text.strip()
        
#     except Exception as e:
#         logger.error(f"OCR Error for page {page_num}: {str(e)}")
#         return ""

# def process_pdf(file_path: str) -> list:
#     """Xử lý file PDF, trích xuất văn bản bằng OCR."""
#     texts = []
    
#     try:
#         if not os.path.exists(file_path):
#             logger.error(f"File not found: {file_path}")
#             raise FileNotFoundError(f"File not found: {file_path}")
        
#         logger.info(f"Processing PDF: {file_path}")
        
#         # Convert PDF to images
#         try:
#             images = convert_from_path(file_path, dpi=300, fmt='PNG')
#             logger.info(f"Converted PDF to {len(images)} images")
#         except Exception as e:
#             logger.error(f"Failed to convert PDF to images: {str(e)}")
#             return texts
        
#         # Process each page
#         for i, img in enumerate(images):
#             page_num = i + 1
#             logger.info(f"Processing page {page_num}/{len(images)}")
            
#             # Extract text using OCR
#             text = extract_text_with_tesseract(img, page_num)
            
#             if text and text.strip():
#                 # Add page information to the text
                
#                 texts.append(text)
#                 logger.info(f"Successfully extracted text from page {page_num}")
#             else:
#                 logger.warning(f"No text extracted from page {page_num}")
        
#         logger.info(f"Extracted text from {len(texts)} pages of {file_path}")
        
#     except Exception as e:
#         logger.error(f"PDF Processing Error for {file_path}: {str(e)}")
    
#     return texts

# def load_new_documents(file_path: str, metadata) -> list:
#     """Load documents from various file formats"""
#     documents = []
    
#     if not os.path.exists(file_path):
#         logger.error(f"File not found: {file_path}")
#         return documents

#     extension = file_path.lower().split('.')[-1]
#     supported_extensions = {
#         'pdf': 'pdf_ocr',  # Special handling for PDF with OCR
#         'txt': TextLoader,
#         'docx': Docx2txtLoader,
#         'csv': CSVLoader,
#         'xlsx': UnstructuredExcelLoader,
#         'xls': UnstructuredExcelLoader
#     }

#     if extension in supported_extensions:
#         try:
#             logger.info(f"Loading document: {file_path} with extension {extension}")
            
#             if extension == 'pdf':
#                 # Special PDF processing with OCR
#                 texts = process_pdf(file_path)
#                 metadata_dict = metadata.dict(by_alias=True) if hasattr(metadata, 'dict') else metadata
                
#                 for text in texts:
#                     if text and text.strip():
#                         documents.append(Document(page_content=text, metadata=metadata_dict))
                        
#             else:
#                 # Standard document loading
#                 loader = supported_extensions[extension](file_path)
#                 loaded_docs = loader.load()
#                 metadata_dict = metadata.dict(by_alias=True) if hasattr(metadata, 'dict') else metadata
                
#                 for doc in loaded_docs:
#                     documents.append(Document(
#                         page_content=doc.page_content, 
#                         metadata=metadata_dict
#                     ))
            
#             logger.info(f"Loaded {len(documents)} documents from {file_path}")
            
#         except Exception as e:
#             logger.error(f"Error loading {file_path}: {str(e)}")
#     else:
#         logger.warning(f"Unsupported file extension: {extension}")
    
#     return documents


# app/services/document_loader.py
import os
import logging
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from langchain.docstore.document import Document
from langchain_community.document_loaders import (
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredExcelLoader
)

logger = logging.getLogger(__name__)

# Global variables for OCR
ocr_initialized = False

def init_tesseract():
    """Khởi tạo và kiểm tra Tesseract OCR"""
    global ocr_initialized
    
    if ocr_initialized:
        return True
    
    try:
        logger.info("Initializing Tesseract OCR...")
        
        # Trong Docker Linux, tesseract được cài đặt system-wide
        # Không cần set đường dẫn cụ thể
        
        # Kiểm tra tesseract có sẵn không
        import subprocess
        try:
            result = subprocess.run(['tesseract', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info(f"Tesseract version: {result.stdout.split()[1]}")
            else:
                logger.error("Tesseract not found in system PATH")
                return False
        except subprocess.TimeoutExpired:
            logger.error("Tesseract version check timed out")
            return False
        except FileNotFoundError:
            logger.error("Tesseract executable not found")
            return False
        
        # Kiểm tra các ngôn ngữ có sẵn
        try:
            result = subprocess.run(['tesseract', '--list-langs'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                available_langs = result.stdout.strip().split('\n')[1:]  # Skip header
                logger.info(f"Available languages: {available_langs}")
                
                # Kiểm tra có tiếng Việt và tiếng Anh không
                has_vie = 'vie' in available_langs
                has_eng = 'eng' in available_langs
                
                if not has_vie and not has_eng:
                    logger.error("Neither Vietnamese nor English language support found")
                    return False
                    
                logger.info(f"Language support - Vietnamese: {has_vie}, English: {has_eng}")
            else:
                logger.warning("Could not list available languages, proceeding with default")
        except subprocess.TimeoutExpired:
            logger.warning("Language list check timed out, proceeding with default")
        except Exception as e:
            logger.warning(f"Language check failed: {e}, proceeding with default")
        
        # Test OCR với hình ảnh đơn giản
        try:
            test_img = Image.new('RGB', (200, 50), color='white')
            from PIL import ImageDraw
            draw = ImageDraw.Draw(test_img)
            draw.text((10, 10), "TEST", fill='black')
            
            # Test với cấu hình cơ bản
            test_result = pytesseract.image_to_string(test_img, lang='eng')
            logger.info("OCR test successful")
        except Exception as e:
            logger.error(f"OCR test failed: {e}")
            return False
        
        ocr_initialized = True
        logger.info("Tesseract OCR initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize Tesseract OCR: {str(e)}")
        return False

def preprocess_image_for_ocr(image):
    """
    Tiền xử lý hình ảnh để cải thiện OCR
    """
    try:
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(img_array.shape) == 3:
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_array
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.medianBlur(enhanced, 3)
        
        # Threshold
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return Image.fromarray(thresh)
        
    except Exception as e:
        logger.warning(f"Image preprocessing failed: {str(e)}, using original image")
        return image

def extract_text_with_tesseract(image, page_num=1):
    """Trích xuất văn bản từ hình ảnh bằng Tesseract OCR"""
    if not init_tesseract():
        logger.error("Cannot initialize Tesseract OCR")
        return ""
    
    try:
        # Preprocess image
        processed_img = preprocess_image_for_ocr(image)
        
        # Kiểm tra ngôn ngữ có sẵn
        import subprocess
        available_langs = []
        try:
            result = subprocess.run(['tesseract', '--list-langs'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                available_langs = result.stdout.strip().split('\n')[1:]
        except:
            available_langs = ['eng']  # fallback
        
        # Chọn ngôn ngữ ưu tiên
        if 'vie' in available_langs:
            selected_lang = 'vie+eng'  # Combine Vietnamese and English
        elif 'eng' in available_langs:
            selected_lang = 'eng'
        else:
            selected_lang = 'eng'  # Default fallback
        
        # Try different PSM modes for better results
        psm_modes = [6, 4, 3, 1]  # Different page segmentation modes
        best_text = ""
        best_length = 0
        
        for psm in psm_modes:
            try:
                config = f'--psm {psm} -c preserve_interword_spaces=1'
                text = pytesseract.image_to_string(processed_img, lang=selected_lang, config=config)
                
                # Choose the result with most content
                if len(text.strip()) > best_length:
                    best_text = text
                    best_length = len(text.strip())
                    
            except Exception as e:
                logger.warning(f"OCR failed with PSM {psm}: {str(e)}")
                continue
        
        logger.info(f"OCR extracted {best_length} characters from page {page_num} using {selected_lang}")
        return best_text.strip()
        
    except Exception as e:
        logger.error(f"OCR Error for page {page_num}: {str(e)}")
        return ""

def process_pdf(file_path: str) -> list:
    """Xử lý file PDF, trích xuất văn bản bằng OCR."""
    texts = []
    
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Processing PDF: {file_path}")
        
        # Convert PDF to images
        try:
            images = convert_from_path(file_path, dpi=300, fmt='PNG')
            logger.info(f"Converted PDF to {len(images)} images")
        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {str(e)}")
            return texts
        
        # Process each page
        for i, img in enumerate(images):
            page_num = i + 1
            logger.info(f"Processing page {page_num}/{len(images)}")
            
            # Extract text using OCR
            text = extract_text_with_tesseract(img, page_num)
            
            if text and text.strip():
                texts.append(text)
                logger.info(f"Successfully extracted text from page {page_num}")
            else:
                logger.warning(f"No text extracted from page {page_num}")
        
        logger.info(f"Extracted text from {len(texts)} pages of {file_path}")
        
    except Exception as e:
        logger.error(f"PDF Processing Error for {file_path}: {str(e)}")
    
    return texts

def load_new_documents(file_path: str, metadata) -> list:
    """Load documents from various file formats"""
    documents = []
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return documents

    extension = file_path.lower().split('.')[-1]
    supported_extensions = {
        'pdf': 'pdf_ocr',  # Special handling for PDF with OCR
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
                # Special PDF processing with OCR
                texts = process_pdf(file_path)
                metadata_dict = metadata.dict(by_alias=True) if hasattr(metadata, 'dict') else metadata
                
                for text in texts:
                    if text and text.strip():
                        documents.append(Document(page_content=text, metadata=metadata_dict))
                        
            else:
                # Standard document loading
                loader = supported_extensions[extension](file_path)
                loaded_docs = loader.load()
                metadata_dict = metadata.dict(by_alias=True) if hasattr(metadata, 'dict') else metadata
                
                for doc in loaded_docs:
                    documents.append(Document(
                        page_content=doc.page_content, 
                        metadata=metadata_dict
                    ))
            
            logger.info(f"Loaded {len(documents)} documents from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
    else:
        logger.warning(f"Unsupported file extension: {extension}")
    
    return documents