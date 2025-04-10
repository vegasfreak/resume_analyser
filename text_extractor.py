import os
import pdfplumber
from docx import Document

def extract_text(file_path: str) -> str:
    """Extracts text from PDF, DOCX, or TXT files with encoding handling."""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == ".pdf":
            return extract_text_from_pdf(file_path)
        elif file_ext == ".docx":
            return extract_text_from_docx(file_path)
        elif file_ext == ".txt":
            return extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    except Exception as e:
        raise ValueError(f"Error extracting text from {file_path}: {str(e)}")

def extract_text_from_pdf(file_path: str) -> str:
    """Extracts text from PDF using pdfplumber with error handling."""
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        raise ValueError(f"PDF extraction error: {str(e)}")
    return text.strip()

def extract_text_from_docx(file_path: str) -> str:
    """Extracts text from DOCX file."""
    try:
        doc = Document(file_path)
        return "\n".join(para.text for para in doc.paragraphs if para.text.strip())
    except Exception as e:
        raise ValueError(f"DOCX extraction error: {str(e)}")

def extract_text_from_txt(file_path: str) -> str:
    """Extracts text from TXT file with encoding fallback."""
    encodings = ['utf-8', 'latin-1', 'windows-1252']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue
    raise ValueError("Failed to decode text file with tried encodings")