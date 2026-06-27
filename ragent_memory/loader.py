import os
import fitz  # PyMuPDF
from docx import Document

class DocumentLoader:
    """Utility to extract clean text from PDF and DOCX files for RAG ingestion."""

    @staticmethod
    def extract_text(file_path: str) -> str:
        """
        Detects file type and extracts all text content.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            return DocumentLoader._parse_pdf(file_path)
        elif ext == ".docx":
            return DocumentLoader._parse_docx(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    @staticmethod
    def _parse_pdf(file_path: str) -> str:
        """Extract text page-by-page using high-performance PyMuPDF."""
        text_blocks = []
        
        # Open the PDF document
        with fitz.open(file_path) as doc:
            for page in doc:
                # "text" layout preserves basic reading order blocks
                page_text = page.get_text("text")
                if page_text.strip(): # type: ignore
                    text_blocks.append(page_text)
                    
        # Join pages with a standard newline separator
        return "\n\n".join(text_blocks)

    @staticmethod
    def _parse_docx(file_path: str) -> str:
        """Extract paragraphs and structural text from a Word document."""
        doc = Document(file_path)
        text_blocks = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_blocks.append(paragraph.text)
                
        return "\n\n".join(text_blocks)