"""
WHAT: Extracts text from PDFs and text files
WHY: SEC filings are PDFs, need to convert to text for RAG
HOW: Uses PyPDF2 for PDFs, handles .txt files directly
"""

import os
from PyPDF2 import PdfReader
from typing import List, Dict

class DocumentProcessor:
    def __init__(self):
        self.supported_formats = ['.pdf', '.txt']
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        WHAT: Extracts all text from a PDF file
        WHY: SEC filings are in PDF format
        RETURNS: String with full text content
        """
        print(f"📄 Processing PDF: {os.path.basename(pdf_path)}")
        
        try:
            reader = PdfReader(pdf_path)
            text = ""
            
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                text += page_text + "\n\n"
            
            print(f"✅ Extracted {len(text)} characters from {len(reader.pages)} pages")
            return text
            
        except Exception as e:
            print(f"❌ Error processing PDF {pdf_path}: {e}")
            return ""
    
    def extract_text_from_txt(self, txt_path: str) -> str:
        """
        WHAT: Reads text file content
        WHY: News articles are saved as .txt
        RETURNS: File content as string
        """
        print(f"📝 Processing TXT: {os.path.basename(txt_path)}")
        
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            print(f"✅ Extracted {len(text)} characters")
            return text
            
        except Exception as e:
            print(f"❌ Error processing TXT {txt_path}: {e}")
            return ""
    
    def process_document(self, file_path: str) -> Dict[str, str]:
        """
        WHAT: Processes any supported document type
        WHY: Unified interface for different file types
        RETURNS: Dict with text and metadata
        """
        _, ext = os.path.splitext(file_path)
        
        if ext.lower() == '.pdf':
            text = self.extract_text_from_pdf(file_path)
        elif ext.lower() == '.txt':
            text = self.extract_text_from_txt(file_path)
        else:
            print(f"⚠️ Unsupported format: {ext}")
            return None
        
        # Extract metadata from filename and content
        filename = os.path.basename(file_path)
        
        # Determine document type
        if 'sec-edgar' in file_path or '10-K' in filename or '10-Q' in filename:
            doc_type = 'SEC_FILING'
        elif 'news' in file_path:
            doc_type = 'NEWS'
        else:
            doc_type = 'OTHER'
        
        return {
            'text': text,
            'source': file_path,
            'filename': filename,
            'doc_type': doc_type,
            'char_count': len(text)
        }
    
    def process_directory(self, directory: str) -> List[Dict]:
        """
        WHAT: Processes all documents in a directory
        WHY: Batch processing for multiple files
        RETURNS: List of processed documents
        """
        print(f"\n🔍 Scanning directory: {directory}")
        
        documents = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                _, ext = os.path.splitext(file)
                
                if ext.lower() in self.supported_formats:
                    doc = self.process_document(file_path)
                    if doc and doc['text']:  # Only add if text was extracted
                        documents.append(doc)
        
        print(f"\n✅ Processed {len(documents)} documents")
        return documents


# USAGE EXAMPLE
if __name__ == "__main__":
    processor = DocumentProcessor()
    
    # Process SEC filings
    sec_docs = processor.process_directory("data/sec_filings")
    print(f"\nSEC Documents: {len(sec_docs)}")
    
    # Process news articles
    news_docs = processor.process_directory("data/news_articles")
    print(f"News Documents: {len(news_docs)}")
    
    # Example: Show first document info
    if sec_docs:
        print("\n" + "="*50)
        print("SAMPLE DOCUMENT:")
        print(f"Source: {sec_docs[0]['filename']}")
        print(f"Type: {sec_docs[0]['doc_type']}")
        print(f"Characters: {sec_docs[0]['char_count']}")
        print(f"Preview: {sec_docs[0]['text'][:500]}...")