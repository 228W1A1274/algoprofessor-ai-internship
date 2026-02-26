"""
ingestion.py - Document loading and chunking pipeline
Enterprise Knowledge Navigator - AlgoProfessor Internship 2026
"""

import os
import re
import hashlib
from pathlib import Path
from tqdm import tqdm


class TeslaDocumentLoader:
    """
    Loads PDF and text documents from a directory.
    Extracts text with metadata: source, page number, category, char count.
    """

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.documents = []
        self.errors = []

    def get_category(self, filename):
        fn = filename.lower()
        if any(k in fn for k in ["model", "cyber", "semi", "roadster"]): return "products"
        if any(k in fn for k in ["giga", "manufactur", "factory"]): return "manufacturing"
        if any(k in fn for k in ["power", "mega", "solar", "energy", "battery"]): return "energy"
        if any(k in fn for k in ["auto", "fsd", "dojo", "techno", "neural", "vision"]): return "technology"
        if any(k in fn for k in ["supercharg", "charg", "connector"]): return "infrastructure"
        if any(k in fn for k in ["elon", "musk", "leader", "founder"]): return "leadership"
        if any(k in fn for k in ["financial", "revenue", "annual", "earning"]): return "financial"
        return "general"

    def make_id(self, text, source, page):
        return hashlib.md5((source + str(page) + text[:80]).encode()).hexdigest()[:16]

    def load_text(self, txt_path):
        docs = []
        try:
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            if len(text.strip()) < 100:
                return []
            cat = self.get_category(txt_path.name)
            sections = re.split(r"(?=^#{1,3} )", text, flags=re.MULTILINE)
            for i, sec in enumerate(sections):
                sec = sec.strip()
                if len(sec) < 50:
                    continue
                docs.append({
                    "text": sec,
                    "metadata": {
                        "source": txt_path.name,
                        "source_type": "text",
                        "page_number": i + 1,
                        "total_pages": len(sections),
                        "category": cat,
                        "char_count": len(sec),
                        "doc_id": self.make_id(sec, txt_path.name, i)
                    }
                })
        except Exception as e:
            self.errors.append({"file": txt_path.name, "error": str(e)})
        return docs

    def load_pdf(self, pdf_path):
        try:
            import fitz
        except ImportError:
            print("pymupdf not installed. Run: pip install pymupdf")
            return []
        docs = []
        try:
            pdf = fitz.open(str(pdf_path))
            cat = self.get_category(pdf_path.name)
            for pg in range(len(pdf)):
                text = pdf[pg].get_text().strip()
                if len(text) < 100:
                    continue
                text = re.sub(r"\n{3,}", "\n\n", text)
                docs.append({
                    "text": text,
                    "metadata": {
                        "source": pdf_path.name,
                        "source_type": "pdf",
                        "page_number": pg + 1,
                        "total_pages": len(pdf),
                        "category": cat,
                        "char_count": len(text),
                        "doc_id": self.make_id(text, pdf_path.name, pg)
                    }
                })
            pdf.close()
        except Exception as e:
            self.errors.append({"file": pdf_path.name, "error": str(e)})
        return docs

    def load_all(self):
        pdf_files = list(self.data_dir.glob("*.pdf"))
        txt_files = list(self.data_dir.glob("*.txt"))
        print(f"Found: {len(pdf_files)} PDFs, {len(txt_files)} text files")
        for f in pdf_files:
            self.documents.extend(self.load_pdf(f))
        for f in txt_files:
            docs = self.load_text(f)
            self.documents.extend(docs)
        print(f"Total document sections loaded: {len(self.documents)}")
        if self.errors:
            print(f"Errors: {len(self.errors)}")
        return self.documents


class RecursiveChunker:
    """
    Splits documents into overlapping chunks.
    Target size: 400-600 words. Overlap: 75 words to avoid boundary splits.
    """

    def __init__(self, chunk_size=500, overlap=75):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, text, metadata):
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
        if not paragraphs:
            return []
        chunks = []
        current_words = []
        chunk_index = 0
        for para in paragraphs:
            para_words = para.split()
            if len(current_words) + len(para_words) > self.chunk_size and current_words:
                chunk_text = " ".join(current_words)
                meta = dict(metadata)
                meta["chunk_index"] = chunk_index
                meta["chunk_word_count"] = len(current_words)
                meta["chunk_id"] = metadata["doc_id"] + "_c" + str(chunk_index)
                chunks.append({"text": chunk_text, "metadata": meta})
                chunk_index += 1
                current_words = current_words[-self.overlap:] if len(current_words) > self.overlap else []
            current_words.extend(para_words)
        if current_words:
            meta = dict(metadata)
            meta["chunk_index"] = chunk_index
            meta["chunk_word_count"] = len(current_words)
            meta["chunk_id"] = metadata["doc_id"] + "_c" + str(chunk_index)
            chunks.append({"text": " ".join(current_words), "metadata": meta})
        return chunks

    def chunk_all(self, documents):
        all_chunks = []
        for doc in tqdm(documents, desc="Chunking"):
            all_chunks.extend(self.chunk_document(doc["text"], doc["metadata"]))
        print(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
