#!/usr/bin/env python3
"""PDF Ingestion Script for NICE NG12 Cancer Guidelines.

Downloads the NG12 PDF, parses it page-by-page, chunks the text with
overlap, and builds a ChromaDB vector store for RAG queries.
"""

import os
import sys

# Allow running as a standalone script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz  # PyMuPDF
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import requests
from app.config import CHROMA_PERSIST_DIR, CHUNK_SIZE, CHUNK_OVERLAP, NG12_PDF_PATH

PDF_URL = (
    "https://www.nice.org.uk/guidance/ng12/resources/"
    "suspected-cancer-recognition-and-referral-pdf-1837268071621"
)

# ── Page-to-section mapping for NG12 PDF ─────────────────────────────────
# (start_page, end_page, section, cancer_type, content_type)
PAGE_SECTION_MAP = [
    (1, 8, "introduction", "general", "background"),
    (9, 11, "1.1", "lung", "recommendation"),
    (12, 13, "1.2", "oesophageal", "recommendation"),
    (13, 14, "1.2", "stomach", "recommendation"),
    (14, 16, "1.3", "colorectal", "recommendation"),
    (17, 17, "1.4", "hepatobiliary", "recommendation"),
    (17, 18, "1.5", "gynaecological", "recommendation"),
    (19, 20, "1.6", "renal", "recommendation"),
    (20, 21, "1.6", "prostate", "recommendation"),
    (21, 22, "1.6", "bladder", "recommendation"),
    (22, 22, "1.6", "testicular", "recommendation"),
    (22, 22, "1.6", "penile", "recommendation"),
    (23, 24, "1.7", "skin", "recommendation"),
    (25, 25, "1.8", "head_and_neck", "recommendation"),
    (26, 28, "1.10", "haematological", "recommendation"),
    (29, 30, "1.11", "sarcoma", "recommendation"),
    (30, 31, "1.12", "brain_cns", "recommendation"),
    (31, 32, "1.13", "unknown_primary", "recommendation"),
    (33, 36, "patient_support", "general", "support"),
    (37, 82, "symptom_tables", "multi", "symptom_lookup"),
    (83, 95, "appendix", "general", "reference"),
]

CANCER_TYPE_KEYWORDS = {
    "lung": "lung",
    "pulmonary": "lung",
    "breast": "breast",
    "colorectal": "colorectal",
    "bowel": "colorectal",
    "rectal": "colorectal",
    "prostate": "prostate",
    "skin": "skin",
    "melanoma": "skin",
    "bladder": "bladder",
    "oesophageal": "oesophageal",
    "stomach": "stomach",
    "gastric": "stomach",
    "ovarian": "gynaecological",
    "endometrial": "gynaecological",
    "cervical": "gynaecological",
    "renal": "renal",
    "kidney": "renal",
    "thyroid": "head_and_neck",
    "leukaemia": "haematological",
    "leukemia": "haematological",
    "lymphoma": "haematological",
    "myeloma": "haematological",
    "sarcoma": "sarcoma",
    "brain": "brain_cns",
    "hepatobiliary": "hepatobiliary",
    "testicular": "testicular",
    "penile": "penile",
}


def _get_section_metadata(page_num: int) -> dict:
    """Return section metadata for a given page number."""
    for start, end, section, cancer_type, content_type in PAGE_SECTION_MAP:
        if start <= page_num <= end:
            return {
                "section": section,
                "cancer_type": cancer_type,
                "content_type": content_type,
            }
    return {"section": "unknown", "cancer_type": "general", "content_type": "reference"}


def _detect_cancer_types_in_text(text: str) -> list[str]:
    """Detect cancer type keywords in chunk text."""
    text_lower = text.lower()
    found = set()
    for keyword, cancer_type in CANCER_TYPE_KEYWORDS.items():
        if keyword in text_lower:
            found.add(cancer_type)
    return sorted(found)


def download_pdf(url: str, save_path: str) -> None:
    """Download the NG12 PDF if it does not already exist locally."""
    if os.path.exists(save_path):
        print(f"PDF already exists at {save_path}, skipping download.")
        return

    print("Downloading NG12 PDF from NICE website...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    response = requests.get(
        url,
        timeout=120,
        headers={"User-Agent": "NG12-ClinicalAssessmentTool/1.0"},
    )
    response.raise_for_status()

    with open(save_path, "wb") as f:
        f.write(response.content)
    print(f"PDF saved to {save_path} ({len(response.content):,} bytes)")


def parse_pdf(pdf_path: str) -> list[dict]:
    """Extract text from each page of the PDF."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            pages.append({"page": page_num + 1, "text": text.strip()})
    doc.close()
    print(f"Extracted text from {len(pages)} pages")
    return pages


def chunk_pages(
    pages: list[dict],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Split page text into overlapping chunks, respecting paragraph boundaries."""
    chunks = []

    for page_data in pages:
        text = page_data["text"]
        page_num = page_data["page"]

        # Get section metadata for this page
        section_meta = _get_section_metadata(page_num)

        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        current_chunk = ""
        chunk_idx = 0

        for para in paragraphs:
            # If adding this paragraph would exceed chunk_size, save current chunk
            if current_chunk and len(current_chunk) + len(para) + 2 > chunk_size:
                chunk_id = f"ng12_p{page_num:03d}_c{chunk_idx:02d}"
                cancer_type = section_meta["cancer_type"]

                # For symptom tables, detect specific cancer types from text
                if section_meta["section"] == "symptom_tables":
                    detected = _detect_cancer_types_in_text(current_chunk)
                    if detected:
                        cancer_type = detected[0] if len(detected) == 1 else "multi"

                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": current_chunk,
                        "page": page_num,
                        "source": "NG12 PDF",
                        "section": section_meta["section"],
                        "cancer_type": cancer_type,
                        "content_type": section_meta["content_type"],
                    }
                )
                chunk_idx += 1

                # Carry overlap from the end of the current chunk
                if len(current_chunk) > chunk_overlap:
                    overlap_text = current_chunk[-chunk_overlap:]
                else:
                    overlap_text = current_chunk
                current_chunk = overlap_text + "\n\n" + para
            else:
                current_chunk = (current_chunk + "\n\n" + para) if current_chunk else para

        # Save the remaining text on this page
        if current_chunk.strip():
            chunk_id = f"ng12_p{page_num:03d}_c{chunk_idx:02d}"
            cancer_type = section_meta["cancer_type"]

            if section_meta["section"] == "symptom_tables":
                detected = _detect_cancer_types_in_text(current_chunk)
                if detected:
                    cancer_type = detected[0] if len(detected) == 1 else "multi"

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": current_chunk,
                    "page": page_num,
                    "source": "NG12 PDF",
                    "section": section_meta["section"],
                    "cancer_type": cancer_type,
                    "content_type": section_meta["content_type"],
                }
            )

    print(f"Created {len(chunks)} chunks from {len(pages)} pages")

    # Print metadata stats
    cancer_types = {}
    content_types = {}
    for c in chunks:
        ct = c["cancer_type"]
        cancer_types[ct] = cancer_types.get(ct, 0) + 1
        ctype = c["content_type"]
        content_types[ctype] = content_types.get(ctype, 0) + 1
    print(f"Cancer types: {cancer_types}")
    print(f"Content types: {content_types}")

    return chunks


def build_vector_store(chunks: list[dict]) -> None:
    """Create embeddings and store chunks in ChromaDB."""
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    print("Creating ChromaDB collection...")
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Remove existing collection if present
    try:
        client.delete_collection("ng12_guidelines")
        print("Deleted existing collection.")
    except (ValueError, Exception):
        pass

    collection = client.create_collection(
        name="ng12_guidelines",
        embedding_function=embedding_fn,
        metadata={"description": "NICE NG12 Suspected Cancer Guidelines"},
    )

    # Add chunks in batches
    batch_size = 50
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.add(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[
                {
                    "page": c["page"],
                    "source": c["source"],
                    "section": c.get("section", "unknown"),
                    "cancer_type": c.get("cancer_type", "general"),
                    "content_type": c.get("content_type", "reference"),
                }
                for c in batch
            ],
        )
        print(f"  Batch {i // batch_size + 1}/{total_batches} ({len(batch)} chunks)")

    print(f"Vector store complete: {collection.count()} chunks indexed")


def main():
    print("=" * 60)
    print("NG12 PDF Ingestion Pipeline")
    print("=" * 60)

    download_pdf(PDF_URL, NG12_PDF_PATH)
    pages = parse_pdf(NG12_PDF_PATH)
    chunks = chunk_pages(pages)
    build_vector_store(chunks)

    print("=" * 60)
    print("Done! Vector store is ready at:", CHROMA_PERSIST_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
