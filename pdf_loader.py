import re
from typing import List
from pypdf import PdfReader

def load_pdf(file) -> str:
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        raw_text = page.extract_text()
        if raw_text:
            text += clean_pdf_text(raw_text)
    return text


def load_pdf_paginated(file) -> List[tuple[int, str]]:
    reader = PdfReader(file)
    pages_text = []

    for i, page in enumerate(reader.pages):
        raw_text = page.extract_text()
        if raw_text:
            pages_text.append((i + 1, clean_pdf_text(raw_text)))

    return pages_text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


def clean_pdf_text(text: str) -> str:

    text = re.sub(r"(\w+)\n(\w+)", r"\1\2", text)  # unisci parole spezzate da newline
    text = re.sub(r"\n+", "\n", text)              # rimuovi newline multipli
    text = re.sub(r"\s+", " ", text)               # rimuovi spazi multipli

    return text.lower()