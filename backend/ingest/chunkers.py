from pathlib import Path
import re
from typing import Iterable, Iterator, Dict


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n".join([p.extract_text() or "" for p in reader.pages])
    except Exception:
        return ""


def _read_docx(path: Path) -> str:
    try:
        import docx
        d = docx.Document(str(path))
        return "\n".join([p.text for p in d.paragraphs])
    except Exception:
        return ""


def _read_html(path: Path) -> str:
    try:
        from bs4 import BeautifulSoup
        html = path.read_text(encoding="utf-8", errors="ignore")
        return BeautifulSoup(html, "html.parser").get_text(separator="\n")
    except Exception:
        return ""


READERS = {
    ".txt": _read_txt,
    ".pdf": _read_pdf,
    ".docx": _read_docx,
    ".html": _read_html,
    ".htm": _read_html,
}


def load_docs_from_folder(folder: str) -> Iterator[Dict[str, str]]:
    folder_p = Path(folder)
    for path in folder_p.rglob("*"):
        reader = READERS.get(path.suffix.lower())
        if reader is None:
            continue
        text = reader(path)
        if not text:
            continue
        yield {"title": path.stem, "source": str(path), "text": clean_text(text)}


def clean_text(text: str) -> str:
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> Iterator[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        overlap = chunk_size // 4
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        yield text[start:end]
        if end >= length:
            break
        start = max(end - overlap, 0)
        if start >= length:
            break


def build_chunks(
    docs: Iterable[Dict[str, str]],
    chunk_size: int = 1200,
    overlap: int = 200,
    domain: str = "unknown",
) -> Iterator[Dict[str, str]]:
    for doc in docs:
        for idx, part in enumerate(chunk_text(doc["text"], chunk_size, overlap)):
            yield {
                "doc_id": doc["source"],
                "chunk_id": idx,
                "domain": domain,
                "title": doc["title"],
                "source": doc["source"],
                "lang": "fr",
                "text": part,
            }
