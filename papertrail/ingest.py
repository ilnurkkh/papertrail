"""Step 1: PDF -> text, one string per page."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path


class IngestError(Exception):
    pass


def normalize(text: str) -> str:
    """One shared cleanup used BOTH for storing pages and for checking quotes."""
    text = unicodedata.normalize("NFKC", text)           # ligatures: ﬁ -> fi
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)   # re-join hyphenated line breaks
    return " ".join(text.split())                         # collapse all whitespace


def read_pdf(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.exists():
        raise IngestError(f"File not found: {path}")
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            raise IngestError("PDF is encrypted; not supported.")
        pages = [normalize(page.extract_text() or "") for page in reader.pages]
    except IngestError:
        raise
    except Exception as error:  # corrupt file, not a PDF, etc.
        raise IngestError(f"Could not read PDF: {error}") from error
    if not any(pages):
        raise IngestError("No text layer found (scanned PDF?). OCR is out of scope.")
    return pages
