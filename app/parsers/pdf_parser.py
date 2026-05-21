"""
PDF text extraction using PyMuPDF (fitz).
"""

from pathlib import Path

import fitz  # PyMuPDF

from app.core.exceptions import ParsingError
from app.core.logging import get_logger
from app.parsers.base import BaseParser, ParsedDocument

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf"}


class PDFParser(BaseParser):
    """Extracts text from PDF documents using PyMuPDF."""

    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() in SUPPORTED_EXTENSIONS

    def parse(self, file_path: Path) -> ParsedDocument:
        """Extract text from every page of a PDF file.

        Each page's content is separated by a clear page marker to preserve
        structural context during chunking.
        """
        logger.info("Parsing PDF: %s", file_path.name)

        if not file_path.exists():
            raise ParsingError(f"PDF file not found: {file_path}")

        try:
            doc = fitz.open(str(file_path))
        except Exception as exc:
            raise ParsingError(
                f"Failed to open PDF: {file_path.name}",
                detail=str(exc),
            ) from exc

        pages_text: list[str] = []
        page_count = len(doc)

        for page_num in range(page_count):
            page = doc[page_num]
            text = page.get_text("text")
            if text and text.strip():
                pages_text.append(
                    f"--- Page {page_num + 1} ---\n{text.strip()}"
                )

        doc.close()

        full_text = "\n\n".join(pages_text)

        if not full_text.strip():
            raise ParsingError(
                f"No extractable text found in PDF: {file_path.name}",
                detail="The PDF may contain only images or scanned content.",
            )

        logger.info(
            "PDF parsed successfully: %d pages, %d characters",
            page_count,
            len(full_text),
        )

        return ParsedDocument(
            text=full_text,
            page_count=page_count,
            metadata={
                "parser": "PyMuPDF",
                "filename": file_path.name,
            },
        )
