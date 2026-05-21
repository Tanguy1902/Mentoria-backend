"""
Abstract base class for all document parsers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParsedDocument:
    """Result of parsing a document."""
    text: str
    page_count: int
    metadata: dict[str, str | int] = field(default_factory=dict)


class BaseParser(ABC):
    """Abstract interface that all document parsers must implement."""

    @abstractmethod
    def parse(self, file_path: Path) -> ParsedDocument:
        """Extract text content from a document file.

        Args:
            file_path: Absolute path to the document file.

        Returns:
            A ``ParsedDocument`` with the extracted text and metadata.

        Raises:
            ParsingError: If text extraction fails.
        """
        ...

    @abstractmethod
    def supports(self, file_extension: str) -> bool:
        """Check whether this parser supports the given file extension.

        Args:
            file_extension: Lowercase file extension including the dot (e.g., ``.pdf``).

        Returns:
            ``True`` if the parser can handle the file type.
        """
        ...
