# src/document_loader.py

import os
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Loads and preprocesses text documents from the input directory.
    Supports .txt and .pdf files.
    """

    SUPPORTED_EXTENSIONS = {".txt", ".pdf"}

    def __init__(self, input_dir: str):
        self.input_dir = Path(input_dir)

    # ── Public interface ───────────────────────────────────────────────

    def load_all(self) -> dict[str, str]:
        """
        Load and clean every supported file in the input directory.
        Returns a dict of { filename: cleaned_text }
        """
        documents = {}

        files = [
            f for f in self.input_dir.iterdir()
            if f.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]

        if not files:
            logger.warning(
                f"No supported files found in {self.input_dir}. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )
            return documents

        for filepath in sorted(files):
            logger.info(f"Loading: {filepath.name}  [{filepath.suffix}]")
            try:
                if filepath.suffix.lower() == ".pdf":
                    raw_text = self._read_pdf(filepath)
                else:
                    raw_text = self._read_file(filepath)

                if not raw_text.strip():
                    logger.warning(f"  Empty content extracted from {filepath.name} — skipping")
                    continue

                clean_text = self._clean_text(raw_text)
                documents[filepath.name] = clean_text

                logger.info(
                    f"  → {len(clean_text.split())} words, "
                    f"{clean_text.count('.')} sentences after cleaning"
                )

            except Exception as e:
                logger.error(f"  Failed to load {filepath.name}: {e}")
                continue

        return documents

    def load_file(self, filename: str) -> str:
        """Load and clean a single file by name."""
        filepath = self.input_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        if filepath.suffix.lower() == ".pdf":
            raw_text = self._read_pdf(filepath)
        else:
            raw_text = self._read_file(filepath)

        return self._clean_text(raw_text)

    # ── PDF reader ─────────────────────────────────────────────────────

    def _read_pdf(self, filepath: Path) -> str:
        """
        Extract text from a PDF using pypdf.

        Handles:
          - Multi-page PDFs (concatenates all pages)
          - Scanned PDFs (warns user — no OCR built in)
          - Encrypted PDFs (attempts decryption with empty password)
          - Per-page extraction errors (skips bad pages, logs warning)
        """
        try:
            import pypdf
        except ImportError:
            raise ImportError(
                "pypdf is not installed. Run: pip install pypdf"
            )

        pages_text = []

        with open(filepath, "rb") as f:
            reader = pypdf.PdfReader(f)

            # Handle encrypted PDFs
            if reader.is_encrypted:
                try:
                    reader.decrypt("")   # try empty password first
                    logger.info(f"  Decrypted {filepath.name} with empty password")
                except Exception:
                    raise ValueError(
                        f"{filepath.name} is encrypted. "
                        f"Please decrypt it before loading."
                    )

            total_pages = len(reader.pages)
            logger.info(f"  PDF has {total_pages} pages")

            empty_pages = 0
            for i, page in enumerate(reader.pages):
                try:
                    text = page.extract_text()
                    if text and text.strip():
                        pages_text.append(text)
                    else:
                        empty_pages += 1
                except Exception as e:
                    logger.warning(f"  Could not extract page {i+1}: {e}")
                    continue

            if empty_pages > 0:
                logger.warning(
                    f"  {empty_pages}/{total_pages} pages had no extractable text. "
                    f"If this is a scanned PDF, you need OCR (e.g. pdf2image + tesseract)."
                )

        if not pages_text:
            raise ValueError(
                f"No text could be extracted from {filepath.name}. "
                f"It may be a scanned image PDF — see OCR note below."
            )

        # Join pages with double newline to preserve page boundaries
        full_text = "\n\n".join(pages_text)
        logger.info(
            f"  Extracted {len(full_text.split())} raw words "
            f"from {len(pages_text)} pages"
        )
        return full_text

    # ── TXT reader ─────────────────────────────────────────────────────

    def _read_file(self, filepath: Path) -> str:
        """Read raw text from a .txt file with encoding fallback."""
        try:
            return filepath.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning(
                f"  UTF-8 failed for {filepath.name}, retrying with latin-1"
            )
            return filepath.read_text(encoding="latin-1")

    # ── Text cleaning ──────────────────────────────────────────────────

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        Works for both .txt and .pdf sources.

        Steps:
          - Normalize unicode punctuation
          - Remove non-printable control characters
          - Fix hyphenated line breaks common in PDFs  (e.g. "learn-\ning")
          - Remove page headers/footers (short repeated lines)
          - Strip and collapse whitespace
        """
        # Normalize unicode punctuation
        text = text.replace("\u2013", "-").replace("\u2014", "-")
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = text.replace("\u00a0", " ")    # non-breaking space

        # Remove non-printable control characters (keep newline/tab)
        text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", "", text)

        # Fix PDF hyphenated line breaks: "learn-\ning" → "learning"
        text = re.sub(r"-\n(\S)", r"\1", text)

        # Remove lines that look like page numbers (lone digits or "Page N of M")
        text = re.sub(r"(?m)^\s*\d+\s*$", "", text)
        text = re.sub(r"(?mi)^page\s+\d+\s*(of\s+\d+)?\s*$", "", text)

        # Collapse 3+ blank lines into two
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Strip each line
        lines = [line.strip() for line in text.splitlines()]
        text  = "\n".join(lines)

        # Collapse multiple spaces
        text = re.sub(r" {2,}", " ", text)

        return text.strip()