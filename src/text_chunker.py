# src/text_chunker.py

import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single text chunk with metadata."""
    chunk_id:   int
    text:       str
    word_count: int
    char_start: int   # character offset in original document
    char_end:   int


class TextChunker:
    """
    Splits a cleaned document into overlapping sentence-aware chunks.

    Strategy:
      1. Split text into individual sentences
      2. Greedily accumulate sentences until we hit the word limit
      3. Step back by `overlap_sentences` sentences before starting the next chunk
      4. Repeat until the full document is covered
    """

    def __init__(self, chunk_size: int = 300, overlap_sentences: int = 2):
        """
        Args:
            chunk_size:        target max words per chunk
            overlap_sentences: how many sentences to repeat at chunk boundaries
        """
        self.chunk_size        = chunk_size
        self.overlap_sentences = overlap_sentences

    # ── Public interface ───────────────────────────────────────────────

    def chunk_document(self, text: str, doc_name: str = "") -> list[Chunk]:
        """
        Split a full document string into a list of Chunk objects.
        """
        sentences = self._split_sentences(text)

        if not sentences:
            logger.warning(f"No sentences found in document: {doc_name}")
            return []

        chunks     = []
        chunk_id   = 0
        sent_idx   = 0               # pointer into sentences list

        while sent_idx < len(sentences):
            accumulated   = []
            word_count    = 0
            current_idx   = sent_idx

            # Greedily add sentences until we hit the word limit
            while current_idx < len(sentences):
                sentence   = sentences[current_idx]
                sent_words = len(sentence.split())

                # If adding this sentence would bust the limit AND
                # we already have some content, stop here
                if word_count + sent_words > self.chunk_size and accumulated:
                    break

                accumulated.append(sentence)
                word_count  += sent_words
                current_idx += 1

            # Build the chunk text
            chunk_text = " ".join(accumulated).strip()

            if chunk_text:
                # Compute character offsets in the original document
                char_start = text.find(accumulated[0])
                char_end   = text.find(accumulated[-1]) + len(accumulated[-1])

                chunks.append(Chunk(
                    chunk_id  = chunk_id,
                    text      = chunk_text,
                    word_count= word_count,
                    char_start= char_start,
                    char_end  = char_end,
                ))
                chunk_id += 1

            # Step forward, but overlap by rewinding `overlap_sentences`
            advance   = max(1, len(accumulated) - self.overlap_sentences)
            sent_idx += advance

        logger.info(
            f"'{doc_name}' → {len(sentences)} sentences → "
            f"{len(chunks)} chunks (target size: {self.chunk_size} words)"
        )
        return chunks

    def chunk_documents(self, documents: dict[str, str]) -> dict[str, list[Chunk]]:
        """
        Chunk every document in a {filename: text} dict.
        Returns {filename: [Chunk, ...]}
        """
        return {
            name: self.chunk_document(text, doc_name=name)
            for name, text in documents.items()
        }

    # ── Private helpers ────────────────────────────────────────────────

    def _split_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences using punctuation heuristics.
        Handles abbreviations, decimals, and ellipses reasonably well.
        """
        # Split on . ! ? followed by whitespace + capital letter
        # but NOT on decimals (3.14) or common abbreviations (Mr. Dr.)
        sentence_endings = re.compile(
            r'(?<!\w\.\w.)'           # not abbreviations like U.S.A.
            r'(?<![A-Z][a-z]\.)'      # not titles like Dr. Mr.
            r'(?<=\.|\!|\?)'          # must end with . ! ?
            r'\s+'                    # followed by whitespace
            r'(?=[A-Z])'              # next word starts with capital
        )

        raw_sentences = sentence_endings.split(text)

        # Strip whitespace and filter out empty / very short fragments
        sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 10]
        return sentences