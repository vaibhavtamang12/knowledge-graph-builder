# src/concept_extractor.py

import re
import logging
import ollama
from tqdm import tqdm
from dataclasses import dataclass, field

from .text_chunker import Chunk

logger = logging.getLogger(__name__)


@dataclass
class ExtractedConcepts:
    """Concepts extracted from a single chunk."""
    chunk_id:   int
    chunk_text: str
    concepts:   list[str] = field(default_factory=list)
    raw_response: str     = ""


class ConceptExtractor:
    """
    Uses a local Mistral model via Ollama to extract key concepts
    from each text chunk. Concepts become nodes in the knowledge graph.
    """

    # ── Prompt templates ──────────────────────────────────────────────

    SYSTEM_PROMPT = """You are an expert knowledge graph builder.
Your job is to extract key concepts, entities, and ideas from text.
You always respond in the exact format requested — nothing more, nothing less.
Never add explanations, preamble, or commentary."""

    USER_PROMPT_TEMPLATE = """Extract the key concepts from the text below.

Rules:
- Return ONLY a comma-separated list of concepts
- Each concept should be 1-4 words maximum
- Concepts should be nouns or noun phrases (entities, ideas, technologies, methods)
- Normalise to lowercase
- Remove duplicates
- Return 5-15 concepts depending on text richness
- Do NOT include verbs, adjectives alone, or full sentences

Text:
{chunk_text}

Concepts (comma-separated):"""

    # ── Constructor ───────────────────────────────────────────────────

    def __init__(self, model: str = "mistral", temperature: float = 0.0):
        self.model       = model
        self.temperature = temperature
        self._verify_connection()

    # ── Public interface ───────────────────────────────────────────────

    def extract_from_chunks(
        self,
        chunks: list[Chunk],
        doc_name: str = ""
    ) -> list[ExtractedConcepts]:
        """
        Run concept extraction on every chunk in a document.
        Returns a list of ExtractedConcepts (one per chunk).
        """
        results = []
        label   = f"Extracting concepts [{doc_name}]"

        for chunk in tqdm(chunks, desc=label, unit="chunk"):
            extracted = self._extract_from_chunk(chunk)
            results.append(extracted)

            logger.info(
                f"  Chunk {chunk.chunk_id}: "
                f"{len(extracted.concepts)} concepts → {extracted.concepts}"
            )

        return results

    def extract_from_all_documents(
        self,
        all_chunks: dict[str, list[Chunk]]
    ) -> dict[str, list[ExtractedConcepts]]:
        """
        Run extraction across all documents.
        Returns {doc_name: [ExtractedConcepts, ...]}
        """
        return {
            doc_name: self.extract_from_chunks(chunks, doc_name=doc_name)
            for doc_name, chunks in all_chunks.items()
        }

    # ── Private helpers ────────────────────────────────────────────────

    def _extract_from_chunk(self, chunk: Chunk) -> ExtractedConcepts:
        """Send one chunk to Mistral and parse the concept list."""
        prompt = self.USER_PROMPT_TEMPLATE.format(chunk_text=chunk.text)

        try:
            response = ollama.chat(
                model   = self.model,
                options = {"temperature": self.temperature},
                messages= [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            )
            raw = response["message"]["content"].strip()

        except Exception as e:
            logger.error(f"Ollama call failed on chunk {chunk.chunk_id}: {e}")
            raw = ""

        concepts = self._parse_concepts(raw)

        return ExtractedConcepts(
            chunk_id     = chunk.chunk_id,
            chunk_text   = chunk.text,
            concepts     = concepts,
            raw_response = raw,
        )

    def _parse_concepts(self, raw: str) -> list[str]:
        """
        Parse and clean the comma-separated concept list from the LLM.

        Handles edge cases like:
          - Numbered lists  ("1. machine learning, 2. neural network")
          - Bullet points   ("- machine learning\n- neural network")
          - Stray newlines  ("machine learning\nneural network")
          - Extra whitespace or punctuation
        """
        if not raw:
            return []

        # Replace newline-based lists with commas
        raw = re.sub(r"\n+", ", ", raw)

        # Strip leading list markers like "1.", "2.", "-", "*", "•"
        raw = re.sub(r"(?:^|(?<=,))\s*[\d]+\.\s*", " ", raw)
        raw = re.sub(r"(?:^|(?<=,))\s*[-\*•]\s*", " ", raw)

        # Split on commas
        parts = raw.split(",")

        concepts = []
        for part in parts:
            # Strip punctuation and whitespace from each concept
            concept = part.strip().strip(".:;'\"-").lower()

            # Filter out empties, pure stopwords, and overly long phrases
            if concept and 2 <= len(concept) <= 60 and len(concept.split()) <= 5:
                concepts.append(concept)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for c in concepts:
            if c not in seen:
                seen.add(c)
                unique.append(c)

        return unique

    # ── Health check ──────────────────────────────────────────────────

    def _verify_connection(self):
        """Quick ping to confirm Ollama is reachable and model is loaded."""
        try:
            ollama.chat(
                model   = self.model,
                options = {"temperature": 0},
                messages= [{"role": "user", "content": "ping"}],
            )
            logger.info(f"Ollama connection OK — model: {self.model}")
        except Exception as e:
            raise ConnectionError(
                f"Cannot reach Ollama. Is 'ollama serve' running?\n  Error: {e}"
            )