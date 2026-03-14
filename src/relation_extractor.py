# src/relation_extractor.py

import re
import logging
import ollama
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass, field

from .text_chunker      import Chunk
from .concept_extractor import ExtractedConcepts

logger = logging.getLogger(__name__)


@dataclass
class Triplet:
    """A single subject → predicate → object relationship."""
    subject:   str
    predicate: str
    obj:       str        # 'object' is a Python builtin — using obj instead
    chunk_id:  int
    doc_name:  str = ""


@dataclass
class ExtractedRelations:
    """All triplets extracted from a single chunk."""
    chunk_id:  int
    triplets:  list[Triplet] = field(default_factory=list)
    raw_response: str        = ""


class RelationExtractor:
    """
    Uses Mistral via Ollama to extract subject-predicate-object triplets
    from each chunk. These triplets become the edges of the knowledge graph.

    Also supports a lightweight co-occurrence fallback — if two concepts
    appear in the same chunk, we add a weak 'co-occurs with' edge so the
    graph stays connected even when the LLM misses a relationship.
    """

    # ── Prompt templates ──────────────────────────────────────────────

    SYSTEM_PROMPT = """You are an expert knowledge graph builder specialising in 
relation extraction. You extract precise subject-predicate-object triplets from text.
You always respond in the exact pipe-delimited format requested.
Never add explanations, preamble, numbering, or bullet points."""

    USER_PROMPT_TEMPLATE = """Extract all relationships from the text below as triplets.

Format rules (STRICT):
- One triplet per line
- Each line: subject | predicate | object
- Use lowercase for all three parts
- Subject and object must be short noun phrases (1-4 words)
- Predicate should be a short verb phrase (1-5 words) like:
    is a type of, is used in, enables, consists of, was introduced by,
    is a subset of, uses, applies to, is part of, relates to
- Only extract relationships explicitly stated or strongly implied in the text
- Do NOT add relationships not present in the text
- Return 3-10 triplets per chunk

Known concepts in this text (use these exactly when they appear):
{concepts}

Text:
{chunk_text}

Triplets (subject | predicate | object):"""

    # ── Constructor ───────────────────────────────────────────────────

    def __init__(
        self,
        model: str             = "mistral",
        temperature: float     = 0.0,
        add_cooccurrence: bool = True,
    ):
        self.model            = model
        self.temperature      = temperature
        self.add_cooccurrence = add_cooccurrence

    # ── Public interface ───────────────────────────────────────────────

    def extract_from_chunks(
        self,
        chunks: list[Chunk],
        concept_extractions: list[ExtractedConcepts],
        doc_name: str = "",
    ) -> list[ExtractedRelations]:
        """
        Extract relations from every chunk, using the already-extracted
        concepts as hints to guide the LLM.
        """
        # Build a lookup: chunk_id → concept list
        concept_map = {
            ex.chunk_id: ex.concepts
            for ex in concept_extractions
        }

        results = []
        label   = f"Extracting relations [{doc_name}]"

        for chunk in tqdm(chunks, desc=label, unit="chunk"):
            concepts  = concept_map.get(chunk.chunk_id, [])
            extracted = self._extract_from_chunk(chunk, concepts, doc_name)

            # Optionally add co-occurrence edges for concepts in the same chunk
            if self.add_cooccurrence and len(concepts) >= 2:
                co_triplets = self._build_cooccurrence_triplets(
                    concepts, chunk.chunk_id, doc_name
                )
                extracted.triplets.extend(co_triplets)

            results.append(extracted)
            logger.info(
                f"  Chunk {chunk.chunk_id}: "
                f"{len(extracted.triplets)} triplets"
            )

        return results

    def extract_from_all_documents(
        self,
        all_chunks: dict[str, list[Chunk]],
        all_concepts: dict[str, list[ExtractedConcepts]],
    ) -> dict[str, list[ExtractedRelations]]:
        """
        Run relation extraction across all documents.
        Returns {doc_name: [ExtractedRelations, ...]}
        """
        return {
            doc_name: self.extract_from_chunks(
                chunks,
                concept_extractions = all_concepts.get(doc_name, []),
                doc_name            = doc_name,
            )
            for doc_name, chunks in all_chunks.items()
        }

    def to_dataframe(
        self,
        all_relations: dict[str, list[ExtractedRelations]],
    ) -> pd.DataFrame:
        """
        Flatten all extracted triplets into a single Pandas DataFrame.

        Columns: subject | predicate | object | chunk_id | doc_name
        This DataFrame is the direct input to the graph builder.
        """
        rows = []
        for doc_name, relation_list in all_relations.items():
            for extraction in relation_list:
                for triplet in extraction.triplets:
                    rows.append({
                        "subject":   triplet.subject,
                        "predicate": triplet.predicate,
                        "object":    triplet.obj,
                        "chunk_id":  triplet.chunk_id,
                        "doc_name":  triplet.doc_name,
                    })

        df = pd.DataFrame(rows, columns=[
            "subject", "predicate", "object", "chunk_id", "doc_name"
        ])

        logger.info(
            f"Total triplets: {len(df)} "
            f"({df[['subject','object']].stack().nunique()} unique concepts)"
        )
        return df

    # ── Private helpers ────────────────────────────────────────────────

    def _extract_from_chunk(
        self,
        chunk: Chunk,
        concepts: list[str],
        doc_name: str,
    ) -> ExtractedRelations:
        """Send one chunk to Mistral and parse the triplet list."""
        concept_hint = ", ".join(concepts) if concepts else "none identified"
        prompt = self.USER_PROMPT_TEMPLATE.format(
            chunk_text = chunk.text,
            concepts   = concept_hint,
        )

        try:
            response = ollama.chat(
                model    = self.model,
                options  = {"temperature": self.temperature},
                messages = [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            )
            raw = response["message"]["content"].strip()

        except Exception as e:
            logger.error(f"Ollama call failed on chunk {chunk.chunk_id}: {e}")
            raw = ""

        triplets = self._parse_triplets(raw, chunk.chunk_id, doc_name)

        return ExtractedRelations(
            chunk_id     = chunk.chunk_id,
            triplets     = triplets,
            raw_response = raw,
        )

    def _parse_triplets(
        self,
        raw: str,
        chunk_id: int,
        doc_name: str,
    ) -> list[Triplet]:
        """
        Parse pipe-delimited triplet lines from LLM output.

        Handles messy output like:
          - Extra spaces around pipes
          - Numbered lines ("1. subject | pred | obj")
          - Lines with more or fewer than 3 parts (skipped)
          - Trailing punctuation
        """
        if not raw:
            return []

        triplets = []

        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue

            # Strip leading list markers
            line = re.sub(r"^\s*[\d]+[\.\)]\s*", "", line)
            line = re.sub(r"^\s*[-\*•]\s*",      "", line)

            parts = [p.strip().lower().strip(".:;'\"") for p in line.split("|")]

            # We need exactly 3 parts
            if len(parts) != 3:
                continue

            subject, predicate, obj = parts

            # Skip if any part is empty or suspiciously long
            if not all([subject, predicate, obj]):
                continue
            if any(len(p.split()) > 8 for p in [subject, predicate, obj]):
                continue

            triplets.append(Triplet(
                subject   = subject,
                predicate = predicate,
                obj       = obj,
                chunk_id  = chunk_id,
                doc_name  = doc_name,
            ))

        return triplets

    def _build_cooccurrence_triplets(
        self,
        concepts: list[str],
        chunk_id: int,
        doc_name: str,
    ) -> list[Triplet]:
        """
        Build lightweight 'co-occurs with' edges between every pair of
        concepts found in the same chunk. These get a weaker weight later.
        Only adds pairs not already covered by LLM-extracted relations.
        """
        co_triplets = []
        for i, c1 in enumerate(concepts):
            for c2 in concepts[i + 1:]:
                co_triplets.append(Triplet(
                    subject   = c1,
                    predicate = "co-occurs with",
                    obj       = c2,
                    chunk_id  = chunk_id,
                    doc_name  = doc_name,
                ))
        return co_triplets