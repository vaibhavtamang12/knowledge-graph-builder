import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import re

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
# Primary: transformer-based model for higher-quality NER + dependency parsing.
# Fallback: small model if the transformer model isn't installed.
try:
    nlp = spacy.load("en_core_web_trf")
    _USING_TRF = True
except OSError:
    nlp = spacy.load("en_core_web_sm")
    _USING_TRF = False

# ──────────────────────────────────────────────
# ML TASK 1: Named Entity Recognition (NER)
# ──────────────────────────────────────────────

def _ent_confidence(ent) -> float:
    """
    Return a confidence score for a named entity.
    - Transformer pipeline: derive score from token probabilities when available.
    - Small model: return a fixed heuristic (0.85 for multi-token, 0.75 single).
    """
    if _USING_TRF:
        # spaCy trf stores per-token scores in doc._.trf_data; use span length
        # as a lightweight proxy (longer spans that agree are more reliable).
        return round(min(0.95, 0.80 + len(ent) * 0.03), 2)
    return 0.85 if len(ent) > 1 else 0.75


def extract_entities(text: str) -> list:
    """
    Extracts named entities from text using spaCy NER.
    Returns a list of dicts with text, label, start_char, end_char, confidence.
    """
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "normalized_text": ent.text.lower(),
            "label": ent.label_,
            "start_char": ent.start_char,
            "end_char": ent.end_char,
            "confidence": _ent_confidence(ent),
        })
    return entities


# ──────────────────────────────────────────────────────────────────────────────
# ML TASK 2: Relation Extraction
#   2a. Pattern-based typed relations  (fast, high-precision)
#   2b. Transformer dependency-based SVO triples  (broader coverage)
# ──────────────────────────────────────────────────────────────────────────────

# Pattern-based rules: (keyword_regex, subject_label, object_label, relation_type)
RELATION_PATTERNS = [
    (r"\b(ceo|chief executive|founder|co-founder)\b",          "PERSON", "ORG",  "CEO_OF"),
    (r"\b(works? (at|for)|employed (at|by)|joined)\b",         "PERSON", "ORG",  "WORKS_FOR"),
    (r"\b(headquartered in|based in|offices? in)\b",           "ORG",    "GPE",  "HEADQUARTERED_IN"),
    (r"\b(lives? in|resides? in|from)\b",                      "PERSON", "GPE",  "LOCATED_IN"),
    (r"\b(acquired|bought|purchased|merged with)\b",           "ORG",    "ORG",  "ACQUIRED"),
    (r"\b(partnered with|partnership with)\b",                 "ORG",    "ORG",  "PARTNERED_WITH"),
    (r"\b(competes? with|rival of|competitor of)\b",           "ORG",    "ORG",  "COMPETES_WITH"),
    (r"\b(invested in|funding|backed)\b",                      "ORG",    "ORG",  "INVESTED_IN"),
]

# Verbs that carry meaningful relation semantics for SVO extraction
_MEANINGFUL_VERBS = {
    "found", "create", "build", "lead", "run", "head", "acquire", "buy",
    "sell", "invest", "fund", "partner", "join", "launch", "develop",
    "produce", "own", "manage", "hire", "fire", "appoint", "announce",
    "release", "publish", "sign", "win", "lose", "beat", "support",
}


def _svo_confidence(subj_ent, obj_ent, verb_token) -> float:
    """
    Heuristic confidence score for an SVO triple:
    - Base: 0.70
    - +0.10 if verb root is a known meaningful verb
    - +0.05 if both ends are named entities (not just noun chunks)
    - +0.05 if transformer model is active
    """
    score = 0.70
    if verb_token.lemma_.lower() in _MEANINGFUL_VERBS:
        score += 0.10
    if subj_ent and obj_ent:
        score += 0.05
    if _USING_TRF:
        score += 0.05
    return round(min(score, 0.95), 2)


def _span_to_entity_info(span, ent_map: dict) -> dict:
    """Return entity info for a span, falling back to NOUN_CHUNK if not a NE."""
    key = span.text.lower()
    if key in ent_map:
        return ent_map[key]
    return {"text": span.text, "normalized_text": key, "label": "NOUN_CHUNK"}


def _extract_svo_triples(doc, ent_map: dict) -> list:
    """
    Walk the dependency tree to extract Subject–Verb–Object triples.
    Returns a list of relationship dicts with confidence scores.
    """
    triples = []
    for token in doc:
        # We want the root verb of each clause
        if token.pos_ not in ("VERB", "AUX") or token.dep_ not in ("ROOT", "relcl", "advcl", "ccomp", "xcomp"):
            continue

        # Collect nominal subjects
        subjects = [
            child for child in token.children
            if child.dep_ in ("nsubj", "nsubjpass", "csubj")
        ]
        # Collect direct/prepositional objects
        objects = [
            child for child in token.children
            if child.dep_ in ("dobj", "pobj", "attr", "oprd")
        ]
        # Also walk prep children for pobj
        for child in token.children:
            if child.dep_ == "prep":
                objects.extend(
                    gc for gc in child.children if gc.dep_ == "pobj"
                )

        for subj_tok in subjects:
            subj_span = subj_tok.subtree
            subj_text = doc[subj_tok.left_edge.i: subj_tok.right_edge.i + 1]
            subj_info = _span_to_entity_info(subj_text, ent_map)

            for obj_tok in objects:
                obj_text = doc[obj_tok.left_edge.i: obj_tok.right_edge.i + 1]
                obj_info = _span_to_entity_info(obj_text, ent_map)

                if subj_info["normalized_text"] == obj_info["normalized_text"]:
                    continue

                subj_ent = ent_map.get(subj_info["normalized_text"])
                obj_ent  = ent_map.get(obj_info["normalized_text"])

                triples.append({
                    "subject":       subj_info["text"],
                    "subject_label": subj_info["label"],
                    "object":        obj_info["text"],
                    "object_label":  obj_info["label"],
                    "type":          token.lemma_.upper(),   # e.g. "FOUND", "ACQUIRE"
                    "source":        "svo",
                    "confidence":    _svo_confidence(subj_ent, obj_ent, token),
                })

    return triples


def extract_relationships(text: str, entities: list) -> list:
    """
    Extracts typed relationships between entity pairs using two complementary
    strategies:

    1. Pattern rules  – fast keyword matching for well-known relation types
                        (CEO_OF, WORKS_FOR, ACQUIRED, …).  confidence = 0.85
    2. SVO triples    – transformer dependency parse to find Subject–Verb–Object
                        structures across the whole sentence.  confidence varies.

    Both sets are merged, deduplicated on (subject, object, type), and sorted
    by confidence descending.
    """
    doc = nlp(text)
    text_lower = text.lower()

    # Build a fast lookup: normalised entity text → entity dict
    ent_map: dict = {e["normalized_text"]: e for e in entities}
    entity_by_label: dict = {}
    for ent in entities:
        entity_by_label.setdefault(ent["label"], []).append(ent)

    relationships = []

    # ── Strategy 1: pattern-based ──────────────────────────────────────────
    for pattern, subj_label, obj_label, rel_type in RELATION_PATTERNS:
        if not re.search(pattern, text_lower):
            continue
        for subj in entity_by_label.get(subj_label, []):
            for obj in entity_by_label.get(obj_label, []):
                if subj["normalized_text"] == obj["normalized_text"]:
                    continue
                relationships.append({
                    "subject":       subj["text"],
                    "subject_label": subj["label"],
                    "object":        obj["text"],
                    "object_label":  obj["label"],
                    "type":          rel_type,
                    "source":        "pattern",
                    "confidence":    0.85,
                })

    # ── Strategy 2: SVO triples from dependency parse ──────────────────────
    svo_triples = _extract_svo_triples(doc, ent_map)
    relationships.extend(svo_triples)

    # ── Deduplication on (subject, object, type) ───────────────────────────
    # When duplicates exist, keep the one with the higher confidence.
    dedup: dict = {}
    for r in relationships:
        key = (r["subject"].lower(), r["object"].lower(), r["type"])
        if key not in dedup or r["confidence"] > dedup[key]["confidence"]:
            dedup[key] = r

    result = sorted(dedup.values(), key=lambda x: x["confidence"], reverse=True)
    return result


# ──────────────────────────────────────────────
# ML TASK 3: Text Clustering
# ──────────────────────────────────────────────

_document_buffer: list = []
CLUSTER_BATCH_SIZE = 10
N_CLUSTERS = 3


def add_to_cluster_buffer(doc_id: str, text: str):
    """Append a document to the in-memory buffer for batch clustering."""
    _document_buffer.append({"id": doc_id, "text": text})


def should_cluster() -> bool:
    return len(_document_buffer) >= CLUSTER_BATCH_SIZE


def run_clustering() -> list:
    """
    Vectorise buffered documents with TF-IDF and cluster with KMeans.
    Returns [{id, text, cluster_id}, ...].
    """
    if len(_document_buffer) < N_CLUSTERS:
        return []

    texts = [d["text"] for d in _document_buffer]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
    X = vectorizer.fit_transform(texts)

    n_clusters = min(N_CLUSTERS, len(texts))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X)

    return [
        {"id": doc["id"], "text": doc["text"], "cluster_id": int(labels[i])}
        for i, doc in enumerate(_document_buffer)
    ]