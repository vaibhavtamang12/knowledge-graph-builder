import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import re

nlp = spacy.load("en_core_web_sm")

# ──────────────────────────────────────────────
# ML TASK 1: Named Entity Recognition (NER)
# ──────────────────────────────────────────────

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
            "confidence": 1.0  # placeholder; swap in a transformer model for real scores
        })

    return entities


# ──────────────────────────────────────────────
# ML TASK 2: Relation Extraction
# ──────────────────────────────────────────────

# Pattern-based rules: (keyword_regex, subject_label, object_label, relation_type)
RELATION_PATTERNS = [
    (r"\b(ceo|chief executive|founder|co-founder)\b",           "PERSON", "ORG",  "CEO_OF"),
    (r"\b(works? (at|for)|employed (at|by)|joined)\b",          "PERSON", "ORG",  "WORKS_FOR"),
    (r"\b(headquartered in|based in|offices? in)\b",            "ORG",    "GPE",  "HEADQUARTERED_IN"),
    (r"\b(lives? in|resides? in|from)\b",                       "PERSON", "GPE",  "LOCATED_IN"),
    (r"\b(acquired|bought|purchased|merged with)\b",            "ORG",    "ORG",  "ACQUIRED"),
    (r"\b(partnered with|partnership with)\b",                  "ORG",    "ORG",  "PARTNERED_WITH"),
    (r"\b(competes? with|rival of|competitor of)\b",            "ORG",    "ORG",  "COMPETES_WITH"),
    (r"\b(invested in|funding|backed)\b",                       "ORG",    "ORG",  "INVESTED_IN"),
]


def extract_relationships(text: str, entities: list) -> list:
    """
    Extracts typed relationships between entity pairs using pattern rules.
    Replaces the old detect_relationships() which only detected CEO_OF.
    """
    text_lower = text.lower()
    entity_by_label: dict = {}
    for ent in entities:
        entity_by_label.setdefault(ent["label"], []).append(ent)

    relationships = []
    for pattern, subj_label, obj_label, rel_type in RELATION_PATTERNS:
        if not re.search(pattern, text_lower):
            continue

        for subj in entity_by_label.get(subj_label, []):
            for obj in entity_by_label.get(obj_label, []):
                if subj["text"].lower() == obj["text"].lower():
                    continue
                relationships.append({
                    "subject": subj["text"],
                    "subject_label": subj["label"],
                    "object": obj["text"],
                    "object_label": obj["label"],
                    "type": rel_type,
                    "confidence": 0.85
                })

    # Deduplicate on (subject, object, type)
    seen = set()
    unique = []
    for r in relationships:
        key = (r["subject"].lower(), r["object"].lower(), r["type"])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    return unique


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
