import uuid
from consumer import get_messages
from producer import send_processed
from coref import resolve_coreferences          # ← NEW (Change 1.2)
from extractor import (
    extract_entities,
    extract_relationships,
    add_to_cluster_buffer,
    should_cluster,
    run_clustering,
)


def run():
    print("NLP Service started...")
    for message in get_messages():
        raw_text = message.get("text", "")
        doc_id   = message.get("message_id", str(uuid.uuid4()))

        # ── Change 1.2: Coreference Resolution ────────────────────────────
        # Resolve pronouns / repeated references before any NLP extraction so
        # that entity and relation extraction see canonical names throughout.
        # Example: "Elon Musk founded Tesla. He later bought Twitter."
        #       →  "Elon Musk founded Tesla. Elon Musk later bought Twitter."
        text = resolve_coreferences(raw_text)

        # ML Task 1 – Named Entity Recognition
        entities = extract_entities(text)

        # ML Task 2 – Relation Extraction
        relationships = extract_relationships(text, entities)

        # ML Task 3 – Clustering (batch-triggered)
        add_to_cluster_buffer(doc_id, text)
        clusters = run_clustering() if should_cluster() else []

        structured_output = {
            "message_id":    doc_id,
            "original_text": raw_text,       # keep original for traceability
            "resolved_text": text,           # coref-resolved version
            "entities":      entities,
            "relationships": relationships,
            "clusters":      clusters,
        }

        print("Processed:", structured_output)
        send_processed(structured_output)


if __name__ == "__main__":
    run()