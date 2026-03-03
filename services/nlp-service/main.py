import uuid
from consumer import get_messages
from producer import send_processed
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
        text = message.get("text", "")
        doc_id = message.get("message_id", str(uuid.uuid4()))

        # ML Task 1 – Named Entity Recognition
        entities = extract_entities(text)

        # ML Task 2 – Relation Extraction
        relationships = extract_relationships(text, entities)

        # ML Task 3 – Clustering (batch-triggered)
        add_to_cluster_buffer(doc_id, text)
        clusters = run_clustering() if should_cluster() else []

        structured_output = {
            "message_id": doc_id,
            "original_text": text,
            "entities": entities,
            "relationships": relationships,
            "clusters": clusters,          # empty list until batch is full
        }

        print("Processed:", structured_output)
        send_processed(structured_output)


if __name__ == "__main__":
    run()
