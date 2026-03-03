# graph/services/graph-builder/main.py
from consumer import get_messages
from neo4j_client import insert_entities, insert_relationships, insert_clusters


def run():
    print("Graph Builder started...")

    for message in get_messages():
        entities      = message.get("entities", [])
        relationships = message.get("relationships", [])
        clusters      = message.get("clusters", [])   # NEW: ML Task 3 results

        if entities:
            insert_entities(entities)

        if relationships:
            insert_relationships(relationships)

        if clusters:
            insert_clusters(clusters)

        print("Inserted entities:", len(entities),
              "| relationships:", len(relationships),
              "| cluster docs:", len(clusters))


if __name__ == "__main__":
    run()
