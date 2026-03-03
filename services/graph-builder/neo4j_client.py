from neo4j import GraphDatabase
import uuid

URI = "bolt://neo4j:7687"
USERNAME = "neo4j"
PASSWORD = "password"

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

def create_entity(tx, entity):
    tx.run(
        """
        MERGE (e:Entity {normalized_text: $normalized_text})
        ON CREATE SET e.entity_id = $entity_id,
                      e.text = $text,
                      e.label = $label
        """,
        entity_id=str(uuid.uuid4()),
        normalized_text=entity["text"].lower(),
        text=entity["text"],
        label=entity["label"]
    )

def insert_entities(entities):
    with driver.session() as session:
        for entity in entities:
            session.execute_write(create_entity, entity)

def create_relationship(tx, rel):
    rel_type = rel["type"].upper().replace(" ", "_")
    query = f"""
        MATCH (a:Entity {{normalized_text: $subject}})
        MATCH (b:Entity {{normalized_text: $object}})
        MERGE (a)-[:`{rel_type}`]->(b)
    """
    tx.run(query, subject=rel["subject"].lower(), object=rel["object"].lower())

def insert_relationships(relationships):
    with driver.session() as session:
        for rel in relationships:
            session.execute_write(create_relationship, rel)

def insert_cluster_result(tx, doc):
    """Store a document's cluster assignment as a :Document node."""
    tx.run(
        """
        MERGE (d:Document {doc_id: $doc_id})
        SET d.text = $text,
            d.cluster_id = $cluster_id
        """,
        doc_id=doc["message_id"],   # Fixed: was doc["id"], nlp-service sends "message_id"
        text=doc["text"],
        cluster_id=doc["cluster_id"]
    )

def insert_clusters(cluster_results: list):
    with driver.session() as session:
        for doc in cluster_results:
            session.execute_write(insert_cluster_result, doc)