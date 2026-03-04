from neo4j import GraphDatabase
import uuid
import json

URI      = "bolt://neo4j:7687"
USERNAME = "neo4j"
PASSWORD = "password"

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------

def create_entity(tx, entity):
    """
    Upsert an Entity node.

    Change 1.3 – Entity Linking: if the entity dict carries enrichment fields
    (wikipedia_url, wikidata_id, description, aliases, metadata, link_confidence)
    they are stored as node properties so the graph becomes a richer, contextual
    knowledge system.
    """
    tx.run(
        """
        MERGE (e:Entity {normalized_text: $normalized_text})
        ON CREATE SET
            e.entity_id       = $entity_id,
            e.text            = $text,
            e.label           = $label
        SET
            e.wikipedia_url     = $wikipedia_url,
            e.wikipedia_id      = $wikipedia_id,
            e.wikidata_id       = $wikidata_id,
            e.description       = $description,
            e.extract           = $extract,
            e.aliases           = $aliases,
            e.metadata          = $metadata,
            e.link_confidence   = $link_confidence
        """,
        entity_id       = str(uuid.uuid4()),
        normalized_text = entity["text"].lower(),
        text            = entity["text"],
        label           = entity["label"],
        # Entity-linking fields (default to empty / zero if not present)
        wikipedia_url   = entity.get("wikipedia_url",   ""),
        wikipedia_id    = entity.get("wikipedia_id",    ""),
        wikidata_id     = entity.get("wikidata_id",     ""),
        description     = entity.get("description",     ""),
        extract         = entity.get("extract",         ""),
        # aliases is a list → store as JSON string (Neo4j CE doesn't support
        # list properties on MERGE nodes without APOC; JSON is portable)
        aliases         = json.dumps(entity.get("aliases", [])),
        # metadata is a dict → store as JSON string for the same reason
        metadata        = json.dumps(entity.get("metadata", {})),
        link_confidence = entity.get("link_confidence", 0.0),
    )


def insert_entities(entities):
    with driver.session() as session:
        for entity in entities:
            session.execute_write(create_entity, entity)


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------

def create_relationship(tx, rel):
    rel_type = rel["type"].upper().replace(" ", "_")
    query = f"""
        MATCH (a:Entity {{normalized_text: $subject}})
        MATCH (b:Entity {{normalized_text: $object}})
        MERGE (a)-[r:`{rel_type}`]->(b)
        SET r.confidence = $confidence,
            r.source     = $source
    """
    tx.run(
        query,
        subject    = rel["subject"].lower(),
        object     = rel["object"].lower(),
        confidence = rel.get("confidence", 0.0),
        source     = rel.get("source", "unknown"),
    )


def insert_relationships(relationships):
    with driver.session() as session:
        for rel in relationships:
            session.execute_write(create_relationship, rel)


# ---------------------------------------------------------------------------
# Clusters
# ---------------------------------------------------------------------------

def insert_cluster_result(tx, doc):
    """Store a document's cluster assignment as a :Document node."""
    tx.run(
        """
        MERGE (d:Document {doc_id: $doc_id})
        SET d.text       = $text,
            d.cluster_id = $cluster_id
        """,
        doc_id     = doc["id"],
        text       = doc["text"],
        cluster_id = doc["cluster_id"],
    )


def insert_clusters(cluster_results: list):
    with driver.session() as session:
        for doc in cluster_results:
            session.execute_write(insert_cluster_result, doc)