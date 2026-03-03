# graph/services/api-service/main.py

import os
import json
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("api-service")
logging.basicConfig(level=logging.INFO)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Configuration from environment variables ────────────────────────────────
# These allow deployment flexibility (Docker, localhost, cloud, etc.)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")

logger.info(f"Configuration loaded: NEO4J_URI={NEO4J_URI}, KAFKA={KAFKA_BOOTSTRAP_SERVERS}")


# ── Lazy Kafka producer ────────────────────────────────────────────────────
# DO NOT create KafkaProducer at module level.
# Uvicorn imports this file before Kafka is ready, causing NoBrokersAvailable.
# Instead we create it once on first use and cache it.

_producer = None

def get_producer():
    """
    Lazy-load Kafka producer with retry logic.
    
    This prevents startup failures when Kafka isn't immediately available.
    The first API request that uses Kafka will wait for it to be ready.
    """
    global _producer
    if _producer is None:
        try:
            from kafka import KafkaProducer
            from kafka.errors import KafkaError
            import time
            
            logger.info(f"Attempting to connect to Kafka at {KAFKA_BOOTSTRAP_SERVERS}")
            
            # Retry logic: wait up to 30 seconds for Kafka to be ready
            max_retries = 6
            for attempt in range(max_retries):
                try:
                    _producer = KafkaProducer(
                        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                        request_timeout_ms=10000,
                        retries=3,
                        # Don't wait too long on first attempt
                        connections_max_idle_ms=5000,
                    )
                    logger.info("✓ Successfully connected to Kafka")
                    return _producer
                except (KafkaError, Exception) as e:
                    if attempt < max_retries - 1:
                        wait_time = 5 * (attempt + 1)  # 5s, 10s, 15s, 20s, 25s
                        logger.warning(
                            f"Kafka connection attempt {attempt + 1}/{max_retries} failed: {e}\n"
                            f"Retrying in {wait_time}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed to connect to Kafka after {max_retries} attempts: {e}")
                        raise
        except Exception as e:
            logger.error(f"Fatal error initializing Kafka producer: {e}", exc_info=True)
            raise
    
    return _producer


# ── Lazy Neo4j driver ──────────────────────────────────────────────────────
# Same principle: don't connect at import time.
# The first query attempt will wait for Neo4j to be available.

_driver = None

def get_driver():
    """
    Lazy-load Neo4j driver with retry logic.
    
    This prevents startup failures when Neo4j isn't immediately available.
    The first API request that queries Neo4j will wait for it to be ready.
    """
    global _driver
    if _driver is None:
        try:
            from neo4j import GraphDatabase
            from neo4j.exceptions import ServiceUnavailable
            import time
            
            logger.info(f"Attempting to connect to Neo4j at {NEO4J_URI}")
            
            # Retry logic: wait up to 30 seconds for Neo4j to be ready
            max_retries = 6
            for attempt in range(max_retries):
                try:
                    _driver = GraphDatabase.driver(
                        NEO4J_URI,
                        auth=(NEO4J_USER, NEO4J_PASSWORD),
                        connection_timeout=10,
                    )
                    # Verify connection works
                    _driver.verify_connectivity()
                    logger.info("✓ Successfully connected to Neo4j")
                    return _driver
                except (ServiceUnavailable, Exception) as e:
                    if attempt < max_retries - 1:
                        wait_time = 5 * (attempt + 1)  # 5s, 10s, 15s, 20s, 25s
                        logger.warning(
                            f"Neo4j connection attempt {attempt + 1}/{max_retries} failed: {e}\n"
                            f"Retrying in {wait_time}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed to connect to Neo4j after {max_retries} attempts: {e}")
                        raise
        except Exception as e:
            logger.error(f"Fatal error initializing Neo4j driver: {e}", exc_info=True)
            raise
    
    return _driver


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """
    Health check endpoint.
    
    Returns 200 if the API service is running.
    Does NOT check Neo4j or Kafka status.
    """
    return {
        "status": "ok",
        "service": "api-service",
        "neo4j_uri": NEO4J_URI,
        "kafka_servers": KAFKA_BOOTSTRAP_SERVERS,
    }


@app.post("/ingest")
def ingest_text(payload: dict):
    """
    Ingest raw text into the system.
    
    Sends the payload to Kafka topic 'raw-text-stream' for processing
    by downstream services (nlp-service, graph-builder).
    
    Returns:
        - 200 OK: Message sent to Kafka
        - 503 Service Unavailable: Kafka is not available
    """
    try:
        producer = get_producer()
        producer.send("raw-text-stream", payload)
        producer.flush()  # Ensure message is sent
        return {"status": "sent", "message": "Text ingestion request queued"}
    except Exception as e:
        logger.error(f"Kafka send failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"Kafka unavailable: {str(e)}"
        )


@app.get("/graph")
def get_graph():
    """
    Retrieve full graph data.
    
    Returns all nodes and relationships from Neo4j.
    Useful for complete graph visualization.
    
    Returns:
        - 200 OK: {nodes: [...], links: [...]}
        - 503 Service Unavailable: Neo4j is not available
    """
    try:
        driver = get_driver()
        with driver.session() as session:
            # Fetch all nodes
            nodes_result = session.run(
                "MATCH (n) RETURN id(n) as id, n.normalized_text as label, n.label as entity_type"
            )
            nodes = [record.data() for record in nodes_result]
            
            # Fetch all relationships
            rels_result = session.run(
                "MATCH (a)-[r]->(b) RETURN id(a) as source, id(b) as target, type(r) as type"
            )
            links = [record.data() for record in rels_result]
            
            return {
                "nodes": nodes,
                "links": links,
                "node_count": len(nodes),
                "link_count": len(links),
            }
    except Exception as e:
        logger.error(f"Neo4j query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"Neo4j unavailable: {str(e)}"
        )


@app.get("/entities")
def get_entities():
    """
    Retrieve all extracted entities.
    
    Returns all Entity nodes with their text, labels, and normalized forms.
    
    Returns:
        - 200 OK: List of entities
        - 503 Service Unavailable: Neo4j is not available
    """
    try:
        driver = get_driver()
        with driver.session() as session:
            result = session.run(
                "MATCH (e:Entity) RETURN e.text as text, e.label as label, e.normalized_text as normalized_text"
            )
            entities = [record.data() for record in result]
            return {
                "entities": entities,
                "count": len(entities),
            }
    except Exception as e:
        logger.error(f"Neo4j query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"Neo4j unavailable: {str(e)}"
        )


@app.get("/relationships")
def get_relationships():
    """
    Retrieve all relationships between entities.
    
    Returns Subject -> Predicate -> Object triples extracted from text.
    
    Returns:
        - 200 OK: List of relationships
        - 503 Service Unavailable: Neo4j is not available
    """
    try:
        driver = get_driver()
        with driver.session() as session:
            result = session.run(
                "MATCH (a:Entity)-[r]->(b:Entity) "
                "RETURN a.text as subject, type(r) as relation, b.text as object"
            )
            relationships = [record.data() for record in result]
            return {
                "relationships": relationships,
                "count": len(relationships),
            }
    except Exception as e:
        logger.error(f"Neo4j query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"Neo4j unavailable: {str(e)}"
        )


@app.get("/clusters")
def get_clusters():
    """
    Retrieve documents grouped by semantic clusters.
    
    Returns documents with assigned cluster IDs for similarity-based grouping.
    
    Returns:
        - 200 OK: List of documents with cluster assignments
        - 503 Service Unavailable: Neo4j is not available
    """
    try:
        driver = get_driver()
        with driver.session() as session:
            result = session.run(
                "MATCH (d:Document) WHERE d.cluster_id IS NOT NULL "
                "RETURN d.doc_id as doc_id, d.text as text, d.cluster_id as cluster_id "
                "ORDER BY d.cluster_id"
            )
            clusters = [record.data() for record in result]
            return {
                "clusters": clusters,
                "count": len(clusters),
            }
    except Exception as e:
        logger.error(f"Neo4j query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"Neo4j unavailable: {str(e)}"
        )


@app.on_event("startup")
async def startup_event():
    """
    Application startup event.
    
    Logs startup information but does NOT wait for dependencies.
    Dependencies will be initialized on first use (lazy loading).
    """
    logger.info("=" * 60)
    logger.info("API Service Starting Up")
    logger.info("=" * 60)
    logger.info(f"Neo4j URI: {NEO4J_URI}")
    logger.info(f"Kafka Servers: {KAFKA_BOOTSTRAP_SERVERS}")
    logger.info("Dependencies will be initialized on first use")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """
    Application shutdown event.
    
    Cleanly closes database connections.
    """
    global _driver, _producer
    
    if _driver is not None:
        logger.info("Closing Neo4j driver")
        _driver.close()
    
    if _producer is not None:
        logger.info("Closing Kafka producer")
        _producer.close()
    
    logger.info("API Service shutdown complete")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)