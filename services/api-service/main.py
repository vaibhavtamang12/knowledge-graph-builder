import os
import json
import logging
from fastapi import FastAPI, HTTPException, Query
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
NEO4J_URI             = os.getenv("NEO4J_URI",             "bolt://neo4j:7687")
NEO4J_USER            = os.getenv("NEO4J_USER",            "neo4j")
NEO4J_PASSWORD        = os.getenv("NEO4J_PASSWORD",        "password")
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")

logger.info(f"Configuration loaded: NEO4J_URI={NEO4J_URI}, KAFKA={KAFKA_BOOTSTRAP_SERVERS}")


# ── Lazy Kafka producer ────────────────────────────────────────────────────
_producer = None

def get_producer():
    global _producer
    if _producer is None:
        try:
            from kafka import KafkaProducer
            from kafka.errors import KafkaError
            import time

            logger.info(f"Attempting to connect to Kafka at {KAFKA_BOOTSTRAP_SERVERS}")
            max_retries = 6
            for attempt in range(max_retries):
                try:
                    _producer = KafkaProducer(
                        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                        request_timeout_ms=10000,
                        retries=3,
                        connections_max_idle_ms=5000,
                    )
                    logger.info("✓ Successfully connected to Kafka")
                    return _producer
                except (KafkaError, Exception) as e:
                    if attempt < max_retries - 1:
                        wait_time = 5 * (attempt + 1)
                        logger.warning(f"Kafka attempt {attempt+1}/{max_retries} failed: {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed to connect to Kafka after {max_retries} attempts: {e}")
                        raise
        except Exception as e:
            logger.error(f"Fatal error initializing Kafka producer: {e}", exc_info=True)
            raise
    return _producer


# ── Lazy Neo4j driver ──────────────────────────────────────────────────────
_driver = None

def get_driver():
    global _driver
    if _driver is None:
        try:
            from neo4j import GraphDatabase
            from neo4j.exceptions import ServiceUnavailable
            import time

            logger.info(f"Attempting to connect to Neo4j at {NEO4J_URI}")
            max_retries = 6
            for attempt in range(max_retries):
                try:
                    _driver = GraphDatabase.driver(
                        NEO4J_URI,
                        auth=(NEO4J_USER, NEO4J_PASSWORD),
                        connection_timeout=10,
                    )
                    _driver.verify_connectivity()
                    logger.info("✓ Successfully connected to Neo4j")
                    return _driver
                except (ServiceUnavailable, Exception) as e:
                    if attempt < max_retries - 1:
                        wait_time = 5 * (attempt + 1)
                        logger.warning(f"Neo4j attempt {attempt+1}/{max_retries} failed: {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed to connect to Neo4j after {max_retries} attempts: {e}")
                        raise
        except Exception as e:
            logger.error(f"Fatal error initializing Neo4j driver: {e}", exc_info=True)
            raise
    return _driver


# ── Helper: check if GDS plugin is available ──────────────────────────────
def _gds_available() -> bool:
    """Return True if the Neo4j Graph Data Science plugin is installed."""
    try:
        with get_driver().session() as session:
            session.run("CALL gds.version() YIELD version RETURN version")
        return True
    except Exception:
        return False


# ── Helper: project an in-memory graph for GDS algorithms ─────────────────
_GDS_GRAPH_NAME = "entity-graph"

def _ensure_gds_projection():
    """
    Create (or recreate) a named GDS in-memory graph projection of all
    Entity nodes and their relationships.  Safe to call multiple times.
    """
    with get_driver().session() as session:
        # Drop if exists so we always have a fresh projection
        session.run(
            """
            CALL gds.graph.exists($name) YIELD exists
            WITH exists WHERE exists = true
            CALL gds.graph.drop($name) YIELD graphName
            RETURN graphName
            """,
            name=_GDS_GRAPH_NAME,
        )
        session.run(
            """
            CALL gds.graph.project(
                $name,
                'Entity',
                {
                    __ALL__: {
                        type: '*',
                        orientation: 'UNDIRECTED'
                    }
                }
            )
            """,
            name=_GDS_GRAPH_NAME,
        )


# ═══════════════════════════════════════════════════════════════════════════
# EXISTING ENDPOINTS (unchanged)
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "api-service",
        "neo4j_uri": NEO4J_URI,
        "kafka_servers": KAFKA_BOOTSTRAP_SERVERS,
        "gds_available": _gds_available(),
    }


@app.post("/ingest")
def ingest_text(payload: dict):
    try:
        producer = get_producer()
        producer.send("raw-text-stream", payload)
        producer.flush()
        return {"status": "sent", "message": "Text ingestion request queued"}
    except Exception as e:
        logger.error(f"Kafka send failed: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Kafka unavailable: {str(e)}")


@app.get("/graph")
def get_graph():
    try:
        driver = get_driver()
        with driver.session() as session:
            nodes_result = session.run(
                "MATCH (n) RETURN id(n) as id, n.normalized_text as label, n.label as entity_type"
            )
            nodes = [record.data() for record in nodes_result]
            rels_result = session.run(
                "MATCH (a)-[r]->(b) RETURN id(a) as source, id(b) as target, type(r) as type"
            )
            links = [record.data() for record in rels_result]
        return {"nodes": nodes, "links": links, "node_count": len(nodes), "link_count": len(links)}
    except Exception as e:
        logger.error(f"Neo4j query failed: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Neo4j unavailable: {str(e)}")


@app.get("/entities")
def get_entities():
    try:
        driver = get_driver()
        with driver.session() as session:
            result = session.run(
                "MATCH (e:Entity) RETURN e.text as text, e.label as label, e.normalized_text as normalized_text"
            )
            entities = [record.data() for record in result]
        return {"entities": entities, "count": len(entities)}
    except Exception as e:
        logger.error(f"Neo4j query failed: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Neo4j unavailable: {str(e)}")


@app.get("/relationships")
def get_relationships():
    try:
        driver = get_driver()
        with driver.session() as session:
            result = session.run(
                "MATCH (a:Entity)-[r]->(b:Entity) "
                "RETURN a.text as subject, type(r) as relation, b.text as object"
            )
            relationships = [record.data() for record in result]
        return {"relationships": relationships, "count": len(relationships)}
    except Exception as e:
        logger.error(f"Neo4j query failed: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Neo4j unavailable: {str(e)}")


@app.get("/clusters")
def get_clusters():
    try:
        driver = get_driver()
        with driver.session() as session:
            result = session.run(
                "MATCH (d:Document) WHERE d.cluster_id IS NOT NULL "
                "RETURN d.doc_id as doc_id, d.text as text, d.cluster_id as cluster_id "
                "ORDER BY d.cluster_id"
            )
            clusters = [record.data() for record in result]
        return {"clusters": clusters, "count": len(clusters)}
    except Exception as e:
        logger.error(f"Neo4j query failed: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Neo4j unavailable: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════
# CHANGE 2.1 – GRAPH ALGORITHM ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/analytics/pagerank")
def get_pagerank(
    top_n: int = Query(default=10, ge=1, le=100, description="Number of top entities to return"),
):
    """
    Run PageRank on the Entity graph and return the most influential nodes.

    PageRank scores entities by how many other important entities point to
    them — a high score means the entity is a central hub in the knowledge
    graph (e.g. a major company or well-connected person).

    Uses GDS if available, falls back to a pure Cypher degree-centrality
    approximation so the endpoint always works even without the GDS plugin.

    Returns:
        List of {entity, label, score} sorted by score descending.
    """
    try:
        if _gds_available():
            _ensure_gds_projection()
            with get_driver().session() as session:
                result = session.run(
                    """
                    CALL gds.pageRank.stream($graph)
                    YIELD nodeId, score
                    WITH gds.util.asNode(nodeId) AS node, score
                    WHERE node:Entity
                    RETURN node.text AS entity,
                           node.label AS label,
                           round(score, 4) AS score
                    ORDER BY score DESC
                    LIMIT $top_n
                    """,
                    graph=_GDS_GRAPH_NAME,
                    top_n=top_n,
                )
                rows = [r.data() for r in result]
        else:
            # Fallback: degree centrality via plain Cypher (no GDS needed)
            with get_driver().session() as session:
                result = session.run(
                    """
                    MATCH (e:Entity)
                    OPTIONAL MATCH (e)-[r]-()
                    WITH e, count(r) AS degree
                    RETURN e.text AS entity,
                           e.label AS label,
                           toFloat(degree) AS score
                    ORDER BY score DESC
                    LIMIT $top_n
                    """,
                    top_n=top_n,
                )
                rows = [r.data() for r in result]

        return {
            "algorithm": "pagerank",
            "results": rows,
            "count": len(rows),
            "note": "GDS PageRank" if _gds_available() else "Degree-centrality fallback (install GDS for true PageRank)",
        }
    except Exception as e:
        logger.error(f"PageRank failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/communities")
def get_communities(
    min_size: int = Query(default=2, ge=1, description="Minimum community size to include"),
):
    """
    Detect communities (clusters) in the Entity graph using the Louvain method.

    Entities in the same community are more densely connected to each other
    than to the rest of the graph — this reveals natural groupings like
    'tech companies', 'politicians', 'geographic regions', etc.

    Uses GDS Louvain if available, falls back to weakly-connected-components
    via plain Cypher.

    Returns:
        List of communities, each with {community_id, members, size}.
    """
    try:
        if _gds_available():
            _ensure_gds_projection()
            with get_driver().session() as session:
                result = session.run(
                    """
                    CALL gds.louvain.stream($graph)
                    YIELD nodeId, communityId
                    WITH gds.util.asNode(nodeId) AS node, communityId
                    WHERE node:Entity
                    WITH communityId,
                         collect(node.text) AS members,
                         count(*) AS size
                    WHERE size >= $min_size
                    RETURN communityId AS community_id,
                           members,
                           size
                    ORDER BY size DESC
                    """,
                    graph=_GDS_GRAPH_NAME,
                    min_size=min_size,
                )
                rows = [r.data() for r in result]
        else:
            # Fallback: weakly connected components via plain Cypher
            with get_driver().session() as session:
                result = session.run(
                    """
                    MATCH (e:Entity)
                    OPTIONAL MATCH (e)-[r]-(neighbor:Entity)
                    WITH e, collect(DISTINCT coalesce(neighbor.normalized_text, e.normalized_text)) AS connected
                    RETURN id(e) AS community_id,
                           [e.text] + [x IN connected WHERE x <> e.normalized_text | x] AS members,
                           size(connected) + 1 AS size
                    ORDER BY size DESC
                    LIMIT 20
                    """
                )
                rows = [r.data() for r in result]

        return {
            "algorithm": "louvain_community_detection",
            "results": rows,
            "count": len(rows),
            "note": "GDS Louvain" if _gds_available() else "Connected-components fallback (install GDS for true Louvain)",
        }
    except Exception as e:
        logger.error(f"Community detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/shortest-path")
def get_shortest_path(
    from_entity: str = Query(..., description="Source entity text (e.g. 'Elon Musk')"),
    to_entity:   str = Query(..., description="Target entity text (e.g. 'Twitter')"),
):
    """
    Find the shortest path between two named entities in the graph.

    Useful for questions like "How is Elon Musk connected to Twitter?" —
    the path through intermediate nodes reveals the chain of relationships.

    Uses GDS shortestPath if available, falls back to Cypher
    shortestPath() which works on any Neo4j instance.

    Returns:
        {path: [entity names], length: int, relationships: [rel types]}
    """
    try:
        with get_driver().session() as session:
            if _gds_available():
                _ensure_gds_projection()
                result = session.run(
                    """
                    MATCH (source:Entity {normalized_text: $from_e}),
                          (target:Entity {normalized_text: $to_e})
                    CALL gds.shortestPath.dijkstra.stream($graph, {
                        sourceNode: source,
                        targetNode: target
                    })
                    YIELD nodeIds, costs
                    RETURN [nodeId IN nodeIds | gds.util.asNode(nodeId).text] AS path,
                           size(nodeIds) - 1 AS length
                    LIMIT 1
                    """,
                    graph=_GDS_GRAPH_NAME,
                    from_e=from_entity.lower(),
                    to_e=to_entity.lower(),
                )
            else:
                # Cypher shortestPath — always available
                result = session.run(
                    """
                    MATCH (a:Entity {normalized_text: $from_e}),
                          (b:Entity {normalized_text: $to_e}),
                          p = shortestPath((a)-[*..10]-(b))
                    RETURN [n IN nodes(p) | n.text] AS path,
                           length(p) AS length,
                           [r IN relationships(p) | type(r)] AS relationships
                    LIMIT 1
                    """,
                    from_e=from_entity.lower(),
                    to_e=to_entity.lower(),
                )

            row = result.single()
            if not row:
                raise HTTPException(
                    status_code=404,
                    detail=f"No path found between '{from_entity}' and '{to_entity}'. "
                           "Make sure both entities exist in the graph.",
                )
            return {
                "algorithm": "shortest_path",
                "from": from_entity,
                "to": to_entity,
                "path": row["path"],
                "length": row["length"],
                "relationships": row.get("relationships", []),
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Shortest path failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/similar-nodes")
def get_similar_nodes(
    entity: str = Query(..., description="Entity text to find similar nodes for (e.g. 'Tesla')"),
    top_n:  int = Query(default=5, ge=1, le=50, description="Number of similar nodes to return"),
):
    """
    Find nodes most similar to a given entity based on shared neighbours
    (Jaccard similarity on the neighbourhood set).

    Useful for questions like "What entities are similar to Tesla?" —
    entities that share many of the same connections will score highly.

    Uses GDS nodeSimilarity if available, falls back to a Cypher
    common-neighbour count.

    Returns:
        List of {entity, similarity_score} sorted by score descending.
    """
    try:
        if _gds_available():
            _ensure_gds_projection()
            with get_driver().session() as session:
                result = session.run(
                    """
                    CALL gds.nodeSimilarity.stream($graph)
                    YIELD node1, node2, similarity
                    WITH gds.util.asNode(node1) AS n1,
                         gds.util.asNode(node2) AS n2,
                         similarity
                    WHERE n1.normalized_text = $entity_lower
                       OR n2.normalized_text = $entity_lower
                    WITH CASE
                           WHEN n1.normalized_text = $entity_lower THEN n2.text
                           ELSE n1.text
                         END AS similar_entity,
                         round(similarity, 4) AS score
                    ORDER BY score DESC
                    LIMIT $top_n
                    RETURN similar_entity AS entity, score
                    """,
                    graph=_GDS_GRAPH_NAME,
                    entity_lower=entity.lower(),
                    top_n=top_n,
                )
                rows = [r.data() for r in result]
        else:
            # Fallback: common-neighbour count as similarity proxy
            with get_driver().session() as session:
                result = session.run(
                    """
                    MATCH (target:Entity {normalized_text: $entity_lower})--(neighbor)
                           --(candidate:Entity)
                    WHERE candidate.normalized_text <> $entity_lower
                    WITH candidate.text AS entity,
                         count(DISTINCT neighbor) AS common_neighbors
                    RETURN entity,
                           toFloat(common_neighbors) AS score
                    ORDER BY score DESC
                    LIMIT $top_n
                    """,
                    entity_lower=entity.lower(),
                    top_n=top_n,
                )
                rows = [r.data() for r in result]

        if not rows:
            raise HTTPException(
                status_code=404,
                detail=f"Entity '{entity}' not found or has no neighbours in the graph.",
            )

        return {
            "algorithm": "node_similarity",
            "query_entity": entity,
            "results": rows,
            "count": len(rows),
            "note": "GDS Jaccard similarity" if _gds_available() else "Common-neighbour fallback (install GDS for true Jaccard)",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Node similarity failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════
# Lifecycle
# ═══════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("API Service Starting Up")
    logger.info("=" * 60)
    logger.info(f"Neo4j URI: {NEO4J_URI}")
    logger.info(f"Kafka Servers: {KAFKA_BOOTSTRAP_SERVERS}")
    logger.info("Dependencies will be initialized on first use")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
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