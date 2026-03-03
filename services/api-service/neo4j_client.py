#graph/services/api-service/neo4j_client.py
"""
Neo4j client module with lazy initialization and connection handling.

Instead of creating a driver at module load time (which fails if Neo4j isn't ready),
this module uses lazy initialization to create connections on demand.

Usage:
    from neo4j_client import get_driver
    
    driver = get_driver()
    with driver.session() as session:
        result = session.run("MATCH (e:Entity) RETURN e")
"""

import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Configuration from environment variables
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Cached driver instance
_driver = None


def get_driver():
    """
    Get or create a Neo4j driver instance.
    
    Uses lazy initialization to avoid connection errors at import time.
    The driver is created on first use and reused for all subsequent calls.
    
    Returns:
        neo4j.GraphDatabase driver instance
        
    Raises:
        Exception: If unable to connect to Neo4j after retries
    """
    global _driver
    
    if _driver is None:
        try:
            from neo4j import GraphDatabase
            from neo4j.exceptions import ServiceUnavailable
            import time
            
            logger.info(f"Initializing Neo4j driver: {NEO4J_URI}")
            
            # Retry logic: wait up to 30 seconds for Neo4j to be available
            max_retries = 6
            for attempt in range(max_retries):
                try:
                    _driver = GraphDatabase.driver(
                        NEO4J_URI,
                        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
                        connection_timeout=10,
                    )
                    # Verify the connection works
                    _driver.verify_connectivity()
                    logger.info(f"✓ Neo4j driver initialized successfully")
                    return _driver
                    
                except (ServiceUnavailable, Exception) as e:
                    if attempt < max_retries - 1:
                        wait_time = 5 * (attempt + 1)
                        logger.warning(
                            f"Neo4j connection attempt {attempt + 1}/{max_retries} failed: {e}\n"
                            f"Retrying in {wait_time}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            f"Failed to initialize Neo4j driver after {max_retries} attempts: {e}"
                        )
                        raise
                        
        except ImportError:
            logger.error("neo4j package not installed. Run: pip install neo4j")
            raise
    
    return _driver


def get_entities() -> List[Dict[str, Any]]:
    """
    Retrieve all Entity nodes from Neo4j.
    
    Returns:
        List of dictionaries containing entity data
        
    Raises:
        Exception: If Neo4j query fails
    """
    try:
        driver = get_driver()
        with driver.session() as session:
            result = session.run("MATCH (e:Entity) RETURN e")
            entities = []
            for record in result:
                entity = record["e"]
                # Convert neo4j node to dictionary
                entities.append({
                    "id": entity.identity,
                    "text": entity.get("text"),
                    "label": entity.get("label"),
                    "normalized_text": entity.get("normalized_text"),
                    "properties": dict(entity)
                })
            logger.info(f"Retrieved {len(entities)} entities from Neo4j")
            return entities
    except Exception as e:
        logger.error(f"Failed to retrieve entities: {e}", exc_info=True)
        raise


def get_relationships() -> List[Dict[str, Any]]:
    """
    Retrieve all relationships between Entity nodes.
    
    Returns relationships as Subject-Predicate-Object triples.
    
    Returns:
        List of dictionaries with keys: 'from', 'type', 'to'
        
    Raises:
        Exception: If Neo4j query fails
    """
    try:
        driver = get_driver()
        with driver.session() as session:
            result = session.run(
                "MATCH (a:Entity)-[r]->(b:Entity) "
                "RETURN a, r, b"
            )
            relationships = []
            for record in result:
                relationships.append({
                    "from": record["a"].get("text"),
                    "type": record["r"].type,
                    "to": record["b"].get("text"),
                    "properties": dict(record["r"])
                })
            logger.info(f"Retrieved {len(relationships)} relationships from Neo4j")
            return relationships
    except Exception as e:
        logger.error(f"Failed to retrieve relationships: {e}", exc_info=True)
        raise


def get_entity_by_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a specific entity by its text value.
    
    Args:
        text: The entity text to search for
        
    Returns:
        Entity dictionary if found, None otherwise
        
    Raises:
        Exception: If Neo4j query fails
    """
    try:
        driver = get_driver()
        with driver.session() as session:
            result = session.run(
                "MATCH (e:Entity) WHERE e.text = $text RETURN e LIMIT 1",
                text=text
            )
            record = result.single()
            if record:
                entity = record["e"]
                return {
                    "id": entity.identity,
                    "text": entity.get("text"),
                    "label": entity.get("label"),
                    "normalized_text": entity.get("normalized_text"),
                    "properties": dict(entity)
                }
            return None
    except Exception as e:
        logger.error(f"Failed to retrieve entity '{text}': {e}", exc_info=True)
        raise


def get_entity_neighbors(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Retrieve all entities connected to a given entity.
    
    Returns:
        Dictionary with 'incoming' and 'outgoing' relationship lists
        
    Raises:
        Exception: If Neo4j query fails
    """
    try:
        driver = get_driver()
        with driver.session() as session:
            # Get outgoing relationships
            outgoing_result = session.run(
                "MATCH (a:Entity)-[r]->(b:Entity) "
                "WHERE a.text = $text "
                "RETURN b.text as text, type(r) as relation",
                text=text
            )
            outgoing = [record.data() for record in outgoing_result]
            
            # Get incoming relationships
            incoming_result = session.run(
                "MATCH (a:Entity)-[r]->(b:Entity) "
                "WHERE b.text = $text "
                "RETURN a.text as text, type(r) as relation",
                text=text
            )
            incoming = [record.data() for record in incoming_result]
            
            logger.info(
                f"Retrieved neighbors for '{text}': "
                f"{len(outgoing)} outgoing, {len(incoming)} incoming"
            )
            
            return {
                "incoming": incoming,
                "outgoing": outgoing,
            }
    except Exception as e:
        logger.error(f"Failed to retrieve neighbors for '{text}': {e}", exc_info=True)
        raise


def close_driver():
    """
    Explicitly close the Neo4j driver connection.
    
    Call this during application shutdown.
    """
    global _driver
    if _driver is not None:
        logger.info("Closing Neo4j driver")
        _driver.close()
        _driver = None