# System Architecture

## Data Flow

1. text-ingestor
   - Fetches raw text (NewsAPI)
   - Sends messages to Kafka topic: raw-text-stream

2. nlp-service
   - Consumes raw-text-stream
   - Extracts entities and relationships
   - Resolves duplicate entities
   - Sends structured output to Kafka topic: nlp-processed

3. graph-builder
   - Consumes nlp-processed
   - Writes nodes and relationships to Neo4j

4. api-service
   - Queries Neo4j
   - Exposes REST endpoints
   - Exposes WebSocket updates

5. frontend
   - Calls API endpoints
   - Renders graph visualization

## Kafka Topics

- raw-text-stream
- nlp-processed


## Service Responsibilities

### text-ingestor
Produces raw text to Kafka.

### nlp-service
Consumes raw text, extracts structured data.

### graph-builder
Writes structured data into Neo4j.

### api-service
Exposes graph data via REST.