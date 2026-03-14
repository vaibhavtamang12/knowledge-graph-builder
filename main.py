# main.py

import logging
import os
from src.document_loader    import DocumentLoader
from src.graph_cleaner      import GraphCleaner
from src.text_chunker       import TextChunker
from src.concept_extractor  import ConceptExtractor
from src.relation_extractor import RelationExtractor
from src.graph_builder      import GraphBuilder
from src.edge_weighter      import EdgeWeighter
from src.visualizer         import GraphVisualizer
from config import (
    DATA_INPUT_DIR, DATA_OUTPUT_DIR,
    CHUNK_SIZE, MIN_EDGE_WEIGHT,
    OLLAMA_MODEL, OLLAMA_TEMPERATURE,
    GRAPH_OUTPUT_FILE, GRAPH_DATA_FILE,
    TOP_K_NODES,
)

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt= "%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_pipeline():
    os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

    # ── Phase 4: Load documents ───────────────────────────
    logger.info("── Phase 4: Loading documents")
    loader = DocumentLoader(DATA_INPUT_DIR)
    docs   = loader.load_all()
    if not docs:
        logger.error("No documents found. Add .txt files to data/input/")
        return

    # ── Phase 5: Chunk ────────────────────────────────────
    logger.info("── Phase 5: Chunking text")
    chunker    = TextChunker(chunk_size=CHUNK_SIZE, overlap_sentences=2)
    all_chunks = chunker.chunk_documents(docs)

    # ── Phase 6: Extract concepts ─────────────────────────
    logger.info("── Phase 6: Extracting concepts")
    concept_extractor = ConceptExtractor(OLLAMA_MODEL, OLLAMA_TEMPERATURE)
    all_concepts      = concept_extractor.extract_from_all_documents(all_chunks)

    # ── Phase 7: Extract relations ────────────────────────
    logger.info("── Phase 7: Extracting relations")
    relation_extractor = RelationExtractor(
        model            = OLLAMA_MODEL,
        temperature      = OLLAMA_TEMPERATURE,
        add_cooccurrence = True,
    )
    all_relations = relation_extractor.extract_from_all_documents(
        all_chunks, all_concepts
    )
    df = relation_extractor.to_dataframe(all_relations)
    df.to_csv(GRAPH_DATA_FILE, index=False)
    logger.info(f"Triplets saved → {GRAPH_DATA_FILE}")

    # ── Phase 8: Build graph ──────────────────────────────
    logger.info("── Phase 8: Building graph")
    builder = GraphBuilder(min_edge_weight=MIN_EDGE_WEIGHT)
    G       = builder.build(df)
    
    # ── Phase 11: Clean graph ─────────────────────────────
    logger.info("── Phase 11: Cleaning graph")
    cleaner  = GraphCleaner(
        min_concept_length   = 2,
        min_concept_words    = 1,
        max_concept_words    = 5,
        min_node_degree      = 1,
        similarity_threshold = 0.85,
    )
    G_clean = cleaner.clean(G)

    # ── Phase 9: Weight edges ─────────────────────────────
    logger.info("── Phase 9: Computing edge weights")
    weighter   = EdgeWeighter()
    G_weighted = weighter.compute_weights(G)
    G_filtered = weighter.filter_weak_edges(G_weighted, min_weight=0.1)

    # ── Phase 10: Visualize ───────────────────────────────
    logger.info("── Phase 10: Rendering visualization")
    visualizer = GraphVisualizer(top_k_nodes=TOP_K_NODES)
    out_path   = visualizer.render(G_filtered, GRAPH_OUTPUT_FILE)

    logger.info(f"\n✓ Done!  Open this file in your browser:")
    logger.info(f"  {os.path.abspath(out_path)}")

    return G_filtered


if __name__ == "__main__":
    run_pipeline()