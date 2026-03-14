# config.py

import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM settings ──────────────────────────────────────────
OLLAMA_MODEL      = "mistral"        # swap to "phi3" if low on RAM
OLLAMA_TEMPERATURE = 0.0             # 0 = deterministic, better for extraction
OLLAMA_BASE_URL   = "http://localhost:11434"

# ── Chunking settings ─────────────────────────────────────
CHUNK_SIZE        = 300              # target tokens per chunk
CHUNK_OVERLAP     = 50               # overlap between chunks to preserve context

# ── Graph settings ────────────────────────────────────────
MIN_EDGE_WEIGHT   = 1                # edges below this are filtered out
TOP_K_NODES       = 50              # max nodes to show in visualization

# ── Paths ─────────────────────────────────────────────────
DATA_INPUT_DIR    = "data/input"
DATA_OUTPUT_DIR   = "data/output"
GRAPH_OUTPUT_FILE = "data/output/knowledge_graph.html"
GRAPH_DATA_FILE   = "data/output/graph_data.csv"

# config.py  — add these lines

# ── Edge weighting ────────────────────────────────────────
LLM_WEIGHT         = 3.0    # multiplier for LLM-extracted edges
COOC_WEIGHT        = 1.0    # multiplier for co-occurrence edges
FREQ_WEIGHT        = 0.5    # multiplier for log-frequency bonus
MIN_WEIGHT_FILTER  = 0.1    # edges below this weight are pruned