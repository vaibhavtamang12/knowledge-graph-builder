# 🧠 Knowledge Graph Extraction

> Convert any text document or PDF into an interactive, queryable knowledge graph using a local LLM — no API keys, no cloud, no cost.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Mistral_7B-black?style=flat)
![NetworkX](https://img.shields.io/badge/NetworkX-Graph-orange?style=flat)
![PyVis](https://img.shields.io/badge/PyVis-Interactive_Viz-blue?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## 📌 What it does

Feed this pipeline any `.txt` or `.pdf` document and it produces:

- An **interactive HTML knowledge graph** you can explore in your browser
- A **CSV of extracted triplets** (subject → relation → object)
- A **CLI query interface** to ask questions about the graph

The entire pipeline runs **locally** using [Ollama](https://ollama.com) + Mistral 7B — your documents never leave your machine.

---

## 🖼️ Demo

```
Input:  Any .txt or .pdf document
         ↓
Output: Interactive graph (nodes = concepts, edges = relationships)
         ↓
Query:  "neighbours machine learning"
        "path deep learning, artificial intelligence"
        "ask what connects transformers to language models?"
```

---

## 🏗️ Architecture

```
data/input/*.pdf or *.txt
        │
        ▼
┌─────────────────┐
│ DocumentLoader  │  Load + clean text (txt / pdf)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  TextChunker    │  Split into overlapping chunks (~800 words)
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ CombinedExtractor   │  Single LLM call per chunk →
│  (Mistral 7B)       │  concepts (nodes) + relations (edges)
└────────┬────────────┘
         │
         ▼
┌─────────────────┐
│  GraphBuilder   │  Build weighted NetworkX DiGraph
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  GraphCleaner   │  Deduplicate, expand abbreviations, prune noise
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  EdgeWeighter   │  LLM edges (×3) + co-occurrence (×1) + freq bonus
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────────┐
│  GraphVisualizer│     │  GraphQueryEngine     │
│  (PyVis HTML)   │     │  CLI + LLM Q&A        │
└─────────────────┘     └──────────────────────┘
```

---

## ⚙️ Tech stack

| Component | Tool | Purpose |
|---|---|---|
| Language | Python 3.9+ | Core pipeline |
| Local LLM | Ollama + Mistral 7B | Concept & relation extraction |
| Graph library | NetworkX | Graph construction & analysis |
| Visualization | PyVis | Interactive HTML graph |
| Data handling | Pandas | Triplet storage & manipulation |
| PDF parsing | pypdf | Text extraction from PDFs |

---

## 🚀 Quick start

### 1. Prerequisites

```bash
# Python 3.9+
python --version

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull Mistral
ollama pull mistral
```

### 2. Clone & install

```bash
git clone https://github.com/yourusername/knowledge_graph.git
cd knowledge_graph

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Add your document

```bash
# Drop any .txt or .pdf into data/input/
cp your_document.pdf data/input/
```

### 4. Run the pipeline

```bash
# Make sure Ollama is running
ollama serve &

# Run
python main.py
```

### 5. Open the graph

```bash
xdg-open data/output/knowledge_graph.html   # Linux
open data/output/knowledge_graph.html        # macOS
```

---

## 🖥️ Interactive CLI

```bash
python cli.py
```

```
╔══════════════════════════════════════════════╗
║       Knowledge Graph Query Interface        ║
╚══════════════════════════════════════════════╝

kg> hubs                              # most connected concepts
kg> neighbours machine learning       # all connections for a concept
kg> path deep learning, ai            # shortest path between concepts
kg> search neural                     # find concepts by keyword
kg> ask what is backpropagation?      # LLM answers using graph context
kg> subgraph transformers             # render focused subgraph HTML
kg> summary                           # graph statistics
```

---

## 📁 Project structure

```
knowledge_graph/
├── data/
│   ├── input/          ← drop .txt or .pdf files here
│   ├── output/         ← generated graph HTML + CSV
│   └── cache/          ← LLM extraction cache (speeds up re-runs)
├── src/
│   ├── document_loader.py      # load & clean txt/pdf
│   ├── text_chunker.py         # sentence-aware chunking with overlap
│   ├── concept_extractor.py    # extract nodes via LLM
│   ├── relation_extractor.py   # extract edges via LLM
│   ├── combined_extractor.py   # single LLM call for both (faster)
│   ├── graph_builder.py        # build NetworkX DiGraph
│   ├── edge_weighter.py        # composite weight formula
│   ├── graph_cleaner.py        # dedup, abbreviation expansion, pruning
│   ├── visualizer.py           # full-screen PyVis HTML output
│   ├── graph_query_engine.py   # query API + LLM context builder
│   └── extraction_cache.py     # disk cache for LLM results
├── config.py           # all tunable parameters in one place
├── main.py             # full pipeline entry point
├── cli.py              # interactive query interface
└── requirements.txt
```

---

## ⚡ Performance

Processing time depends on document length and hardware.

| Document size | Chunks | First run | Re-run (cached) |
|---|---|---|---|
| 5-page PDF | ~3 | ~30s | ~3s |
| 20-page PDF | ~6 | ~2 min | ~5s |
| 100-page PDF | ~25 | ~8 min | ~5s |

**Speed optimisations built in:**
- Combined concept + relation extraction (1 LLM call per chunk instead of 2)
- Larger chunk size (800 words) reduces total LLM calls
- Disk-based extraction cache — second run on same document is near-instant

---

## 🎛️ Configuration

All parameters live in `config.py`:

```python
OLLAMA_MODEL   = "mistral"   # swap to "llama3" or "phi3"
CHUNK_SIZE     = 800         # words per chunk (larger = faster, less granular)
MIN_EDGE_WEIGHT = 1          # prune edges below this raw count
TOP_K_NODES    = 50          # max nodes in visualization
LLM_WEIGHT     = 3.0         # multiplier for LLM-extracted edges
COOC_WEIGHT    = 1.0         # multiplier for co-occurrence edges
```

---

## 🔍 How edge weighting works

Each edge gets a composite weight combining three signals:

```
weight = (llm_count × 3.0)            ← explicit LLM-extracted relation
       + (cooccurrence_count × 1.0)   ← concepts in same chunk
       + (log(node_frequency) × 0.5)  ← how often nodes appear
```

Then normalised to `[0, 1]`. LLM-extracted edges with strong predicates
(`is a subset of`, `consists of`, `introduced by`, etc.) get an additional
`×1.2` strength bonus.

---

## 🧹 Graph cleaning pipeline

Before visualisation the graph goes through 5 cleaning passes:

1. **Normalise** — lowercase, strip punctuation, expand abbreviations (`ml` → `machine learning`)
2. **Filter** — remove stopwords, single-character nodes, overly long phrases
3. **Deduplicate** — merge plural/singular variants, substring matches, trigram-similar labels
4. **Prune edges** — remove redundant bidirectional co-occurrence back-edges
5. **Prune isolates** — remove nodes with no remaining connections

---

## 📄 PDF support notes

- Works with digitally-created PDFs (Word exports, LaTeX, etc.)
- Scanned/image PDFs require OCR — see `src/document_loader.py` for the `pdf2image` + `tesseract` fallback
- Encrypted PDFs: decrypt before use, or the loader will attempt empty-password decryption

---

## 🗺️ Roadmap

- [ ] Web UI (Flask/FastAPI)
- [ ] Multi-document graph merging
- [ ] Neo4j export
- [ ] Named entity type classification (Person / Organisation / Concept)
- [ ] BERT-based deduplication for better concept merging
- [ ] Graph diff — compare knowledge graphs across document versions

---

## 🤝 Contributing

Pull requests welcome. For major changes please open an issue first.

```bash
# Run the test scripts to verify your changes
python test_loader.py
python test_chunker.py
python test_concept_extractor.py
python test_relation_extractor.py
python test_graph_builder.py
python test_edge_weighter.py
python test_graph_cleaner.py
```

---

## 📜 License

MIT — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

Inspired by [rahulnyk/knowledge_graph](https://github.com/rahulnyk/knowledge_graph).
Built with [Ollama](https://ollama.com), [NetworkX](https://networkx.org), and [PyVis](https://pyvis.readthedocs.io).