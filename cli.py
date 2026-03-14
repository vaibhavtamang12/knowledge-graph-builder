# cli.py

import sys
import os
sys.path.insert(0, ".")

import logging
logging.basicConfig(level=logging.WARNING)   # quiet during interactive use

import ollama
from src.document_loader    import DocumentLoader
from src.text_chunker       import TextChunker
from src.concept_extractor  import ConceptExtractor
from src.relation_extractor import RelationExtractor
from src.graph_builder      import GraphBuilder
from src.edge_weighter      import EdgeWeighter
from src.graph_cleaner      import GraphCleaner
from src.visualizer         import GraphVisualizer
from src.graph_query_engine import GraphQueryEngine
from config import (
    DATA_INPUT_DIR, DATA_OUTPUT_DIR, GRAPH_DATA_FILE,
    CHUNK_SIZE, MIN_EDGE_WEIGHT, TOP_K_NODES,
    OLLAMA_MODEL, OLLAMA_TEMPERATURE, GRAPH_OUTPUT_FILE,
)
import pandas as pd


# ── ANSI colours for terminal ──────────────────────────────────────────
class C:
    BOLD   = "\033[1m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    PURPLE = "\033[95m"
    RESET  = "\033[0m"
    DIM    = "\033[2m"


def banner():
    print(f"""
{C.CYAN}{C.BOLD}
╔══════════════════════════════════════════════╗
║       Knowledge Graph Query Interface        ║
║       Powered by Mistral + NetworkX          ║
╚══════════════════════════════════════════════╝
{C.RESET}""")


def help_text():
    print(f"""
{C.BOLD}Available commands:{C.RESET}
  {C.GREEN}neighbours <concept>{C.RESET}       — show all connections for a concept
  {C.GREEN}path <concept> <concept>{C.RESET}   — shortest path between two concepts
  {C.GREEN}hubs{C.RESET}                       — show the most connected concepts
  {C.GREEN}search <keyword>{C.RESET}           — find concepts containing a keyword
  {C.GREEN}summary{C.RESET}                    — show graph statistics
  {C.GREEN}ask <question>{C.RESET}             — ask the LLM a question about the graph
  {C.GREEN}subgraph <concept>{C.RESET}         — render a subgraph HTML around a concept
  {C.GREEN}help{C.RESET}                       — show this help
  {C.GREEN}quit{C.RESET}                       — exit
""")


def build_or_load_graph():
    """Build the graph fresh or load from saved CSV if it exists."""
    os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

    if os.path.exists(GRAPH_DATA_FILE):
        print(f"{C.DIM}Loading saved graph data from {GRAPH_DATA_FILE}...{C.RESET}")
        df = pd.read_csv(GRAPH_DATA_FILE)
    else:
        print(f"{C.YELLOW}No saved graph found. Running full pipeline...{C.RESET}")
        loader     = DocumentLoader(DATA_INPUT_DIR)
        docs       = loader.load_all()
        chunker    = TextChunker(chunk_size=CHUNK_SIZE, overlap_sentences=2)
        all_chunks = chunker.chunk_documents(docs)

        concept_extractor  = ConceptExtractor(OLLAMA_MODEL, OLLAMA_TEMPERATURE)
        all_concepts       = concept_extractor.extract_from_all_documents(all_chunks)

        relation_extractor = RelationExtractor(OLLAMA_MODEL, OLLAMA_TEMPERATURE)
        all_relations      = relation_extractor.extract_from_all_documents(
            all_chunks, all_concepts
        )
        df = relation_extractor.to_dataframe(all_relations)
        df.to_csv(GRAPH_DATA_FILE, index=False)

    builder    = GraphBuilder(min_edge_weight=MIN_EDGE_WEIGHT)
    G          = builder.build(df)

    cleaner    = GraphCleaner()
    G_clean    = cleaner.clean(G)

    weighter   = EdgeWeighter()
    G_weighted = weighter.compute_weights(G_clean)
    G_final    = weighter.filter_weak_edges(G_weighted, min_weight=0.1)

    return G_final


def ask_llm(question: str, engine: GraphQueryEngine) -> str:
    """
    Answer a natural language question using graph context
    injected into the Mistral prompt.
    """
    context = engine.build_context_for_query(question)
    summary = engine.get_graph_summary()

    prompt = f"""You are an expert at analysing knowledge graphs.
Use the graph context below to answer the question concisely and accurately.
Only use information present in the graph context.

Graph overview:
  - {summary['nodes']} concepts, {summary['edges']} relationships
  - Key concepts: {summary['top_concepts']}
  - Relation types: {', '.join(summary['relation_types'][:8])}

{context}

Question: {question}

Answer:"""

    try:
        response = ollama.chat(
            model    = OLLAMA_MODEL,
            options  = {"temperature": 0.3},
            messages = [{"role": "user", "content": prompt}],
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"LLM error: {e}"


def run_cli():
    banner()

    print(f"{C.DIM}Building graph...{C.RESET}")
    G      = build_or_load_graph()
    engine = GraphQueryEngine(G)

    summary = engine.get_graph_summary()
    print(f"{C.GREEN}✓ Graph loaded:{C.RESET} "
          f"{summary['nodes']} nodes, "
          f"{summary['edges']} edges\n")

    help_text()

    while True:
        try:
            raw = input(f"{C.CYAN}kg>{C.RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{C.DIM}Goodbye!{C.RESET}")
            break

        if not raw:
            continue

        parts   = raw.split(maxsplit=1)
        command = parts[0].lower()
        args    = parts[1] if len(parts) > 1 else ""

        # ── neighbours ────────────────────────────────────────────
        if command in ("neighbours", "neighbors", "n"):
            if not args:
                print(f"{C.RED}Usage: neighbours <concept>{C.RESET}")
                continue

            result = engine.get_neighbours(args)
            if "error" in result:
                print(f"{C.RED}{result['error']}{C.RESET}")
                continue

            node = result["node"]
            print(f"\n{C.BOLD}── {node} ──{C.RESET}")

            if result["outgoing"]:
                print(f"\n  {C.GREEN}Outgoing ({len(result['outgoing'])}):{C.RESET}")
                for target, label, weight in result["outgoing"]:
                    bar = "█" * max(1, int(weight * 10))
                    print(f"    → {target:<35} [{label}]  w={weight:.3f} {C.DIM}{bar}{C.RESET}")

            if result["incoming"]:
                print(f"\n  {C.PURPLE}Incoming ({len(result['incoming'])}):{C.RESET}")
                for source, label, weight in result["incoming"]:
                    bar = "█" * max(1, int(weight * 10))
                    print(f"    ← {source:<35} [{label}]  w={weight:.3f} {C.DIM}{bar}{C.RESET}")
            print()

        # ── path ──────────────────────────────────────────────────
        elif command in ("path", "p"):
            if not args or len(args.split(",")) < 2:
                print(f"{C.RED}Usage: path <source>, <target>{C.RESET}")
                continue

            src, tgt = [x.strip() for x in args.split(",", 1)]
            result   = engine.shortest_path(src, tgt)

            if "error" in result:
                print(f"{C.RED}{result['error']}{C.RESET}")
                continue

            print(f"\n{C.BOLD}Path ({result['path_length']} hops):{C.RESET}  "
                  f"{result['source']}  →  {result['target']}")
            for u, label, v in result["steps"]:
                print(f"  {C.GREEN}{u}{C.RESET} "
                      f"--[{C.DIM}{label}{C.RESET}]--> "
                      f"{C.GREEN}{v}{C.RESET}")
            print()

        # ── hubs ──────────────────────────────────────────────────
        elif command in ("hubs", "h"):
            hubs = engine.get_hubs(10)
            print(f"\n{C.BOLD}── Top 10 hub concepts ──{C.RESET}")
            for i, hub in enumerate(hubs, 1):
                bar = "█" * hub["degree"]
                print(
                    f"  {i:>2}. {hub['node']:<40} "
                    f"degree={hub['degree']:<4} "
                    f"freq={hub['frequency']:<4} "
                    f"{C.DIM}{bar[:20]}{C.RESET}"
                )
            print()

        # ── search ────────────────────────────────────────────────
        elif command in ("search", "s"):
            if not args:
                print(f"{C.RED}Usage: search <keyword>{C.RESET}")
                continue
            results = engine.search(args)
            if results:
                print(f"\n{C.BOLD}Matching concepts ({len(results)}):{C.RESET}")
                for r in results:
                    print(f"  • {r}")
            else:
                print(f"{C.YELLOW}No concepts found matching '{args}'{C.RESET}")
            print()

        # ── summary ───────────────────────────────────────────────
        elif command in ("summary", "stats"):
            s = engine.get_graph_summary()
            print(f"""
{C.BOLD}── Graph Summary ──{C.RESET}
  Nodes         : {s['nodes']}
  Edges         : {s['edges']}
  Density       : {s['density']}
  Top concepts  : {s['top_concepts']}
  Relation types: {', '.join(s['relation_types'][:6])}
""")

        # ── ask ───────────────────────────────────────────────────
        elif command in ("ask", "a"):
            if not args:
                print(f"{C.RED}Usage: ask <question>{C.RESET}")
                continue
            print(f"{C.DIM}Thinking...{C.RESET}")
            answer = ask_llm(args, engine)
            print(f"\n{C.BOLD}Answer:{C.RESET}\n  {answer}\n")

        # ── subgraph ──────────────────────────────────────────────
        elif command in ("subgraph", "sub"):
            if not args:
                print(f"{C.RED}Usage: subgraph <concept>{C.RESET}")
                continue
            out  = f"data/output/subgraph_{args.replace(' ', '_')}.html"
            viz  = GraphVisualizer(top_k_nodes=TOP_K_NODES)
            path = viz.render_subgraph(G, args, depth=2, output_path=out)
            print(f"{C.GREEN}✓ Subgraph saved:{C.RESET} {os.path.abspath(path)}")
            os.system(f"xdg-open {path}")

        # ── help ──────────────────────────────────────────────────
        elif command in ("help", "?"):
            help_text()

        # ── quit ──────────────────────────────────────────────────
        elif command in ("quit", "exit", "q"):
            print(f"{C.DIM}Goodbye!{C.RESET}")
            break

        else:
            print(f"{C.RED}Unknown command '{command}'. Type 'help' for options.{C.RESET}")


if __name__ == "__main__":
    run_cli()