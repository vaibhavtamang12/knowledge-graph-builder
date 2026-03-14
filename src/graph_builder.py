# src/graph_builder.py

import logging
import networkx as nx
import pandas as pd
from collections import defaultdict

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Converts a DataFrame of subject-predicate-object triplets into a
    weighted, directed NetworkX graph.

    Node attributes:
        - frequency   : how many times this concept appeared across all triplets
        - doc_sources : set of documents this concept appeared in
        - in_degree   : number of incoming edges
        - out_degree  : number of outgoing edges

    Edge attributes:
        - weight      : number of times this (subject, object) pair appeared
        - predicates  : list of all predicate labels for this pair
        - chunk_ids   : list of chunk IDs where this edge was found
        - doc_name    : source document
    """

    def __init__(self, min_edge_weight: int = 1):
        """
        Args:
            min_edge_weight: edges with weight below this are pruned
        """
        self.min_edge_weight = min_edge_weight

    # ── Public interface ───────────────────────────────────────────────

    def build(self, df: pd.DataFrame) -> nx.DiGraph:
        """
        Build a directed, weighted graph from a triplets DataFrame.

        Steps:
          1. Normalise node names
          2. Aggregate duplicate edges (increment weight, collect predicates)
          3. Add all nodes with frequency metadata
          4. Add all edges above the weight threshold
          5. Compute and attach degree statistics
        """
        if df.empty:
            logger.warning("Empty DataFrame — returning empty graph")
            return nx.DiGraph()

        df = self._normalise_df(df)

        # ── Aggregate edges ────────────────────────────────────────────
        # For each (subject, object) pair collect all predicates + counts
        edge_data = defaultdict(lambda: {
            "weight":     0,
            "predicates": [],
            "chunk_ids":  [],
            "doc_names":  [],
        })

        node_freq     = defaultdict(int)
        node_docs     = defaultdict(set)

        for _, row in df.iterrows():
            subj = row["subject"]
            obj  = row["object"]
            pred = row["predicate"]

            # Update edge aggregation
            edge_data[(subj, obj)]["weight"]     += 1
            edge_data[(subj, obj)]["predicates"].append(pred)
            edge_data[(subj, obj)]["chunk_ids"].append(int(row["chunk_id"]))
            edge_data[(subj, obj)]["doc_names"].append(row["doc_name"])

            # Update node frequency
            node_freq[subj] += 1
            node_freq[obj]  += 1
            node_docs[subj].add(row["doc_name"])
            node_docs[obj].add(row["doc_name"])

        # ── Build graph ────────────────────────────────────────────────
        G = nx.DiGraph()

        # Add nodes
        for node, freq in node_freq.items():
            G.add_node(
                node,
                frequency   = freq,
                doc_sources = list(node_docs[node]),
                label       = node,
            )

        # Add edges (apply weight threshold)
        skipped = 0
        for (subj, obj), attrs in edge_data.items():
            if attrs["weight"] < self.min_edge_weight:
                skipped += 1
                continue

            # Use the most common predicate as the display label
            main_predicate = max(
                set(attrs["predicates"]),
                key=attrs["predicates"].count
            )

            G.add_edge(
                subj, obj,
                weight     = attrs["weight"],
                label      = main_predicate,
                predicates = list(set(attrs["predicates"])),
                chunk_ids  = list(set(attrs["chunk_ids"])),
                doc_name   = attrs["doc_names"][0],
            )

        # ── Attach degree stats to nodes ───────────────────────────────
        for node in G.nodes():
            G.nodes[node]["in_degree"]  = G.in_degree(node)
            G.nodes[node]["out_degree"] = G.out_degree(node)
            G.nodes[node]["degree"]     = G.degree(node)

        self._log_stats(G, skipped)
        return G

    def get_subgraph(
        self,
        G: nx.DiGraph,
        node: str,
        depth: int = 2
    ) -> nx.DiGraph:
        """
        Extract a subgraph centred on `node` up to `depth` hops away.
        Useful for querying the graph around a specific concept.
        """
        if node not in G:
            logger.warning(f"Node '{node}' not in graph")
            return nx.DiGraph()

        # BFS to collect nodes within depth hops
        neighbours = nx.single_source_shortest_path_length(
            G.to_undirected(), node, cutoff=depth
        )
        return G.subgraph(neighbours.keys()).copy()

    def top_nodes_by_degree(
        self,
        G: nx.DiGraph,
        n: int = 20
    ) -> list[tuple[str, int]]:
        """Return the top-n nodes sorted by total degree (most connected first)."""
        return sorted(
            G.degree(),
            key=lambda x: x[1],
            reverse=True
        )[:n]

    def summary(self, G: nx.DiGraph) -> dict:
        """Return a dict of key graph statistics."""
        if G.number_of_nodes() == 0:
            return {"nodes": 0, "edges": 0}

        return {
            "nodes":               G.number_of_nodes(),
            "edges":               G.number_of_edges(),
            "density":             round(nx.density(G), 4),
            "avg_degree":          round(
                sum(d for _, d in G.degree()) / G.number_of_nodes(), 2
            ),
            "top_5_nodes":         self.top_nodes_by_degree(G, 5),
            "weakly_connected_components": nx.number_weakly_connected_components(G),
        }

    # ── Private helpers ────────────────────────────────────────────────

    def _normalise_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the DataFrame before building:
          - Strip whitespace from subject / predicate / object
          - Lowercase everything
          - Drop rows with empty subject or object
          - Drop self-loops (subject == object)
        """
        df = df.copy()

        for col in ["subject", "predicate", "object"]:
            df[col] = df[col].astype(str).str.strip().str.lower()

        # Fill missing doc_name / chunk_id
        df["doc_name"]  = df["doc_name"].fillna("unknown")
        df["chunk_id"]  = df["chunk_id"].fillna(0).astype(int)

        # Drop bad rows
        df = df[df["subject"].str.len() > 0]
        df = df[df["object"].str.len()  > 0]
        df = df[df["subject"] != df["object"]]   # no self-loops

        return df.reset_index(drop=True)

    def _log_stats(self, G: nx.DiGraph, skipped: int):
        """Log a summary of the built graph."""
        stats = self.summary(G)
        logger.info(f"Graph built successfully:")
        logger.info(f"  Nodes   : {stats['nodes']}")
        logger.info(f"  Edges   : {stats['edges']}")
        logger.info(f"  Density : {stats['density']}")
        logger.info(f"  Skipped : {skipped} edges (below weight threshold)")
        logger.info(f"  Top nodes by degree: {stats['top_5_nodes']}")