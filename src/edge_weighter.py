# src/edge_weighter.py

import logging
import math
import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


class EdgeWeighter:
    """
    Computes a rich composite weight for every edge in the graph.

    Weight formula (before normalisation):
        w = (llm_count   × llm_weight)
          + (cooc_count  × cooc_weight)
          + (log(node_freq + 1) × freq_weight)

    After computing raw weights, everything is normalised to [0, 1]
    so the visualiser can map weight → thickness / colour uniformly.

    The predicate type also influences weight — some relation types
    are semantically stronger than others (e.g. 'is a type of' is
    more informative than 'relates to').
    """

    # Predicates considered "strong" LLM relations (get an extra bump)
    STRONG_PREDICATES = {
        "is a type of", "is a subset of", "is a subtype of",
        "consists of",  "is part of",     "is used in",
        "enables",      "introduced by",  "was created by",
        "is defined as","is an example of","uses",
        "is based on",  "is built on",
    }

    def __init__(
        self,
        llm_weight:   float = 3.0,
        cooc_weight:  float = 1.0,
        freq_weight:  float = 0.5,
        cooc_predicate: str = "co-occurs with",
    ):
        """
        Args:
            llm_weight:     multiplier for LLM-extracted relations
            cooc_weight:    multiplier for co-occurrence edges
            freq_weight:    multiplier for the log-frequency bonus
            cooc_predicate: the predicate label used for co-occurrence edges
        """
        self.llm_weight     = llm_weight
        self.cooc_weight    = cooc_weight
        self.freq_weight    = freq_weight
        self.cooc_predicate = cooc_predicate

    # ── Public interface ───────────────────────────────────────────────

    def compute_weights(self, G: nx.DiGraph) -> nx.DiGraph:
        """
        Compute and attach composite weights to every edge in G.
        Returns the same graph with updated edge attributes:
            - raw_weight       : unnormalised composite score
            - weight           : normalised score in [0, 1]
            - is_llm_edge      : True if extracted by LLM (not co-occurrence)
            - predicate_strength: 'strong' or 'weak'
        """
        if G.number_of_edges() == 0:
            logger.warning("Graph has no edges — nothing to weight")
            return G

        G = G.copy()

        # ── Pass 1: compute raw composite weight per edge ──────────────
        raw_weights = {}

        for u, v, data in G.edges(data=True):
            predicates  = data.get("predicates", [data.get("label", "")])
            edge_count  = data.get("weight", 1)
            is_cooc     = self.cooc_predicate in predicates

            # Split count into LLM vs co-occurrence contributions
            if is_cooc:
                cooc_count = edge_count
                llm_count  = 0
            else:
                llm_count  = edge_count
                cooc_count = 0

            # Predicate strength bonus
            strength_bonus = 1.2 if any(
                p in self.STRONG_PREDICATES for p in predicates
            ) else 1.0

            # Node frequency bonus (average of both endpoints)
            freq_u = G.nodes[u].get("frequency", 1)
            freq_v = G.nodes[v].get("frequency", 1)
            freq_bonus = math.log(((freq_u + freq_v) / 2) + 1)

            raw = (
                (llm_count  * self.llm_weight  * strength_bonus) +
                (cooc_count * self.cooc_weight) +
                (freq_bonus * self.freq_weight)
            )

            raw_weights[(u, v)] = raw

            # Attach interim attributes
            G[u][v]["is_llm_edge"]       = not is_cooc
            G[u][v]["predicate_strength"] = (
                "strong" if any(p in self.STRONG_PREDICATES for p in predicates)
                else "weak"
            )
            G[u][v]["raw_weight"] = raw

        # ── Pass 2: normalise to [0, 1] ────────────────────────────────
        max_w = max(raw_weights.values()) if raw_weights else 1.0
        min_w = min(raw_weights.values()) if raw_weights else 0.0
        span  = max_w - min_w if max_w != min_w else 1.0

        for (u, v), raw in raw_weights.items():
            normalised      = (raw - min_w) / span
            G[u][v]["weight"] = round(normalised, 4)

        # ── Pass 3: attach normalised weight to nodes too ──────────────
        # A node's weight = average of its incident edge weights
        for node in G.nodes():
            incident = list(G.in_edges(node, data=True)) + \
                       list(G.out_edges(node, data=True))
            if incident:
                avg_w = sum(d["weight"] for _, _, d in incident) / len(incident)
                G.nodes[node]["weight"] = round(avg_w, 4)
            else:
                G.nodes[node]["weight"] = 0.0

        self._log_weight_stats(G)
        return G

    def filter_weak_edges(
        self,
        G: nx.DiGraph,
        min_weight: float = 0.1,
        keep_llm_edges: bool = True,
    ) -> nx.DiGraph:
        """
        Remove edges below min_weight from the graph.

        Args:
            min_weight:     threshold — edges below this are removed
            keep_llm_edges: if True, never remove LLM-extracted edges
                            regardless of weight (they're always meaningful)
        """
        G = G.copy()
        to_remove = []

        for u, v, data in G.edges(data=True):
            w          = data.get("weight", 0)
            is_llm     = data.get("is_llm_edge", False)

            if w < min_weight:
                if keep_llm_edges and is_llm:
                    continue   # always keep explicit LLM relations
                to_remove.append((u, v))

        G.remove_edges_from(to_remove)

        # Remove isolated nodes (no edges left)
        isolated = list(nx.isolates(G))
        G.remove_nodes_from(isolated)

        logger.info(
            f"Filtered graph: removed {len(to_remove)} edges, "
            f"{len(isolated)} isolated nodes. "
            f"Remaining: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges"
        )
        return G

    def get_weight_distribution(self, G: nx.DiGraph) -> dict:
        """
        Return a breakdown of edge weights by bucket for inspection.
        """
        weights = [d["weight"] for _, _, d in G.edges(data=True)]
        if not weights:
            return {}

        buckets = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0,
                   "0.6-0.8": 0, "0.8-1.0": 0}

        for w in weights:
            if   w < 0.2: buckets["0.0-0.2"] += 1
            elif w < 0.4: buckets["0.2-0.4"] += 1
            elif w < 0.6: buckets["0.4-0.6"] += 1
            elif w < 0.8: buckets["0.6-0.8"] += 1
            else:         buckets["0.8-1.0"] += 1

        return {
            "min":         round(min(weights), 4),
            "max":         round(max(weights), 4),
            "mean":        round(sum(weights) / len(weights), 4),
            "distribution": buckets,
        }

    # ── Private helpers ────────────────────────────────────────────────

    def _log_weight_stats(self, G: nx.DiGraph):
        dist = self.get_weight_distribution(G)
        llm_edges  = sum(
            1 for _, _, d in G.edges(data=True) if d.get("is_llm_edge")
        )
        cooc_edges = G.number_of_edges() - llm_edges

        logger.info(f"Edge weights computed:")
        logger.info(f"  LLM edges        : {llm_edges}")
        logger.info(f"  Co-occurrence    : {cooc_edges}")
        logger.info(f"  Weight min/max   : {dist.get('min')} / {dist.get('max')}")
        logger.info(f"  Weight mean      : {dist.get('mean')}")
        logger.info(f"  Distribution     : {dist.get('distribution')}")