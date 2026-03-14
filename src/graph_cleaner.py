# src/graph_cleaner.py

import re
import logging
import networkx as nx
from collections import defaultdict

logger = logging.getLogger(__name__)


class GraphCleaner:
    """
    Improves knowledge graph quality through five cleaning passes:

    Pass 1 — Normalize     : lowercase, strip punctuation, collapse whitespace
    Pass 2 — Filter        : remove stopwords, fragments, and junk nodes
    Pass 3 — Deduplicate   : merge nodes that refer to the same concept
                             (abbreviations, plurals, substrings)
    Pass 4 — Prune edges   : remove redundant bidirectional co-occurrence pairs
    Pass 5 — Prune isolates: remove nodes with no remaining edges
    """

    # Concepts that should never appear as graph nodes
    STOPWORDS = {
        "the", "a", "an", "this", "that", "these", "those",
        "it", "its", "they", "them", "their", "we", "our",
        "is", "are", "was", "were", "be", "been", "being",
        "has", "have", "had", "do", "does", "did",
        "for", "of", "in", "on", "at", "by", "with", "from",
        "to", "and", "or", "but", "not", "no", "so",
        "as", "if", "then", "than", "also", "each", "all",
        "text", "data", "information", "system", "approach",
        "method", "way", "type", "kind", "form", "example",
        "use", "used", "using", "based", "new", "many",
        "large", "small", "different", "specific", "various",
    }

    # Known abbreviation → full form mappings
    # Add domain-specific ones here as needed
    ABBREV_MAP = {
        "ml":    "machine learning",
        "dl":    "deep learning",
        "ai":    "artificial intelligence",
        "nlp":   "natural language processing",
        "llm":   "large language model",
        "llms":  "large language model",
        "nn":    "neural network",
        "nns":   "neural networks",
        "rl":    "reinforcement learning",
        "cv":    "computer vision",
        "gd":    "gradient descent",
        "bp":    "backpropagation",
        "tl":    "transfer learning",
        "kg":    "knowledge graph",
        "kgs":   "knowledge graph",
    }

    def __init__(
        self,
        min_concept_length:  int   = 2,
        min_concept_words:   int   = 1,
        max_concept_words:   int   = 5,
        min_node_degree:     int   = 1,
        similarity_threshold: float = 0.85,
    ):
        """
        Args:
            min_concept_length:   minimum character length for a valid node
            min_concept_words:    minimum word count for a valid node
            max_concept_words:    maximum word count — prunes overly long phrases
            min_node_degree:      nodes below this degree are removed
            similarity_threshold: string similarity cutoff for deduplication
        """
        self.min_concept_length   = min_concept_length
        self.min_concept_words    = min_concept_words
        self.max_concept_words    = max_concept_words
        self.min_node_degree      = min_node_degree
        self.similarity_threshold = similarity_threshold

    # ── Public interface ───────────────────────────────────────────────

    def clean(self, G: nx.DiGraph) -> nx.DiGraph:
        """
        Run all five cleaning passes in sequence.
        Returns a cleaned copy of the graph.
        """
        logger.info(
            f"Cleaning graph: "
            f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
        )

        G = self._pass1_normalize(G)
        G = self._pass2_filter_nodes(G)
        G = self._pass3_deduplicate(G)
        G = self._pass4_prune_edges(G)
        G = self._pass5_prune_isolates(G)

        logger.info(
            f"Cleaned graph:  "
            f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
        )
        return G

    def get_node_report(self, G: nx.DiGraph) -> dict:
        """
        Return a quality report: degree distribution, top nodes,
        isolated count, etc.
        """
        degrees    = dict(G.degree())
        isolates   = [n for n, d in degrees.items() if d == 0]
        top_nodes  = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "total_nodes":     G.number_of_nodes(),
            "total_edges":     G.number_of_edges(),
            "isolated_nodes":  len(isolates),
            "top_nodes":       top_nodes,
            "avg_degree":      round(
                sum(degrees.values()) / max(len(degrees), 1), 2
            ),
            "density":         round(nx.density(G), 4),
        }

    # ── Pass 1: Normalize ──────────────────────────────────────────────

    def _pass1_normalize(self, G: nx.DiGraph) -> nx.DiGraph:
        """
        Normalize every node label:
          - Strip leading/trailing whitespace and punctuation
          - Collapse internal whitespace
          - Lowercase
          - Expand known abbreviations
        Returns a new graph with relabelled nodes.
        """
        mapping = {}
        for node in G.nodes():
            normalized = self._normalize_concept(str(node))
            # Expand abbreviations
            normalized = self.ABBREV_MAP.get(normalized, normalized)
            if normalized != node:
                mapping[node] = normalized

        if mapping:
            G = nx.relabel_nodes(G, mapping, copy=True)
            # Merge any nodes that became identical after normalization
            G = self._merge_duplicate_nodes(G)
            logger.info(f"  Pass 1 normalized {len(mapping)} node labels")

        return G

    # ── Pass 2: Filter junk nodes ──────────────────────────────────────

    def _pass2_filter_nodes(self, G: nx.DiGraph) -> nx.DiGraph:
        """Remove stopwords, single characters, and overly long phrases."""
        G         = G.copy()
        to_remove = []

        for node in G.nodes():
            reason = self._should_remove_node(node)
            if reason:
                to_remove.append(node)
                logger.debug(f"  Removing '{node}': {reason}")

        G.remove_nodes_from(to_remove)
        logger.info(f"  Pass 2 removed {len(to_remove)} junk nodes")
        return G

    def _should_remove_node(self, node: str) -> str | None:
        """Return reason string if node should be removed, else None."""
        words = node.strip().split()

        if len(node) < self.min_concept_length:
            return "too short"

        if len(words) < self.min_concept_words:
            return "too few words"

        if len(words) > self.max_concept_words:
            return "too many words"

        if node.lower() in self.STOPWORDS:
            return "stopword"

        if all(len(w) <= 1 for w in words):
            return "single characters"

        # Mostly numbers / punctuation
        if re.match(r'^[\d\s\.\,\-\_\/\(\)]+$', node):
            return "numeric/punctuation only"

        return None

    # ── Pass 3: Deduplicate similar concepts ───────────────────────────

    def _pass3_deduplicate(self, G: nx.DiGraph) -> nx.DiGraph:
        """
        Merge nodes that refer to the same concept using three strategies:

        Strategy A — Exact substring: if node A is a substring of node B
                     and they share significant overlap, keep the longer form.
                     e.g. "neural net" → "neural networks"

        Strategy B — Plural/singular: strip trailing 's' and compare.
                     e.g. "neural networks" ↔ "neural network"

        Strategy C — Edit distance similarity: if two labels are very similar
                     (Jaccard on character trigrams ≥ threshold), merge them.
        """
        nodes   = list(G.nodes())
        merges  = {}    # node_to_remove → canonical_node

        for i, n1 in enumerate(nodes):
            if n1 in merges:
                continue
            for n2 in nodes[i + 1:]:
                if n2 in merges:
                    continue
                if n1 == n2:
                    continue

                canonical = self._find_merge_target(n1, n2)
                if canonical:
                    removed = n2 if canonical == n1 else n1
                    merges[removed] = canonical

        if merges:
            G = self._apply_merges(G, merges)
            logger.info(f"  Pass 3 merged {len(merges)} duplicate concepts")

        return G

    def _find_merge_target(self, n1: str, n2: str) -> str | None:
        """
        Return the canonical node name if n1 and n2 should be merged,
        or None if they are distinct concepts.
        """
        s1, s2 = n1.strip(), n2.strip()

        # Strategy B — plural/singular
        if s1.rstrip('s') == s2.rstrip('s') and abs(len(s1) - len(s2)) <= 2:
            # Keep the longer (more descriptive) form
            return s1 if len(s1) >= len(s2) else s2

        # Strategy A — substring containment
        if len(s1) > 4 and len(s2) > 4:
            if s1 in s2:
                return s2   # keep the longer form
            if s2 in s1:
                return s1

        # Strategy C — trigram similarity
        if self._trigram_similarity(s1, s2) >= self.similarity_threshold:
            # Keep the more frequent / longer form
            return s1 if len(s1) >= len(s2) else s2

        return None

    def _trigram_similarity(self, s1: str, s2: str) -> float:
        """Jaccard similarity over character 3-grams."""
        def trigrams(s):
            s = f"  {s}  "   # pad so edge characters get trigrams
            return set(s[i:i+3] for i in range(len(s) - 2))

        t1, t2 = trigrams(s1), trigrams(s2)
        if not t1 or not t2:
            return 0.0
        intersection = len(t1 & t2)
        union        = len(t1 | t2)
        return intersection / union if union else 0.0

    # ── Pass 4: Prune redundant edges ─────────────────────────────────

    def _pass4_prune_edges(self, G: nx.DiGraph) -> nx.DiGraph:
        """
        Remove redundant co-occurrence back-edges.

        When both A→B and B→A exist and BOTH are co-occurrence only,
        keep only A→B (the one with the higher weight) and drop B→A.
        This halves the visual noise from co-occurrence flooding.
        """
        G         = G.copy()
        to_remove = []
        seen      = set()

        for u, v, data in G.edges(data=True):
            pair = tuple(sorted([u, v]))
            if pair in seen:
                continue
            seen.add(pair)

            # Check if the reverse edge exists
            if not G.has_edge(v, u):
                continue

            fwd_data = data
            rev_data = G[v][u]

            fwd_is_cooc = not fwd_data.get("is_llm_edge", False)
            rev_is_cooc = not rev_data.get("is_llm_edge", False)

            # Only prune if BOTH directions are co-occurrence
            if fwd_is_cooc and rev_is_cooc:
                fwd_w = fwd_data.get("weight", 0)
                rev_w = rev_data.get("weight", 0)
                # Drop the weaker direction
                if fwd_w >= rev_w:
                    to_remove.append((v, u))
                else:
                    to_remove.append((u, v))

        G.remove_edges_from(to_remove)
        logger.info(f"  Pass 4 pruned {len(to_remove)} redundant back-edges")
        return G

    # ── Pass 5: Remove isolates ────────────────────────────────────────

    def _pass5_prune_isolates(self, G: nx.DiGraph) -> nx.DiGraph:
        """Remove nodes with degree below min_node_degree."""
        G         = G.copy()
        to_remove = [
            n for n in G.nodes()
            if G.degree(n) < self.min_node_degree
        ]
        G.remove_nodes_from(to_remove)
        logger.info(f"  Pass 5 removed {len(to_remove)} isolated/low-degree nodes")
        return G

    # ── Merge helpers ──────────────────────────────────────────────────

    def _apply_merges(
        self,
        G:      nx.DiGraph,
        merges: dict[str, str],
    ) -> nx.DiGraph:
        """
        Redirect all edges from merged nodes to their canonical target,
        then remove the merged nodes.
        """
        G = G.copy()

        for removed, canonical in merges.items():
            if removed not in G or canonical not in G:
                continue

            # Carry over incoming edges
            for pred in list(G.predecessors(removed)):
                if pred == canonical:
                    continue
                if not G.has_edge(pred, canonical):
                    G.add_edge(pred, canonical, **G[pred][removed])
                else:
                    # Merge weights
                    G[pred][canonical]["weight"] = (
                        G[pred][canonical].get("weight", 0) +
                        G[pred][removed].get("weight", 0)
                    )

            # Carry over outgoing edges
            for succ in list(G.successors(removed)):
                if succ == canonical:
                    continue
                if not G.has_edge(canonical, succ):
                    G.add_edge(canonical, succ, **G[removed][succ])
                else:
                    G[canonical][succ]["weight"] = (
                        G[canonical][succ].get("weight", 0) +
                        G[removed][succ].get("weight", 0)
                    )

            # Merge node frequency
            if canonical in G.nodes() and removed in G.nodes():
                G.nodes[canonical]["frequency"] = (
                    G.nodes[canonical].get("frequency", 0) +
                    G.nodes[removed].get("frequency", 0)
                )

        G.remove_nodes_from(
            [n for n in merges.keys() if n in G]
        )
        return G

    def _merge_duplicate_nodes(self, G: nx.DiGraph) -> nx.DiGraph:
        """
        After relabelling, two nodes may share the same name.
        Merge them by summing their edge weights.
        """
        return self._apply_merges(
            G,
            {n: n for n in G.nodes()},   # identity — triggers dedup logic
        )

    # ── Utility ────────────────────────────────────────────────────────

    def _normalize_concept(self, concept: str) -> str:
        """Strip, lowercase, collapse spaces, remove surrounding punctuation."""
        concept = concept.lower().strip()
        concept = re.sub(r'["\'\`\*\_\#\@\!\?]+', '', concept)
        concept = re.sub(r'\s+', ' ', concept)
        concept = concept.strip('.,;:-()')
        return concept