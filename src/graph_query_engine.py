# src/graph_query_engine.py

import logging
import networkx as nx
from difflib import get_close_matches

logger = logging.getLogger(__name__)


class GraphQueryEngine:
    """
    Provides structured query operations over a knowledge graph.

    Supports:
      - Fuzzy node lookup       (handles typos / partial names)
      - Neighbour exploration   (what connects to X?)
      - Shortest path finding   (how does X relate to Y?)
      - Hub discovery           (what are the most important concepts?)
      - Keyword search          (find all nodes matching a term)
      - Subgraph extraction     (neighbourhood around a concept)
      - Natural language context (build graph context for LLM prompts)
    """

    def __init__(self, G: nx.DiGraph):
        self.G = G
        self._node_list = list(G.nodes())

    # ── Node lookup ────────────────────────────────────────────────────

    def find_node(self, query: str, cutoff: float = 0.6) -> list[str]:
        """
        Fuzzy-match a query string against all node names.
        Returns a ranked list of close matches.
        """
        query   = query.strip().lower()
        matches = []

        # Exact match first
        if query in self.G:
            return [query]

        # Substring match
        substr_matches = [
            n for n in self._node_list
            if query in n.lower() or n.lower() in query
        ]
        matches.extend(substr_matches)

        # Fuzzy match via difflib
        fuzzy = get_close_matches(
            query,
            [n.lower() for n in self._node_list],
            n=5,
            cutoff=cutoff,
        )
        for f in fuzzy:
            # Map back to original case
            for node in self._node_list:
                if node.lower() == f and node not in matches:
                    matches.append(node)

        return list(dict.fromkeys(matches))   # deduplicate, preserve order

    # ── Neighbourhood queries ──────────────────────────────────────────

    def get_neighbours(
        self,
        node:      str,
        direction: str = "both",   # "in" | "out" | "both"
        depth:     int = 1,
    ) -> dict:
        """
        Return all neighbours of node up to depth hops away.

        Returns:
            {
              "node":      canonical node name,
              "outgoing":  [(target, predicate, weight), ...],
              "incoming":  [(source, predicate, weight), ...],
            }
        """
        matches = self.find_node(node)
        if not matches:
            return {"error": f"Node '{node}' not found in graph"}

        canonical = matches[0]

        outgoing, incoming = [], []

        if direction in ("out", "both"):
            for _, succ, data in self.G.out_edges(canonical, data=True):
                outgoing.append((
                    succ,
                    data.get("label", "relates to"),
                    round(data.get("weight", 0), 3),
                ))
            outgoing.sort(key=lambda x: x[2], reverse=True)

        if direction in ("in", "both"):
            for pred, _, data in self.G.in_edges(canonical, data=True):
                incoming.append((
                    pred,
                    data.get("label", "relates to"),
                    round(data.get("weight", 0), 3),
                ))
            incoming.sort(key=lambda x: x[2], reverse=True)

        return {
            "node":     canonical,
            "outgoing": outgoing,
            "incoming": incoming,
        }

    def get_subgraph(self, node: str, depth: int = 2) -> nx.DiGraph:
        """Extract subgraph centred on node up to depth hops away."""
        matches = self.find_node(node)
        if not matches:
            return nx.DiGraph()

        canonical   = matches[0]
        neighbours  = nx.single_source_shortest_path_length(
            self.G.to_undirected(), canonical, cutoff=depth
        )
        return self.G.subgraph(neighbours.keys()).copy()

    # ── Path queries ───────────────────────────────────────────────────

    def shortest_path(self, source: str, target: str) -> dict:
        """
        Find the shortest path between two concepts.
        Returns the path as a list of (node, edge_label) tuples.
        """
        src_matches = self.find_node(source)
        tgt_matches = self.find_node(target)

        if not src_matches:
            return {"error": f"Source '{source}' not found"}
        if not tgt_matches:
            return {"error": f"Target '{target}' not found"}

        src = src_matches[0]
        tgt = tgt_matches[0]

        try:
            path = nx.shortest_path(
                self.G.to_undirected(), src, tgt
            )
        except nx.NetworkXNoPath:
            return {"error": f"No path between '{src}' and '{tgt}'"}
        except nx.NodeNotFound as e:
            return {"error": str(e)}

        # Annotate each step with the edge label
        steps = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.G.has_edge(u, v):
                label = self.G[u][v].get("label", "→")
            elif self.G.has_edge(v, u):
                label = self.G[v][u].get("label", "←")
            else:
                label = "—"
            steps.append((u, label, v))

        return {
            "source":      src,
            "target":      tgt,
            "path_length": len(path) - 1,
            "steps":       steps,
            "full_path":   path,
        }

    # ── Discovery queries ──────────────────────────────────────────────

    def get_hubs(self, top_n: int = 10) -> list[dict]:
        """Return the top-n most connected nodes with their stats."""
        hubs = []
        for node, degree in sorted(
            self.G.degree(), key=lambda x: x[1], reverse=True
        )[:top_n]:
            data = self.G.nodes[node]
            hubs.append({
                "node":      node,
                "degree":    degree,
                "in_degree": self.G.in_degree(node),
                "out_degree":self.G.out_degree(node),
                "frequency": data.get("frequency", 0),
                "weight":    round(data.get("weight", 0), 3),
            })
        return hubs

    def search(self, keyword: str) -> list[str]:
        """Find all nodes whose label contains the keyword."""
        keyword = keyword.strip().lower()
        return [
            n for n in self._node_list
            if keyword in n.lower()
        ]

    def get_graph_summary(self) -> dict:
        """Return a high-level summary of the graph's contents."""
        hubs    = self.get_hubs(5)
        hub_str = ", ".join(h["node"] for h in hubs)

        # Collect all unique predicates
        predicates = set()
        for _, _, d in self.G.edges(data=True):
            predicates.add(d.get("label", ""))
        predicates.discard("")
        predicates.discard("co-occurs with")

        return {
            "nodes":       self.G.number_of_nodes(),
            "edges":       self.G.number_of_edges(),
            "density":     round(nx.density(self.G), 4),
            "top_concepts": hub_str,
            "relation_types": sorted(predicates),
        }

    # ── LLM context builder ────────────────────────────────────────────

    def build_context_for_query(
        self,
        query: str,
        max_nodes: int = 8,
        depth:     int = 2,
    ) -> str:
        """
        Given a natural language query, find the most relevant graph
        neighbourhood and format it as a context string for the LLM.
        """
        # Extract keywords from query (simple word-based)
        stopwords = {"what", "how", "why", "is", "are", "the", "a",
                     "an", "does", "do", "tell", "me", "about", "between",
                     "and", "or", "in", "of", "to", "from"}
        keywords = [
            w.lower() for w in query.split()
            if w.lower() not in stopwords and len(w) > 2
        ]

        # Find relevant nodes from keywords
        relevant_nodes = set()
        for kw in keywords:
            matches = self.find_node(kw)
            relevant_nodes.update(matches[:2])   # top 2 matches per keyword

        if not relevant_nodes:
            # Fall back to hub nodes
            relevant_nodes = {h["node"] for h in self.get_hubs(5)}

        # Build neighbourhood context
        context_lines = ["Knowledge graph context:"]
        nodes_added   = set()

        for node in list(relevant_nodes)[:max_nodes]:
            result = self.get_neighbours(node, direction="both", depth=depth)
            if "error" in result:
                continue

            canonical = result["node"]
            if canonical in nodes_added:
                continue
            nodes_added.add(canonical)

            # Outgoing relationships
            for target, label, weight in result["outgoing"][:4]:
                if label != "co-occurs with":
                    context_lines.append(
                        f"  {canonical} --[{label}]--> {target}"
                    )

            # Incoming relationships
            for source, label, weight in result["incoming"][:4]:
                if label != "co-occurs with":
                    context_lines.append(
                        f"  {source} --[{label}]--> {canonical}"
                    )

        if len(context_lines) == 1:
            context_lines.append("  (no specific context found)")

        return "\n".join(context_lines)