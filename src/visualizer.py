# src/visualizer.py

import logging
import re
import networkx as nx
from pyvis.network import Network
from pathlib import Path

logger = logging.getLogger(__name__)


class GraphVisualizer:
    """
    Renders a weighted NetworkX graph as a fully full-screen
    interactive PyVis HTML file.

    Visual encoding:
      - Node size    → degree (more connections = bigger node)
      - Node color   → degree tier (hub / major / normal / minor)
      - Edge width   → composite weight from EdgeWeighter
      - Edge color   → LLM edge (teal) vs co-occurrence (gray)
      - Hover label  → predicate, weight, sources
    """

    NODE_COLORS = {
        "hub":    "#E8593C",
        "major":  "#7F77DD",
        "normal": "#378ADD",
        "minor":  "#888780",
    }

    EDGE_COLORS = {
        "llm_strong": "#1D9E75",
        "llm_weak":   "#5DCAA5",
        "cooc":       "#555555",
    }

    def __init__(
        self,
        min_node_size:  int   = 10,
        max_node_size:  int   = 60,
        min_edge_width: float = 0.5,
        max_edge_width: float = 8.0,
        top_k_nodes:    int   = 50,
    ):
        self.min_node_size  = min_node_size
        self.max_node_size  = max_node_size
        self.min_edge_width = min_edge_width
        self.max_edge_width = max_edge_width
        self.top_k_nodes    = top_k_nodes

    # ── Public interface ───────────────────────────────────────────────

    def render(
        self,
        G:           nx.DiGraph,
        output_path: str = "data/output/knowledge_graph.html",
    ) -> str:
        """
        Render the graph to a full-screen interactive HTML file.
        Returns the output path.
        """
        if G.number_of_nodes() == 0:
            logger.warning("Empty graph — nothing to render")
            return output_path

        G = self._trim_graph(G)

        node_sizes  = self._compute_node_sizes(G)
        node_colors = self._compute_node_colors(G)
        edge_widths = self._compute_edge_widths(G)

        net = self._build_pyvis_network(G, node_sizes, node_colors, edge_widths)
        self._configure_physics(net)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save raw PyVis output then post-process for full-screen
        net.save_graph(output_path)
        self._make_fullscreen(output_path)

        logger.info(
            f"Graph rendered → {output_path}  "
            f"({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)"
        )
        return output_path

    def render_subgraph(
        self,
        G:           nx.DiGraph,
        focus_node:  str,
        depth:       int = 2,
        output_path: str = "data/output/subgraph.html",
    ) -> str:
        """Render a neighbourhood subgraph around focus_node."""
        if focus_node not in G:
            # fuzzy fallback
            matches = [n for n in G.nodes() if focus_node.lower() in n.lower()]
            if not matches:
                logger.warning(f"Node '{focus_node}' not found in graph")
                return output_path
            focus_node = matches[0]

        neighbours = nx.single_source_shortest_path_length(
            G.to_undirected(), focus_node, cutoff=depth
        )
        sub = G.subgraph(neighbours.keys()).copy()
        logger.info(
            f"Subgraph around '{focus_node}' "
            f"(depth={depth}): {sub.number_of_nodes()} nodes"
        )
        return self.render(sub, output_path)

    # ── Full-screen post-processor ─────────────────────────────────────

    def _make_fullscreen(self, html_path: str):
        """
        Post-process the PyVis-generated HTML to make the graph
        fill the entire browser viewport with no scrollbars,
        no margins, and a dark background.

        PyVis hard-codes a fixed-size <div id="mynetwork"> and adds
        body margins — we surgically replace both.
        """
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()

        # ── 1. Inject a <style> block right before </head> ─────────────
        fullscreen_css = """
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }

  html, body {
    width: 100%;
    height: 100%;
    overflow: hidden;
    background: #0d1117;
  }

  /* The main canvas div PyVis generates */
  #mynetwork {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    border: none !important;
  }

  /* Legend panel — sits on top of the graph */
  #legend {
    position: fixed;
    top: 16px;
    right: 16px;
    background: rgba(13, 17, 23, 0.85);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 10px;
    padding: 14px 18px;
    color: #e6edf3;
    font-family: Arial, sans-serif;
    font-size: 13px;
    line-height: 1.8;
    z-index: 9999;
    backdrop-filter: blur(6px);
    min-width: 190px;
  }

  #legend h3 {
    margin-bottom: 8px;
    font-size: 13px;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    border-bottom: 1px solid rgba(255,255,255,0.15);
    padding-bottom: 6px;
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 4px 0;
  }

  .legend-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  .legend-line {
    width: 24px;
    height: 3px;
    border-radius: 2px;
    flex-shrink: 0;
  }

  .legend-section {
    margin-top: 10px;
    font-size: 11px;
    color: rgba(255,255,255,0.45);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  /* Stats badge — bottom left */
  #stats {
    position: fixed;
    bottom: 16px;
    left: 16px;
    background: rgba(13, 17, 23, 0.80);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 8px;
    padding: 8px 14px;
    color: rgba(255,255,255,0.55);
    font-family: Arial, sans-serif;
    font-size: 12px;
    z-index: 9999;
    backdrop-filter: blur(4px);
  }

  #stats span {
    color: rgba(255,255,255,0.85);
    font-weight: 600;
  }

  /* Help hint — bottom right */
  #hint {
    position: fixed;
    bottom: 16px;
    right: 16px;
    color: rgba(255,255,255,0.30);
    font-family: Arial, sans-serif;
    font-size: 11px;
    z-index: 9999;
  }

  /* Hide PyVis default toolbar chrome if present */
  div.vis-network div.vis-navigation { bottom: 50px !important; }
</style>
"""
        html = html.replace("</head>", fullscreen_css + "\n</head>")

        # ── 2. Replace PyVis hard-coded width/height on #mynetwork ─────
        # PyVis writes something like:
        #   <div id="mynetwork" style="width: 100%; height: 750px; ...">
        # We replace the entire inline style to be safe.
        html = re.sub(
            r'(<div\s+id=["\']mynetwork["\'])[^>]*>',
            r'\1 style="width:100vw;height:100vh;border:none;background:#0d1117;">',
            html,
        )

        # ── 3. Inject legend + stats + hint right after <body> ─────────
        legend_html = f"""
<div id="legend">
  <h3>Knowledge Graph</h3>

  <div class="legend-section">Nodes</div>
  <div class="legend-item">
    <div class="legend-dot" style="background:#E8593C;"></div>
    <span>Hub concept</span>
  </div>
  <div class="legend-item">
    <div class="legend-dot" style="background:#7F77DD;"></div>
    <span>Major concept</span>
  </div>
  <div class="legend-item">
    <div class="legend-dot" style="background:#378ADD;"></div>
    <span>Regular concept</span>
  </div>
  <div class="legend-item">
    <div class="legend-dot" style="background:#888780;"></div>
    <span>Peripheral concept</span>
  </div>

  <div class="legend-section">Edges</div>
  <div class="legend-item">
    <div class="legend-line" style="background:#1D9E75;"></div>
    <span>Strong LLM relation</span>
  </div>
  <div class="legend-item">
    <div class="legend-line" style="background:#5DCAA5;height:2px;"></div>
    <span>Weak LLM relation</span>
  </div>
  <div class="legend-item">
    <div class="legend-line" style="background:#555555;height:1px;"></div>
    <span>Co-occurrence</span>
  </div>
</div>

<div id="hint">
  scroll to zoom &nbsp;·&nbsp; drag to pan &nbsp;·&nbsp; hover for details
</div>
"""
        html = html.replace("<body>", "<body>\n" + legend_html)

        # ── 4. Inject stats counter via JS (runs after vis loads) ──────
        stats_js = """
<script>
  // Wait for vis-network to finish stabilising then show node/edge count
  document.addEventListener('DOMContentLoaded', function() {
    setTimeout(function() {
      try {
        var container = document.getElementById('mynetwork');
        var visNet    = container && container.__vis_network__;

        // PyVis exposes the network object as a global called `network`
        var net = (typeof network !== 'undefined') ? network : null;

        var nodeCount = net ? Object.keys(net.body.nodes).length : '?';
        var edgeCount = net ? Object.keys(net.body.edges).length : '?';

        var statsDiv = document.getElementById('stats');
        if (statsDiv) {
          statsDiv.innerHTML =
            '<span>' + nodeCount + '</span> nodes &nbsp;·&nbsp; ' +
            '<span>' + edgeCount + '</span> edges';
        }
      } catch(e) { /* silently ignore */ }
    }, 2000);
  });
</script>
"""
        # Add stats div + script before </body>
        stats_html = '\n<div id="stats">loading graph...</div>\n' + stats_js
        html = html.replace("</body>", stats_html + "\n</body>")

        # ── 5. Write back ──────────────────────────────────────────────
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info("  Full-screen post-processing applied")

    # ── PyVis network builder ──────────────────────────────────────────

    def _build_pyvis_network(
        self,
        G:           nx.DiGraph,
        node_sizes:  dict,
        node_colors: dict,
        edge_widths: dict,
    ) -> Network:
        # Use 100% / 100% here — _make_fullscreen will override to vw/vh
        net = Network(
            height    = "100%",
            width     = "100%",
            bgcolor   = "#0d1117",
            font_color= "#ffffff",
            directed  = True,
            notebook  = False,
        )

        for node in G.nodes():
            data      = G.nodes[node]
            size      = node_sizes.get(node, self.min_node_size)
            color     = node_colors.get(node, self.NODE_COLORS["normal"])
            degree    = data.get("degree", 0)
            frequency = data.get("frequency", 0)
            weight    = data.get("weight", 0)
            docs      = ", ".join(data.get("doc_sources", []))

            tooltip = (
                f"<b>{node}</b><br>"
                f"Degree: {degree}<br>"
                f"Frequency: {frequency}<br>"
                f"Weight: {weight:.3f}<br>"
                f"Sources: {docs}"
            )

            net.add_node(
                node,
                label = node,
                size  = size,
                color = color,
                title = tooltip,
                font  = {
                    "size":  max(10, int(size * 0.4)),
                    "color": "#ffffff",
                },
            )

        for u, v, data in G.edges(data=True):
            width      = edge_widths.get((u, v), self.min_edge_width)
            is_llm     = data.get("is_llm_edge", False)
            strength   = data.get("predicate_strength", "weak")
            label      = data.get("label", "")
            weight     = data.get("weight", 0)
            predicates = data.get("predicates", [label])

            if is_llm and strength == "strong":
                color = self.EDGE_COLORS["llm_strong"]
            elif is_llm:
                color = self.EDGE_COLORS["llm_weak"]
            else:
                color = self.EDGE_COLORS["cooc"]

            tooltip = (
                f"<b>{u} → {v}</b><br>"
                f"Relation: {label}<br>"
                f"All: {', '.join(set(predicates))}<br>"
                f"Weight: {weight:.3f}<br>"
                f"Type: {'LLM' if is_llm else 'co-occurrence'}"
            )

            net.add_edge(
                u, v,
                width  = width,
                color  = color,
                title  = tooltip,
                label  = label if is_llm else "",
                arrows = "to",
                smooth = {"type": "curvedCW", "roundness": 0.2},
            )

        return net

    def _configure_physics(self, net: Network):
        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "barnesHut": {
              "gravitationalConstant": -8000,
              "centralGravity": 0.3,
              "springLength": 200,
              "springConstant": 0.04,
              "damping": 0.09,
              "avoidOverlap": 0.5
            },
            "maxVelocity": 50,
            "minVelocity": 0.75,
            "solver": "barnesHut",
            "stabilization": {
              "enabled": true,
              "iterations": 200,
              "updateInterval": 25
            }
          },
          "edges": {
            "color": { "inherit": false },
            "smooth": { "enabled": true, "type": "curvedCW" },
            "arrows": { "to": { "enabled": true, "scaleFactor": 0.6 } },
            "font":   { "size": 9, "align": "middle", "color": "#aaaaaa" }
          },
          "nodes": {
            "shape": "dot",
            "shadow": false,
            "font":  { "face": "arial" }
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "hideEdgesOnDrag": true,
            "navigationButtons": true,
            "keyboard": true
          }
        }
        """)

    # ── Scaling helpers ────────────────────────────────────────────────

    def _compute_node_sizes(self, G: nx.DiGraph) -> dict:
        degrees = dict(G.degree())
        max_d   = max(degrees.values()) if degrees else 1
        min_d   = min(degrees.values()) if degrees else 0
        span    = max_d - min_d if max_d != min_d else 1
        return {
            node: int(
                self.min_node_size +
                ((deg - min_d) / span) *
                (self.max_node_size - self.min_node_size)
            )
            for node, deg in degrees.items()
        }

    def _compute_node_colors(self, G: nx.DiGraph) -> dict:
        degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)
        n       = len(degrees)
        colors  = {}
        for rank, (node, _) in enumerate(degrees):
            p = rank / n
            if   p <= 0.10: colors[node] = self.NODE_COLORS["hub"]
            elif p <= 0.25: colors[node] = self.NODE_COLORS["major"]
            elif p <= 0.60: colors[node] = self.NODE_COLORS["normal"]
            else:           colors[node] = self.NODE_COLORS["minor"]
        return colors

    def _compute_edge_widths(self, G: nx.DiGraph) -> dict:
        weights = {
            (u, v): d.get("weight", 0)
            for u, v, d in G.edges(data=True)
        }
        max_w = max(weights.values()) if weights else 1.0
        min_w = min(weights.values()) if weights else 0.0
        span  = max_w - min_w if max_w != min_w else 1.0
        return {
            (u, v): round(
                self.min_edge_width +
                ((w - min_w) / span) *
                (self.max_edge_width - self.min_edge_width),
                2
            )
            for (u, v), w in weights.items()
        }

    def _trim_graph(self, G: nx.DiGraph) -> nx.DiGraph:
        if G.number_of_nodes() <= self.top_k_nodes:
            return G
        top_nodes = sorted(
            G.degree(), key=lambda x: x[1], reverse=True
        )[:self.top_k_nodes]
        return G.subgraph([n for n, _ in top_nodes]).copy()