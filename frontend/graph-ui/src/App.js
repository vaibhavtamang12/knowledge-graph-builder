import React, { useState, useEffect, useCallback } from "react";
import ForceGraph2D from "react-force-graph-2d";

const API = "http://localhost:8000";

const ENTITY_COLORS = {
  PERSON: "#4f8ef7", ORG: "#f7a44f", GPE: "#4fd97b", LOC: "#4fd9d9",
  DATE: "#b44ff7", MONEY: "#f74f4f", PRODUCT: "#f7e24f", EVENT: "#f74fb4",
};
const CLUSTER_COLORS = ["#4f8ef7", "#f7a44f", "#4fd97b", "#f74f4f", "#b44ff7"];

async function apiFetch(path, opts) {
  const res = await fetch(`${API}${path}`, opts);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

function EntityBadge({ label, text }) {
  return (
    <span style={{
      display: "inline-block", background: ENTITY_COLORS[label] || "#94a3b8",
      color: "#fff", borderRadius: 4, padding: "2px 7px", margin: "2px 3px",
      fontSize: 12, fontWeight: 600,
    }}>
      {text}
      <span style={{ opacity: 0.7, marginLeft: 4, fontSize: 10 }}>[{label}]</span>
    </span>
  );
}

function StatusBar({ apiOk, error }) {
  if (apiOk) return (
    <span style={{ fontSize: 12, color: "#4fd97b", marginLeft: 8 }}>● API connected</span>
  );
  return (
    <span style={{ fontSize: 12, color: "#f74f4f", marginLeft: 8 }}>
      ● API unreachable{error ? ` — ${error}` : ""}
    </span>
  );
}

export default function App() {
  const [text, setText] = useState("");
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [entities, setEntities] = useState([]);
  const [relationships, setRelationships] = useState([]);
  const [clusters, setClusters] = useState([]);
  const [activeTab, setActiveTab] = useState("graph");
  const [status, setStatus] = useState("");
  const [apiOk, setApiOk] = useState(null);   // null = unknown, true/false after first fetch
  const [apiError, setApiError] = useState("");

  const fetchAll = useCallback(async () => {
    try {
      // Check health first so we give a clear message
      await apiFetch("/health");
      setApiOk(true);
      setApiError("");
    } catch (e) {
      setApiOk(false);
      setApiError(e.message);
      return; // Don't try the other endpoints if API is down
    }

    // Fetch all data; failures per-endpoint are silently ignored
    const results = await Promise.allSettled([
      apiFetch("/graph"),
      apiFetch("/entities"),
      apiFetch("/relationships"),
      apiFetch("/clusters"),
    ]);

    // Handle graph data (returns {nodes, links} directly)
    if (results[0].status === "fulfilled") {
      setGraphData(results[0].value);
    }

    // Handle entities (returns {entities: [...], count: N})
    if (results[1].status === "fulfilled") {
      const data = results[1].value;
      // Extract the array from the response object
      setEntities(Array.isArray(data) ? data : (data.entities || []));
    }

    // Handle relationships (returns {relationships: [...], count: N})
    if (results[2].status === "fulfilled") {
      const data = results[2].value;
      // Extract the array from the response object
      setRelationships(Array.isArray(data) ? data : (data.relationships || []));
    }

    // Handle clusters (returns {clusters: [...], count: N})
    if (results[3].status === "fulfilled") {
      const data = results[3].value;
      // Extract the array from the response object
      setClusters(Array.isArray(data) ? data : (data.clusters || []));
    }
  }, []);

  const sendText = async () => {
    if (!text.trim()) return;
    setStatus("Sending…");
    try {
      await apiFetch("/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      setText("");
      setStatus("Sent ✓ — graph refreshing in 2 s…");
      setTimeout(() => { fetchAll(); setStatus(""); }, 2000);
    } catch (e) {
      setStatus(`Send failed: ${e.message}`);
    }
  };

  useEffect(() => {
    fetchAll();
    const id = setInterval(fetchAll, 10000); // auto-refresh every 10 s
    return () => clearInterval(id);
  }, [fetchAll]);

  const tabStyle = (tab) => ({
    padding: "6px 16px", marginRight: 6, borderRadius: 6, border: "none",
    cursor: "pointer", fontWeight: activeTab === tab ? 700 : 400,
    background: activeTab === tab ? "#4f8ef7" : "#e5e7eb",
    color: activeTab === tab ? "#fff" : "#333",
  });

  const tabLabels = {
    graph: "📊 Graph",
    ner: `🏷 NER (${entities.length})`,
    relations: `🔗 Relations (${relationships.length})`,
    clusters: `🗂 Clusters (${clusters.length})`,
  };

  return (
    <div style={{ fontFamily: "sans-serif", height: "100vh", display: "flex", flexDirection: "column" }}>

      {/* Header */}
      <div style={{ background: "#1e293b", color: "#fff", padding: "10px 16px", display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
        <strong style={{ fontSize: 18, whiteSpace: "nowrap" }}>🧠 Knowledge Graph</strong>
        <input
          value={text}
          onChange={e => setText(e.target.value)}
          onKeyDown={e => e.key === "Enter" && sendText()}
          placeholder='e.g. "Elon Musk is the CEO of Tesla, based in Austin."'
          style={{ flex: 1, minWidth: 200, padding: "6px 10px", borderRadius: 6, border: "none", fontSize: 14 }}
        />
        <button onClick={sendText} disabled={!apiOk} style={{
          padding: "6px 16px", background: apiOk ? "#4f8ef7" : "#94a3b8",
          color: "#fff", border: "none", borderRadius: 6,
          cursor: apiOk ? "pointer" : "not-allowed", fontWeight: 700,
        }}>Send</button>
        <StatusBar apiOk={apiOk} error={apiError} />
        {status && <span style={{ fontSize: 12, opacity: 0.8 }}>{status}</span>}
      </div>

      {/* Offline banner */}
      {apiOk === false && (
        <div style={{ background: "#fef2f2", borderBottom: "1px solid #fecaca", padding: "10px 16px", color: "#dc2626", fontSize: 13 }}>
          ⚠️ Cannot reach the API at <code>{API}</code>. Make sure the FastAPI service is running (<code>uvicorn main:app --reload --port 8000</code>) and Kafka + Neo4j are up via Docker Compose.
        </div>
      )}

      {/* Tabs */}
      <div style={{ padding: "8px 16px", background: "#f1f5f9", borderBottom: "1px solid #e2e8f0" }}>
        {Object.entries(tabLabels).map(([tab, label]) => (
          <button key={tab} style={tabStyle(tab)} onClick={() => setActiveTab(tab)}>{label}</button>
        ))}
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflow: "hidden", position: "relative" }}>

        {activeTab === "graph" && (
          graphData.nodes && graphData.nodes.length === 0
            ? <EmptyState msg="No graph data yet. Send a sentence above." />
            : <ForceGraph2D
                graphData={graphData}
                nodeLabel="label"
                linkLabel="type"
                nodeColor={node => ENTITY_COLORS[node.entity_type] || "#94a3b8"}
                nodeRelSize={6}
              />
        )}

        {activeTab === "ner" && (
          <ScrollPane>
            <h2 style={{ marginTop: 0 }}>Named Entities</h2>
            {entities.length === 0
              ? <EmptyState msg="No entities yet." />
              : Object.entries(
                  entities.reduce((acc, e) => { (acc[e.label] = acc[e.label] || []).push(e); return acc; }, {})
                ).map(([label, ents]) => (
                  <div key={label} style={{ marginBottom: 14 }}>
                    <strong style={{ color: ENTITY_COLORS[label] || "#333", marginRight: 8 }}>{label}</strong>
                    {ents.map((e, i) => <EntityBadge key={i} label={e.label} text={e.text} />)}
                  </div>
                ))
            }
          </ScrollPane>
        )}

        {activeTab === "relations" && (
          <ScrollPane>
            <h2 style={{ marginTop: 0 }}>Extracted Relations</h2>
            {relationships.length === 0
              ? <EmptyState msg="No relationships yet." />
              : (
                <table style={{ borderCollapse: "collapse", width: "100%" }}>
                  <thead>
                    <tr style={{ background: "#f1f5f9" }}>
                      {["Subject", "Relation", "Object"].map(h => (
                        <th key={h} style={{ padding: "8px 12px", textAlign: "left", borderBottom: "2px solid #e2e8f0" }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {relationships.map((r, i) => (
                      <tr key={i} style={{ borderBottom: "1px solid #f1f5f9" }}>
                        <td style={{ padding: "7px 12px" }}><EntityBadge label="PERSON" text={r.subject} /></td>
                        <td style={{ padding: "7px 12px", fontWeight: 600, color: "#64748b", fontSize: 13 }}>{r.relation}</td>
                        <td style={{ padding: "7px 12px" }}><EntityBadge label="ORG" text={r.object} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )
            }
          </ScrollPane>
        )}

        {activeTab === "clusters" && (
          <ScrollPane>
            <h2 style={{ marginTop: 0 }}>Text Clusters</h2>
            <p style={{ color: "#64748b", fontSize: 13, marginTop: 0 }}>
              Documents are clustered in batches of 10 using TF-IDF + KMeans. Send at least 10 sentences to see results.
            </p>
            {clusters.length === 0
              ? <EmptyState msg="No cluster data yet. Send 10+ sentences." />
              : Object.entries(
                  clusters.reduce((acc, d) => { (acc[d.cluster_id] = acc[d.cluster_id] || []).push(d); return acc; }, {})
                ).map(([cid, docs]) => (
                  <div key={cid} style={{ marginBottom: 20, background: "#f8fafc", borderRadius: 8, padding: 14 }}>
                    <div style={{ marginBottom: 10 }}>
                      <span style={{
                        background: CLUSTER_COLORS[Number(cid) % CLUSTER_COLORS.length],
                        color: "#fff", borderRadius: 12, padding: "2px 10px", fontWeight: 700, fontSize: 13
                      }}>Cluster {cid}</span>
                      <span style={{ marginLeft: 8, color: "#94a3b8", fontSize: 12 }}>{docs.length} document{docs.length !== 1 ? "s" : ""}</span>
                    </div>
                    {docs.map((d, i) => (
                      <div key={i} style={{ fontSize: 13, color: "#334155", padding: "4px 0", borderBottom: "1px solid #e2e8f0" }}>
                        {d.text}
                      </div>
                    ))}
                  </div>
                ))
            }
          </ScrollPane>
        )}

      </div>
    </div>
  );
}

function ScrollPane({ children }) {
  return <div style={{ padding: 20, overflowY: "auto", height: "100%", boxSizing: "border-box" }}>{children}</div>;
}

function EmptyState({ msg }) {
  return <p style={{ color: "#94a3b8", fontStyle: "italic" }}>{msg}</p>;
}