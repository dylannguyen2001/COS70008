
import pandas as pd
import json
from pathlib import Path

# === Paths ===
# Edges.parquet path
EDGE_PATH = Path("C:/Users/Hai/Downloads/Edges.parquet")

# NodeMetrics.parquet path
NODE_PATH = Path("C:/Users/Hai/Downloads/NodeMetrics.parquet")
OUT_PATH  = Path("graph.json")

# === Load data ===
edges = pd.read_parquet(EDGE_PATH)
nodes = pd.read_parquet(NODE_PATH)
print(f"Loaded {len(nodes)} nodes and {len(edges)} edges")
edge_cols = ["src_person_id", "dst_person_id", "weight"]
edges = edges[edge_cols].dropna()
edges = edges.rename(columns={
    "src_person_id": "source",
    "dst_person_id": "target"
})
node_cols = ["person_id", "degree", "pagerank", "clustering_coef", "kcore"]
nodes = nodes[node_cols].dropna()
nodes = nodes.rename(columns={"person_id": "id"})
nodes_json = nodes.to_dict(orient="records")
edges_json = edges.to_dict(orient="records")
graph = {"nodes": nodes_json, "edges": edges_json}
with open(OUT_PATH, "w") as f:
    json.dump(graph, f, indent=2)

print(f"✅ Saved graph JSON: {OUT_PATH.resolve()}")
print(f"  Nodes: {len(nodes_json):,} | Edges: {len(edges_json):,}")
