# ===============================================================
#  SCRIPT TO BUILD CORE EDGES/NODE METRICS EACH YEAR (1999–2002, INTERNAL ONLY)
# ===============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx
import re, ast

# Base folders
BASE_DIR = Path("data")
FOLDERS = ["all", "1999", "2000", "2001", "2002"]
INPUT_CLEAN = Path("data/Emails_clean_9902.parquet")

for year in FOLDERS:
    OUT_DIR = Path("data") / str(year)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading emails: {year}")
    emails = pd.read_parquet(INPUT_CLEAN, engine="pyarrow", memory_map=False)
    if(year != "all"):
        emails = emails[emails["year"] == int(year)]
    print(f"Loaded {len(emails):,} emails")

    # Helper to normalize recipient lists
    def to_list_clean(x):
        if x is None:
            return []
        if hasattr(x, "as_py"):
            x = x.as_py()
        if hasattr(x, "to_pylist"):
            x = x.to_pylist()
        if isinstance(x, (list, tuple, set)):
            lst = list(x)
        elif isinstance(x, np.ndarray):
            lst = x.tolist()
        elif isinstance(x, str):
            s = x.strip()
            if not s:
                return []
            if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
                try:
                    v = ast.literal_eval(s)
                    lst = list(v) if isinstance(v, (list, tuple, set, np.ndarray)) else [str(v)]
                except Exception:
                    lst = [p.strip() for p in re.split(r"[;,]", s) if p.strip()]
            else:
                lst = [p.strip() for p in re.split(r"[;,]", s) if p.strip()]
        else:
            if pd.isna(x):
                return []
            lst = [str(x).strip()]
        return [str(v).strip().lower() for v in lst if str(v).strip()]

    # Normalize recipient columns
    for col in ["to_norm", "cc_norm", "bcc_norm"]:
        if col not in emails.columns:
            emails[col] = [[]] * len(emails)
        emails[col] = emails[col].apply(to_list_clean)

    # Compute year & filter to internal Enron emails 
    emails["dt_utc"] = pd.to_datetime(emails["dt_utc"], errors="coerce")
    emails["year"] = emails["dt_utc"].dt.year
    emails = emails[emails["person_id"].str.endswith("@enron.com", na=False)].copy()
    print(f"Filtered to {len(emails):,} internal sender emails")

    # Recipient explosion
    emails["recipient_count"] = (
        emails["to_norm"].str.len()
        + emails["cc_norm"].str.len()
        + emails["bcc_norm"].str.len()
    ).astype("int32")

    edges = emails[["email_id", "person_id", "dt_utc", "year", "recipient_count"]].copy()
    edges["recipient"] = emails["to_norm"] + emails["cc_norm"] + emails["bcc_norm"]
    edges = edges.explode("recipient", ignore_index=True)
    edges = edges.rename(columns={"person_id": "src_person_id", "recipient": "dst_person_id"})

    # Drop empty or self-loops
    edges["dst_person_id"] = edges["dst_person_id"].astype(str).str.lower()
    edges = edges.dropna(subset=["dst_person_id"])
    edges = edges[edges["dst_person_id"].str.len() > 0]
    edges = edges[edges["src_person_id"] != edges["dst_person_id"]]

    # Filter destination to internal Enron as well

    edges = edges[edges["dst_person_id"].str.endswith("@enron.com", na=False)]
    print(f"Remaining internal edges: {len(edges):,}")

    # Weight edges
    edges["weight_unit"] = 1.0
    edges["weight_mass"] = 1.0 / (1.0 + edges["recipient_count"].fillna(0).astype(float))

    # Temporal aggregation per directed pair 
    edges_agg = (
        edges.groupby(["src_person_id", "dst_person_id"])
            .agg(
                weight=("weight_unit", "sum"),
                first_date=("dt_utc", "min"),
                last_date=("dt_utc", "max"),
                years=("year", lambda x: sorted(x.dropna().unique().tolist()))
            )
            .reset_index()
    )
    edges_agg["directed"] = True


    # Reciprocal (active two-way actors only) -- Ensuring A <-> B 
    print(f"Before reciprocity filter: {len(edges_agg):,} directed pairs")

    pairs = set(zip(edges_agg["src_person_id"], edges_agg["dst_person_id"]))
    rev_pairs = set(zip(edges_agg["dst_person_id"], edges_agg["src_person_id"]))
    mutual_pairs = pairs & rev_pairs

    edges_agg = edges_agg[edges_agg.apply(
        lambda r: (r["src_person_id"], r["dst_person_id"]) in mutual_pairs, axis=1
    )].copy()

    print(f"After reciprocity filter: {len(edges_agg):,} directed pairs (mutual only)")

    active_nodes = set(edges_agg["src_person_id"]) | set(edges_agg["dst_person_id"])
    edges_agg = edges_agg[
        edges_agg["src_person_id"].isin(active_nodes)
        & edges_agg["dst_person_id"].isin(active_nodes)
    ]
    print(f"After stricter filter: {len(edges_agg):,} directed pairs (mutual only)")
    print(f"Filtering raw edges to reciprocal pairs...")
    edges = edges[
        edges.apply(lambda r: (r["src_person_id"], r["dst_person_id"]) in mutual_pairs, axis=1)
    ].copy()
    print(f"Remaining raw reciprocal edges: {len(edges):,}")



    # Save detailed edge list (each message)
    edges_out = edges[["email_id", "src_person_id", "dst_person_id", "dt_utc", "year"]].copy()
    edges_out["weight"] = edges["weight_unit"].astype("float32")
    edges_out["directed"] = True
    edges_out.to_parquet(OUT_DIR / "Edges_internal.parquet", index=False)

    print(f"Saved detailed edges: {len(edges_out):,}")
    print(f"Aggregated edges: {len(edges_agg):,}")

    # Build undirected aggregation (for clustering) 
    u = edges[["src_person_id", "dst_person_id", "weight_mass"]].copy()
    u["a"] = np.where(u["src_person_id"] < u["dst_person_id"], u["src_person_id"], u["dst_person_id"])
    u["b"] = np.where(u["src_person_id"] < u["dst_person_id"], u["dst_person_id"], u["src_person_id"])
    edges_undir = (
        u.groupby(["a", "b"])
        .agg(weight=("weight_mass", "sum"))
        .reset_index()
        .rename(columns={"a": "src_person_id", "b": "dst_person_id"})
    )
    edges_undir["directed"] = False

    edges_agg.to_parquet(OUT_DIR / "Edges_directed_agg_internal.parquet", index=False)
    edges_undir.to_parquet(OUT_DIR / "Edges_undirected_agg_internal.parquet", index=False)

    print(f"Unique internal nodes: {len(pd.unique(pd.concat([edges['src_person_id'], edges['dst_person_id']])))})")
    print(f"Directed edges: {len(edges_agg)} | Undirected edges: {len(edges_undir)}")
    print("Year range:", emails['year'].min(), "-", emails['year'].max())

    # Build Node Index (re-added section) 
    print("Building Node Index...")
    senders = emails[["person_id"]].drop_duplicates().rename(columns={"person_id": "email_norm"})
    senders["domain"] = senders["email_norm"].str.extract(r'@(.+)$')[0].str.lower()
    senders["internal"] = senders["domain"].eq("enron.com")
    senders["person_id"] = senders["email_norm"]

    recips = edges[["dst_person_id"]].drop_duplicates().rename(columns={"dst_person_id": "person_id"})
    recips["email_norm"] = recips["person_id"]
    recips["domain"] = recips["email_norm"].str.extract(r'@(.+)$')[0].str.lower()
    recips["internal"] = recips["domain"].eq("enron.com")

    node_index = pd.concat([senders[["person_id", "email_norm", "internal", "domain"]],
                            recips[["person_id", "email_norm", "internal", "domain"]]],
                        ignore_index=True).drop_duplicates("person_id")

    node_index.to_parquet(OUT_DIR / "NodeIndex_internal.parquet", index=False)
    print(f"Saved NodeIndex_internal.parquet with {len(node_index):,} unique nodes")

    # Directed graph
    Gd = nx.DiGraph()
    Gd.add_nodes_from(node_index["person_id"])
    for r in edges_agg.itertuples(index=False):
        Gd.add_edge(r.src_person_id, r.dst_person_id, weight=float(r.weight))

    in_deg = dict(Gd.in_degree())
    out_deg = dict(Gd.out_degree())
    deg = {n: in_deg.get(n, 0) + out_deg.get(n, 0) for n in Gd.nodes()}
    pagerank = nx.pagerank(Gd, alpha=0.85, weight="weight") if Gd.number_of_edges() else {n: 0.0 for n in Gd.nodes()}

    # Undirected graph
    Gu = nx.Graph()
    Gu.add_nodes_from(node_index["person_id"])
    for r in edges_undir.itertuples(index=False):
        Gu.add_edge(r.src_person_id, r.dst_person_id, weight=float(r.weight))

    clust = nx.clustering(Gu) if Gu.number_of_edges() else {n: 0.0 for n in Gu.nodes()}
    try:
        core = nx.core_number(Gu) if Gu.number_of_edges() else {n: 0 for n in Gu.nodes()}
    except nx.NetworkXError:
        core = {n: 0 for n in Gu.nodes()}

    # Weighted degree
    wdeg = {n: 0.0 for n in node_index["person_id"]}
    for r in edges_undir.itertuples(index=False):
        wdeg[r.src_person_id] += float(r.weight)
        wdeg[r.dst_person_id] += float(r.weight)

    # Node metrics dataframe 
    nm = pd.DataFrame({"person_id": node_index["person_id"]})
    nm["degree"] = nm["person_id"].map(deg).fillna(0).astype("int32")
    nm["in_degree"] = nm["person_id"].map(in_deg).fillna(0).astype("int32")
    nm["out_degree"] = nm["person_id"].map(out_deg).fillna(0).astype("int32")
    nm["w_degree"] = nm["person_id"].map(wdeg).fillna(0).astype("float32")
    nm["pagerank"] = nm["person_id"].map(pagerank).fillna(0).astype("float32")
    nm["clustering_coef"] = nm["person_id"].map(clust).fillna(0).astype("float32")
    nm["kcore"] = nm["person_id"].map(core).fillna(0).astype("int32")

    # Add temporal coverage per node
    temporal = (
        edges.groupby("src_person_id")
            .agg(
                first_date=("dt_utc", "min"),
                last_date=("dt_utc", "max"),
                years=("year", lambda x: sorted(x.dropna().unique().tolist()))
            )
            .reset_index()
    )
    nm = nm.merge(temporal, left_on="person_id", right_on="src_person_id", how="left").drop(columns=["src_person_id"])

    # Save
    nm.to_parquet(OUT_DIR / "NodeMetrics_internal.parquet", index=False)
    print("Done! Saved internal-only graph metrics and edges.")


for yr in FOLDERS:
    if yr != "all":
        yr = int(yr)
    ni = pd.read_parquet(f"data/{yr}/NodeIndex_internal.parquet")
    nm = pd.read_parquet(f"data/{yr}/NodeMetrics_internal.parquet")
    print(f"{yr}: nodes={len(ni)}, metrics={len(nm)} → match: {len(ni)==len(nm)}")
