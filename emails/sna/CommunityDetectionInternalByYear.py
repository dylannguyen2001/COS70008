from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx

INTERNAL_ONLY     = True         
MIN_EDGE_WEIGHT   = 0.0         
RESOLUTION        = 1.0         
FOLDERS = ["all", "1999", "2000", "2001", "2002"]

for year in FOLDERS:
    IN_DIR = Path("data") / str(year)
    OUT_DIR = Path("data") / str(year)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    INTERNAL_ONLY     = True         
    MIN_EDGE_WEIGHT   = 0.0         
    RESOLUTION        = 1.0         

    node_index = pd.read_parquet(IN_DIR / "NodeIndex_internal.parquet", engine="pyarrow", memory_map=False)
    node_index["internal"] = node_index["internal"]

    p_undir = IN_DIR / "Edges_undirected_agg_internal.parquet"
    p_edges = IN_DIR / "Edges_internal.parquet"

    if p_undir.exists():
        edges = pd.read_parquet(p_undir, engine="pyarrow", memory_map=False)
    else:
        e = pd.read_parquet(p_edges, engine="pyarrow", memory_map=False)
        if "weight" not in e.columns:
            e["weight"] = 1.0
        a = np.where(e["src_person_id"] < e["dst_person_id"], e["src_person_id"], e["dst_person_id"])
        b = np.where(e["src_person_id"] < e["dst_person_id"], e["dst_person_id"], e["src_person_id"])
        edges = (pd.DataFrame({"a": a, "b": b, "w": e["weight"].astype(float)})
                .groupby(["a","b"], as_index=False)["w"].sum()
                .rename(columns={"a":"src_person_id","b":"dst_person_id","w":"weight"}))
        edges["directed"] = False

    if INTERNAL_ONLY:
        keep = set(node_index.loc[node_index.internal, "person_id"])
        edges = edges[edges["src_person_id"].isin(keep) & edges["dst_person_id"].isin(keep)]

    edges["weight"] = edges["weight"].astype(float)
    edges = edges[edges["weight"] > MIN_EDGE_WEIGHT]
    edges = edges[edges["src_person_id"] != edges["dst_person_id"]]  # no self-loops
    edges = edges.dropna(subset=["src_person_id","dst_person_id","weight"])
    edges = edges.reset_index(drop=True)

    G = nx.Graph()
    nodes = set(edges["src_person_id"]).union(set(edges["dst_person_id"]))
    G.add_nodes_from(nodes)
    for r in edges.itertuples(index=False):
        G.add_edge(r.src_person_id, r.dst_person_id, weight=float(r.weight))

    print(f"Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    algo_used = ""
    partition = None  

    try:
        import community as community_louvain  
        partition = community_louvain.best_partition(G, weight="weight", resolution=RESOLUTION)
        algo_used = "louvain-python-louvain"
    except Exception as e:
        from networkx.algorithms.community import greedy_modularity_communities
        comms = list(greedy_modularity_communities(G, weight="weight"))
        partition = {}
        for cid, members in enumerate(comms):
            for n in members:
                partition[n] = cid
        algo_used = "greedy_modularity"

    comm_df = pd.DataFrame({
        "person_id": list(partition.keys()),
        "community_id": list(partition.values())
    })

    sizes = comm_df.groupby("community_id").size().rename("community_size")
    comm_df = comm_df.merge(sizes, on="community_id", how="left")
    comm_df["algo"] = algo_used
    comm_df["resolution"] = RESOLUTION if "louvain" in algo_used else np.nan

    comm_meta = comm_df.merge(node_index[["person_id","email_norm","domain","internal"]],
                            on="person_id", how="left")

    comm_df.to_parquet(OUT_DIR / "Communities_internal.parquet", index=False)
    comm_meta.to_parquet(OUT_DIR / "Communities_with_meta_internal.parquet", index=False)

    summary = pd.DataFrame({
        "n_nodes":        [G.number_of_nodes()],
        "n_edges":        [G.number_of_edges()],
        "n_communities":  [comm_df["community_id"].nunique()],
        "min_size":       [sizes.min()],
        "median_size":    [sizes.median()],
        "mean_size":      [sizes.mean()],
        "max_size":       [sizes.max()],
        "algo":           [algo_used],
        "resolution":     [RESOLUTION if "louvain" in algo_used else np.nan],
        "internal_only":  [INTERNAL_ONLY],
        "min_edge_weight":[MIN_EDGE_WEIGHT],
    })
    summary.to_csv(OUT_DIR / "Community_Summary_internal.csv", index=False)

    print("Wrote:",
        OUT_DIR / "Communities_internal.parquet",
        OUT_DIR / "Communities_with_meta_internal.parquet",
        OUT_DIR / "Community_Summary_internal.csv")
