# ===============================================================
#  SCRIPT TO BUILD ENRON GRAPHS EACH YEAR (1999–2002, INTERNAL ONLY)
# ===============================================================
import pandas as pd, json
from pathlib import Path
import numpy as np

folders = ["all", "1999", "2000", "2001", "2002"]

def dominant_label(df, label_col, score_col, neutral_label=None, top_frac=0.1, min_threshold=0.1):
    """
    Return (dominant_label, mean_intensity) based on the top X% highest scores.
    Keeps results in [0,1]. Filters neutrals unless all entries are neutral.
    """
    valid = df.dropna(subset=[label_col, score_col])
    if valid.empty:
        return "None", 0.0

    # Remove neutral rows unless everything is neutral
    if neutral_label:
        non_neutral = valid[valid[label_col] != neutral_label]
        if not non_neutral.empty:
            valid = non_neutral

    # sort descending by score
    valid = valid.sort_values(score_col, ascending=False)
    top_n = max(1, int(len(valid) * top_frac))
    top = valid.head(top_n)

    # determine dominant label among top fraction
    dom_label = top[label_col].value_counts().idxmax()
    dom_intensity = top[score_col].mean()

    if dom_intensity < min_threshold:
        return neutral_label or "None", 0.0
    return str(dom_label), float(dom_intensity)

for folder in folders:
    IN_DIR = Path("data") / folder
    OUT_DIR = IN_DIR
    edge_path  = IN_DIR / "Edges_directed_agg_internal.parquet"
    nodes_path = IN_DIR / "NodeMetrics_internal.parquet"
    comms_path = IN_DIR / "Communities_internal.parquet"
    risks_path = "data/risk_expanded_results/RiskScores_zeroshot_full.parquet"
    sent_path  = "data/sentiment_results/Email_SentimentScores_internal_multi_9902_both.parquet"
    emails_path= "data/Emails_clean_9902.parquet"

    edges  = pd.read_parquet(edge_path)
    nodes  = pd.read_parquet(nodes_path)
    comms  = pd.read_parquet(comms_path)
    risks  = pd.read_parquet(risks_path)
    sent   = pd.read_parquet(sent_path)
    emails = pd.read_parquet(emails_path)

    if folder != "all":
        emails = emails[emails["year"] == int(folder)]

    print(f"Loaded edges={len(edges):,}, nodes={len(nodes):,}, comms={len(comms):,}, emails={len(emails):,}")

    # Merge all e-mail info
    merged = (
        emails
        .merge(risks[["email_id", "risk_label", "final_score"]], on="email_id", how="left")
        .merge(sent[["email_id", "sentiment_label", "sentiment_score", "emotion_label", "emotion_score"]],
               on="email_id", how="left")
    )

    # Aggregate per actor (sentiment + risk info)
    agg_rows = []
    for pid, sub in merged.groupby("person_id"):
        dom_risk, risk_intensity = dominant_label(
            sub, "risk_label", "final_score",
            neutral_label="This email appears routine, compliant, and shows no indication of risk or wrongdoing.", top_frac = 0.025
        )
        dom_sent, sentiment_intensity = dominant_label(
            sub, "sentiment_label", "sentiment_score",
            neutral_label="neutral", top_frac=1.0, min_threshold=0.0
        )
        dom_emot, emotion_intensity = dominant_label(
            sub, "emotion_label", "emotion_score",
            neutral_label="neutral", top_frac=1.0, min_threshold=0.0
        )
        agg_rows.append({
            "person_id": pid,
            "dom_risk_label": dom_risk,
            "risk_intensity": risk_intensity,
            "dom_sentiment_label": dom_sent,
            "sentiment_intensity": sentiment_intensity,
            "dom_emotion_label": dom_emot,
            "emotion_intensity": emotion_intensity,
            "total_emails": len(sub),
            "risk_emails": sub["risk_label"].notna().sum()
        })
    agg_person = pd.DataFrame(agg_rows)

    # Alias map
    xfrom_map = (
        emails.groupby("person_id")["x_from"]
        .agg(lambda x: sorted(set(a for a in x if pd.notna(a))))
        .reset_index()
    )
    xfrom_map["aliases"] = xfrom_map["x_from"].map(lambda lst: lst if lst else ["N/A"])
    xfrom_map = xfrom_map.drop(columns=["x_from"])

    # Merge with node + community info
    nodes = (
        nodes.merge(agg_person, on="person_id", how="left")
             .merge(comms[["person_id", "community_id"]], on="person_id", how="left")
             .merge(xfrom_map, on="person_id", how="left")
    )

    fill_0 = ["risk_intensity","sentiment_intensity","emotion_intensity"]
    for c in fill_0: nodes[c] = nodes[c].fillna(0.0)

    fill_none = ["dom_risk_label","dom_sentiment_label","dom_emotion_label"]
    for c in fill_none: nodes[c] = nodes[c].fillna("None")

    nodes["total_emails"] = nodes["total_emails"].fillna(0).astype(int)
    nodes["risk_emails"]  = nodes["risk_emails"].fillna(0).astype(int)
    nodes["aliases"]      = nodes["aliases"].apply(lambda x: x if isinstance(x,list) and len(x)>0 else ["N/A"])

    # Helpers for JSON
    def safe_date(r, col):
        if col in r and pd.notna(r[col]): return r[col].isoformat()
        return None
    def safe_years(r):
        if "years" in r and isinstance(r["years"],(list,np.ndarray)):
            return [int(y) for y in r["years"]]
        return []

    # Build JSON
    graph = {
        "directed": True,
        "nodes": [
            {
                "id": r.person_id,
                "aliases": r.aliases,
                "community": int(r.community_id) if pd.notna(r.community_id) else None,
                "risk_label": r.dom_risk_label,
                "risk_intensity": float(r.risk_intensity),
                "sentiment_label": r.dom_sentiment_label,
                "sentiment_intensity": float(r.sentiment_intensity),
                "emotion_label": r.dom_emotion_label,
                "emotion_intensity": float(r.emotion_intensity),
                "total_emails": int(r.total_emails),
                "risk_emails": int(r.risk_emails),
                "degree": int(r.degree),
                "in_degree": int(r.in_degree),
                "out_degree": int(r.out_degree),
                "w_degree": float(r.w_degree),
                "pagerank": float(r.pagerank),
                "clustering_coef": float(r.clustering_coef),
                "kcore": int(r.kcore),
                "first_date": safe_date(r,"first_date"),
                "last_date": safe_date(r,"last_date"),
                "years": safe_years(r)
            }
            for _, r in nodes.iterrows()
        ],
        "edges": [
            {
                "source": r.src_person_id,
                "target": r.dst_person_id,
                "weight": int(r.weight),
                "first_date": safe_date(r,"first_date"),
                "last_date": safe_date(r,"last_date"),
                "years": safe_years(r)
            }
            for _, r in edges.iterrows()
        ]
    }

    with open(OUT_DIR / "graph_internal.json","w",encoding="utf-8") as f:
        json.dump(graph,f,indent=2)

    print(f"Graph saved: {folder} → {len(graph['nodes']):,} nodes, {len(graph['edges']):,} edges")