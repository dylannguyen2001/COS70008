# ===============================================================
#  SCRIPT TO BUILD THREAD JSONS (1999–2002, INTERNAL ONLY)
# ===============================================================

import pandas as pd, json, numpy as np, ast

risk = pd.read_parquet("data/risk_expanded_results/RiskScores_zeroshot_threads_full.parquet")
print(f"Loaded {len(risk):,} thread risk rows.")

threads_meta = pd.read_parquet("data/threads/Threads_internal_multi_9902.parquet")
print(f"Loaded {len(threads_meta):,} thread metadata rows.")

sentiments_score = pd.read_parquet("data/sentiment_results/Thread_SentimentScores_internal_multi_9902_both.parquet")
print(f"Loaded {len(sentiments_score):,} sentiment+emotion rows.")

# Keep only multi-email threads (non-singletons, with replies)
threads_meta = threads_meta[
    (~threads_meta["is_singleton"]) & (threads_meta["has_replies"])
].copy()
print(f"Filtered to {len(threads_meta):,} multi-email threads.")
print("Sample participants before merge:")
print(threads_meta["participants"].head(5))

def safe_list(x):
    """Convert a stringified list or set to a Python list of lowercase emails."""
    return list(x)

# Compute thread-level aggregates
risk_agg = (
    risk.groupby("thread_id")
    .agg(
        mean_risk=("model_confidence", "mean"),
        risk_label_zeroshot=("risk_label_zeroshot", lambda x: x.value_counts().idxmax() if x.notna().any() else None),
        risk_entries=("thread_id", "count")
    )
    .reset_index()
)

sent_agg = (
    sentiments_score.groupby("thread_id")
    .agg(
        mean_sentiment_score=("sentiment_score", "mean"),
        dominant_sentiment=("sentiment_label", lambda x: x.value_counts().idxmax() if x.notna().any() else None),
        mean_emotion_score=("emotion_score", "mean"),
        dominant_emotion=("emotion_label", lambda x: x.value_counts().idxmax() if x.notna().any() else None),
    )
    .reset_index()
)

# Merge 
threads = (
    threads_meta[["thread_id", "n_emails", "is_singleton", "has_replies", "participants"]]
    .merge(risk_agg, on="thread_id", how="left")
    .merge(sent_agg, on="thread_id", how="left")
)

# Normalize and fill
threads["participants"] = threads["participants"].apply(safe_list)
threads["mean_risk"] = threads["mean_risk"].fillna(0)
threads["mean_sentiment_score"] = threads["mean_sentiment_score"].fillna(0)
threads["mean_emotion_score"] = threads["mean_emotion_score"].fillna(0)
threads["risk_label_zeroshot"] = threads["risk_label_zeroshot"].fillna("None")
threads["dominant_sentiment"] = threads["dominant_sentiment"].fillna("neutral")
threads["dominant_emotion"] = threads["dominant_emotion"].fillna("neutral")
threads["risk_entries"] = threads["risk_entries"].fillna(0)

print("Sample participants after normalization:")
print(threads["participants"].head(5))


threads_json = {
    "threads": [
        {
            "id": row.thread_id,
            "n_emails": int(row.n_emails),
            "mean_risk": float(row.mean_risk),
            "risk_label_zeroshot": row.risk_label_zeroshot,
            "risk_entries": int(row.risk_entries),
            "mean_sentiment_score": float(row.mean_sentiment_score),
            "dominant_sentiment": row.dominant_sentiment,
            "mean_emotion_score": float(row.mean_emotion_score),
            "dominant_emotion": row.dominant_emotion,
            "is_singleton": bool(row.is_singleton),
            "has_replies": bool(row.has_replies),
            "participants": row.participants
        }
        for _, row in threads.iterrows()
    ]
}

# Save
out_path = "data/threads/Threads_internal_multi_9902.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(threads_json, f, indent=2)

print(f"threads.json created with {len(threads):,} threads (non-singleton only, with participants)")


# Load parquet
thread_text = pd.read_parquet("data/threads/ThreadText_internal_multi_9902.parquet")
print(f"Loaded {len(thread_text):,} rows with columns: {thread_text.columns.tolist()}")

# Convert to list of dicts
records = thread_text.to_dict(orient="records")

# Save as formatted JSON (one file)
out_path = "data/threads/ThreadText_internal_multi_9902.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"Saved {out_path} with {len(records):,} thread records.")
