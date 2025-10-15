# -*- coding: utf-8 -*-
"""
Zero-Shot Risk Classification (Threads Only - Flagged)
Uses facebook/bart-large-mnli on threads flagged by the hybrid model.
Inputs:
    ThreadText.parquet
    RiskTaxonomy.json
    risk_outputs/RiskScores_threads.parquet   (for flagged threads)
Outputs:
    risk_outputs/RiskScores_zeroshot_threads.parquet
    risk_outputs/RiskScores_zeroshot_threads.csv
"""

import pandas as pd, json, time
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline

# -------------------- Paths --------------------
BASE_DIR = Path("/content/drive/MyDrive/maildir")
THREAD_PATH = BASE_DIR / "ThreadText.parquet"
TAX_PATH = BASE_DIR / "RiskTaxonomy.json"
HYBRID_PATH = BASE_DIR / "risk_outputs" / "RiskScores_threads.parquet"

OUT_PARQ = BASE_DIR / "risk_outputs" / "RiskScores_zeroshot_threads.parquet"
OUT_CSV  = BASE_DIR / "risk_outputs" / "RiskScores_zeroshot_threads.csv"
OUT_PARQ.parent.mkdir(parents=True, exist_ok=True)

# -------------------- Load data --------------------
print("\nLoading thread and flagged hybrid data...")

threads_df = pd.read_parquet(THREAD_PATH)
hybrid_df = pd.read_parquet(HYBRID_PATH)

# keep only threads with hits_total > 0
flagged_ids = set(hybrid_df.loc[hybrid_df["hits_total"] > 0, "thread_id"])
df = threads_df[threads_df["thread_id"].isin(flagged_ids)].copy()
print(f"Loaded {len(df)} flagged threads for zero-shot classification.")

# ✅ combine subject and body into one text column
subj = df["subject_norm"] if "subject_norm" in df.columns else ""
body = df["body_concat"] if "body_concat" in df.columns else ""
df["__text__"] = (subj.fillna("") + " " + body.fillna("")).str.strip()

# -------------------- Load taxonomy --------------------
with open(TAX_PATH, "r", encoding="utf-8") as f:
    tax = json.load(f)

categories = []
for c in tax.get("categories", []):
    if isinstance(c, dict) and "name" in c:
        categories.append(c["name"])
    elif isinstance(c, str):
        categories.append(c)
print("Categories:", categories)

# -------------------- Load model --------------------
print("\nLoading zero-shot classification model (facebook/bart-large-mnli)...")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
device = classifier.device
print(f"🚀 Model loaded on {device} — starting inference...\n")

# -------------------- Run classification --------------------
rows, start = [], time.time()
total = len(df)

print("Columns in df right before inference:", list(df.columns)[:10])
print("First few rows:")
print(df.head(3)[["thread_id", "__text__"]].to_string())

for i, (_, row) in enumerate(tqdm(df.iterrows(), total=total, desc="Processing threads")):
    text = row["__text__"]
    thread_id = row["thread_id"]

    try:
        res = classifier(text, candidate_labels=categories, multi_label=False)
        label = res["labels"][0]
        score = round(float(res["scores"][0]), 4)
    except Exception as e:
        label, score = None, 0.0
        print(f"⚠️ Error on {thread_id}: {e}")

    rows.append({
        "thread_id": thread_id,
        "risk_label_zeroshot": label,
        "model_confidence": score
    })

    # Save progress every 100 threads
    if (i + 1) % 100 == 0:
        pd.DataFrame(rows).to_parquet(OUT_PARQ, index=False)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        percent = ((i + 1) / total) * 100
        if int(percent) % 10 == 0:
            print(f"✅ {int(percent)}% done ({i+1}/{total})")

# Final save
pd.DataFrame(rows).to_parquet(OUT_PARQ, index=False)
pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

print(f"\n✅ Done. Processed {len(rows)} threads in {round(time.time() - start, 2)} seconds.")
print(f"Outputs saved to:\n  {OUT_PARQ}\n  {OUT_CSV}")
