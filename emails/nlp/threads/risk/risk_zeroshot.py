# -*- coding: utf-8 -*-
"""
Zero-Shot Risk Classification on flagged emails only
Uses facebook/bart-large-mnli to predict the main risk label
Runs only on emails where hits_total > 0 (from the hybrid model)
This version points to my local OneDrive maildir.
"""

import pandas as pd, json, time
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline

# -------------------- Paths --------------------
BASE_DIR = Path("C:/Users/petermak11/OneDrive - Swinburne University/Documents/Master of IT/2025 Semester 2/COS70008 - Technology Innovation Research and Project/enron_mail_20150507.tar/enron_mail_20150507/maildir")

TEXT_PATH   = BASE_DIR / "TextBase.parquet"
HYBRID_PATH = BASE_DIR / "RiskScores.parquet"     # hybrid keyword scoring results
TAX_PATH    = BASE_DIR / "RiskTaxonomy.json"

OUT_DIR = BASE_DIR / "risk_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PARQ = OUT_DIR / "RiskScores_zeroshot_flagged.parquet"
OUT_CSV  = OUT_DIR / "RiskScores_zeroshot_flagged.csv"

# -------------------- Load text + hybrid data --------------------
print("\nLoading text data and hybrid keyword results...")
text_df = pd.read_parquet(TEXT_PATH)
hybrid_df = pd.read_parquet(HYBRID_PATH)

# Filter only the emails that had any keyword hits
flagged_ids = set(hybrid_df.loc[hybrid_df["hits_total"] > 0, "email_id"])
df = text_df[text_df["email_id"].isin(flagged_ids)].copy()

# Combine subject and body for model input
df["__text__"] = (df["subject_norm"].fillna("") + " " + df["body_clean"].fillna("")).str.strip()
print(f"Loaded {len(df)} flagged emails for zero-shot classification.")

# -------------------- Load taxonomy --------------------
with open(TAX_PATH, "r", encoding="utf-8") as f:
    tax = json.load(f)

# Handle both flat and nested taxonomy structures
categories = [c["name"] if isinstance(c, dict) else c for c in tax.get("categories", [])]
print("Categories:", categories)

# -------------------- Load model --------------------
print("\nLoading zero-shot classification model (facebook/bart-large-mnli)...")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)  # runs on CPU
print("Model loaded — starting inference...\n")

# -------------------- Run classification --------------------
rows = []
start = time.time()
total = len(df)

for i, (_, row) in enumerate(tqdm(df.iterrows(), total=total, desc="Processing emails")):
    text = row["__text__"]
    email_id = row["email_id"]

    try:
        res = classifier(text, candidate_labels=categories, multi_label=False)
        label = res["labels"][0]
        score = round(float(res["scores"][0]), 4)
    except Exception as e:
        label, score = None, 0.0
        print(f"⚠️ {email_id}: {e}")

    rows.append({
        "email_id": email_id,
        "risk_label_zeroshot": label,
        "model_confidence": score
    })

    # Quick progress checkpoints (5%, 50%, 100%)
    pct = int(((i + 1) / total) * 100)
    if pct in [5, 50, 100]:
        print(f"{pct}% done ({i+1}/{total})")
        pd.DataFrame(rows).to_parquet(OUT_PARQ, index=False)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

# -------------------- Final save --------------------
pd.DataFrame(rows).to_parquet(OUT_PARQ, index=False)
pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

print(f"\nProcessed {len(rows)} flagged emails in {round(time.time() - start, 2)} seconds.")
print(f"Outputs saved to:\n  {OUT_PARQ}\n  {OUT_CSV}")
