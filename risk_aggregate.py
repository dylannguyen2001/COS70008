# -*- coding: utf-8 -*-
"""
Aggregate Risk Scores (Flagged Emails Only)
-------------------------------------------
Combines Hybrid + Zero-Shot outputs for emails that were flagged (hits_total > 0)

Inputs:
    RiskScores.parquet           (hybrid taxonomy model)
    RiskScores_zeroshot.parquet  (zero-shot BART model)
Outputs:
    RiskScores_aggregate.parquet
    RiskScores_aggregate.csv
"""

import pandas as pd
from pathlib import Path

# -------------------- Paths --------------------
BASE_DIR = Path(r"C:\Users\petermak11\OneDrive - Swinburne University\Documents\Master of IT\2025 Semester 2\COS70008 - Technology Innovation Research and Project\enron_mail_20150507.tar\enron_mail_20150507\maildir")

HYBRID_PATH = BASE_DIR / "risk_outputs" / "RiskScores.parquet"
ZERO_PATH   = BASE_DIR / "risk_outputs" / "RiskScores_zeroshot.parquet"
FLAGGED_SRC = BASE_DIR / "risk_outputs" / "TopRisk.csv"   # contains hits_total > 0
OUT_PARQ    = BASE_DIR / "risk_outputs" / "RiskScores_aggregate.parquet"
OUT_CSV     = BASE_DIR / "risk_outputs" / "RiskScores_aggregate.csv"

# -------------------- Load Data --------------------
print("\n📥 Loading input data...")

hybrid_df = pd.read_parquet(HYBRID_PATH)
zero_df   = pd.read_parquet(ZERO_PATH)
flag_df   = pd.read_csv(FLAGGED_SRC)

print(f"Hybrid shape: {hybrid_df.shape}")
print(f"Zero-shot shape: {zero_df.shape}")
print(f"Flagged shape: {flag_df.shape}")

# -------------------- Filter for flagged emails --------------------
flagged_ids = set(flag_df.loc[flag_df["hits_total"] > 0, "email_id"])
print(f"✅ Found {len(flagged_ids)} flagged emails")

hybrid_df = hybrid_df[hybrid_df["email_id"].isin(flagged_ids)].copy()
zero_df   = zero_df[zero_df["email_id"].isin(flagged_ids)].copy()

# -------------------- Rename & Merge --------------------
hybrid_df = hybrid_df.rename(columns={
    "risk_label": "risk_label_hybrid",
    "risk_score": "risk_score_hybrid"
}).drop_duplicates("email_id")

zero_df = zero_df.rename(columns={
    "risk_label_zeroshot": "risk_label_zero",
    "model_confidence": "risk_score_zero"
}).drop_duplicates("email_id")

merged = hybrid_df.merge(zero_df, on="email_id", how="outer")
print(f"🔗 Merged shape: {merged.shape}")

# -------------------- Aggregate Scores --------------------
merged["risk_score_hybrid"] = merged["risk_score_hybrid"].clip(0, 1)
merged["risk_score_zero"] = merged["risk_score_zero"].clip(0, 1)

# weighted combination (adjust as needed)
W_HYBRID = 0.4
W_ZERO = 0.6

merged["aggregate_score"] = (
    merged["risk_score_hybrid"].fillna(0) * W_HYBRID +
    merged["risk_score_zero"].fillna(0) * W_ZERO
)

# Prefer zero-shot label if present
merged["final_label"] = merged["risk_label_zero"].combine_first(merged["risk_label_hybrid"])

# -------------------- Save Outputs --------------------
merged.to_parquet(OUT_PARQ, index=False)
merged.to_csv(OUT_CSV, index=False)

print(f"\n✅ Aggregate scores saved to:\n  {OUT_PARQ}\n  {OUT_CSV}")
print(f"Total rows: {len(merged)}")

# -------------------- Preview --------------------
print("\n🔍 Sample output:")
print(merged.head(10))
