# -*- coding: utf-8 -*-
"""
Simple Aggregate: averages the final_score from Hybrid + Zero-Shot
for the same email_id (flagged emails only)
"""

import pandas as pd
from pathlib import Path

# --- paths ---
BASE = Path(r"C:\Users\petermak11\OneDrive - Swinburne University\Documents\Master of IT\2025 Semester 2\COS70008 - Technology Innovation Research and Project\enron_mail_20150507.tar\enron_mail_20150507\maildir")
HYBRID = BASE / "risk_outputs" / "RiskScores.parquet"
ZERO   = BASE / "risk_outputs" / "RiskScores_zeroshot.parquet"
OUTP   = BASE / "risk_outputs" / "RiskScores_aggregate.parquet"
OUTC   = BASE / "risk_outputs" / "RiskScores_aggregate.csv"

print("📥 Loading hybrid and zero-shot results…")
hy = pd.read_parquet(HYBRID)
ze = pd.read_parquet(ZERO)
print(f"Hybrid shape: {hy.shape}, Zero-shot shape: {ze.shape}")

# --- keep only emails flagged by the hybrid model ---
hy_flag = hy.loc[hy["hits_total"] > 0, ["email_id", "risk_label", "final_score"]].copy()
print(f"✅ Flagged emails: {len(hy_flag)}")

# --- merge both datasets on email_id ---
merged = hy_flag.merge(
    ze[["email_id", "risk_label", "final_score"]],
    on="email_id",
    how="left",
    suffixes=("_hybrid", "_zeroshot")
)
print(f"🔗 Merged shape: {merged.shape}")

# --- take the average of hybrid + zero-shot final_score ---
merged["risk_score_aggregate"] = merged[["final_score_hybrid", "final_score_zeroshot"]].mean(axis=1)

# If zero-shot label is missing, fall back to the hybrid label
merged["risk_label_aggregate"] = merged["risk_label_zeroshot"].fillna(merged["risk_label_hybrid"])

# --- save outputs ---
merged.to_parquet(OUTP, index=False)
merged.to_csv(OUTC, index=False)

print(f"\n✅ Done — aggregate results saved to:")
print(f"  {OUTP}")
print(f"  {OUTC}")
