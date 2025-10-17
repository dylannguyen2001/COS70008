# -*- coding: utf-8 -*-
"""
Simple Aggregate (Threads):
Averages the final_score from Hybrid + Zero-Shot
for the same thread_id (flagged threads only)
"""

import pandas as pd
from pathlib import Path

# --- paths ---
BASE = Path(r"C:\Users\petermak11\OneDrive - Swinburne University\Documents\Master of IT\2025 Semester 2\COS70008 - Technology Innovation Research and Project\enron_mail_20150507.tar\enron_mail_20150507\maildir")
HYBRID = BASE / "risk_outputs" / "RiskScores_threads.parquet"
ZERO   = BASE / "risk_outputs" / "RiskScores_zeroshot_threads.parquet"
OUTP   = BASE / "risk_outputs" / "RiskScores_aggregate_threads.parquet"
OUTC   = BASE / "risk_outputs" / "RiskScores_aggregate_threads.csv"

print("📥 Loading hybrid and zero-shot results for threads…")
hy = pd.read_parquet(HYBRID)
ze = pd.read_parquet(ZERO)
print(f"Hybrid shape: {hy.shape}, Zero-shot shape: {ze.shape}")

# --- keep only threads flagged by the hybrid model ---
hy_flag = hy.loc[hy["hits_total"] > 0, ["thread_id", "risk_label", "final_score"]].copy()
print(f"✅ Flagged threads: {len(hy_flag)}")

# --- merge both datasets on thread_id ---
merged = hy_flag.merge(
    ze[["thread_id", "risk_label", "final_score"]],
    on="thread_id",
    how="left",
    suffixes=("_hybrid", "_zeroshot")
)
print(f"🔗 Merged shape: {merged.shape}")

# --- compute aggregate score ---
merged["risk_score_aggregate"] = merged[["final_score_hybrid", "final_score_zeroshot"]].mean(axis=1)

# --- combine labels ---
merged["risk_label_aggregate"] = merged["risk_label_zeroshot"].fillna(merged["risk_label_hybrid"])

# --- save outputs ---
merged.to_parquet(OUTP, index=False)
merged.to_csv(OUTC, index=False)

print(f"\n✅ Done — aggregate thread results saved to:")
print(f"  {OUTP}")
print(f"  {OUTC}")
