import pandas as pd
from supabase import create_client, Client
from pathlib import Path
import time
import numpy as np

PARQUET_PATH = Path("data/Emails_clean_9902.parquet")
CSV_PATH = Path("data/Emails_clean_9902.csv")
BASE = Path("data/risk_sentiment_results")
SENT_PATH = BASE / "Email_SentimentScores_internal_multi_9902_both.parquet"
RISK_PATH = BASE / "Email_ZeroShot_RiskScores_internal_multi_9902_both.parquet"


df = pd.read_parquet(PARQUET_PATH)
print(f"Loaded {len(df):,} rows and {len(df.columns)} columns.")
sent_df = pd.read_parquet(SENT_PATH)
print(sent_df.info())
risk_df = pd.read_parquet(RISK_PATH)
print(risk_df.info())

merged = (
    df
    .merge(risk_df[["email_id", "risk_label", "final_score"]], on="email_id", how="left")
    .merge(
        sent_df[["email_id", "sentiment_label", "sentiment_score", "emotion_label", "emotion_score"]],
        on="email_id",
        how="left"
    )
)

print(len(merged.dropna()))
