import pandas as pd
from supabase import create_client, Client
from pathlib import Path
import time

SUPABASE_URL = "https://klzdfxpuxqgfdwvgrbvr.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtsemRmeHB1eHFnZmR3dmdyYnZyIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MDIxNjEzMywiZXhwIjoyMDc1NzkyMTMzfQ.ngiFc2MVovqgsbGrLY4z0sqsM8mvYkSfxkQ-ayVZ4pg";
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


PARQUET_PATH = Path("data/Emails_clean_9902.parquet")
CSV_PATH = Path("data/Emails_clean_9902.csv")
BASE = Path("data/sentiment_results")
SENT_PATH = BASE / "Email_SentimentScores_internal_multi_9902_both.parquet"
RISK_PATH = Path("data/risk_expanded_results") / "RiskScores_zeroshot_full.parquet"


df = pd.read_parquet(PARQUET_PATH)
print(f"Loaded {len(df):,} rows and {len(df.columns)} columns.")
sent_df = pd.read_parquet(SENT_PATH)
risk_df = pd.read_parquet(RISK_PATH)

merged = (
    df
    .merge(risk_df[["email_id", "risk_label", "final_score"]], on="email_id", how="left")
    .merge(
        sent_df[["email_id", "sentiment_label", "sentiment_score", "emotion_label", "emotion_score"]],
        on="email_id",
        how="left"
    )
)
merged = merged.dropna()

# Convert non-string serializables (lists, dicts, etc.)
for c in df.columns:
    if df[c].dtype == "object":
        df[c] = df[c].apply(lambda x: str(x) if isinstance(x, (list, dict)) else x)

merged.to_csv(CSV_PATH, index=False)

def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if "datetime" in str(df[col].dtype):
            df[col] = df[col].astype(str)
        elif df[col].dtype == "bool":
            df[col] = df[col].astype(bool)
        elif "int" in str(df[col].dtype):
            df[col] = df[col].astype(int)
        elif df[col].dtype == "object":
            df[col] = df[col].astype(str)
    return df.where(pd.notnull(df), None)


# Upload CSV Supabase
def upload_csv_to_supabase(table_name: str, file_path: Path, batch_size=250):
    print(f"\nUploading {file_path.name} -> '{table_name}' ...")

    df = pd.read_csv(file_path)
    df = sanitize_df(df)
    total = len(df)
    print(f"Prepared {total:,} rows x {len(df.columns)} cols.")

    # Optional: clear old data
    try:
        supabase.table(table_name).delete().execute()
        print("Cleared existing rows.")
    except Exception as e:
        print("Skipped clearing:", e)

    # Upload in batches
    for start in range(0, total, batch_size):
        batch = df.iloc[start:start + batch_size].to_dict(orient="records")
        response = supabase.table(table_name).insert(batch, count="exact").execute()
        print(f"Uploaded batch {start + len(batch)} / {total}")
        time.sleep(0.1)  # avoid rate limit

    print(f"Finished uploading '{table_name}' ({total:,} rows).")

upload_csv_to_supabase("emails_clean_9902", CSV_PATH)
