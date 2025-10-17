import pandas as pd
from pathlib import Path

# P aths
OUT_DIR = Path("data/threads")
OUT_DIR.mkdir(parents=True, exist_ok=True)

THREADS_PATH   = OUT_DIR / ("Threads_internal_9902.parquet")
TEXT_PATH      = OUT_DIR / ("ThreadText_internal_9902.parquet")
SENT_PATH      = OUT_DIR / ("ThreadSentence_internal_9902.parquet")

OUT_THREADS    = OUT_DIR / ("Threads_internal_multi_9902.parquet")
OUT_TEXT       = OUT_DIR / ("ThreadText_internal_multi_9902.parquet")
OUT_SENT       = OUT_DIR / ("ThreadSentence_internal_multi_9902.parquet")

def main():
    # Load threads
    df_threads = pd.read_parquet(THREADS_PATH)
    print(f"Loaded Threads: {df_threads.shape}")

    # Flag singletons
    df_threads["is_singleton"] = df_threads["n_emails"] == 1
    df_threads["has_replies"]  = df_threads["n_emails"] >= 2

    # Stats
    total = len(df_threads)
    singletons = df_threads["is_singleton"].sum()
    multi = df_threads["has_replies"].sum()

    print("\n=== Thread Stats ===")
    print(f"Total threads: {total:,}")
    print(f"Singletons   : {singletons:,} ({singletons/total:.1%})")
    print(f"Multi-email  : {multi:,} ({multi/total:.1%})")

    # Distribution
    dist = df_threads["n_emails"].value_counts().sort_index()
    print("\nThread size distribution (first 20 sizes):")
    print(dist.head(20))

    # Keep only multi-email threads
    multi_threads = df_threads[df_threads["has_replies"]].copy()
    multi_ids = set(multi_threads["thread_id"])

    df_text = pd.read_parquet(TEXT_PATH)
    df_sent = pd.read_parquet(SENT_PATH)

    df_text_multi = df_text[df_text["thread_id"].isin(multi_ids)].copy()
    df_sent_multi = df_sent[df_sent["thread_id"].isin(multi_ids)].copy()

    multi_threads.to_parquet(OUT_THREADS, index=False)
    df_text_multi.to_parquet(OUT_TEXT, index=False)
    df_sent_multi.to_parquet(OUT_SENT, index=False)

    print(f"\nSaved multi-email threads: {len(multi_threads):,}")
    print(f"Saved ThreadText_multi   : {len(df_text_multi):,}")
    print(f"Saved ThreadSentence_multi: {len(df_sent_multi):,}")

if __name__ == "__main__":
    main()
