import os, re, hashlib
from pathlib import Path
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

INPUT_CLEAN = Path("data/Emails_clean_9902.parquet")    # already cleaned emails
OUT_DIR = Path("data/threads")
OUT_DIR.mkdir(parents=True, exist_ok=True)

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

def normalize_subject(subj: str) -> str:
    if pd.isna(subj): return ""
    s = subj.lower().strip()
    s = re.sub(r'^\s*(re|fwd)\s*:\s*', '', s)
    s = re.sub(r'^\[[^\]]{1,40}\]\s*', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def format_email_block(sub):
    """Format emails within one thread in chronological order with separators."""
    sub = sub.sort_values("dt_utc")
    blocks = []
    for i, row in enumerate(sub.itertuples(index=False)):
        header = (
            f"\n--- EMAIL {i+1}/{len(sub)} ---\n"
            f"From: {row.from_norm}\n"
            f"Date: {row.dt_utc}\n"
            f"Subject: {row.subject_root}\n\n"
        )
        body = (row.body_raw)
        if body.strip():
            blocks.append(header + body)
    return "\n\n".join(blocks).strip()


# Load cleaned emails
df = pd.read_parquet(INPUT_CLEAN)
print("Loaded:", df.shape)

print(f"Original length: {len(df):,} emails")

if "internal_sender" in df.columns:
    df = df[df["internal_sender"] == True].copy()
else:
    df = df[df["person_id"].str.endswith("@enron.com", na=False)].copy()

print(f"Filtered to internal senders only: {len(df):,} emails")


# subject normalization at email-level (helps thread grouping)
df["subject_root"] = df["subject"].map(normalize_subject)

# participants string for grouping
df["participants"] = df.apply(
    lambda r: sorted(set([r["from_norm"]] + list(r["to_norm"]) + list(r["cc_norm"]) + list(r["bcc_norm"]))),
    axis=1,
)
df["participants_str"] = df["participants"].map(lambda lst: "|".join(lst))

# stable thread_id (hash of subject_root + participants)
df["thread_id"] = df.apply(
    lambda r: hashlib.md5((r["subject_root"] + "|" + r["participants_str"]).encode()).hexdigest(),
    axis=1
)

# Threads.parquet
threads = df.groupby("thread_id").agg(
    n_emails=("email_id", "count"),
    participants=("participants", lambda x: sorted(set(p for sub in x for p in sub))),
    start_dt=("dt_utc", "min"),
    end_dt=("dt_utc", "max"),
    subject_root=("subject_root", "first"),
    root_email_id=("email_id", "first"),
    email_ids=("email_id", lambda x: list(x))
).reset_index()

threads.to_parquet(OUT_DIR / "Threads_internal_9902.parquet", index=False)
print("Threads_9902.parquet:", threads.shape)

# build ThreadText.parquet
# thread_text = df.groupby("thread_id").agg(
#     subject_norm=("subject_root", "first"),
#     body_concat=("body_raw", lambda x: "\n\n".join(clean_body(b) for b in x if b.strip())),
# ).reset_index()

thread_text = (
    df.groupby("thread_id")
      .apply(format_email_block)
      .reset_index(name="body_concat")
)


thread_text["n_tokens"] = thread_text["body_concat"].map(lambda t: len(t.split()))
thread_text["has_text"] = thread_text["n_tokens"] > 0

thread_text.to_parquet(OUT_DIR / "ThreadText_internal_9902.parquet", index=False)
print("ThreadText.parquet:", thread_text.shape)

# ThreadSentence.parquet
sent_rows = []
for _, row in thread_text[thread_text["has_text"]].iterrows():
    for i, s in enumerate(sent_tokenize(row["body_concat"])):
        s = s.strip()
        if s:
            sent_rows.append({
                "sentence_id": f"{row['thread_id']}|s{i}",
                "thread_id": row["thread_id"],
                "sent_idx": i,
                "sentence_text": s
            })
sent_df = pd.DataFrame(sent_rows)
sent_df.to_parquet(OUT_DIR / "ThreadSentence_internal_9902.parquet", index=False)
print("ThreadSentence_9902.parquet:", sent_df.shape)

print("\nQA:")
print("  Total threads:", len(threads))
print("  With text    :", thread_text['has_text'].sum())
print("  Total thread sentences:", len(sent_df))

