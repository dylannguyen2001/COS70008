# ==== Preparing the NLP Corpus from the CLEAN file (OneDrive \ maildir) ====

import os, re
from pathlib import Path
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

# My OneDrive path to the extracted Enron maildir folder
MAILDIR = r"C:\Users\petermak11\OneDrive - Swinburne University\Documents\Master of IT\2025 Semester 2\COS70008 - Technology Innovation Research and Project\enron_mail_20150507.tar\enron_mail_20150507\maildir"

# Path to the cleaned parquet file (I double-checked this filename manually)
INPUT_CLEAN = os.path.join(MAILDIR, "Emails_clean.parquet")

# Output folder for the processed files (stored under maildir\outputs)
OUT_DIR = os.path.join(MAILDIR, "outputs")
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# Quick path check just to make sure the cleaned file exists
assert os.path.exists(INPUT_CLEAN), f"Not found: {INPUT_CLEAN}\nCheck filename and path."

# Download the NLTK tokenizer (used for splitting sentences)
nltk.download("punkt", quiet=True)

# ------------ Helper functions ------------
def normalize_subject(subj: str) -> str:
    """Standardises subject lines by removing reply/forward tags, brackets, and extra spaces."""
    if pd.isna(subj): return ""
    s = subj.lower().strip()
    s = re.sub(r'^\s*(re|fwd)\s*:\s*', '', s)
    s = re.sub(r'^\[[^\]]{1,40}\]\s*', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def clean_body(text: str) -> str:
    """Cleans email body text — removes quoted replies, signatures, disclaimers, and extra whitespace."""
    if pd.isna(text): return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = "\n".join([ln for ln in t.splitlines() if not ln.strip().startswith(">")])
    t = re.split(r"(?im)^on .{1,200} wrote:$", t)[0]
    t = re.sub(r"(?is)\n(--\s*$|best regards|kind regards|regards,|cheers,|thanks,).*$", "", t)
    t = re.sub(r"(?is)(this e-mail.*confidential|disclaimer:.*)$", "", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()

# ------------ Load the cleaned dataset ------------
df = pd.read_parquet(INPUT_CLEAN)
print("Loaded cleaned dataset:", df.shape)

# ------------ NLP prep transforms ------------
# Here I’m creating standardised subject and cleaned body fields, 
# plus some useful metadata for length and text presence.
df["subject_norm"]    = df["subject"].apply(normalize_subject)
df["body_clean"]      = df["body_raw"].apply(clean_body)
df["text_len_chars"]  = df["body_clean"].str.len()
df["text_len_tokens"] = df["body_clean"].str.split().str.len()
df["has_text"]        = (df["text_len_tokens"].fillna(0) >= 3)

# ------------ Sentence-level table ------------
# This breaks each email body into individual sentences and assigns IDs to each one.
sent_rows = []
for _, row in df[df["has_text"]].iterrows():
    for i, s in enumerate(sent_tokenize(row["body_clean"])):
        s = s.strip()
        if s:
            sent_rows.append({
                "sentence_id": f"{row['email_id']}|s{i}",
                "email_id": row["email_id"],
                "sent_idx": i,
                "sentence_text": s
            })
sent_df = pd.DataFrame(sent_rows)

# ------------ Email-level text table ------------
# Keeps only the main text columns I need for NLP models.
textbase = df[[
    "email_id", "subject_norm", "body_clean", "text_len_chars", "text_len_tokens", "has_text"
]].copy()

# ------------ Metadata / document index ------------
# Here I’m keeping metadata fields to help with joins and filtering later on.
doc_cols = [c for c in [
    "email_id", "person_id", "from_norm", "dt_utc",
    "domain_sender", "internal_sender", "mass_mail",
    "employee_dir", "folder", "path", "subject"
] if c in df.columns]
docindex = df[doc_cols].copy()
docindex["reply_flag"]   = df["subject"].str.lower().str.startswith("re:")
docindex["forward_flag"] = df["subject"].str.lower().str.startswith("fwd:")

# ------------ Save outputs ------------
# I’m exporting three parquet files for later analysis or NLP input.
textbase.to_parquet(os.path.join(OUT_DIR, "TextBase.parquet"), index=False)
sent_df.to_parquet(os.path.join(OUT_DIR, "Sentence.parquet"), index=False)
docindex.to_parquet(os.path.join(OUT_DIR, "DocIndex.parquet"), index=False)

print("Saved output files to:", OUT_DIR)

# ------------ Quick quality check ------------
# Just summarising how many emails and sentences were processed successfully.
n_total = len(df)
n_text  = int(df["has_text"].sum())
n_sent  = len(sent_df)
median_tokens = int(df.loc[df["has_text"], "text_len_tokens"].median() if n_text else 0)

print("\n QA:")
print(f"  Emails total: {n_total:,}")
print(f"  With text   : {n_text:,}")
print(f"  Sentences   : {n_sent:,}")
print(f"  Median tokens/email (has_text): {median_tokens}")
