# ==========================
# Two outputs, both models
# 1) EmailScores_both.parquet   (email_id + real email text)
# 2) ThreadScores_both.parquet  (thread_id + concatenated thread text)
# ==========================

!pip -q install pandas pyarrow tqdm transformers torch

import os, time, re
import pandas as pd
import pyarrow.parquet as pq
import torch, torch.nn.functional as F
from google.colab import files
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("CUDA:", torch.cuda.is_available(), "|",
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# Models
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"     # pos/neu/neg
EMOTION_MODEL   = "j-hartmann/emotion-english-distilroberta-base"        # anger, joy, sadness, fear, disgust, surprise, neutral

# Batch and length
BATCH_SIZE = 2048             # drop to 1024 or 512 if you hit CUDA OOM
MAXLEN_EMAIL  = 128
MAXLEN_THREAD = 256
LIMIT = None                  # set to 1000 for a smoke test

# Upload if missing
need_upload = not (os.path.exists("TextBase.parquet") and os.path.exists("ThreadText.parquet"))
if need_upload:
    print("Please upload TextBase.parquet and ThreadText.parquet …")
    files.upload()

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_pipe(model_name):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    id2label = getattr(mdl.config, "id2label", {})
    return tok, mdl, id2label

sent_tok, sent_mdl, sent_id2label = load_pipe(SENTIMENT_MODEL)
emo_tok,  emo_mdl,  emo_id2label  = load_pipe(EMOTION_MODEL)

def _pick_text_cols(parquet_path):
    cols = set(pq.ParquetFile(parquet_path).schema.names)
    subj = next((c for c in ["subject_norm","subject","thread_subject","title","topic"] if c in cols), None)
    body = next((c for c in ["body_clean","body_norm","body","text","content","message","thread_text"] if c in cols), None)
    print(f"[{parquet_path}] subject={subj or '<empty>'}, body={body or '<empty>'}")
    if subj is None and body is None:
        raise ValueError(f"No suitable text columns found in {parquet_path}.")
    return subj, body

def _softmax_top(logits, id2label):
    probs = F.softmax(logits.float(), dim=-1)
    conf, idx = probs.max(dim=-1)
    labels = [id2label[int(i)] if not str(id2label.get(int(i),"")).lower().startswith("label_")
              else f"LABEL_{int(i)}" for i in idx]
    return labels, conf

@torch.inference_mode()
def _score_texts(texts, tok, mdl, max_len):
    enc = tok(texts, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
    enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
    logits = mdl(**enc).logits
    labels, conf = _softmax_top(logits, getattr(mdl.config, "id2label", {}))
    return labels, conf.tolist()

def _token_lengths(texts, tok, max_len):
    return tok(texts, truncation=True, max_length=max_len, return_length=True, padding=False)["length"]

def _clean_join(a, b=None):
    if b is None or b == "":
        t = str(a)
    else:
        t = f"{a}. {b}"
    return re.sub(r"\s+", " ", str(t)).strip()

def _iter_parquet(path, id_col, text_cols, max_rows=None):
    pf = pq.ParquetFile(path)
    remaining = pf.metadata.num_rows if max_rows is None else min(pf.metadata.num_rows, max_rows)
    seen = 0
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=[id_col] + text_cols):
        df = batch.to_pandas()
        df[text_cols] = df[text_cols].fillna("")
        texts = [_clean_join(row[text_cols[0]], row[text_cols[1]] if len(text_cols) > 1 else "")
                 for _, row in df.iterrows()]
        ids = df[id_col].tolist()
        if max_rows is not None and seen + len(ids) > max_rows:
            cut = max_rows - seen
            yield ids[:cut], texts[:cut]
            break
        else:
            yield ids, texts
            seen += len(ids)

# 1) EMAILS
def score_emails():
    if not os.path.exists("TextBase.parquet"):
        print("TextBase.parquet not found, skipping emails")
        return None
    subj_col, body_col = _pick_text_cols("TextBase.parquet")
    text_cols = [c for c in [subj_col, body_col] if c is not None]

    rows, seen, t0 = [], 0, time.time()
    for ids, texts in _iter_parquet("TextBase.parquet", "email_id", text_cols, max_rows=LIMIT):
        # sentiment
        s_labels, s_scores = _score_texts(texts, sent_tok, sent_mdl, MAXLEN_EMAIL)
        # emotion
        e_labels, e_scores = _score_texts(texts, emo_tok, emo_mdl, MAXLEN_EMAIL)
        # lengths
        lens = _token_lengths(texts, sent_tok, MAXLEN_EMAIL)

        for pid, st, ss, et, es, L, txt in zip(ids, s_labels, s_scores, e_labels, e_scores, lens, texts):
            rows.append({
                "email_id": pid,
                "email_text": txt,                         # real email text for your presentation
                "sentiment_label": st,
                "sentiment_score": float(ss),
                "emotion_label": et,
                "emotion_score": float(es),
                "text_len_tokens": int(L),
                "sentiment_model": SENTIMENT_MODEL,
                "emotion_model": EMOTION_MODEL,
            })
        seen += len(ids)
        rps = seen / max(time.time() - t0, 1e-6)
        print(f"Email   [{seen}] | {rps:.1f} rows/s")

    df = pd.DataFrame(rows)
    df.to_parquet("EmailScores_both.parquet", index=False)
    print(f"Saved EmailScores_both.parquet with {len(df):,} rows")
    return df

# 2) THREADS (concatenate messages per thread)
def score_threads():
    if not os.path.exists("ThreadText.parquet"):
        print("ThreadText.parquet not found, skipping threads")
        return None
    tt = pd.read_parquet("ThreadText.parquet").fillna("")
    if "date" in tt.columns:
        tt = tt.sort_values(["thread_id","date"])
    else:
        tt = tt.sort_values(["thread_id"])

    # Build one text per thread
    joined = (tt["subject_norm"].astype(str) + ". " + tt["body_clean"].astype(str)).str.replace(r"\s+"," ", regex=True)
    thread_text = tt.assign(_text=joined).groupby("thread_id")["_text"].agg(" [SEP] ".join).reset_index()
    thread_text = thread_text.rename(columns={"_text":"thread_text"})

    # Score in chunks
    ids_all = thread_text["thread_id"].tolist()
    texts_all = thread_text["thread_text"].tolist()

    rows, t0 = [], time.time()
    for i in range(0, len(ids_all), BATCH_SIZE):
        ids = ids_all[i:i+BATCH_SIZE]
        texts = texts_all[i:i+BATCH_SIZE]

        # sentiment
        s_labels, s_scores = _score_texts(texts, sent_tok, sent_mdl, MAXLEN_THREAD)
        # emotion
        e_labels, e_scores = _score_texts(texts, emo_tok, emo_mdl, MAXLEN_THREAD)
        # lengths
        lens = _token_lengths(texts, sent_tok, MAXLEN_THREAD)

        for pid, st, ss, et, es, L, txt in zip(ids, s_labels, s_scores, e_labels, e_scores, lens, texts):
            rows.append({
                "thread_id": pid,
                "thread_text": txt,                        # concatenated text for transparency
                "sentiment_label": st,
                "sentiment_score": float(ss),
                "emotion_label": et,
                "emotion_score": float(es),
                "text_len_tokens": int(L),
                "sentiment_model": SENTIMENT_MODEL,
                "emotion_model": EMOTION_MODEL,
            })

        done = min(i + BATCH_SIZE, len(ids_all))
        rps = done / max(time.time() - t0, 1e-6)
        print(f"Thread  [{done}/{len(ids_all)}] | {rps:.1f} rows/s")

        if LIMIT is not None and done >= LIMIT:
            rows = rows[:LIMIT]
            break

    df = pd.DataFrame(rows)
    df.to_parquet("ThreadScores_both.parquet", index=False)
    print(f"Saved ThreadScores_both.parquet with {len(df):,} rows")
    return df

emails_df = score_emails()
threads_df = score_threads()
