#!/usr/bin/env python3
"""
Sentiment Pipeline Prototype (Task D)
Reads TextBase.parquet and writes EmailSentiment.parquet with:
- email_id: str
- sentiment_score: float32  (VADER compound, −1..+1)
- sentiment_label: str      (negative | neutral | positive)
- text_len_tokens: int32
"""

import argparse
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


TEXT_CANDIDATES = [
    "body_clean", "body_norm", "body_text",
    "text", "body", "body_raw", "message", "content"
]
ID_DEFAULT = "email_id"

NEG_T = -0.05   # VADER recommended thresholds
POS_T =  0.05


def pick_text_column(df, preferred=None):
    """Choose a text column. Respect user preferred if valid."""
    if preferred:
        if preferred in df.columns:
            return preferred
        raise ValueError(f"Preferred text column '{preferred}' not found. "
                         f"Available: {list(df.columns)}")
    for c in TEXT_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not find a text column. Looked for: {TEXT_CANDIDATES}. "
        f"Columns present: {list(df.columns)}"
    )


def label_from_score(compound: float) -> str:
    if compound <= NEG_T:
        return "negative"
    if compound >= POS_T:
        return "positive"
    return "neutral"


def simple_tokenise(s: str):
    # Lowercase, split on non-alphanumerics, keep words and numbers
    return [t for t in re.split(r"[^A-Za-z0-9']+", s.lower()) if t]


def clean_email_text(text: str) -> str:
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)

    # Remove common quoted header lines and long footers
    text = re.sub(r"-{2,}\s*original message\s*-{2,}.*", "", text,
                  flags=re.I | re.S)
    text = re.sub(r"from:.*?\n|\bsubject:.*?\n|\bdate:.*?\n|\bto:.*?\n",
                  "", text, flags=re.I)
    return text.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Sentiment Pipeline Prototype"
    )
    parser.add_argument("--in", dest="in_path", default="TextBase.parquet",
                        help="Input parquet path (default: TextBase.parquet)")
    parser.add_argument("--out", dest="out_path",
                        default="EmailSentiment.parquet",
                        help="Output parquet path "
                             "(default: EmailSentiment.parquet)")
    parser.add_argument("--id-col", dest="id_col", default=ID_DEFAULT,
                        help=f"ID column name (default: {ID_DEFAULT})")
    parser.add_argument("--text-col", dest="text_col", default=None,
                        help="Text column to use (optional). If not set, "
                             "the script will auto-detect.")
    parser.add_argument("--sample", dest="sample", type=int, default=None,
                        help="Optional number of rows to sample for a quick run.")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    id_col = args.id_col

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path.resolve()}")

    # Load
    df = pd.read_parquet(in_path)

    if id_col not in df.columns:
        raise ValueError(f"Missing required column '{id_col}' in {in_path}")

    text_col = pick_text_column(df, preferred=args.text_col)

    # Narrow to columns of interest
    df = df[[id_col, text_col]].copy()
    df[text_col] = df[text_col].fillna("").astype(str)

    if args.sample:
        df = df.sample(n=args.sample, random_state=42).reset_index(drop=True)

    analyzer = SentimentIntensityAnalyzer()

    email_ids = []
    scores = []
    labels = []
    tok_lens = []

    print(f"Scoring sentiment on {len(df):,} emails "
          f"(text column: '{text_col}')...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Sentiment"):
        text = clean_email_text(row[text_col])

        vs = analyzer.polarity_scores(text)
        compound = float(vs["compound"])
        label = label_from_score(compound)
        tok_len = len(simple_tokenise(text))

        email_ids.append(row[id_col])
        scores.append(compound)
        labels.append(label)
        tok_lens.append(tok_len)

    out = pd.DataFrame({
        id_col: email_ids,
        "sentiment_score": pd.Series(scores, dtype="float32"),
        "sentiment_label": labels,
        "text_len_tokens": pd.Series(tok_lens, dtype="int32"),
    })

    # Reorder columns exactly as spec
    out = out[[id_col, "sentiment_score", "sentiment_label", "text_len_tokens"]]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    # Quick summary
    counts = out["sentiment_label"].value_counts(dropna=False)
    print(f"\nWrote {len(out):,} rows → {out_path.resolve()}")
    print("Label distribution:")
    for k, v in counts.items():
        pct = 100.0 * v / len(out)
        print(f"  {k:8s}: {v:>8,d}  ({pct:5.1f}%)")


if __name__ == "__main__":
    main()


