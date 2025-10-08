# -*- coding: utf-8 -*-
"""
Keyword/Rules Risk Scoring (emails + threads)
- Inputs (in same folder as this file):
    TextBase.parquet               (emails, Week 1)
    ThreadText.parquet            (optional, Hai)
    RiskTaxonomy.json             (your taxonomy & keywords)
    LabeledSeed.parquet           (optional, your 50 labels)
- Outputs (created under ./risk_outputs):
    RiskScores.parquet
    TopRisk.csv
    RiskScores_threads.parquet    (if threads exist)
    TopRisk_threads.csv           (if threads exist)
"""

import os, re, json
from pathlib import Path
import pandas as pd

# ----------------- paths -----------------
BASE_DIR = Path(__file__).resolve().parent
IN_TEXTBASE = BASE_DIR / "TextBase.parquet"
IN_THREADTEXT = BASE_DIR / "ThreadText.parquet"
IN_TAXON = BASE_DIR / "RiskTaxonomy.json"
IN_SEED = BASE_DIR / "LabeledSeed.parquet"   # optional

OUT_DIR = BASE_DIR / "risk_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------- helpers -----------------
def load_taxonomy(tax_path: Path):
    with open(tax_path, "r", encoding="utf-8") as f:
        tax = json.load(f)
    cats = tax.get("categories", [])
    keywords = tax.get("keywords", {})
    # compile case-insensitive regex list per category
    compiled = {
        cat: [re.compile(pat, re.IGNORECASE) for pat in keywords.get(cat, [])]
        for cat in cats
    }
    return cats, compiled

def prepare_email_text(df: pd.DataFrame) -> pd.DataFrame:
    # Expect subject_norm, body_clean; fallbacks if needed
    subj = df["subject_norm"] if "subject_norm" in df.columns else df.get("subject", "")
    body = df["body_clean"] if "body_clean" in df.columns else df.get("body_raw", "")
    df = df.copy()
    df["__text__"] = (subj.fillna("") + " " + body.fillna("")).str.strip()
    return df

def prepare_thread_text(df: pd.DataFrame) -> pd.DataFrame:
    # Expect subject_norm, body_concat
    subj = df["subject_norm"] if "subject_norm" in df.columns else ""
    body = df["body_concat"] if "body_concat" in df.columns else ""
    df = df.copy()
    df["__text__"] = (subj.fillna("") + " " + body.fillna("")).str.strip()
    return df

def keyword_score(text: str, compiled_patterns: dict):
    counts = {}
    hits_terms = {}
    for cat, pats in compiled_patterns.items():
        total = 0
        terms = []
        for pat in pats:
            found = pat.findall(text)
            if found:
                total += len(found)
                terms.append(pat.pattern)
        counts[cat] = total
        hits_terms[cat] = terms
    return counts, hits_terms

def decide_label(counts: dict, seed_label: str | None = None):
    # If a seed label exists, trust it; otherwise argmax of keyword counts
    if seed_label:
        label = seed_label
        by_cat = counts.get(seed_label, 0)
    else:
        label = max(counts, key=lambda k: counts[k]) if counts else None
        by_cat = counts.get(label, 0) if label else 0
    total = sum(counts.values())
    # final_score = 0..1 simple normalization (avoid div by zero)
    final_score = (by_cat / total) if total > 0 else 0.0
    return label, total, final_score

def run_block(label: str,
              in_path: Path,
              id_col: str,
              prep_fn,
              compiled_patterns: dict,
              seed_map: dict | None):
    print(f"\n--- Processing {label} ---")
    df = pd.read_parquet(in_path)
    df = prep_fn(df)

    rows = []
    for _, r in df.iterrows():
        rid = r[id_col]
        txt = r["__text__"] or ""
        counts, terms = keyword_score(txt, compiled_patterns)
        seed_lbl = seed_map.get(rid) if seed_map else None
        risk_label, hits_total, final_score = decide_label(counts, seed_lbl)

        # pack matched terms only for winning category to keep row compact
        matched_terms_json = {}
        for k, v in terms.items():
            if v:
                matched_terms_json[k] = v

        # Track per-category counts too (flat columns)
        out = {
            id_col: rid,
            "risk_label": risk_label,
            "hits_total": hits_total,
            "final_score": round(float(final_score), 4),
            "matched_terms_json": json.dumps(matched_terms_json, ensure_ascii=False)
        }
        for cat in compiled_patterns.keys():
            out[f"hits_{cat}"] = counts.get(cat, 0)
        rows.append(out)

    out_df = pd.DataFrame(rows)

    # top risk pick per item (already 1-row per item here; also write CSV quick view)
    csv_name = "TopRisk.csv" if id_col == "email_id" else "TopRisk_threads.csv"
    pq_name  = "RiskScores.parquet" if id_col == "email_id" else "RiskScores_threads.parquet"
    out_df.sort_values(["final_score", id_col], ascending=[False, True]).to_csv(OUT_DIR / csv_name, index=False)
    out_df.to_parquet(OUT_DIR / pq_name, index=False)

    print(f"Saved: {OUT_DIR / pq_name}")
    print(f"Saved: {OUT_DIR / csv_name}")
    return out_df

# ----------------- main -----------------
def main():
    # Check inputs
    assert IN_TEXTBASE.exists(), f"Missing {IN_TEXTBASE}"
    assert IN_TAXON.exists(), f"Missing {IN_TAXON}"

    categories, compiled_patterns = load_taxonomy(IN_TAXON)

    # Optional seed labels to “lock in” some true categories
    seed_map = {}
    if IN_SEED.exists():
        seed_df = pd.read_parquet(IN_SEED)
        # Expect columns: email_id, risk_label
        if "email_id" in seed_df.columns and "risk_label" in seed_df.columns:
            seed_map = dict(zip(seed_df["email_id"], seed_df["risk_label"].astype(str)))
            print(f"Loaded seed labels: {len(seed_map)} items")

    # Emails pass
    run_block(
        label="emails",
        in_path=IN_TEXTBASE,
        id_col="email_id",
        prep_fn=prepare_email_text,
        compiled_patterns=compiled_patterns,
        seed_map=seed_map
    )

    # Threads pass
    if IN_THREADTEXT.exists():
        run_block(
            label="threads",
            in_path=IN_THREADTEXT,
            id_col="thread_id",
            prep_fn=prepare_thread_text,
            compiled_patterns=compiled_patterns,
            seed_map=None  # seeds are per email_id; skip for threads
        )
    else:
        print("\n(No ThreadText.parquet found — skipping threads block.)")

    print("\nDone.")

if __name__ == "__main__":
    main()
