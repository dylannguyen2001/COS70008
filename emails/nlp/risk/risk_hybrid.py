# -*- coding: utf-8 -*-
"""
Keyword/Rules Risk Scoring (emails + threads)
- Inputs (all in the same folder as this file):
    TextBase.parquet              (emails, from Week 1)
    ThreadText.parquet            (Hai’s thread data)
    RiskTaxonomy.json             (my keyword taxonomy)
    LabeledSeed.parquet           (manual seed labels)
- Outputs (saved under ./risk_outputs):
    RiskScores.parquet
    TopRisk.csv
    RiskScores_threads.parquet
    TopRisk_threads.csv
"""

import os, re, json
from pathlib import Path
import pandas as pd

# ----------------- paths -----------------
BASE_DIR = Path(__file__).resolve().parent
IN_TEXTBASE = BASE_DIR / "TextBase.parquet"
IN_THREADTEXT = BASE_DIR / "ThreadText.parquet"
IN_TAXON = BASE_DIR / "RiskTaxonomy.json"
IN_SEED = BASE_DIR / "LabeledSeed.parquet"

OUT_DIR = BASE_DIR / "risk_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------- load taxonomy -----------------
def load_taxonomy(tax_path: Path):
    """Loads my taxonomy JSON and compiles all regex patterns (handles both nested and flat)."""
    with open(tax_path, "r", encoding="utf-8") as f:
        tax = json.load(f)

    compiled = {}
    # Handles both nested and older flat taxonomy formats
    if isinstance(tax.get("categories", [])[0], dict):
        # Nested JSON version
        for cat in tax["categories"]:
            cat_name = cat["name"]
            subpatterns = []
            for sub in cat.get("subcategories", []):
                subpatterns.extend(sub.get("keywords", []))
            compiled[cat_name] = [re.compile(p, re.IGNORECASE) for p in subpatterns]
    else:
        # Flat JSON version (what I’m mainly using)
        for cat in tax.get("categories", []):
            patterns = tax.get("keywords", {}).get(cat, [])
            compiled[cat] = [re.compile(p, re.IGNORECASE) for p in patterns]

    return list(compiled.keys()), compiled

# ----------------- prep text fields -----------------
def prepare_email_text(df: pd.DataFrame) -> pd.DataFrame:
    """Joins subject and body together for scanning."""
    subj = df["subject_norm"] if "subject_norm" in df.columns else df.get("subject", "")
    body = df["body_clean"] if "body_clean" in df.columns else df.get("body_raw", "")
    df = df.copy()
    df["__text__"] = (subj.fillna("") + " " + body.fillna("")).str.strip()
    return df

def prepare_thread_text(df: pd.DataFrame) -> pd.DataFrame:
    """Does the same for thread-level data (subject + concatenated body)."""
    subj = df["subject_norm"] if "subject_norm" in df.columns else ""
    body = df["body_concat"] if "body_concat" in df.columns else ""
    df = df.copy()
    df["__text__"] = (subj.fillna("") + " " + body.fillna("")).str.strip()
    return df

# ----------------- keyword matching -----------------
def keyword_score(text: str, compiled_patterns: dict):
    """Counts matches per category and stores the matching terms."""
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

# ----------------- label + scoring logic -----------------
def decide_label(counts: dict, seed_label: str | None = None):
    """Chooses the risk label (manual if available, else based on max hits)."""
    if seed_label:
        label = seed_label
        by_cat = counts.get(seed_label, 0)
    else:
        label = max(counts, key=lambda k: counts[k]) if counts else None
        by_cat = counts.get(label, 0) if label else 0
    total = sum(counts.values())
    final_score = (by_cat / total) if total > 0 else 0.0
    return label, total, final_score

# ----------------- main scoring function -----------------
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

        matched_terms_json = {k: v for k, v in terms.items() if v}

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

    csv_name = "TopRisk.csv" if id_col == "email_id" else "TopRisk_threads.csv"
    pq_name  = "RiskScores.parquet" if id_col == "email_id" else "RiskScores_threads.parquet"
    out_df.sort_values(["final_score", id_col], ascending=[False, True]).to_csv(OUT_DIR / csv_name, index=False)
    out_df.to_parquet(OUT_DIR / pq_name, index=False)

    print(f"Saved: {OUT_DIR / pq_name}")
    print(f"Saved: {OUT_DIR / csv_name}")
    return out_df

# ----------------- main entry -----------------
def main():
    assert IN_TEXTBASE.exists(), f"Missing {IN_TEXTBASE}"
    assert IN_TAXON.exists(), f"Missing {IN_TAXON}"

    categories, compiled_patterns = load_taxonomy(IN_TAXON)

    # Load any manual labels I’ve created in LabeledSeed.parquet
    seed_map = {}
    if IN_SEED.exists():
        seed_df = pd.read_parquet(IN_SEED)
        if "email_id" in seed_df.columns and "risk_label" in seed_df.columns:
            seed_map = dict(zip(seed_df["email_id"], seed_df["risk_label"].astype(str)))
            print(f"Loaded seed labels: {len(seed_map)} items")

    # Run scoring for emails
    run_block(
        label="emails",
        in_path=IN_TEXTBASE,
        id_col="email_id",
        prep_fn=prepare_email_text,
        compiled_patterns=compiled_patterns,
        seed_map=seed_map
    )

    # Run scoring for threads if available
    if IN_THREADTEXT.exists():
        run_block(
            label="threads",
            in_path=IN_THREADTEXT,
            id_col="thread_id",
            prep_fn=prepare_thread_text,
            compiled_patterns=compiled_patterns,
            seed_map=None
        )
    else:
        print("\n(No ThreadText.parquet found — skipping threads block.)")

    print("\nDone.")

    # --- Quick coverage summary ---
    try:
        rs_path = OUT_DIR / "RiskScores.parquet"
        if rs_path.exists():
            df = pd.read_parquet(rs_path)
            coverage = (df["hits_total"] > 0).mean() * 100
            avg_score = df["final_score"].mean()
            print(f"\n[Coverage] {coverage:.2f}% of emails matched at least one keyword.")
            print(f"[Average final_score] {avg_score:.4f}")

            # Optional per-category breakdown
            per_cat = {c: (df[f"hits_{c}"] > 0).mean() * 100 for c in compiled_patterns.keys()}
            print("\n[Per-category coverage]")
            for k, v in per_cat.items():
                print(f"  {k:15s}: {v:6.2f}%")
        else:
            print("\n[Coverage] RiskScores.parquet not found — skipping coverage check.")
    except Exception as e:
        print(f"\n[Coverage] Error during coverage evaluation: {e}")

if __name__ == "__main__":
    main()
