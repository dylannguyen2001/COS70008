# -*- coding: utf-8 -*-
"""
Keyword/Rules Risk Scoring (Threads Only)
- Inputs:
    ThreadText.parquet
    RiskTaxonomy.json
- Outputs:
    ./risk_outputs/RiskScores_threads.parquet
    ./risk_outputs/TopRisk_threads.csv
"""

import re, json
from pathlib import Path
import pandas as pd

# ----------------- paths -----------------
BASE_DIR = Path(__file__).resolve().parent
THREAD_PATH = BASE_DIR / "ThreadText.parquet"
TAX_PATH = BASE_DIR / "RiskTaxonomy.json"
OUT_DIR = BASE_DIR / "risk_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------- load taxonomy -----------------
def load_taxonomy(tax_path: Path):
    """Loads my risk taxonomy JSON and compiles all regex patterns."""
    with open(tax_path, "r", encoding="utf-8") as f:
        tax = json.load(f)

    compiled = {}
    # Works for both nested and flat JSON structures
    if isinstance(tax.get("categories", [])[0], dict):
        for cat in tax["categories"]:
            cat_name = cat["name"]
            subpatterns = []
            for sub in cat.get("subcategories", []):
                subpatterns.extend(sub.get("keywords", []))
            compiled[cat_name] = [re.compile(p, re.IGNORECASE) for p in subpatterns]
    else:
        for cat in tax.get("categories", []):
            patterns = tax.get("keywords", {}).get(cat, [])
            compiled[cat] = [re.compile(p, re.IGNORECASE) for p in patterns]
    return list(compiled.keys()), compiled

# ----------------- prepare thread text -----------------
def prepare_thread_text(df: pd.DataFrame) -> pd.DataFrame:
    """Combines the thread subject and concatenated body into a single text field."""
    subj = df["subject_norm"] if "subject_norm" in df.columns else ""
    body = df["body_concat"] if "body_concat" in df.columns else ""
    df = df.copy()
    df["__text__"] = (subj.fillna("") + " " + body.fillna("")).str.strip()
    return df

# ----------------- keyword scanning -----------------
def keyword_score(text: str, compiled_patterns: dict):
    """Counts how many times each category’s keywords appear."""
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

# ----------------- label + scoring -----------------
def decide_label(counts: dict):
    """Picks the top risk category and calculates a proportional score."""
    label = max(counts, key=lambda k: counts[k]) if counts else None
    by_cat = counts.get(label, 0) if label else 0
    total = sum(counts.values())
    final_score = (by_cat / total) if total > 0 else 0.0
    return label, total, final_score

# ----------------- main -----------------
def main():
    assert THREAD_PATH.exists(), f"Missing {THREAD_PATH}"
    assert TAX_PATH.exists(), f"Missing {TAX_PATH}"

    print("Loading thread data and taxonomy...")
    df = pd.read_parquet(THREAD_PATH)
    df = prepare_thread_text(df)

    categories, compiled_patterns = load_taxonomy(TAX_PATH)

    rows = []
    for _, r in df.iterrows():
        rid = r["thread_id"]
        txt = r["__text__"] or ""
        counts, terms = keyword_score(txt, compiled_patterns)
        risk_label, hits_total, final_score = decide_label(counts)

        matched_terms_json = {k: v for k, v in terms.items() if v}

        out = {
            "thread_id": rid,
            "risk_label": risk_label,
            "hits_total": hits_total,
            "final_score": round(float(final_score), 4),
            "matched_terms_json": json.dumps(matched_terms_json, ensure_ascii=False)
        }
        for cat in compiled_patterns.keys():
            out[f"hits_{cat}"] = counts.get(cat, 0)
        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_df.sort_values(["final_score", "thread_id"], ascending=[False, True], inplace=True)

    pq_path = OUT_DIR / "RiskScores_threads.parquet"
    csv_path = OUT_DIR / "TopRisk_threads.csv"
    out_df.to_parquet(pq_path, index=False)
    out_df.to_csv(csv_path, index=False)

    print(f"\nSaved thread scores:")
    print(f"  {pq_path}")
    print(f"  {csv_path}")
    print(f"Total threads processed: {len(out_df)}")

if __name__ == "__main__":
    main()
