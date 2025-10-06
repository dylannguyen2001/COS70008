# risk_taxonomy_seed.py
# Week 2 – Part C (Peter): Risk taxonomy JSON + seed labelling table from TextBase.parquet

from pathlib import Path
import json, re
import pandas as pd

# -------- paths (Downloads) --------
DL = Path.home() / "Downloads"
TEXTBASE = DL / "TextBase.parquet"           # must already exist
OUT_DIR   = DL / "risk_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TAXONOMY_JSON = OUT_DIR / "RiskTaxonomy.json"
SEED_CSV      = OUT_DIR / "LabeledSeed.csv"
SEED_PARQUET  = OUT_DIR / "LabeledSeed.parquet"

# -------- sanity checks --------
if not TEXTBASE.exists():
    raise FileNotFoundError(f"Missing TextBase.parquet at:\n{TEXTBASE}\nPut it in Downloads and rerun.")

# -------- define taxonomy (edit keywords anytime) --------
taxonomy = {
    "categories": ["fraud", "manipulation", "collusion", "reputational"],
    "severity_levels": ["low", "medium", "high"],
    "keywords": {
        "fraud": [
            r"\bfraud(ulent)?\b", r"\bbribe(s|d|ry)?\b", r"\bbackdate(d|)\b",
            r"\bkickback(s|)\b", r"\bundisclosed\b", r"\binflat(e|ed|ion)\b"
        ],
        "manipulation": [
            r"\bconceal\b", r"\bhide the\b", r"\bspin\b", r"\bmislead(ing|)?\b",
            r"\bcover up\b"
        ],
        "collusion": [
            r"\bfix(ed)? price(s|)?\b", r"\binsider\b", r"\bcoordina(te|ting)\b",
            r"\bbid(-| )?rig(ging|)\b"
        ],
        "reputational": [
            r"\bpress\b", r"\bheadline(s|)?\b", r"\bscandal\b", r"\bembarrass(ment|)?\b",
            r"\bPR\b"
        ]
    }
}

# Save taxonomy JSON
with open(TAXONOMY_JSON, "w", encoding="utf-8") as f:
    json.dump(taxonomy, f, indent=2, ensure_ascii=False)

print(f"✅ Wrote taxonomy: {TAXONOMY_JSON}")

# -------- load TextBase --------
tb = pd.read_parquet(TEXTBASE)
# Expect at least: email_id, subject_norm, body_clean, text_len_tokens, has_text
needed = {"email_id", "subject_norm", "body_clean", "text_len_tokens"}
missing = needed - set(tb.columns)
if missing:
    raise ValueError(f"TextBase.parquet is missing columns: {missing}")

# Keep only rows with text
tb = tb[tb["text_len_tokens"].fillna(0) >= 3].copy()
tb["body_clean"] = tb["body_clean"].fillna("")

# -------- regex match counts per category --------
def count_hits(text: str, patterns):
    return sum(bool(re.search(p, text, flags=re.IGNORECASE)) for p in patterns)

for cat, patterns in taxonomy["keywords"].items():
    tb[f"hits_{cat}"] = tb["body_clean"].apply(lambda t: count_hits(t, patterns))

hit_cols = [c for c in tb.columns if c.startswith("hits_")]
tb["hits_total"] = tb[hit_cols].sum(axis=1)

# -------- make a quick snippet for labelling UI/help --------
def make_snippet(text: str, max_chars: int = 500):
    t = re.sub(r"\s+", " ", text).strip()
    return t[:max_chars]

tb["snippet"] = tb["body_clean"].apply(make_snippet)

# -------- choose seed set (top keyword hits, then sample extras) --------
seed_core = tb.sort_values("hits_total", ascending=False).head(300)

# Optional: add a few no-hit examples for negatives (comment out if you don’t want)
no_hit = tb[tb["hits_total"] == 0].sample(n=min(50, len(tb[tb["hits_total"] == 0])), random_state=13)
seed_df = pd.concat([seed_core, no_hit], ignore_index=True).drop_duplicates(subset=["email_id"])

# Label columns (to fill manually)
seed_df["risk_label"] = ""     # one of taxonomy["categories"] or "" for none
seed_df["severity"]   = ""     # one of taxonomy["severity_levels"] or ""
seed_df["notes"]      = ""     # free text

# Compact JSON of which patterns matched (optional for reviewer context)
def matched_json(row):
    found = {}
    for cat, pats in taxonomy["keywords"].items():
        found_terms = [p for p in pats if re.search(p, row["body_clean"], re.IGNORECASE)]
        if found_terms:
            found[cat] = found_terms
    return json.dumps(found)

seed_df["matched_terms_json"] = seed_df.apply(matched_json, axis=1)

# Reorder/select columns for the labelling file
out_cols = [
    "email_id", "subject_norm", "snippet", "text_len_tokens",
    "hits_total"
] + hit_cols + ["matched_terms_json", "risk_label", "severity", "notes"]

seed_out = seed_df[out_cols].copy()

# -------- save outputs --------
seed_out.to_csv(SEED_CSV, index=False, encoding="utf-8")
seed_out.to_parquet(SEED_PARQUET, index=False)
print(f"✅ Wrote seed table: {SEED_CSV}")
print(f"✅ Wrote seed parquet: {SEED_PARQUET}")

print("\nSummary:")
print(f"  Rows in TextBase with text: {len(tb):,}")
print(f"  Seed rows for labelling    : {len(seed_out):,}")
print("  Columns in seed            :", list(seed_out.columns))
