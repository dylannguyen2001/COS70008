from pathlib import Path
import pandas as pd

INPUT = Path("data/all/Emails_clean_9902.parquet")
OUT_DIR = Path("data")

emails = pd.read_parquet(INPUT, engine="pyarrow", memory_map=False)
print(f"Loaded {len(emails):,} emails")
print(emails.info())
print(emails["year"].value_counts())

for year in ["1999","2000","2001","2002"]:
    print(emails[emails["year"] == int(year)].info())

