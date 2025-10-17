from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path("data")
IN_DIR   = BASE_DIR / "all"
OUT_DIRS = {year: BASE_DIR / str(year) for year in [1999, 2000, 2001, 2002]}
edges_all = pd.read_parquet(IN_DIR / "Edges_internal.parquet", engine="pyarrow", memory_map=False)

# Loop through each year and save
for year, OUT_DIR in OUT_DIRS.items():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    e = edges_all[edges_all["year"] == year].copy()
    nodes_year = pd.unique(pd.concat([e["src_person_id"], e["dst_person_id"]], ignore_index=True))
    e.to_parquet(OUT_DIR / "Edges_internal.parquet", index=False)
    edges_dir = (
        e.groupby(["src_person_id", "dst_person_id"], as_index=False)["weight"]
         .sum().rename(columns={"weight": "weight"})
    )
    edges_dir["directed"] = True
    edges_dir.to_parquet(OUT_DIR / "Edges_directed_agg_internal.parquet", index=False)

    u = e[["src_person_id", "dst_person_id", "weight"]].copy()
    u["a"] = np.where(u["src_person_id"] < u["dst_person_id"], u["src_person_id"], u["dst_person_id"])
    u["b"] = np.where(u["src_person_id"] < u["dst_person_id"], u["dst_person_id"], u["src_person_id"])
    edges_undir = (
        u.groupby(["a", "b"], as_index=False)["weight"]
         .sum().rename(columns={"a": "src_person_id", "b": "dst_person_id", "weight": "weight"})
    )
    edges_undir["directed"] = False
    edges_undir.to_parquet(OUT_DIR / "Edges_undirected_agg_internal.parquet", index=False)

    print(f"Saved into folder {OUT_DIR}/")

print("\nAll yearly edge files have been created.")
