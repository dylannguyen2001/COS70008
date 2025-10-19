import pandas as pd

df = pd.read_parquet("data/threads/ThreadText_internal_9902.parquet")
print(df.info())
sample = df["body_concat"].iloc[10]
print(sample)  # print first 500 chars
