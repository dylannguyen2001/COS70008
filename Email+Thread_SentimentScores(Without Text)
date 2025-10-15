import os
import pandas as pd
from google.colab import drive

drive.mount('/content/drive')

base_path = "/content/drive/MyDrive"  # adjust if your files are deeper

email_file = os.path.join(base_path, "EmailScores_final.parquet")
thread_file = os.path.join(base_path, "ThreadScores_final.parquet")

# --- Email ---
if os.path.exists(email_file):
    email_df = pd.read_parquet(email_file)
    if "email_text" in email_df.columns:
        email_df = email_df.drop(columns=["email_text"])
        email_df.to_parquet(os.path.join(base_path, "EmailScores_final_clean.parquet"), index=False)
        email_df.to_csv(os.path.join(base_path, "EmailScores_final_clean.csv"), index=False)
        print("✅ Saved EmailScores_final_clean.[parquet/csv] without email_text")
    else:
        print("No 'email_text' column found.")
else:
    print(f"⚠️ File not found: {email_file}")

# --- Thread ---
if os.path.exists(thread_file):
    thread_df = pd.read_parquet(thread_file)
    if "thread_text" in thread_df.columns:
        thread_df = thread_df.drop(columns=["thread_text"])
        thread_df.to_parquet(os.path.join(base_path, "ThreadScores_final_clean.parquet"), index=False)
        thread_df.to_csv(os.path.join(base_path, "ThreadScores_final_clean.csv"), index=False)
        print("✅ Saved ThreadScores_final_clean.[parquet/csv] without thread_text")
    else:
        print("No 'thread_text' column found.")
else:
    print(f"⚠️ File not found: {thread_file}")
