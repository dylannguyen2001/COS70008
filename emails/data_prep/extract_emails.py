# extract_emails.py
import os, hashlib
import pandas as pd
from email import policy
from email.parser import BytesParser
from email.utils import parsedate_to_datetime
from datetime import timezone

DATA_DIR = "C:/Users/Hai/Downloads/maildir_fixed/maildir"      # <- cleaned tree
OUTPUT   = "Emails.parquet"
LIMIT    = None              # set to None for full run after testing

def parse_list(field: str):
    if not field:
        return []
    parts = []
    for chunk in field.replace("\n", " ").split(","):
        parts.extend([p.strip().lower() for p in chunk.split(";") if p.strip()])
    return parts

def parse_dt_utc(date_str: str):
    if not date_str:
        return None
    try:
        dt = parsedate_to_datetime(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return None

def owner_from_root(root: str):
    # work for both "maildir" and "maildir_fixed"
    parts = os.path.normpath(root).split(os.sep)
    for anchor in ("maildir_fixed", "maildir"):
        if anchor in parts:
            i = parts.index(anchor)
            if i + 1 < len(parts):
                return parts[i + 1]
    return None

rows, ok, skip = [], 0, 0

if not os.path.isdir(DATA_DIR):
    raise SystemExit(f"DATA_DIR not found: {DATA_DIR}")

for root, _, files in os.walk(DATA_DIR):
    folder = os.path.basename(root).lower()
    owner  = owner_from_root(root)

    for fname in files:
        path = os.path.join(root, fname)

        # open file normally (names are cleaned now)
        try:
            with open(path, "rb") as fp:
                msg = BytesParser(policy=policy.default).parse(fp)
        except Exception:
            skip += 1
            continue

        try:
            part = msg.get_body(preferencelist=("plain", "html"))
            body = part.get_content() if part else msg.get_content()

            rec = {
                "email_id": hashlib.md5((os.path.abspath(path) + str(msg.get("Message-ID"))).encode("utf-8")).hexdigest(),
                "msg_id": msg.get("Message-ID"),
                "from_raw": (msg.get("From") or "").lower(),
                "x_from": (msg.get("X-From") or "").strip(),   # <-- NEW
                "x_to": (msg.get("X-To") or "").strip(),       # <-- NEW
                "to_list": parse_list(msg.get("To")),
                "cc_list": parse_list(msg.get("Cc")),
                "bcc_list": parse_list(msg.get("Bcc")),
                "dt_utc": parse_dt_utc(msg.get("Date")),
                "subject": msg.get("Subject"),
                "body_raw": body,
                "employee_dir": owner,
                "folder": folder,
                "path": os.path.abspath(path),
            }
            rows.append(rec)
            ok += 1
            if ok % 5000 == 0:
                print(f"Parsed {ok} emails… (skipped {skip})")
            if LIMIT and ok >= LIMIT:
                break
        except Exception:
            skip += 1
            continue
    if LIMIT and ok >= LIMIT:
        break

print(f"Finished: parsed={ok}, skipped={skip}")

# ---- Save to Parquet ----
df = pd.DataFrame(rows)
cols = [
    "email_id","msg_id","from_raw","x_from","x_to",
    "to_list","cc_list","bcc_list",
    "dt_utc","subject","body_raw","employee_dir","folder","path"
]
for c in cols:
    if c not in df.columns:
        df[c] = None
df = df[cols]
df.to_parquet(OUTPUT, index=False)
print(f"Saved {len(df)} emails -> {OUTPUT}")
