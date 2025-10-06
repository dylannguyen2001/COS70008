# untar_fix_windows.py
import tarfile, os, shutil, re

SRC = "enron_mail_20150507.tar.gz"
DST = "maildir_fixed"  # new, cleaned output

def sanitize(path: str) -> str:
    # normalize slashes and remove leading "./"
    path = path.replace("\\", "/")
    if path.startswith("./"):
        path = path[2:]
    # only keep the maildir subtree
    # the archive stores paths like "maildir/allen-p/_sent_mail/1."
    # strip trailing dots/spaces on every path segment (Windows can't handle them)
    parts = []
    for seg in path.split("/"):
        if seg in ("", ".", ".."):
            continue
        seg = re.sub(r"[\. ]+$", "", seg)  # remove trailing dots/spaces
        parts.append(seg)
    return os.path.join(*parts) if parts else ""

# clean target if it exists
if os.path.isdir(DST):
    shutil.rmtree(DST)
os.makedirs(DST, exist_ok=True)

count = 0
with tarfile.open(SRC, "r:gz") as tf:
    for m in tf.getmembers():
        if not m.name.startswith("maildir/"):
            continue
        clean_rel = sanitize(m.name)
        if not clean_rel:
            continue
        out_path = os.path.join(DST, clean_rel)

        if m.isdir():
            os.makedirs(out_path, exist_ok=True)
            continue

        # ensure parent dir exists
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # extract file content safely
        f = tf.extractfile(m)
        if f is None:
            continue
        with open(out_path, "wb") as w:
            w.write(f.read())
        count += 1
        if count % 10000 == 0:
            print(f"Extracted {count} files...")

print(f"Done. Extracted {count} files into {DST}")
