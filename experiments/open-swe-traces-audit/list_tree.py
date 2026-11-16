"""List a HF dataset tree (paginated) at a fixed revision; dump JSON entries (path, size, lfs)."""
import json, os, re, sys, urllib.request

repo, rev, out = sys.argv[1], sys.argv[2], sys.argv[3]
tok = os.environ.get("HF_TOKEN")
url = f"https://huggingface.co/api/datasets/{repo}/tree/{rev}?recursive=true&expand=false&limit=1000"
items = []
while url:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"} if tok else {})
    with urllib.request.urlopen(req) as r:
        items += json.load(r)
        m = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link", "") or "")
        url = m.group(1) if m else None
json.dump(items, open(out, "w"))
print(len(items), "entries")
