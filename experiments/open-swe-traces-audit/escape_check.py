"""How much does JSON \\uXXXX escaping inflate stored tool outputs?

Open-SWE mini:  escape_check.py osw <parquet>... --tok DIR
DeepSWE mini:   escape_check.py deepswe <run_dir> --tok DIR
For each tool message: tokens as stored vs tokens after un-escaping (json.loads → json.dumps(ensure_ascii=False, indent=2)).
"""
import argparse, glob, gzip, json, re
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("kind"); ap.add_argument("paths", nargs="+"); ap.add_argument("--tok", required=True)
ap.add_argument("--limit", type=int, default=400)
a = ap.parse_args()
T = AutoTokenizer.from_pretrained(a.tok)
ESC = re.compile(r"\\u[0-9a-fA-F]{4}")


def texts():
    if a.kind == "osw":
        import pyarrow.parquet as pq
        n = 0
        for p in a.paths:
            for r in pq.read_table(p, columns=["messages"]).to_pylist():
                yield [m["content"] or "" for m in r["messages"] if m["role"] == "tool"]
                n += 1
                if n >= a.limit:
                    return
    else:
        for f in sorted(glob.glob(f"{a.paths[0]}/tasks/*/trajectory.json.gz"))[: a.limit]:
            d = json.load(gzip.open(f, "rt"))
            out = []
            for s in d["steps"]:
                for res in (s.get("observation") or {}).get("results") or []:
                    c = res.get("content")
                    out.append(c if isinstance(c, str) else json.dumps(c))
            yield out


n_msg = n_esc = tok_st = tok_un = 0
examples = []
for tl in texts():
    for s in tl:
        n_msg += 1
        k = len(ESC.findall(s))
        if not k:
            continue
        n_esc += 1
        try:
            u = json.dumps(json.loads(s), ensure_ascii=False, indent=2)
        except Exception:
            continue
        a1 = len(T(s, add_special_tokens=False)["input_ids"]); a2 = len(T(u, add_special_tokens=False)["input_ids"])
        tok_st += a1; tok_un += a2
        if len(examples) < 2:
            examples.append(s[:200])
print(f"{a.kind}: tool msgs {n_msg}, with \\uXXXX {n_esc} ({n_esc / max(n_msg, 1):.1%}); "
      f"tokens in those msgs: stored {tok_st}, unescaped {tok_un}, inflation {tok_st / max(tok_un, 1):.3f}x")
for e in examples:
    print("  e.g.", repr(e))
