"""Read parquet footers of v1.2 (qwen38_27b) shards remotely (range requests only).

usage: parquet_meta.py <tree.json> <rev> <out.json>
"""
import json, sys
from concurrent.futures import ThreadPoolExecutor
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

tree, rev, out = sys.argv[1:4]
fs = HfFileSystem()
paths = [e["path"] for e in json.load(open(tree))
         if e["type"] == "file" and "/qwen38_27b/" in e["path"] and e["path"].endswith(".parquet")]


def meta(p):
    with fs.open(f"datasets/nvidia/Open-SWE-Traces@{rev}/{p}", "rb", block_size=1 << 20) as f:
        pf = pq.ParquetFile(f)
        md = pf.metadata
        rgs = [(md.row_group(i).num_rows, md.row_group(i).total_byte_size) for i in range(md.num_row_groups)]
        return dict(path=p, num_rows=md.num_rows, row_groups=rgs, schema=str(pf.schema_arrow),
                    kv={k.decode(): v.decode()[:3000] for k, v in (md.metadata or {}).items()},
                    created_by=md.created_by)


with ThreadPoolExecutor(8) as ex:
    res = list(ex.map(meta, paths))
json.dump(res, open(out, "w"), indent=1)
print(len(res), "files,", sum(r["num_rows"] for r in res), "rows")
