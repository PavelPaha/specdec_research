"""Print a compact view of one trajectory. usage: peek.py <parquet> <row> [maxchars] [msg_from] [msg_to]"""
import json, sys
import pyarrow.parquet as pq

path, row = sys.argv[1], int(sys.argv[2])
mx = int(sys.argv[3]) if len(sys.argv) > 3 else 600
a = int(sys.argv[4]) if len(sys.argv) > 4 else 0
b = int(sys.argv[5]) if len(sys.argv) > 5 else 6
t = pq.read_table(path).slice(row, 1).to_pylist()[0]
md = dict(t["metadata"]); md.pop("reference_patch"); mp = md.pop("model_patch")
print({k: t[k] for k in ("instance_id", "trajectory_id", "repo", "language", "resolved", "hf_dataset_name")}, md,
      "model_patch_lines", (mp or {}).get("num_modified_lines"))
print("n_msgs", len(t["messages"]))
for i, m in enumerate(t["messages"][a:b], a):
    print(f"--- [{i}] {m['role']}  content={len(m['content'] or '')}ch reasoning={len(m['reasoning_content'] or '') if m['reasoning_content'] is not None else None}")
    if m["reasoning_content"]:
        print("  R>", repr(m["reasoning_content"][:mx // 2]))
    print("  C>", repr((m["content"] or "")[:mx]))
    for tc in m["tool_calls"] or []:
        print("  TC>", tc["id"], tc["type"], tc["function"]["name"], repr(tc["function"]["arguments"][:mx // 2]))
