"""Inspect anomalous trajectories. usage: inspect_anomalies.py <shards_root> <tokenizer_dir> <spec.json>
spec: list of {"kind": ..., "harness": ..., "source": ..., "file": ..., "row": ...}
kinds: big_tools (largest tool outputs), long_output (largest assistant turn), mid_user (user messages after the first)
"""
import json, os, sys
import pyarrow.parquet as pq
from transformers import AutoTokenizer

root, tokdir, spec = sys.argv[1], sys.argv[2], json.load(open(sys.argv[3]))
T = AutoTokenizer.from_pretrained(tokdir)
ntok = lambda s: len(T(s or "", add_special_tokens=False)["input_ids"])
cut = lambda s, n=300: repr((s or "")[:n]) + (f" …[{len(s)} ch]… " + repr(s[-n:]) if s and len(s) > 2 * n else "")

for sp in spec:
    p = f"{root}/data/{sp['harness']}/qwen38_27b/{sp['source']}/{sp['file']}"
    r = pq.read_table(p, columns=["trajectory_id", "messages"]).slice(sp["row"], 1).to_pylist()[0]
    ms = r["messages"]
    print("=" * 100); print(sp["kind"], r["trajectory_id"], sp["harness"], sp["source"], "n_msgs", len(ms))
    if sp["kind"] == "big_tools":
        sizes = sorted(((ntok(m["content"]), i) for i, m in enumerate(ms) if m["role"] == "tool"), reverse=True)
        print("tool msgs", len(sizes), "top sizes", sizes[:8], "sum", sum(s for s, _ in sizes))
        for s, i in sizes[:2]:
            prev = ms[i - 1] if ms[i - 1]["role"] == "assistant" else None
            k = i - 1
            while ms[k]["role"] != "assistant": k -= 1
            print(f"-- tool msg {i}: {s} tok; call: {[tc['function']['arguments'][:200] for tc in ms[k]['tool_calls'] or []]}")
            print("   ", cut(ms[i]["content"], 400))
    elif sp["kind"] == "long_output":
        sizes = sorted(((ntok(m["reasoning_content"]) + ntok(m["content"]), i) for i, m in enumerate(ms) if m["role"] == "assistant"), reverse=True)
        s, i = sizes[0]
        m = ms[i]
        print(f"-- assistant msg {i}: ~{s} tok; reasoning {len(m['reasoning_content'] or '')} ch, content {len(m['content'] or '')} ch, "
              f"tool_calls {[tc['function']['name'] for tc in m['tool_calls'] or []]}")
        print("   reasoning tail:", repr((m["reasoning_content"] or "")[-600:]))
        print("   content:", cut(m["content"], 300))
        for tc in m["tool_calls"] or []:
            print("   tool_call args:", tc["function"]["arguments"][:300])
        if i + 1 < len(ms):
            print(f"   next msg [{ms[i+1]['role']}]:", cut(ms[i + 1]["content"], 300))
        # are there other near-cap turns?
        print("   top-5 assistant sizes:", sizes[:5])
    elif sp["kind"] == "mid_user":
        for i, m in enumerate(ms):
            if m["role"] == "user" and i > 1:
                print(f"-- user msg {i} (prev role {ms[i-1]['role']}):", cut(m["content"], 500))
                pm = ms[i - 1]
                if pm["role"] == "assistant":
                    print("   prev assistant content:", cut(pm["content"], 300), "tool_calls:", len(pm["tool_calls"] or []))
