"""Debug one trajectory: compare prefix render vs slice of full render for every call; print placeholders.

usage: check_exact.py <parquet> <row> <tokenizer_dir>
"""
import json, os, re, sys
import pyarrow.parquet as pq
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit import conv_msgs, conv_tools, GEN_PROMPT, ASSIST_START, REDACT_RX  # noqa: E402

path, row, tokdir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
T = AutoTokenizer.from_pretrained(tokdir)
T.chat_template = open(os.path.join(tokdir, "chat_template.jinja")).read()
r = pq.read_table(path, columns=["trajectory_id", "messages", "tools"]).slice(row, 1).to_pylist()[0]
raw = json.dumps(r["messages"], ensure_ascii=False)
for m in REDACT_RX.finditer(raw):
    print("PLACEHOLDER:", repr(raw[max(0, m.start() - 160): m.end() + 120]))
msgs, _ = conv_msgs(r["messages"]); tools = conv_tools(r["tools"])
text = T.apply_chat_template(msgs, tools=tools, tokenize=False, reasoning_effort="xhigh")
ids = T(text, add_special_tokens=False)["input_ids"]
n_gen = len(T(GEN_PROMPT, add_special_tokens=False)["input_ids"])
a_idx = [j for j, m in enumerate(msgs) if m["role"] == "assistant"]
pre_full = T.apply_chat_template(msgs[:a_idx[-1]], tools=tools, tokenize=False, add_generation_prompt=True,
                                 reasoning_effort="xhigh")
bad = 0
for q, j in enumerate(a_idx):
    pref = T.apply_chat_template(msgs[:j], tools=tools, tokenize=False, add_generation_prompt=True, reasoning_effort="xhigh")
    p = T(pref, add_special_tokens=False)["input_ids"]
    if p != ids[:len(p)]:
        bad += 1
        k = next(i for i, (a, b) in enumerate(zip(p, ids)) if a != b) if any(a != b for a, b in zip(p, ids)) else min(len(p), len(ids))
        if bad <= 2:
            print(f"call {q} (msg {j}): len prefix {len(p)}, first diff at token {k}")
            print("  prefix :", repr(T.decode(p[max(0, k - 8):k + 8])))
            print("  full   :", repr(T.decode(ids[max(0, k - 8):k + 8])))
            print("  text equal as strings:", text.startswith(pref[:-len("<|im_start|>assistant\n<think>\n")]))
print("calls", len(a_idx), "mismatching", bad)
