"""Light survey of ALL v1.2 trajectories using only small leaf columns (remote range reads).

usage: survey_light.py <parquet_meta.json> <rev> <out_dir>
writes: light.parquet (one row per trajectory), tools_variants.json
"""
import collections, hashlib, json, os, sys
from concurrent.futures import ThreadPoolExecutor
import pyarrow as pa, pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

meta, rev, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
fs = HfFileSystem()
COLS = ["instance_id", "repo", "language", "trajectory_id", "resolved", "hf_dataset_name",
        "messages.list.element.role",
        "messages.list.element.tool_calls.list.element.function.name",
        "messages.list.element.tool_calls.list.element.id",
        "tools.list.element",
        "metadata.category", "metadata.teacher_model.name", "metadata.teacher_model.enable_thinking",
        "metadata.teacher_model.reasoning_effort",
        "metadata.model_patch.num_modified_lines", "metadata.reference_patch.num_modified_lines"]

tools_variants = {}


def one(p):
    with fs.open(f"datasets/nvidia/Open-SWE-Traces@{rev}/{p}", "rb", block_size=4 << 20) as f:
        t = pq.ParquetFile(f).read(columns=COLS).to_pylist()
    harness, source = p.split("/")[1], p.split("/")[3]
    rows = []
    for i, r in enumerate(t):
        msgs = r["messages"]
        roles = [m["role"] for m in msgs]
        names = collections.Counter()
        ids = []
        for m in msgs:
            for tc in (m.get("tool_calls") or []):
                names[tc["function"]["name"]] += 1
                ids.append(tc["id"])
        tools = r["tools"] or []
        th = hashlib.sha1(json.dumps(tools).encode()).hexdigest()[:12]
        tools_variants.setdefault((harness, th), (len(tools), tools))
        tm = r["metadata"]["teacher_model"] or {}
        rows.append(dict(
            harness=harness, source=source, file=p, row=i, trajectory_id=r["trajectory_id"],
            instance_id=r["instance_id"], repo=r["repo"], language=r["language"], resolved=r["resolved"],
            hf_dataset_name=r["hf_dataset_name"], category=r["metadata"]["category"],
            teacher=tm.get("name"), enable_thinking=tm.get("enable_thinking"), effort=tm.get("reasoning_effort"),
            n_msgs=len(msgs), n_assistant=roles.count("assistant"), n_tool=roles.count("tool"),
            n_user=roles.count("user"), n_system=roles.count("system"),
            first_roles=",".join(roles[:3]), last_role=roles[-1] if roles else None,
            role_set=",".join(sorted(set(roles))),
            n_tool_calls=len(ids), n_tool_call_ids_unique=len(set(ids)),
            n_tool_call_id_empty=sum(1 for x in ids if not x),
            tool_names=json.dumps(dict(names)), n_tools_schemas=len(tools), tools_hash=th,
            model_patch_lines=(r["metadata"]["model_patch"] or {}).get("num_modified_lines"),
        ))
    return rows


files = [x["path"] for x in json.load(open(meta))]
with ThreadPoolExecutor(8) as ex:
    allrows = [row for rows in ex.map(one, files) for row in rows]
pq.write_table(pa.Table.from_pylist(allrows), f"{out}/light.parquet")
json.dump([dict(harness=h, hash=k, n=n, tools=t) for (h, k), (n, t) in tools_variants.items()],
          open(f"{out}/tools_variants.json", "w"), indent=1)
print(len(allrows), "rows;", len(tools_variants), "tool-schema variants")
