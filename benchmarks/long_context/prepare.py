"""Freeze real, disjoint-domain prompts and their source revisions."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer


SOURCES = {
    "math": ("HuggingFaceH4/MATH-500", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"),
    "code": ("openai/openai_humaneval", "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544"),
    "chat": ("HuggingFaceH4/ultrachat_200k", "8049631c405ae6576f93f445c6b8166f76f5505a"),
    "reasoning": ("lukaemon/bbh", "982bb89fd79532a8ac676a61fc42eb1aeec63f99"),
}
REASONING_TASKS = [
    "causal_judgement", "disambiguation_qa", "formal_fallacies",
    "logical_deduction_five_objects", "logical_deduction_seven_objects",
    "logical_deduction_three_objects", "tracking_shuffled_objects_three_objects",
    "tracking_shuffled_objects_five_objects", "tracking_shuffled_objects_seven_objects",
    "web_of_lies",
]


def load(domain, filename):
    repo, revision = SOURCES[domain]
    path = hf_hub_download(repo, filename, repo_type="dataset", revision=revision)
    if filename.endswith("jsonl"):
        return [json.loads(line) for line in Path(path).read_text().splitlines()]
    return pq.read_table(path).to_pylist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="benchmarks/long_context/config.json")
    parser.add_argument("--output", default="data/long_context_prompts.jsonl")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    rng = random.Random(config["seed"])
    count = config["prompts_per_domain"]
    tokenizer = AutoTokenizer.from_pretrained(
        config["target"], revision=config["target_revision"]
    )
    candidates = {}
    candidates["math"] = [
        (str(i), [{"role": "user", "content": row["problem"]}], "test")
        for i, row in enumerate(load("math", "test.jsonl"))
    ]
    candidates["code"] = [
        (row["task_id"], [{"role": "user", "content":
          "Complete the following Python function.\n\n" + row["prompt"]}], "test")
        for row in load("code", "openai_humaneval/test-00000-of-00001.parquet")
    ]
    # Use held-out synthetic UltraChat conversations before a later assistant
    # turn, preserving conversation history. Exclude explicit math/code prompts.
    chat_rows = load("chat", "data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet")
    rng.shuffle(chat_rows)
    forbidden = ("python", "javascript", "coding", "programming", "equation",
                 "calculate", "algorithm", "sql", "function", "mathematical")
    candidates["chat"] = []
    for row in chat_rows:
        messages = row["messages"]
        user_turns = [i for i, m in enumerate(messages) if m["role"] == "user"]
        if len(user_turns) < 2:
            continue
        end = user_turns[min(1, len(user_turns) - 1)]
        context = messages[:end + 1]
        text = " ".join(m["content"] for m in context).lower()
        if any(word in text for word in forbidden):
            continue
        candidates["chat"].append((str(row["prompt_id"]), context, "test_sft"))
        if len(candidates["chat"]) >= count * 10:
            break
    candidates["reasoning"] = []
    # Equal representation from ten non-arithmetic reasoning tasks.
    assert count % len(REASONING_TASKS) == 0
    for task in REASONING_TASKS:
        rows = load("reasoning", f"{task}/test-00000-of-00001.parquet")
        for i in rng.sample(range(len(rows)), count // len(REASONING_TASKS)):
            candidates["reasoning"].append((
                f"{task}/{i}", [{"role": "user", "content": rows[i]["input"]}],
                f"{task}/test",
            ))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    selected = {domain: rng.sample(rows, count) for domain, rows in candidates.items()}
    seen = set()
    with output.open("w") as handle:
        # Interleave domains so partial runs still cover all four domains.
        for index in range(count):
            for domain, rows in selected.items():
                source_id, messages, split = rows[index]
                ids = tokenizer.apply_chat_template(
                    messages, tokenize=True, return_dict=False, add_generation_prompt=True,
                    enable_thinking=config["enable_thinking"],
                )
                digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
                assert digest not in seen, "duplicate prompt"
                seen.add(digest)
                record = {
                    "id": f"{domain}-{index:03d}", "domain": domain,
                    "source": SOURCES[domain][0], "revision": SOURCES[domain][1],
                    "source_id": source_id, "split": split, "messages": messages,
                    "prompt_token_ids": ids, "prompt_tokens": len(ids),
                    "sha256": digest,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {"seed": config["seed"], "count": len(seen),
                "sources": SOURCES, "target_revision": config["target_revision"],
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
