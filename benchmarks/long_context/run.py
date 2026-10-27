"""Run one method/budget in an isolated vLLM process on one explicitly selected GPU."""

import argparse
from contextlib import ExitStack
import gzip
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="benchmarks/long_context/config.json")
    parser.add_argument("--prompts", default="data/long_context_prompts.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", choices=["ar", "mtp", "dflash", "eagle2", "eagle3"], required=True)
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--prefill-chunk-limit", type=int, default=0,
                        help="Diagnostic fair-share prefill cap; 0 keeps vLLM defaults")
    parser.add_argument("--gpu-memory-utilization", type=float)
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "Select exactly one GPU explicitly"
    assert "," not in os.environ["CUDA_VISIBLE_DEVICES"]
    config = json.loads(Path(args.config).read_text())
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metadata.json").exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    source_snapshot = output / "source_snapshot"
    source_snapshot.mkdir()
    source_hashes = {}
    for source in Path(__file__).parent.iterdir():
        if source.is_file() and source.suffix in (".py", ".json", ".txt", ".md"):
            shutil.copy2(source, source_snapshot / source.name)
            source_hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    os.environ["VLLM_COMPUTE_NANS_IN_LOGITS"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["SPECDEC_TRACE_PATH"] = str(output / "rounds.jsonl.gz")
    os.environ["SPECDEC_POSITION_CACHE_LEN"] = str(config["max_model_len"])
    os.environ["SPECDEC_MIN_KV_CAPACITY"] = str(
        config["max_num_seqs"] * config.get("min_kv_capacity_factor", 0))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    from transformers import AutoTokenizer, GenerationConfig
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from benchmarks.long_context.positions import extend_position_cache

    tokenizer = AutoTokenizer.from_pretrained(config["target"], revision=config["target_revision"])
    gen_config = GenerationConfig.from_pretrained(config["target"], revision=config["target_revision"])
    eos_ids = gen_config.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    eos_ids = sorted(set((eos_ids or []) + [tokenizer.eos_token_id]))
    os.environ["SPECDEC_EOS_IDS"] = json.dumps(eos_ids)
    spec = None
    adaptation = None
    if args.method != "ar":
        spec = dict(config["drafters"][args.method])
        if args.method == "eagle3" and spec.get("model", "").startswith("Dogacel/specdrift-"):
            from benchmarks.long_context.checkpoints import adapt_eagle_checkpoint
            spec["model"], adaptation = adapt_eagle_checkpoint(spec)
            spec.pop("revision")
        spec.update(method="eagle" if args.method == "eagle2" else args.method,
                    num_speculative_tokens=args.budget,
                    max_model_len=config["max_model_len"])
    engine_args = dict(
        model=config["target"], revision=config["target_revision"],
        tokenizer_revision=config["target_revision"], dtype=config["dtype"],
        tensor_parallel_size=1, max_model_len=config["max_model_len"],
        max_num_seqs=config["max_num_seqs"], max_num_batched_tokens=config.get("max_num_batched_tokens", 4096),
        gpu_memory_utilization=(args.gpu_memory_utilization if args.gpu_memory_utilization is not None
                                else config.get("gpu_memory_utilization", 0.85)),
        enable_prefix_caching=False, async_scheduling=config.get("async_scheduling", False),
        language_model_only=True, speculative_config=spec,
        scheduler_cls=("benchmarks.long_context.trace.TracingAsyncScheduler"
                       if config.get("async_scheduling", False)
                       else "benchmarks.long_context.trace.TracingScheduler"),
        seed=config["seed"], enforce_eager=args.enforce_eager,
        disable_log_stats=False, generation_config="vllm",
        hf_overrides=extend_position_cache,
    )
    if args.prefill_chunk_limit:
        engine_args["long_prefill_token_threshold"] = args.prefill_chunk_limit
        engine_args["long_prefill_token_threshold_adaptive"] = False
    metadata = {
        "config": config, "args": vars(args),
        "engine_args": {**engine_args, "hf_overrides": "benchmarks.long_context.positions.extend_position_cache"},
        "source_sha256": source_hashes,
        "checkpoint_adaptation": adaptation,
        "sampling_eos_bad_words": ([tokenizer.convert_ids_to_tokens(i) for i in eos_ids]
                                   if config.get("eos_policy") == "bad_words" else []),
        "eos_token_ids": eos_ids, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "environment": {key: os.environ.get(key) for key in
                        ["VLLM_ALLOW_LONG_MAX_MODEL_LEN", "VLLM_COMPUTE_NANS_IN_LOGITS",
                         "HF_HUB_OFFLINE", "CUDA_HOME", "SPECDEC_POSITION_CACHE_LEN",
                         "SPECDEC_MIN_KV_CAPACITY"]},
        "packages": {p: importlib.metadata.version(p) for p in
                     ["vllm", "torch", "transformers", "huggingface_hub"]},
        "prompt_file_sha256": hashlib.sha256(Path(args.prompts).read_bytes()).hexdigest(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,driver_version", "--format=csv"], text=True),
        "started_at": time.time(), "status": "initializing",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    llm = LLM(**engine_args)
    (output / "effective_vllm_config.txt").write_text(str(llm.llm_engine.vllm_config))
    effective = llm.llm_engine.vllm_config
    (output / "effective_target_hf_config.json").write_text(
        json.dumps(effective.model_config.hf_config.to_dict(), indent=2)
    )
    if effective.speculative_config is not None:
        (output / "effective_draft_hf_config.json").write_text(
            json.dumps(effective.speculative_config.draft_model_config.hf_config.to_dict(), indent=2)
        )
    prompts = [json.loads(line) for line in Path(args.prompts).read_text().splitlines()]
    prompts = prompts[args.offset:]
    if args.limit:
        prompts = prompts[:args.limit]
    metadata["status"] = "running"
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    if config.get("batching", "fixed") == "continuous":
        from benchmarks.long_context.continuous import run_continuous
        run_continuous(llm, tokenizer, prompts, config, args, eos_ids, output)
        llm.llm_engine.engine_core.shutdown()
        metadata.update(status="complete", finished_at=time.time())
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
        return
    summaries = output / "requests.jsonl"
    concurrency = config["max_num_seqs"]
    workload_started = time.perf_counter()
    workload_tokens = 0
    workload_finished = workload_started
    for batch_start in range(0, len(prompts), concurrency):
        batch = prompts[batch_start:batch_start + concurrency]
        states = {}
        lookup = {}
        batch_started = time.perf_counter()
        with ExitStack() as stack:
            for prompt in batch:
                request_id = prompt["id"]
                prompt_ids = prompt["prompt_token_ids"]
                max_tokens = config["max_model_len"] - len(prompt_ids)
                if args.max_new_tokens:
                    max_tokens = min(max_tokens, args.max_new_tokens)
                sampling = SamplingParams(
                    temperature=config["temperature"], top_p=1.0, top_k=-1,
                    seed=config["seed"], ignore_eos=True, max_tokens=max_tokens,
                    detokenize=False, output_kind=RequestOutputKind.DELTA,
                    bad_words=([tokenizer.convert_ids_to_tokens(i) for i in eos_ids]
                               if config.get("eos_policy") == "bad_words" else None),
                )
                state = dict(prompt=prompt, generated_ids=[], started=time.perf_counter(),
                             first_token_time=None, finished=None, finish_reason=None,
                             max_tokens=max_tokens)
                state["stream"] = stack.enter_context(gzip.open(
                    output / f"{request_id}.tokens.jsonl.gz", "wt", compresslevel=1))
                states[request_id] = state
                print(f"START {request_id} input={len(prompt_ids)} max_new={max_tokens}", flush=True)
                internal_id = llm.llm_engine.add_request(
                    request_id, {"prompt_token_ids": prompt_ids}, sampling)
                state["engine_request_id"] = internal_id
                lookup[request_id] = lookup[internal_id] = state
                with (output / "requests.started.jsonl").open("a") as handle:
                    handle.write(json.dumps({"id": request_id, "engine_request_id": internal_id,
                                             "domain": prompt["domain"], "max_new_tokens": max_tokens,
                                             "batch_size": len(batch)}) + "\n")
            while llm.llm_engine.has_unfinished_requests():
                for result in llm.llm_engine.step():
                    state = lookup[result.request_id]
                    for completion in result.outputs:
                        tokens = list(completion.token_ids)
                        if tokens:
                            now = time.perf_counter()
                            state["first_token_time"] = state["first_token_time"] or now
                            state["generated_ids"].extend(tokens)
                            state["stream"].write(json.dumps({
                                "elapsed": now - state["started"], "token_ids": tokens}) + "\n")
                        if completion.finish_reason:
                            state["finish_reason"] = completion.finish_reason
                            state["finished"] = time.perf_counter()
        batch_elapsed = time.perf_counter() - batch_started
        workload_finished = max(s["finished"] for s in states.values())
        with (output / "batches.jsonl").open("a") as handle:
            total = sum(len(s["generated_ids"]) for s in states.values())
            workload_tokens += total
            handle.write(json.dumps({"ids": list(states), "seconds": batch_elapsed,
                                     "generated_tokens": total,
                                     "aggregate_tokens_per_second": total / batch_elapsed}) + "\n")
        for request_id, state in states.items():
            prompt = state["prompt"]
            generated_ids = state["generated_ids"]
            assert state["finished"] is not None, f"{request_id}: missing finish event"
            elapsed = state["finished"] - state["started"]
            first_eos = next((i for i, token in enumerate(generated_ids) if token in eos_ids), None)
            with gzip.open(output / f"{request_id}.text.txt.gz", "wt") as handle:
                handle.write(tokenizer.decode(generated_ids, skip_special_tokens=False))
            summary = {
                "id": request_id, "engine_request_id": state["engine_request_id"],
                "domain": prompt["domain"], "prompt_tokens": prompt["prompt_tokens"],
                "generated_tokens": len(generated_ids), "final_context": prompt["prompt_tokens"] + len(generated_ids),
                "max_new_tokens": state["max_tokens"], "seconds": elapsed,
                "tokens_per_second": len(generated_ids) / elapsed, "batch_size": len(batch),
                "ttft_seconds": state["first_token_time"] - state["started"] if state["first_token_time"] else None,
                "first_eos_output_index": first_eos, "finish_reason": state["finish_reason"],
                "output_sha256": hashlib.sha256(json.dumps(generated_ids).encode()).hexdigest(),
                "completed_requested_length": len(generated_ids) == state["max_tokens"],
            }
            with summaries.open("a") as handle:
                handle.write(json.dumps(summary) + "\n")
            print(json.dumps(summary), flush=True)
            if not summary["completed_requested_length"]:
                raise RuntimeError(f"{request_id}: stopped before requested length")
    workload_seconds = workload_finished - workload_started
    (output / "workload.json").write_text(json.dumps(dict(
        batching="fixed", requests=len(prompts), generated_tokens=workload_tokens,
        seconds=workload_seconds, aggregate_tokens_per_second=workload_tokens/workload_seconds,
    ), indent=2))
    llm.llm_engine.engine_core.shutdown()
    metadata["status"] = "complete"
    metadata["finished_at"] = time.time()
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if "--output" in sys.argv:
            path = Path(sys.argv[sys.argv.index("--output") + 1]) / "metadata.json"
            if path.exists():
                metadata = json.loads(path.read_text())
                metadata.update(status="failed", error=repr(error), finished_at=time.time())
                path.write_text(json.dumps(metadata, indent=2))
        raise
