"""Run independent TP1 configurations on explicitly selected GPUs.

Each stage is a barrier: all smoke/boundary checks must pass before full runs.
No timed automatic retries, overwrites, or plotting inside the GPU monitor.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
from queue import Queue, Empty
import signal
import subprocess
import sys
import threading
import time

from benchmarks.long_context.sweep import stop_worker


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--gpus", nargs=2, required=True)
    parser.add_argument("--stages", nargs="+", default=["smoke", "boundary", "full"],
                        choices=["smoke", "boundary", "full"])
    parser.add_argument("--methods", nargs="+", default=["dflash", "eagle3", "ar"])
    parser.add_argument("--smoke-tokens", type=int, default=2048)
    args = parser.parse_args()
    assert len(set(args.gpus)) == 2
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text())
    locks = []
    for gpu in args.gpus:
        lock = open(f"/tmp/specdec-{gpu}.lock", "w")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    active = {}
    abort = threading.Event()

    def terminate(signum, frame):
        abort.set()
        for process in list(active.values()):
            stop_worker(process)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    records = [json.loads(line) for line in Path(args.prompts).read_text().splitlines()]
    corpus = [t for r in records for t in r["prompt_token_ids"]]
    length = config["max_model_len"] - 128
    ids = (corpus * (length // len(corpus) + 1))[:length]
    boundary = root / "boundary-prompts.jsonl"
    boundary.write_text("".join(json.dumps(dict(id=f"boundary-{i:03d}", domain="diagnostic",
        prompt_token_ids=ids, prompt_tokens=length, exclude_from_benchmark=True,
        source="Artificial repetition; capacity/range test only")) + "\n"
        for i in range(config["max_num_seqs"])))

    def lane(gpu, pending, stage):
        while True:
            try:
                method = pending.get_nowait()
            except Empty:
                return
            if abort.is_set():
                return
            budget = 0 if method == "ar" else config["draft_budgets"][0]
            destination = root / stage / f"{method}-k{budget}"
            if destination.exists() and any(destination.iterdir()):
                raise RuntimeError(f"Refusing to overwrite {destination}")
            memory, utilization = map(int, subprocess.check_output([
                "nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits"], text=True).strip().split(","))
            if memory > 1024 or utilization > 5:
                raise RuntimeError(f"GPU busy: {gpu}: {memory} MiB, {utilization}%")
            destination.mkdir(parents=True)
            command = [sys.executable, "-m", "benchmarks.long_context.run", "--config", args.config,
                "--prompts", str(boundary) if stage == "boundary" else args.prompts,
                "--output", str(destination), "--method", method, "--budget", str(budget)]
            if stage == "smoke":
                command += ["--limit", str(config["max_num_seqs"] + 4),
                            "--max-new-tokens", str(args.smoke_tokens)]
            elif stage == "boundary":
                slots = budget if method == "dflash" else 0
                chunk = config.get("max_num_batched_tokens", 4096) // config["max_num_seqs"] - slots
                command += ["--max-new-tokens", "128", "--prefill-chunk-limit", str(chunk)]
            state = dict(stage=stage, method=method, gpu=gpu, command=command,
                         status="running", started_at=time.time())
            status_path = root / f"status-{gpu}.json"
            with (destination / "console.log").open("w") as console:
                process = subprocess.Popen(command, stdout=console, stderr=subprocess.STDOUT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1"),
                    start_new_session=True)
                active[gpu] = process
                state["worker_pid"] = process.pid
                write_json(status_path, state)
                print(f"START {stage} {method} {gpu} pid={process.pid}", flush=True)
                while process.poll() is None:
                    if abort.wait(10):
                        stop_worker(process)
                        return
                    progress = destination / "progress.json"
                    if progress.exists():
                        state["progress"] = json.loads(progress.read_text())
                    summary = destination / "requests.jsonl"
                    state["completed_requests"] = len(summary.read_text().splitlines()) if summary.exists() else 0
                    values = subprocess.check_output(["nvidia-smi", f"--id={gpu}",
                        "--query-gpu=utilization.gpu,power.draw,memory.used", "--format=csv,noheader,nounits"], text=True)
                    sample = dict(zip(["utilization_percent", "power_watts", "allocated_mib"], map(float, values.split(","))))
                    sample["unix_time"] = time.time()
                    with (destination / "gpu.jsonl").open("a") as handle:
                        handle.write(json.dumps(sample) + "\n")
                    state.update(gpu_sample=sample, heartbeat_at=time.time())
                    write_json(status_path, state)
                stop_worker(process)
                active.pop(gpu, None)
                state.update(status="complete" if process.returncode == 0 else "failed",
                             returncode=process.returncode, finished_at=time.time())
                write_json(status_path, state)
                if process.returncode:
                    raise RuntimeError(f"Failed {destination}: exit {process.returncode}")
                subprocess.run([sys.executable, "-m", "benchmarks.long_context.validate", str(destination)],
                               stdout=console, stderr=subprocess.STDOUT, check=True)
                if stage == "boundary":
                    progress = json.loads((destination / "progress.json").read_text())
                    required = config["max_num_seqs"] * (config["max_model_len"] - 256)
                    assert progress["peak_running_computed_tokens"] >= required, progress
            print(f"COMPLETE {stage} {method}", flush=True)

    try:
        for stage in args.stages:
            write_json(root / "pipeline-status.json", dict(status="running", stage=stage,
                       controller_pid=os.getpid(), updated_at=time.time()))
            with ThreadPoolExecutor(max_workers=2) as executor:
                pending = Queue()
                for method in args.methods:
                    pending.put(method)
                futures = [executor.submit(lane, gpu, pending, stage) for gpu in args.gpus]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except BaseException:
                        abort.set()
                        raise
        if "full" in args.stages:
            paths = sorted((root / "full").glob("*/metadata.json"))
            subprocess.run([sys.executable, "-m", "benchmarks.long_context.plot",
                *[str(p.parent) for p in paths], "--output", str(root / "plots"), "--bin-size", "2048"], check=True)
        write_json(root / "pipeline-status.json", dict(status="complete", stages=args.stages, finished_at=time.time()))
    except BaseException as error:
        abort.set()
        for process in list(active.values()):
            stop_worker(process)
        write_json(root / "pipeline-status.json", dict(status="failed", error=repr(error), finished_at=time.time()))
        raise


if __name__ == "__main__":
    main()
