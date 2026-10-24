"""Durable sequential sweep: smoke-test all methods, then run the full experiment."""

import argparse
import fcntl
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time


def stop_worker(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    time.sleep(2)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def boundary_prompt(root, config, count=1):
    """Artificial input only for CUDA/RoPE range validation, never a data point."""
    source = Path("data/long_context_prompts.jsonl")
    prompts = [json.loads(line) for line in source.read_text().splitlines()]
    corpus = [token for prompt in prompts for token in prompt["prompt_token_ids"]]
    length = config["max_model_len"] - 128
    ids = (corpus * (length // len(corpus) + 1))[:length]
    record = dict(domain="diagnostic", prompt_tokens=length,
                  prompt_token_ids=ids, source="artificial concatenation and repetition",
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  exclude_from_benchmark=True)
    path = root / "boundary-prompt.jsonl"
    with path.open("w") as handle:
        for index in range(count):
            handle.write(json.dumps(dict(record, id=f"boundary-diagnostic-{index:03d}")) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="benchmarks/long_context/config.json")
    parser.add_argument("--gpu", required=True, help="GPU UUID; exactly one device")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--stages", nargs="+", choices=["smoke", "boundary", "pilot", "full"])
    parser.add_argument("--variants", nargs="+", help="Optional method:budget filter, e.g. dflash:7")
    parser.add_argument("--smoke-prompts", type=int)
    parser.add_argument("--smoke-new-tokens", type=int, default=1024)
    parser.add_argument("--full-offset", type=int, help="Use 0 for a fresh full 200-prompt cohort")
    parser.add_argument("--boundary-requests", type=int, default=1)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = open(f"/tmp/specdec-{args.gpu}.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED="1")
    active = [None]

    def terminate(signum, frame):
        stop_worker(active[0])
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    config = json.loads(Path(args.config).read_text())
    pilot_count = max(4, config["max_num_seqs"])
    variants = [("ar", 0)] + [(method, budget) for budget in config["draft_budgets"]
                              for method in ["mtp", "dflash", "eagle3"]]
    # Run every budget through initialization, sampling and logging before the
    # multi-day experiment. All stages are resumable at configuration boundaries.
    stages = args.stages or (["smoke"] if args.smoke_only else ["smoke", "boundary", "pilot", "full"])
    selected = {tuple(name.split(":")) for name in args.variants} if args.variants else None
    if selected is not None and not selected <= {(m, str(k)) for m, k in variants}:
        raise ValueError(f"Unknown variants: {selected}")
    for stage in stages:
        ordered = variants if stage == "smoke" else [
            ("dflash", 7), ("mtp", 3), ("eagle3", 3), ("ar", 0),
            ("dflash", 3), ("mtp", 7), ("eagle3", 7),
        ]
        if stage == "boundary":
            ordered = [("ar", 0)] + [(method, max(config["draft_budgets"]))
                                     for method in ["mtp", "dflash", "eagle3"]]
        for method, budget in ordered:
            if selected is not None and (method, str(budget)) not in selected:
                continue
            name = f"{method}-k{budget}"
            destination = root / stage / name
            metadata_path = destination / "metadata.json"
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text())
                if metadata.get("status") == "complete":
                    print(f"SKIP complete {destination}", flush=True)
                    continue
                raise RuntimeError(f"Incomplete run exists: {destination}; inspect before resuming")
            if destination.exists() and any(destination.iterdir()):
                raise RuntimeError(f"Startup artifacts exist: {destination}; preserve them before retrying")
            usage = subprocess.check_output([
                "nvidia-smi", f"--id={args.gpu}", "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ], text=True).strip()
            memory, utilization = [int(x.strip()) for x in usage.split(",")]
            if memory > 1024 or utilization > 5:
                raise RuntimeError(f"Selected GPU is busy before start: {usage}")
            destination.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-m", "benchmarks.long_context.run", "--method", method,
                       "--budget", str(budget), "--output", str(destination),
                       "--config", args.config]
            if stage == "smoke":
                command += ["--limit", str(args.smoke_prompts or pilot_count),
                            "--max-new-tokens", str(args.smoke_new_tokens)]
            elif stage == "boundary":
                command += ["--prompts", str(boundary_prompt(root, config, args.boundary_requests)),
                            "--limit", str(args.boundary_requests),
                            "--max-new-tokens", "128"]
                if args.boundary_requests > 1:
                    command += ["--prefill-chunk-limit", str(4096 // args.boundary_requests)]
            elif stage == "pilot":
                command += ["--limit", str(pilot_count)]
            else:
                command += ["--offset", str(pilot_count if args.full_offset is None else args.full_offset)]
            state = dict(stage=stage, method=method, budget=budget, command=command,
                         started_at=time.time(), status="running")
            (root / "status.json").write_text(json.dumps(state, indent=2))
            print(f"START {stage} {name}", flush=True)
            with (destination / "console.log").open("w") as console:
                process = subprocess.Popen(command, env=env, stdout=console,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                active[0] = process
                state["worker_pid"] = process.pid
                (root / "status.json").write_text(json.dumps(state, indent=2))
                last_plot = time.monotonic()
                while process.poll() is None:
                    time.sleep(10)
                    summary_path = destination / "requests.jsonl"
                    state["completed_requests"] = len(summary_path.read_text().splitlines()) if summary_path.exists() else 0
                    progress_path = destination / "progress.json"
                    if progress_path.exists():
                        state["progress"] = json.loads(progress_path.read_text())
                    gpu_values = subprocess.check_output([
                        "nvidia-smi", f"--id={args.gpu}",
                        "--query-gpu=utilization.gpu,utilization.memory,power.draw,memory.used",
                        "--format=csv,noheader,nounits"], text=True).strip().split(",")
                    state["gpu"] = dict(zip(["utilization_percent", "memory_utilization_percent",
                                              "power_watts", "allocated_mib"],
                                             map(float, gpu_values)))
                    with (destination / "gpu.jsonl").open("a") as handle:
                        handle.write(json.dumps({"unix_time": time.time(), **state["gpu"]}) + "\n")
                    state["heartbeat_at"] = time.time()
                    (root / "status.json").write_text(json.dumps(state, indent=2))
                    if stage != "boundary" and time.monotonic() - last_plot >= 600:
                        paths = (sorted((root / stage).glob("*/metadata.json")) if stage == "smoke"
                                 else sorted(root.glob("pilot/*/metadata.json")) + sorted(root.glob("full/*/metadata.json")))
                        plot_dir = root / "smoke" / "plots" if stage == "smoke" else root / "plots"
                        subprocess.run([sys.executable, "-m", "benchmarks.long_context.plot",
                                        *[str(p.parent) for p in paths], "--output",
                                        str(plot_dir), "--include-partial"],
                                       stdout=console, stderr=subprocess.STDOUT, check=False)
                        last_plot = time.monotonic()
            if process.returncode:
                stop_worker(process)
                state.update(status="failed", returncode=process.returncode, finished_at=time.time())
                (root / "status.json").write_text(json.dumps(state, indent=2))
                raise RuntimeError(f"Failed {destination}; see console.log")
            stop_worker(process)
            active[0] = None
            subprocess.run([sys.executable, "-m", "benchmarks.long_context.validate",
                            str(destination)], check=True)
            paths = (sorted((root / stage).glob("*/metadata.json")) if stage == "smoke"
                     else sorted(root.glob("pilot/*/metadata.json")) + sorted(root.glob("full/*/metadata.json")))
            plot_dir = root / "smoke" / "plots" if stage == "smoke" else root / "plots"
            if stage != "boundary":
                subprocess.run([sys.executable, "-m", "benchmarks.long_context.plot",
                                *[str(p.parent) for p in paths], "--output", str(plot_dir)],
                               check=True)
            print(f"COMPLETE {stage} {name}", flush=True)
            # A reversible boundary stop lets an operator migrate the controller
            # to a frozen checkout without interrupting an active GPU run.
            if (root / "STOP_AFTER_CONFIGURATION").exists():
                state.update(status="stopped_at_configuration_boundary", finished_at=time.time())
                (root / "status.json").write_text(json.dumps(state, indent=2))
                return
    (root / "status.json").write_text(json.dumps({"status": "complete", "finished_at": time.time()}))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if "--output" in sys.argv:
            path = Path(sys.argv[sys.argv.index("--output") + 1]) / "status.json"
            if path.exists():
                state = json.loads(path.read_text())
                state.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit))
                             else "failed", error=repr(error), finished_at=time.time())
                path.write_text(json.dumps(state, indent=2))
        raise
