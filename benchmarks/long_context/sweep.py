"""Durable sequential sweep: smoke-test all methods, then run the full experiment."""

import argparse
import fcntl
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="benchmarks/long_context/config.json")
    parser.add_argument("--gpu", required=True, help="GPU UUID; exactly one device")
    parser.add_argument("--smoke-only", action="store_true")
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
    stages = ["smoke"] if args.smoke_only else ["smoke", "pilot", "full"]
    for stage in stages:
        ordered = variants if stage == "smoke" else [
            ("dflash", 7), ("mtp", 3), ("eagle3", 3), ("ar", 0),
            ("dflash", 3), ("mtp", 7), ("eagle3", 7),
        ]
        for method, budget in ordered:
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
                command += ["--limit", str(pilot_count), "--max-new-tokens", "1024"]
            elif stage == "pilot":
                command += ["--limit", str(pilot_count)]
            else:
                command += ["--offset", str(pilot_count)]
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
                    state["heartbeat_at"] = time.time()
                    (root / "status.json").write_text(json.dumps(state, indent=2))
                    if time.monotonic() - last_plot >= 600:
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
    main()
