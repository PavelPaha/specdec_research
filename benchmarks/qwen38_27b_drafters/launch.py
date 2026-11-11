"""Run specbench on one machine: split the visible GPUs into tensor-parallel groups and keep every
group busy with (method, shard) jobs until all are done. Finished requests are skipped on restart
(run.py resumes), so the command can simply be repeated after an interruption.

  python launch.py --tp 2 --methods mtp dflash2 -- \
      --config configs/ref.json --prompts data/prompts/forced/{method}.jsonl \
      --output results/forced2/{method} --ref results/ref/ar --kv-fill 0.92 --max-num-seqs 48

Everything after `--` is passed to `python -m specbench.run` ({method} is substituted);
--shards defaults to the number of GPU groups, so each method is spread over all of them.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def gpus():
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return [g for g in env.split(",") if g.strip()]
    out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
    return [str(i) for i, line in enumerate(out.splitlines()) if line.startswith("GPU ")]


def main():
    argv = sys.argv[1:]
    rest = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[: argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=2, help="GPUs per vLLM instance")
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--shards", type=int, help="shards per method (default: number of GPU groups)")
    ap.add_argument("--logs", default="logs")
    a = ap.parse_args(argv)
    devices = gpus()
    groups = [",".join(devices[i:i + a.tp]) for i in range(0, len(devices) - a.tp + 1, a.tp)]
    if not groups:
        sys.exit(f"need at least {a.tp} GPUs, found {len(devices)}")
    shards = a.shards or len(groups)
    # interleave methods so that all groups start with different work
    jobs = [(m, k) for k in range(shards) for m in a.methods]
    Path(a.logs).mkdir(parents=True, exist_ok=True)
    running, failed = {}, []
    print(f"{len(groups)} GPU groups x TP{a.tp}, {len(jobs)} jobs", flush=True)
    while jobs or running:
        for g in groups:
            if g not in running and jobs:
                m, k = jobs.pop(0)
                args = [x.replace("{method}", m) for x in rest]
                cmd = [sys.executable, "-m", "specbench.run", "--method", m,
                       "--shard", str(k), "--num-shards", str(shards), *args]
                log = open(Path(a.logs) / f"{m}-s{k}of{shards}.log", "a")
                running[g] = (m, k, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                                     env=dict(os.environ, CUDA_VISIBLE_DEVICES=g)))
                print(f"start {m} shard {k} on GPUs {g}", flush=True)
        for g, (m, k, p) in list(running.items()):
            if p.poll() is not None:
                del running[g]
                print(f"{'done' if p.returncode == 0 else 'FAILED'} {m} shard {k} (rc={p.returncode})", flush=True)
                if p.returncode:
                    failed.append((m, k))
        time.sleep(10)
    if failed:
        sys.exit(f"failed: {failed}; rerun the same command to resume")


if __name__ == "__main__":
    main()
