#!/usr/bin/env python
"""Sequential run queue (PLAN §8 fase 3).

    .venv/bin/python scripts/run_queue.py queues/fase3_grid.txt [--epochs 100] [--var LR=5e-4 ...]

Queue file: one run per line, `<config> [overrides...] --out runs/<name>_seed<k>`; blank lines and
`#` comments are ignored. `NAME=VALUE` lines (before the runs) define variables; `{NAME}` in a run
line is replaced (e.g. `--epochs {EPOCHS}`). CLI --epochs / --var override the file's values.

Per run: skipped if <out>/evaluation.json exists; waits (polling 60 s, never kills) while a foreign
process uses the GPU; stops with a warning if free disk < 10 GB; stdout/stderr go to
<out>/train.log; a failing run is recorded in logs/queue.log and the queue continues.

NOTE: deliberately NOT named queue.py: a scripts/queue.py shadows the stdlib `queue` module for
every other script in scripts/ (scripts/train.py then crashes inside torch with an ImportError).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import shlex  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402

MIN_FREE_GB = 10.0
POLL_S = 60


def parse_queue(text, overrides=None):
    """-> (variables, [argv lists]). Raises ValueError on a bad/unresolved line."""
    vars_, runs = {}, []
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.fullmatch(r"([A-Z][A-Z0-9_]*)=(\S+)", line)
        if m:
            vars_[m.group(1)] = m.group(2)
            continue
        runs.append((n, line))
    vars_.update(overrides or {})
    out = []
    for n, line in runs:
        for k, v in vars_.items():
            line = line.replace("{" + k + "}", str(v))
        left = re.findall(r"\{[A-Z0-9_]+\}", line)
        if left:
            raise ValueError(f"line {n}: unresolved placeholder(s) {left}")
        argv = shlex.split(line)
        if "--out" not in argv:
            raise ValueError(f"line {n}: missing --out")
        out.append(argv)
    return vars_, out


def out_dir_of(argv):
    return Path(argv[argv.index("--out") + 1])


def is_done(argv, root=ROOT):
    return (root / out_dir_of(argv) / "evaluation.json").exists()


def check_conflict(argv, root=ROOT):
    """Compare the line's resolved config with the stored evaluation.json (trainer's own identity
    check: seed, eval_split, source, epochs, lr, batch size, lambda, ema, reg, smoke).
    Returns None if identical, else a message. Unresolvable lines count as conflicts."""
    root = Path(root)
    try:
        sys.path.insert(0, str(ROOT / "src"))
        from ijepa_sphere import trainer
        cfg_path = Path(argv[0])
        a = trainer.parse_args(["--config", str(cfg_path if cfg_path.is_absolute() else ROOT / cfg_path)]
                               + list(argv[1:]))
        resolved = trainer.resolve_config(a)
        stored = json.load(open(root / out_dir_of(argv) / "evaluation.json")).get("config", {})
        trainer.check_same_run(resolved, stored, out_dir_of(argv) / "evaluation.json")
    except RuntimeError as e:
        return str(e).replace("\n", " ")
    except (SystemExit, Exception) as e:  # noqa: BLE001
        return f"could not compare configs: {type(e).__name__}: {e}"
    return None


def gpu_busy():
    """List of GPU compute processes (any process is foreign: we run one job at a time)."""
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name",
                        "--format=csv,noheader"], capture_output=True, text=True)
    return [x for x in r.stdout.splitlines() if x.strip()] if r.returncode == 0 else []


def free_gb(path):
    return shutil.disk_usage(path).free / 2**30


class Log:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg):
        line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
        print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")


def run_queue(runs, root=ROOT, log=None, train_cmd=None, gpu_busy_fn=gpu_busy, free_gb_fn=free_gb,
              poll_s=POLL_S, min_free_gb=MIN_FREE_GB, sleep=time.sleep, conflict_fn=check_conflict):
    """Run the queue. Returns dict(done, skipped, failed, conflict, stopped). A finished run whose
    stored config differs from the line is reported as CONFLICT: not skipped silently, not rerun."""
    root = Path(root)
    log = log or Log(root / "logs" / "queue.log")
    train_cmd = train_cmd or [sys.executable, str(root / "scripts" / "train.py")]
    res = {"done": [], "skipped": [], "failed": [], "conflict": [], "stopped": None}
    log(f"queue start: {len(runs)} runs")
    for argv in runs:
        out = out_dir_of(argv)
        if is_done(argv, root):
            msg = conflict_fn(argv, root)
            if msg:
                log(f"CONFLICT {out}: finished run has a different config than the queue line "
                    f"(not skipped, not rerun; use another --out or delete it): {msg}")
                res["conflict"].append(str(out))
                continue
            log(f"SKIP {out} (evaluation.json exists)")
            res["skipped"].append(str(out))
            continue
        g = free_gb_fn(root)
        if g < min_free_gb:
            log(f"STOP: only {g:.1f} GB free (< {min_free_gb} GB). Free disk space and relaunch.")
            print(f"WARNING: free disk {g:.1f} GB < {min_free_gb} GB, queue stopped", file=sys.stderr)
            res["stopped"] = "disk"
            return res
        while True:
            busy = gpu_busy_fn()
            if not busy:
                break
            log(f"WAIT: GPU in use by {busy}; retry in {poll_s}s")
            sleep(poll_s)
        (root / out).mkdir(parents=True, exist_ok=True)
        cmd = train_cmd + ["--config", argv[0]] + argv[1:]
        log(f"START {out}: {' '.join(cmd)}")
        t0 = time.time()
        with open(root / out / "train.log", "a") as lf:
            lf.write(f"# {datetime.now().isoformat(timespec='seconds')} {' '.join(cmd)}\n")
            lf.flush()
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=root).returncode
        dt = (time.time() - t0) / 3600
        if rc == 0 and is_done(argv, root):
            log(f"DONE {out} ({dt:.2f} h)")
            res["done"].append(str(out))
        else:
            log(f"FAILED {out} rc={rc} ({dt:.2f} h); see {out}/train.log")
            res["failed"].append(str(out))
    log(f"queue end: done={len(res['done'])} skipped={len(res['skipped'])} failed={len(res['failed'])} conflict={len(res['conflict'])}")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("queue_file")
    ap.add_argument("--epochs", type=int, help="overrides EPOCHS in the file")
    ap.add_argument("--var", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--dry-run", action="store_true", help="print resolved commands and status")
    args = ap.parse_args(argv)
    ov = dict(kv.split("=", 1) for kv in args.var)
    if args.epochs is not None:
        ov["EPOCHS"] = str(args.epochs)
    _, runs = parse_queue(Path(args.queue_file).read_text(), ov)
    if args.dry_run:
        for r in runs:
            tag = "TODO "
            if is_done(r):
                tag = "CONFLICT" if check_conflict(r) else "DONE "
            print(tag + " " + " ".join(r))
        return 0
    res = run_queue(runs)
    if res["conflict"]:
        print("CONFLICTS (finished runs with a different config):", *res["conflict"], sep="\n  ", file=sys.stderr)
    return 3 if res["stopped"] else (4 if res["conflict"] else (1 if res["failed"] else 0))


if __name__ == "__main__":
    sys.exit(main())
