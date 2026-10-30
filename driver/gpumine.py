#!/usr/bin/env python3
"""Argonmine GPU miner driver.

Keeps one CUDA miner process alive and feeds it work through a file, so a new
block (new template) costs a file rewrite instead of a process restart.

    python3 gpumine.py --address arg1... [--api URL] [--batch 1024]

job file format (three lines): job_id, header_prefix (224 hex), target (hex).
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request


def http_json(url, data=None, timeout=20):
    req = urllib.request.Request(url, method="POST" if data is not None else "GET")
    body = None
    if data is not None:
        body = json.dumps(data).encode()
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, body, timeout=timeout) as r:
        return json.loads(r.read().decode())


def write_job(path, job):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(job["job_id"] + "\n" + job["header_prefix"] + "\n" + job["target"] + "\n")
    os.replace(tmp, path)


def reader(proc, args):
    for line in proc.stdout:
        line = line.rstrip()
        if line.startswith("HIT "):
            parts = line.split()
            job_id, nonce = parts[1], int(parts[2])
            print(f"*** hit job={job_id} nonce={nonce}", flush=True)
            try:
                res = http_json(args.api + "/api/work/submit",
                                {"job_id": job_id, "nonce": nonce, "address": args.address})
            except Exception as e:                                  # noqa: BLE001
                print(f"[submit] error {e}", flush=True)
                continue
            if res.get("accepted"):
                print(f"*** BLOCK ACCEPTED *** hash={res.get('hash')}", flush=True)
            else:
                print(f"[submit] {res}", flush=True)
        else:
            print(line, flush=True)


def supervise(args):
    """Keep the CUDA miner alive: it must outlive job swaps and host hiccups."""
    while True:
        proc = subprocess.Popen(
            [args.bin, "--jobfile", args.jobfile, "--batch", args.batch, "--seconds", "0"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        threading.Thread(target=reader, args=(proc, args), daemon=True).start()
        proc.wait()
        print("[miner] exited (rc=%s), restarting in 3s" % proc.returncode, flush=True)
        time.sleep(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", required=True, help="arg1... payout address")
    ap.add_argument("--api", default="https://explorer.argonmine.com")
    ap.add_argument("--bin", default=os.path.join(os.path.dirname(__file__), "..", "argon_gpu"))
    ap.add_argument("--jobfile", default="/tmp/argon_job.txt")
    ap.add_argument("--batch", default="1024")
    ap.add_argument("--interval", type=float, default=2.0, help="tip poll interval, seconds")
    args = ap.parse_args()

    # 0 = run until stopped: the miner must outlive job swaps
    threading.Thread(target=supervise, args=(args,), daemon=True).start()

    last_height, last_fetch = None, 0.0
    while True:
        now = time.time()
        height = None
        try:
            height = http_json(args.api + "/api/health", timeout=10).get("indexedHeight")
        except Exception:                                           # noqa: BLE001
            pass
        if height != last_height or now - last_fetch > 30:
            try:
                job = http_json(args.api + "/api/work?address=" + urllib.parse.quote(args.address))
                if job.get("header_prefix"):
                    write_job(args.jobfile, job)
                    last_height, last_fetch = height, now
                    print(f"[job] height={job.get('height')} target={job['target'][:12]}", flush=True)
                else:
                    print(f"[work] {job}", flush=True)
            except Exception as e:                                  # noqa: BLE001
                print(f"[work] fetch failed: {e}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
