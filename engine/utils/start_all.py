#!/usr/bin/env python3
import signal
import subprocess
import sys
import time

COMMANDS = [
    ["uv", "run", "gabber/main.py", "editor"],
    ["uv", "run", "gabber/main.py", "repository"],
    ["uv", "run", "gabber/main.py", "engine"],
]

processes = []
stop_requested = False
exit_code = 0

def request_stop(signum=None, frame=None):
    global stop_requested
    stop_requested = True

def shutdown():
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()
    deadline = time.time() + 10
    while any(proc.poll() is None for proc in processes) and time.time() < deadline:
        time.sleep(0.2)
    for proc in processes:
        if proc.poll() is None:
            proc.kill()
    for proc in processes:
        try:
            proc.wait()
        except Exception:
            pass

signal.signal(signal.SIGTERM, request_stop)
signal.signal(signal.SIGINT, request_stop)

try:
    for cmd in COMMANDS:
        processes.append(subprocess.Popen(cmd))
    while processes and not stop_requested:
        for proc in list(processes):
            rc = proc.poll()
            if rc is not None:
                processes.remove(proc)
                proc.wait()
                if rc != 0 and exit_code == 0:
                    exit_code = rc
                    stop_requested = True
        if processes and not stop_requested:
            time.sleep(0.2)
finally:
    shutdown()

sys.exit(exit_code)
