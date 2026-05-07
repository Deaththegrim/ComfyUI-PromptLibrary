#!/usr/bin/env python3
"""Live monitor for Smart Detailer runs.

Polls ComfyUI's /queue endpoint to detect when a workflow starts running,
samples the ComfyUI process's RSS + CPU + AMD GPU VRAM/temp every 0.5s,
and prints a per-run summary (duration, RSS peak, RSS delta, VRAM peak)
when the queue empties. Designed for the Smart Detailer bug-and-monitor
matrix — exposes RAM spikes that would otherwise OOM-kill silently.

Run alongside ComfyUI (no impact on the runtime), then queue your detailer
workflow and watch the per-run summary print to this terminal.

  python3 tools/detailer_monitor.py [--interval 0.5] [--port 8188]

Ctrl+C to exit.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path


def _get_comfy_pid(port: int) -> int | None:
    """Return the pid of the process listening on `port`. Most reliable
    detection — venv pythons resolve to /usr/bin/python3 via symlink, so
    cwd / exe / cmdline heuristics all miss. The port owner is unambiguous."""
    try:
        with os.popen(f"ss -tlnp 2>/dev/null | grep :{port} ") as p:
            for line in p:
                # Format: ... users:(("python",pid=12345,fd=34)) ...
                if "pid=" in line:
                    chunk = line.split("pid=", 1)[1]
                    pid_str = chunk.split(",", 1)[0].split(")", 1)[0]
                    if pid_str.isdigit():
                        return int(pid_str)
    except OSError:
        pass
    return None


def _proc_rss_kb(pid: int) -> int:
    try:
        for line in (Path("/proc") / str(pid) / "status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return 0


def _proc_cpu_pct(pid: int, prev_state: dict) -> float:
    """Approximate %CPU since last sample. prev_state is mutated."""
    try:
        stat = (Path("/proc") / str(pid) / "stat").read_text().split()
        utime, stime = int(stat[13]), int(stat[14])
        proc_total = utime + stime
        cpu_stat = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
        sys_total = sum(int(x) for x in cpu_stat)
        prev_proc, prev_sys = prev_state.get("proc", proc_total), prev_state.get("sys", sys_total)
        prev_state["proc"], prev_state["sys"] = proc_total, sys_total
        d_sys = sys_total - prev_sys
        if d_sys <= 0:
            return 0.0
        ncpu = os.cpu_count() or 1
        return 100.0 * (proc_total - prev_proc) / d_sys * ncpu
    except (OSError, ValueError, IndexError):
        return 0.0


def _amd_gpu_stats() -> tuple[int | None, int | None]:
    """Return (vram_used_mb, edge_temp_c) from /sys/class/drm/cardN — picks the
    first card with a hwmon entry (typically the dGPU)."""
    try:
        for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
            dev = card / "device"
            if not (dev / "hwmon").exists():
                continue
            vram = None
            temp = None
            try:
                vram = int((dev / "mem_info_vram_used").read_text().strip()) // (1024 * 1024)
            except (OSError, ValueError):
                pass
            for hwmon in (dev / "hwmon").iterdir():
                t1 = hwmon / "temp1_input"
                if t1.exists():
                    try:
                        temp = int(t1.read_text().strip()) // 1000
                    except (OSError, ValueError):
                        pass
                    break
            return vram, temp
    except OSError:
        pass
    return None, None


def _queue_state(port: int) -> tuple[int, int]:
    """(running, pending). 0/0 = idle. Returns (-1, -1) on connection failure."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/queue", timeout=1.5) as r:
            data = json.load(r)
            return (len(data.get("queue_running", [])),
                    len(data.get("queue_pending", [])))
    except Exception:
        return -1, -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--port", type=int, default=8188)
    args = ap.parse_args()

    pid = _get_comfy_pid(args.port)
    if pid is None:
        print("[monitor] no ComfyUI process found — start it first.", file=sys.stderr)
        return 1
    print(f"[monitor] watching ComfyUI pid {pid} on port {args.port}, "
          f"sampling every {args.interval}s. Ctrl+C to exit.")
    print(f"[monitor] {'time':>8} {'rss_GB':>7} {'cpu%':>5} {'vram_MB':>8} {'temp_C':>6}  q_run/pend")
    print("[monitor] " + "-" * 56)

    cpu_state: dict = {}
    run_start: float | None = None
    rss_peak = 0
    vram_peak = 0
    rss_at_start = 0

    while True:
        try:
            now = time.monotonic()
            rss = _proc_rss_kb(pid)
            if rss == 0:  # process gone
                print("[monitor] ComfyUI process exited.")
                return 0
            cpu = _proc_cpu_pct(pid, cpu_state)
            vram, temp = _amd_gpu_stats()
            running, pending = _queue_state(args.port)

            # Per-run lifecycle: detect transition idle -> running and back.
            if running > 0 and run_start is None:
                run_start = now
                rss_at_start = rss
                rss_peak = rss
                vram_peak = vram or 0
                print(f"[monitor] === RUN STARTED (rss={rss/1024/1024:.2f} GB) ===")
            elif run_start is not None:
                rss_peak = max(rss_peak, rss)
                if vram is not None:
                    vram_peak = max(vram_peak, vram)
                if running == 0 and pending == 0:
                    dur = now - run_start
                    delta_gb = (rss_peak - rss_at_start) / 1024 / 1024
                    print(f"[monitor] === RUN DONE in {dur:.1f}s — "
                          f"rss peak={rss_peak/1024/1024:.2f} GB "
                          f"(+{delta_gb:.2f} GB delta) "
                          f"vram peak={vram_peak} MB ===")
                    run_start = None

            ts = time.strftime("%H:%M:%S")
            print(f"[monitor] {ts} {rss/1024/1024:7.2f} {cpu:5.1f} "
                  f"{(vram or 0):8d} {(temp or 0):6d}  {running}/{pending}",
                  flush=True)
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n[monitor] stopped.")
            return 0


if __name__ == "__main__":
    sys.exit(main())
