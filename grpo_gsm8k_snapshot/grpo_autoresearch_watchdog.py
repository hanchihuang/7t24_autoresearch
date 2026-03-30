#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO = Path("/home/user/图片")
SKILL_ROOT = REPO / ".agents" / "skills" / "codex-autoresearch" / "scripts"
RUNTIME_CTL = SKILL_ROOT / "autoresearch_runtime_ctl.py"
STATUS_CMD = ["python3", str(RUNTIME_CTL), "status", "--repo", str(REPO)]
START_CMD = ["python3", str(RUNTIME_CTL), "start", "--repo", str(REPO)]
STOP_CMD = ["python3", str(RUNTIME_CTL), "stop", "--repo", str(REPO)]

POLL_SECONDS = 300
STALL_SECONDS = 90 * 60
RESTART_COOLDOWN_SECONDS = 15 * 60

STATE_PATH = REPO / "grpo_watchdog_state.json"
LOG_PATH = REPO / "grpo_watchdog.log"
PID_PATH = REPO / "grpo_watchdog.pid"
RESULTS_PATH = REPO / "research-results.tsv"
RUN_STATE_PATH = REPO / "autoresearch-state.json"
RUNTIME_LOG_PATH = REPO / "autoresearch-runtime.log"


@dataclass
class WatchdogState:
    last_restart_at: float = 0.0
    last_seen_iteration: int = 0
    last_seen_status: str = ""
    last_progress_at: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "last_restart_at": self.last_restart_at,
            "last_seen_iteration": self.last_seen_iteration,
            "last_seen_status": self.last_seen_status,
            "last_progress_at": self.last_progress_at,
        }


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_state() -> WatchdogState:
    if not STATE_PATH.exists():
        return WatchdogState(last_progress_at=time.time())
    try:
        raw = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return WatchdogState(last_progress_at=time.time())
    return WatchdogState(
        last_restart_at=float(raw.get("last_restart_at", 0.0) or 0.0),
        last_seen_iteration=int(raw.get("last_seen_iteration", 0) or 0),
        last_seen_status=str(raw.get("last_seen_status", "") or ""),
        last_progress_at=float(raw.get("last_progress_at", 0.0) or 0.0),
    )


def save_state(state: WatchdogState) -> None:
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATE_PATH)


def run_json(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"command failed: {cmd}")
    return json.loads(proc.stdout)


def read_run_iteration() -> tuple[int, str]:
    if not RUN_STATE_PATH.exists():
        return 0, ""
    try:
        payload = json.loads(RUN_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return 0, ""
    state = payload.get("state", {})
    return int(state.get("iteration", 0) or 0), str(state.get("last_status", "") or "")


def newest_activity_ts() -> float:
    candidates = []
    for path in (RESULTS_PATH, RUN_STATE_PATH, RUNTIME_LOG_PATH):
        if path.exists():
            candidates.append(path.stat().st_mtime)
    return max(candidates) if candidates else 0.0


def can_restart(state: WatchdogState) -> bool:
    return (time.time() - state.last_restart_at) >= RESTART_COOLDOWN_SECONDS


def start_runtime(state: WatchdogState, reason: str) -> None:
    payload = run_json(START_CMD)
    state.last_restart_at = time.time()
    save_state(state)
    log(f"restart_ok reason={reason} status={payload.get('status')} pid={payload.get('pid')}")


def stop_runtime(reason: str) -> None:
    payload = run_json(STOP_CMD)
    log(f"stop_ok reason={reason} status={payload.get('status')} pid={payload.get('pid')}")


def ensure_singleton() -> None:
    if PID_PATH.exists():
        try:
            old_pid = int(PID_PATH.read_text(encoding="utf-8").strip())
            os.kill(old_pid, 0)
        except Exception:
            pass
        else:
            raise SystemExit(f"watchdog already running: pid={old_pid}")
    PID_PATH.write_text(str(os.getpid()), encoding="utf-8")


def cleanup_pid(*_: Any) -> None:
    try:
        if PID_PATH.exists():
            PID_PATH.unlink()
    finally:
        raise SystemExit(0)


def main() -> int:
    ensure_singleton()
    signal.signal(signal.SIGTERM, cleanup_pid)
    signal.signal(signal.SIGINT, cleanup_pid)
    state = load_state()
    if state.last_progress_at <= 0:
        state.last_progress_at = time.time()
        save_state(state)
    log("watchdog_started")

    while True:
        try:
            status = run_json(STATUS_CMD)
            runtime_status = str(status.get("status", "unknown"))
            iteration, last_status = read_run_iteration()
            activity_ts = newest_activity_ts()
            now = time.time()

            if iteration > state.last_seen_iteration or last_status != state.last_seen_status:
                state.last_seen_iteration = iteration
                state.last_seen_status = last_status
                state.last_progress_at = now
                save_state(state)
                log(
                    f"progress iteration={iteration} last_status={last_status or '-'} "
                    f"runtime_status={runtime_status}"
                )

            if activity_ts and activity_ts > state.last_progress_at:
                state.last_progress_at = activity_ts
                save_state(state)

            stalled = (now - state.last_progress_at) >= STALL_SECONDS
            log(
                f"heartbeat runtime_status={runtime_status} iteration={iteration} "
                f"last_status={last_status or '-'} stalled={str(stalled).lower()}"
            )

            if runtime_status == "running":
                if stalled and can_restart(state):
                    log("stall_detected runtime_status=running action=restart")
                    stop_runtime("stalled_runtime")
                    start_runtime(state, "stalled_runtime")
                time.sleep(POLL_SECONDS)
                continue

            if runtime_status in {"needs_human", "stopped", "idle", "terminal"}:
                if can_restart(state):
                    log(f"runtime_issue status={runtime_status} action=restart")
                    start_runtime(state, f"status_{runtime_status}")
                else:
                    log(f"runtime_issue status={runtime_status} action=cooldown_skip")
                time.sleep(POLL_SECONDS)
                continue

            log(f"runtime_issue status={runtime_status} action=observe_only")
        except Exception as exc:
            log(f"watchdog_error error={exc}")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
